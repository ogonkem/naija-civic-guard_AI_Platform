#!/bin/sh
set -e

wait_for() {  # host port name
    echo "Waiting for $3 at $1:$2..."
    until python -c "import socket,sys
try:
    s=socket.socket(); s.settimeout(2)
    sys.exit(0 if s.connect_ex(('$1', int('$2')))==0 else 1)
except OSError:
    sys.exit(1)"; do
        sleep 1
    done
    echo "  $3 up."
}

[ -n "$POSTGRES_HOST" ] && wait_for "$POSTGRES_HOST" "${POSTGRES_PORT:-5432}" Postgres
[ -n "$CHROMA_HOST" ]   && wait_for "$CHROMA_HOST" "${CHROMA_PORT:-8000}" ChromaDB

# The Celery worker container passes its command as args - run it directly.
# Only migrations need to have run (the gateway does that); the worker skips
# migrate / collectstatic / gunicorn.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "Applying database migrations..."
python manage.py migrate --noinput

if [ "${AUTO_INGEST:-0}" = "1" ]; then
    python manage.py ingest --if-empty
fi

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn..."
# One worker + threads: the gateway is I/O-bound (streams, waits on the LLM),
# and a single process keeps the prometheus_client registry consistent for
# /metrics without needing multiprocess mode.
exec gunicorn civic_guard.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 8 \
    --timeout 120
