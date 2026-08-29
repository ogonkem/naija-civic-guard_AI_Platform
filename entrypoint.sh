#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Applying database migrations..."
python manage.py migrate

echo "collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn server..."
# --threads matters: a streaming response holds its worker for the whole
# generation, so sync workers alone would serialize all users.
exec gunicorn civic_guard.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --threads 4 \
    --timeout 120