import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class RagEngineConfig(AppConfig):
    name = 'rag_engine'

    def ready(self):
        """Warm the RAG service at process start for real server processes only.

        Building RagService loads torch + the sentence-transformers embedding
        model (~10-15s). Doing it here means the first user request doesn't pay
        that cost. We deliberately skip it for management commands (migrate,
        collectstatic, test, shell, makemigrations, ...) so those stay fast.

        Set RAG_WARMUP=0 to disable (handy for fast runserver auto-reloads).
        """
        if os.environ.get("RAG_WARMUP", "1") == "0":
            return

        argv = sys.argv
        prog = os.path.basename(argv[0]) if argv else ""
        is_runserver = "runserver" in argv
        is_wsgi_server = prog.startswith(("gunicorn", "uvicorn", "daphne"))
        if not (is_runserver or is_wsgi_server):
            return

        # Under `runserver` with auto-reload, ready() runs in both the reloader
        # parent and the worker child. Only the child sets RUN_MAIN=="true";
        # warm there (or when --noreload makes it a single process).
        if is_runserver and "--noreload" not in argv and os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from .views import get_rag_service
            get_rag_service()
            logger.info("RAG service warmed up on boot.")
        except Exception as exc:  # best-effort - don't block startup on failure
            logger.warning(f"RAG warm-up on boot failed: {exc}")
