"""Celery application for civic_guard.

Broker + result backend are Redis (see CELERY_* in settings.py). Used only for
the asynchronous RAG evaluation pipeline (rag_engine.tasks) - the request path
itself never waits on Celery.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "civic_guard.settings")

app = Celery("civic_guard")

# All Celery settings live in Django settings under the CELERY_ namespace.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Discover tasks.py in every installed app (finds rag_engine/tasks.py).
app.autodiscover_tasks()

# Register the worker's Prometheus metrics HTTP server + queue-depth poller
# (connects a worker_ready signal handler).
import rag_engine.celery_metrics  # noqa: E402,F401
