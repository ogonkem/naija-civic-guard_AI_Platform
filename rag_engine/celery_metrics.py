"""Prometheus for the Celery eval worker.

The worker is its own process, so django-prometheus' /metrics on the gateway
can't see it. On worker start we:
  * expose the worker's registry over HTTP on WORKER_METRICS_PORT (default 9540)
  * poll the broker for the eval queue depth into celery_queue_depth
Prometheus scrapes this target in addition to the gateway.
"""

import logging
import os
import threading
import time

from celery.signals import worker_ready

from .metrics_prom import celery_queue_depth

logger = logging.getLogger(__name__)

_PORT = int(os.getenv("WORKER_METRICS_PORT", "9540"))
_QUEUE = os.getenv("CELERY_TASK_DEFAULT_QUEUE", "eval")
_started = False


def _poll_queue_depth():
    import redis  # celery[redis] dependency
    url = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(url)
    while True:
        try:
            celery_queue_depth.set(client.llen(_QUEUE))
        except Exception as exc:  # noqa: BLE001
            logger.debug("queue-depth poll failed: %s", exc)
        time.sleep(int(os.getenv("QUEUE_DEPTH_POLL_SECONDS", "5")))


@worker_ready.connect
def _start_worker_metrics(**_):
    global _started
    if _started:
        return
    _started = True
    from prometheus_client import start_http_server

    start_http_server(_PORT)
    threading.Thread(target=_poll_queue_depth, name="queue-depth", daemon=True).start()
    logger.info("worker Prometheus metrics on :%d (queue=%s)", _PORT, _QUEUE)
