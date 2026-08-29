"""Asynchronous RAG evaluation, decoupled from the request path.

The gateway enqueues one `evaluate_request_task` per request *after* the HTTP
response and the synchronous metrics row are already done. If Redis / the
worker is unavailable the enqueue fails softly (see rag_engine.views) and this
task simply never runs - the request's eval_results row stays absent, which is
a normal state, not an error.
"""

import logging
import time

from celery import shared_task

from .eval_core import evaluate_retrieval
from .metrics_prom import (
    celery_eval_task_duration_seconds,
    celery_eval_task_failures_total,
    eval_keyword_coverage,
)

logger = logging.getLogger(__name__)


@shared_task(
    name="rag_engine.evaluate_request_task",
    bind=True,
    acks_late=True,
    max_retries=0,          # keep it simple: a failure means "no row", nothing more
)
def evaluate_request_task(self, request_id, query, retrieved_context,
                          retrieved_section_ids, response_text):
    """Score one already-executed RAG request and store it in eval_results.

    Reuses rag_engine.eval_core (same code the offline retrieval_eval.py CLI
    uses) - no retrieval or LLM call happens here, only scoring of the payload.
    """
    from .models import EvalResult  # lazy import: app registry must be ready

    t0 = time.perf_counter()
    try:
        scored = evaluate_retrieval(
            query=query,
            retrieved_context=retrieved_context,
            retrieved_section_ids=retrieved_section_ids,
            response_text=response_text,
        )

        EvalResult.objects.create(
            request_id=request_id,
            matched_ground_truth=scored["matched_ground_truth"],
            keyword_coverage=scored["keyword_coverage"],
            keyword_source=scored["keyword_source"],
            retrieved_section_ids=scored["retrieved_section_ids"],
            response_chars=scored["response_chars"],
            target_section=scored["target_section"] or "",
            hit=scored["hit"],
            reciprocal_rank=scored["reciprocal_rank"],
        )

        # gauge: last eval-set query's coverage (only when there's ground truth)
        if scored["matched_ground_truth"] and scored["keyword_coverage"] is not None:
            eval_keyword_coverage.set(scored["keyword_coverage"])

        logger.info(
            "eval ok request_id=%s matched_gt=%s coverage=%s hit=%s source=%s",
            request_id, scored["matched_ground_truth"],
            scored["keyword_coverage"], scored["hit"], scored["keyword_source"],
        )
        return {"request_id": str(request_id), "ok": True}

    except Exception:
        celery_eval_task_failures_total.inc()
        # Visible in the Celery worker log; the request itself is long gone and
        # unaffected.
        logger.exception("evaluate_request_task failed request_id=%s", request_id)
        raise
    finally:
        celery_eval_task_duration_seconds.observe(time.perf_counter() - t0)
