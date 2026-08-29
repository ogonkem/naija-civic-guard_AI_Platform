import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from django.http import StreamingHttpResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .authentication import ApiKeyAuthentication
from .metrics import RequestMetrics
from .metrics_prom import record_request_metrics
from .throttling import ApiKeyRateThrottle

logger = logging.getLogger(__name__)

# Lazily-created singleton. Building RagService imports torch / sentence-
# transformers and touches the network, so we must NOT do it at import time -
# and we defer the `services` import itself so manage.py commands, migrations
# and tests never pay that ~15s import cost.
_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        from .services import RagService
        _rag_service = RagService()
    return _rag_service


def _browser_api_key() -> str:
    """API key the bundled browser chat page authenticates with.

    Operator can pin one via settings.BROWSER_API_KEY; otherwise a dedicated
    'browser-ui' key is auto-provisioned on first page load so the page just
    works after `migrate`. It's rate-limited like any other key and can be
    deactivated in the admin to lock the page down.
    """
    from django.conf import settings
    from .models import ApiKey

    pinned = getattr(settings, "BROWSER_API_KEY", "") or ""
    if pinned:
        return pinned
    key = (ApiKey.objects.filter(owner="browser-ui", is_active=True)
           .order_by("created_at").first())
    if key is None:
        key = ApiKey.objects.create(owner="browser-ui", requests_per_minute=30)
    return key.key


def chat_page(request):
    """Renders the chat interface, embedding the browser's API key."""
    return render(request, "chat.html", {"browser_api_key": _browser_api_key()})


class ChatView(APIView):
    """Streams RAG answers as newline-delimited JSON.

    Gateway: X-API-Key auth (401 if missing/invalid) + per-key rate limit
    (429 over the limit). Past those, behaviour is unchanged - one metadata
    line, token lines, a final ``done`` line, and a RequestMetrics row.
    ``X-Request-ID`` response header carries the metrics request_id so the
    audit-log middleware can join the two.
    """

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [ApiKeyRateThrottle]

    def post(self, request):
        user_query = request.data.get("query")

        if not user_query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)

        service = get_rag_service()
        metrics = RequestMetrics(
            query_text=user_query,
            provider=service.llm_provider,
            model=service.llm_model,
        )

        def event_stream():
            request_start = time.perf_counter()
            try:
                for line in service.query_stream(user_query, metrics=metrics):
                    yield line

                metrics.total_time_ms = (time.perf_counter() - request_start) * 1000.0
                metrics.finalize()
                yield json.dumps({
                    "type": "done",
                    "request_id": metrics.request_id,
                    "duration": round((metrics.total_time_ms or 0) / 1000.0, 4),
                    "timings_ms": {
                        "classify": _round(metrics.classify_ms),
                        "retrieve": _round(metrics.retrieve_ms),
                        "chain": _round(metrics.chain_ms),
                        "verify": _round(metrics.verify_ms),
                        "generation": _round(metrics.generation_time_ms),
                        "total": _round(metrics.total_time_ms),
                    },
                    "agent": {
                        "classify_label": metrics.classify_label or None,
                        "retrieval_calls": metrics.retrieval_calls,
                        "verify_retry": metrics.verify_retry,
                    },
                    "mcp_tool_calls": metrics.tool_calls,
                    "tokens_generated": metrics.tokens_generated,
                    "tokens_generated_is_estimate": metrics.tokens_generated_is_estimate,
                    "tokens_per_second": _round(metrics.tokens_per_second),
                }) + "\n"

            except Exception as e:  # noqa: BLE001 - record then surface
                metrics.error = f"{type(e).__name__}: {e}"
                metrics.total_time_ms = (time.perf_counter() - request_start) * 1000.0
                metrics.finalize()
                logger.exception("chat stream failed (request_id=%s)", metrics.request_id)
                try:
                    yield json.dumps({
                        "type": "error", "error": str(e), "request_id": metrics.request_id,
                    }) + "\n"
                except Exception:
                    pass
            finally:
                if metrics.total_time_ms is None:  # client disconnected mid-stream
                    metrics.total_time_ms = (time.perf_counter() - request_start) * 1000.0
                    metrics.finalize()
                metrics.persist()               # Phase 2a: single inline INSERT
                record_request_metrics(metrics)  # Phase 7: same numbers -> Prometheus
                _enqueue_eval(metrics)          # Phase 2b: fire-and-forget hand-off

        response = StreamingHttpResponse(event_stream(), content_type="application/x-ndjson")
        # Defeat proxy/browser buffering so tokens arrive as they are produced.
        response["X-Accel-Buffering"] = "no"
        response["Cache-Control"] = "no-cache"
        # Join key for the audit-log middleware <-> RequestMetrics row.
        response["X-Request-ID"] = metrics.request_id
        return response


def _round(v, ndigits=2):
    return None if v is None else round(v, ndigits)


# Small bounded pool so the request thread never does broker I/O itself. If
# Redis is unreachable the publish blocks *this* thread (bounded to 4), not the
# request; the request already returned. Tasks that can't be queued are logged
# and dropped - a missing eval_results row is a normal, non-fatal state.
_enqueue_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="eval-enqueue")


def _publish_eval(payload, request_id):
    try:
        from .tasks import evaluate_request_task

        evaluate_request_task.apply_async(kwargs=payload, retry=False)
    except Exception:
        logger.warning(
            "eval task not enqueued for request_id=%s (broker unavailable?)",
            request_id, exc_info=True,
        )


def _enqueue_eval(metrics):
    """Hand the request off to the async evaluator. Fire-and-forget.

    Called from the stream's finally block - after the whole response body is
    sent and the metrics row is written - and it does not even do the broker
    publish inline: that is handed to a background thread so the request path
    cannot block on Redis regardless of whether Redis is up, slow or down.
    """
    payload = {
        "request_id": metrics.request_id,
        "query": metrics.query_text,
        "retrieved_context": metrics.retrieved_contexts,
        "retrieved_section_ids": metrics.retrieved_section_ids,
        "response_text": metrics.response_text,
    }
    try:
        _enqueue_pool.submit(_publish_eval, payload, metrics.request_id)
    except Exception:  # pool full / shutting down - never fatal
        logger.warning("could not submit eval enqueue for request_id=%s",
                       metrics.request_id, exc_info=True)
