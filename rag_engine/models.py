import secrets
import uuid

from django.db import models


def generate_api_key() -> str:
    return "ncg_" + secrets.token_urlsafe(32)


class ApiKey(models.Model):
    """A single API key that authenticates callers of the gateway.

    Create one with `python manage.py create_api_key --owner "<name>"` or in
    the Django admin - never by hand-editing the DB.
    """

    key = models.CharField(max_length=64, unique=True, db_index=True, default=generate_api_key)
    owner = models.CharField(max_length=120, help_text="Owner / team name")
    is_active = models.BooleanField(default=True)
    # Per-key rate limit. NULL -> use the project default (REST_FRAMEWORK
    # DEFAULT_THROTTLE_RATES["api_key"]).
    requests_per_minute = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "api_keys"
        ordering = ["-created_at"]

    def __str__(self):
        state = "" if self.is_active else " (inactive)"
        return f"{self.owner}{state}"


class RequestAuditLog(models.Model):
    """One row per gateway request, written by AuditLogMiddleware (Django ORM).

    `request_id` is the same UUID as the RequestMetric row for successful
    requests, so audit log and metrics join on it. 401/429 requests never
    reach the agent, so they have no request_id and no RequestMetric row.
    """

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    api_key = models.ForeignKey(
        ApiKey, null=True, blank=True, on_delete=models.SET_NULL, related_name="audit_logs"
    )
    api_key_owner = models.CharField(max_length=120, blank=True, default="")
    # For a rejected/unknown key we can't FK it; keep a short hint instead.
    api_key_hint = models.CharField(max_length=20, blank=True, default="")

    method = models.CharField(max_length=8)
    endpoint = models.CharField(max_length=255, db_index=True)
    status_code = models.PositiveSmallIntegerField()
    # Same UUID as the RequestMetric row (both UUIDField, so they store
    # identically and join cleanly). NULL for 401/429 - no agent, no metrics.
    request_id = models.UUIDField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = "request_audit_log"
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.method} {self.endpoint} -> {self.status_code}"


class RequestMetric(models.Model):
    """One row per RAG request: latency breakdown + throughput.

    Written once at the end of each request (inline INSERT). The async eval
    step (separate pipeline) joins its results back here on ``request_id``.
    """

    request_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    timestamp = models.DateTimeField(db_index=True)

    query_text = models.TextField()
    provider = models.CharField(max_length=32, db_index=True)   # groq / ollama / openai / ...
    model = models.CharField(max_length=128, db_index=True)

    embedding_time_ms = models.FloatField(null=True, blank=True)
    retrieval_time_ms = models.FloatField(null=True, blank=True)
    generation_time_ms = models.FloatField(null=True, blank=True)
    total_time_ms = models.FloatField(null=True, blank=True)

    tokens_generated = models.IntegerField(null=True, blank=True)
    tokens_generated_is_estimate = models.BooleanField(default=False)
    tokens_per_second = models.FloatField(null=True, blank=True)

    # LangGraph retrieval agent (classify -> retrieve -> chain -> verify)
    classify_label = models.CharField(max_length=32, blank=True, default="", db_index=True)
    retrieval_calls = models.IntegerField(null=True, blank=True)  # >1 once chaining fires
    verify_retry = models.BooleanField(default=False)
    classify_ms = models.FloatField(null=True, blank=True)
    retrieve_ms = models.FloatField(null=True, blank=True)
    chain_ms = models.FloatField(null=True, blank=True)
    verify_ms = models.FloatField(null=True, blank=True)

    # Nested per-MCP-tool-call log on the same row (not a separate table):
    # [{"tool_name", "tool_latency_ms", "ok", "error"}, ...]
    tool_calls = models.JSONField(default=list, blank=True)

    error = models.TextField(blank=True, default="")

    class Meta:
        db_table = "rag_request_metrics"
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.request_id} {self.provider}/{self.model} {self.total_time_ms:.0f}ms" \
            if self.total_time_ms is not None else str(self.request_id)


class EvalResult(models.Model):
    """One row per evaluated request, written by the async Celery task.

    Deliberately a SEPARATE table from RequestMetric: the request path and the
    eval worker are different processes, so they each own their own row and we
    join on ``request_id`` rather than have two writers touch one row.

    A request with no row here just means "not evaluated (yet / worker down /
    task failed)" - never an error condition for anything else.
    """

    request_id = models.UUIDField(db_index=True)  # -> RequestMetric.request_id
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    # Present for every request.
    matched_ground_truth = models.BooleanField(default=False)
    keyword_coverage = models.FloatField(null=True, blank=True)
    keyword_source = models.CharField(max_length=16, blank=True, default="")  # ground_truth | query
    retrieved_section_ids = models.JSONField(default=list, blank=True)
    response_chars = models.IntegerField(null=True, blank=True)

    # Only when the query has ground truth in evaluation_set.jsonl; else null.
    target_section = models.CharField(max_length=64, blank=True, default="")
    hit = models.BooleanField(null=True, blank=True)
    reciprocal_rank = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "eval_results"
        ordering = ["-created_at"]

    def __str__(self):
        return f"eval({self.request_id}) coverage={self.keyword_coverage} hit={self.hit}"
