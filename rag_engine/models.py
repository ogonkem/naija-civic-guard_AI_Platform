import uuid

from django.db import models


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
