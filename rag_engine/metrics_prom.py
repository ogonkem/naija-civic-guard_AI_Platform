"""Custom Prometheus metrics, defined alongside django-prometheus' automatic
request/response series.

The request-side metrics are fed from the SAME RequestMetrics object that the
Postgres row is written from (rag_engine.views) - latency and throughput are
computed once, in rag_engine.services, then observed here. Nothing is timed
twice.

The gateway process serves these at /metrics (django-prometheus). The Celery
worker is a separate process, so its eval metrics live in ITS registry and it
runs its own tiny metrics HTTP server (see rag_engine.celery_metrics);
Prometheus scrapes both targets.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- request path (gateway) ------------------------------------------------- #
request_latency_seconds = Histogram(
    "request_latency_seconds",
    "RAG request latency by pipeline stage",
    ["stage"],  # embedding | retrieval | generation | total
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 60),
)
generation_tokens_per_second = Histogram(
    "generation_tokens_per_second",
    "LLM output tokens per second",
    ["provider"],  # ollama | openai
    buckets=(10, 25, 50, 100, 150, 200, 300, 450, 600, 900, 1200),
)
mcp_tool_calls_total = Counter(
    "mcp_tool_calls_total", "MCP tool calls made by the retrieve/chain nodes", ["tool_name"],
)
agent_retries_total = Counter(
    "agent_retries_total", "Times the verify node triggered its one reformulated retry",
)
llm_provider_requests_total = Counter(
    "llm_provider_requests_total", "Generation requests by LLM provider", ["provider"],
)

# --- async eval (Celery worker registry) ---------------------------------- #
eval_keyword_coverage = Gauge(
    "eval_keyword_coverage", "Keyword coverage of the most recent eval-set query",
)
celery_eval_task_duration_seconds = Histogram(
    "celery_eval_task_duration_seconds", "Wall time of evaluate_request_task itself",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
celery_queue_depth = Gauge(
    "celery_queue_depth", "Eval jobs waiting in the broker queue",
)
celery_eval_task_failures_total = Counter(
    "celery_eval_task_failures_total", "evaluate_request_task failures",
)


def normalize_provider(p: str) -> str:
    """Collapse the concrete client name to the ollama|openai label the
    dashboards split on. Groq is an OpenAI-compatible hosted API."""
    p = (p or "").lower()
    if "ollama" in p:
        return "ollama"
    return "openai"


def record_request_metrics(m) -> None:
    """Push one finished request's already-computed numbers into Prometheus.
    Call once, right where the Postgres RequestMetric row is written."""
    provider = normalize_provider(m.provider)
    for stage, ms in (
        ("embedding", m.embedding_time_ms),
        ("retrieval", m.retrieve_ms),
        ("generation", m.generation_time_ms),
        ("total", m.total_time_ms),
    ):
        if ms is not None:
            request_latency_seconds.labels(stage=stage).observe(ms / 1000.0)

    if m.tokens_per_second:
        generation_tokens_per_second.labels(provider=provider).observe(m.tokens_per_second)

    llm_provider_requests_total.labels(provider=provider).inc()

    for tc in m.tool_calls or []:
        mcp_tool_calls_total.labels(tool_name=tc.get("tool_name", "unknown")).inc()

    if m.verify_retry:
        agent_retries_total.inc()
