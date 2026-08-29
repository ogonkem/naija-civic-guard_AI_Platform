"""Per-request latency / throughput metrics for the RAG pipeline.

A single ``RequestMetrics`` dataclass is created at the start of each request,
populated incrementally as the request moves through its stages (embedding ->
retrieval -> generation), and written exactly once at the end from a ``finally``
block so a row is persisted even when the request errors partway through.

Persistence is a single inline INSERT (see ``persist()``); it is deliberately
NOT deferred to a background worker. Only the downstream eval step is async.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    # Generated at the very start of the request. Downstream async eval results
    # are joined back to this row by request_id.
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    query_text: str = ""
    provider: str = ""          # "groq" / "ollama" / "openai" / ...
    model: str = ""

    # All stage timings in milliseconds. None means "stage did not run".
    embedding_time_ms: float | None = None      # query-vector embedding (agent folds this into retrieve_ms)
    retrieval_time_ms: float | None = None      # kept for continuity == retrieve_ms
    generation_time_ms: float | None = None     # LLM call (streamed end-to-end)
    total_time_ms: float | None = None          # whole request, set in finally

    # --- LangGraph retrieval agent (classify -> retrieve -> chain -> verify) ---
    classify_label: str = ""                    # direct_lookup | cross_reference | interpretive
    retrieval_calls: int | None = None          # total retrieve() calls; >1 when chaining fires
    verify_retry: bool = False                  # did the verify node trigger its one reformulated retry
    classify_ms: float | None = None            # per-node latency (cumulative if a node runs twice)
    retrieve_ms: float | None = None
    chain_ms: float | None = None
    verify_ms: float | None = None

    # One entry per MCP tool call the retrieve/chain nodes made, in order:
    # {"tool_name", "tool_latency_ms", "ok", "error"}. Persisted as a JSON
    # column on the same rag_request_metrics row (nested, not a separate table).
    tool_calls: list = field(default_factory=list)

    tokens_generated: int | None = None
    # True when tokens_generated is a whitespace-split estimate because the
    # provider returned no exact usage count on the streamed response.
    tokens_generated_is_estimate: bool = False
    tokens_per_second: float | None = None

    error: str | None = None    # "ExceptionType: message" when the request failed

    # --- Not persisted to rag_request_metrics. Captured during the request so
    # the gateway can hand this off to the async evaluator (see rag_engine.tasks)
    # without re-parsing the response stream. ---
    retrieved_section_ids: list = field(default_factory=list, repr=False)
    retrieved_contexts: list = field(default_factory=list, repr=False)
    response_text: str = field(default="", repr=False)

    def record_tool_call(self, tool_name: str, tool_latency_ms: float,
                         ok: bool, error: str | None = None) -> None:
        self.tool_calls.append({
            "tool_name": tool_name,
            "tool_latency_ms": round(tool_latency_ms, 2),
            "ok": bool(ok),
            "error": error,
        })

    def set_tokens_from_text(self, text: str, exact_count: int | None = None) -> None:
        """Record output token count, preferring a provider-reported exact value."""
        if exact_count is not None:
            self.tokens_generated = int(exact_count)
            self.tokens_generated_is_estimate = False
        else:
            # Estimate: whitespace split. Cheap and provider-agnostic; good
            # enough for a throughput signal. Off from true BPE tokens by
            # roughly +25-35% for English prose.
            self.tokens_generated = len(text.split())
            self.tokens_generated_is_estimate = True

    def finalize(self) -> "RequestMetrics":
        """Derive tokens_per_second. Idempotent; safe to call more than once."""
        if (
            self.tokens_per_second is None
            and self.tokens_generated
            and self.generation_time_ms
        ):
            self.tokens_per_second = self.tokens_generated / (self.generation_time_ms / 1000.0)
        return self

    def persist(self) -> None:
        """Write this row once. A single SQLite/Postgres INSERT, done inline.

        Never raises: a metrics failure must not break the user's response.
        """
        from .models import RequestMetric

        try:
            RequestMetric.objects.create(
                request_id=self.request_id,
                timestamp=self.timestamp,
                query_text=self.query_text,
                provider=self.provider,
                model=self.model,
                embedding_time_ms=self.embedding_time_ms,
                retrieval_time_ms=self.retrieval_time_ms,
                generation_time_ms=self.generation_time_ms,
                total_time_ms=self.total_time_ms,
                tokens_generated=self.tokens_generated,
                tokens_generated_is_estimate=self.tokens_generated_is_estimate,
                tokens_per_second=self.tokens_per_second,
                error=self.error or "",
                classify_label=self.classify_label or "",
                retrieval_calls=self.retrieval_calls,
                verify_retry=self.verify_retry,
                classify_ms=self.classify_ms,
                retrieve_ms=self.retrieve_ms,
                chain_ms=self.chain_ms,
                verify_ms=self.verify_ms,
                tool_calls=self.tool_calls,
            )
        except Exception:
            logger.exception("Failed to persist RequestMetrics %s", self.request_id)
        else:
            tools_summary = ",".join(
                f"{t['tool_name']}:{_fmt(t['tool_latency_ms'])}ms{'' if t['ok'] else '!'}"
                for t in self.tool_calls
            ) or "-"
            logger.info(
                "metrics request_id=%s provider=%s model=%s classify=%s retrieval_calls=%s "
                "verify_retry=%s classify_ms=%s retrieve_ms=%s chain_ms=%s verify_ms=%s "
                "mcp_tools=[%s] generation_ms=%s total_ms=%s tokens=%s tok_per_s=%s estimate=%s error=%s",
                self.request_id, self.provider, self.model, self.classify_label,
                self.retrieval_calls, self.verify_retry,
                _fmt(self.classify_ms), _fmt(self.retrieve_ms), _fmt(self.chain_ms),
                _fmt(self.verify_ms), tools_summary,
                _fmt(self.generation_time_ms), _fmt(self.total_time_ms),
                self.tokens_generated, _fmt(self.tokens_per_second),
                self.tokens_generated_is_estimate, bool(self.error),
            )


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"
