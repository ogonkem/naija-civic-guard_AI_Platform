"""LangGraph retrieval agent: classify -> retrieve -> chain -> verify.

Replaces the old single-shot "embed + similarity_search" step. The generation
step (streaming the answer over the retrieved context) is unchanged and still
lives in RagService.

    classify  cheap/fast LLM call -> direct_lookup | cross_reference | interpretive
    retrieve  existing hybrid retrieval (ChromaDB vector + BM25)
    chain     if a retrieved chunk references another section, retrieve that too
    verify    lightweight self-check; at most ONE reformulated retry

The graph is built once per RagService and compiled. `run(query)` returns the
final state; `RagService` reads `docs` / `sources` off it and copies the
per-node timings + labels onto the request's RequestMetrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .sections import find_section_references

logger = logging.getLogger(__name__)

CLASSES = ("direct_lookup", "cross_reference", "interpretive")
MAX_CHAINED_SECTIONS = 2   # cap chain fan-out so latency stays bounded
MAX_VERIFY_RETRIES = 1     # never loop forever


class AgentState(TypedDict, total=False):
    query: str                 # current (possibly reformulated) query
    original_query: str        # what the user actually asked
    classification: str
    docs: list                 # list[langchain_core.documents.Document]
    sources: list              # list[str] section labels, deduped
    retrieval_calls: int       # total retrieve() calls (>1 once chaining fires)
    chained_sections: list     # section labels pulled in by the chain node
    needs_retry: bool
    retry_count: int
    verify_retry: bool         # did the verify step ever trigger a retry
    # cumulative per-node latency (ms); a node that runs twice adds up
    classify_ms: float
    retrieve_ms: float
    chain_ms: float
    verify_ms: float


def _dedupe_sources(docs) -> list:
    out, seen = [], set()
    for d in docs:
        s = d.metadata.get("section")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #
def _classify_node(state: AgentState, service) -> dict[str, Any]:
    t0 = time.perf_counter()
    label = service.classify_query(state["query"])
    return {
        "classification": label,
        "classify_ms": state.get("classify_ms", 0.0) + (time.perf_counter() - t0) * 1000.0,
    }


def _retrieve_node(state: AgentState, service) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = service.retrieve(state["query"])
    return {
        "docs": docs,
        "sources": _dedupe_sources(docs),
        "retrieval_calls": state.get("retrieval_calls", 0) + 1,
        "retrieve_ms": state.get("retrieve_ms", 0.0) + (time.perf_counter() - t0) * 1000.0,
    }


def _chain_node(state: AgentState, service) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = list(state["docs"])
    have = set(state["sources"])
    text = "\n\n".join(d.page_content for d in docs)

    refs = find_section_references(text, exclude=have)[:MAX_CHAINED_SECTIONS]
    extra_calls = 0
    chained: list[str] = []
    for ref in refs:
        # Targeted follow-up lookup for the referenced section.
        hits = service.retrieve(f"{ref} of the Constitution of Nigeria")
        extra_calls += 1
        picked = [d for d in hits if d.metadata.get("section") == ref] or hits[:1]
        if picked:
            docs.extend(picked)
            chained.append(ref)

    if extra_calls:
        logger.info(
            "chain: retrieved text references %s -> %d extra retrieval call(s), pulled in %s",
            refs, extra_calls, chained or "nothing tagged",
        )

    return {
        "docs": docs,
        "sources": _dedupe_sources(docs),
        "chained_sections": chained,
        "retrieval_calls": state.get("retrieval_calls", 0) + extra_calls,
        "chain_ms": state.get("chain_ms", 0.0) + (time.perf_counter() - t0) * 1000.0,
    }


def _verify_node(state: AgentState, service) -> dict[str, Any]:
    t0 = time.perf_counter()
    retry_count = state.get("retry_count", 0)
    text = "\n\n".join(d.page_content for d in state["docs"])

    adequate, reformulated = service.verify_retrieval(
        state["original_query"], state.get("classification", "direct_lookup"), text
    )

    out: dict[str, Any] = {
        "verify_ms": state.get("verify_ms", 0.0) + (time.perf_counter() - t0) * 1000.0,
    }
    if not adequate and retry_count < MAX_VERIFY_RETRIES:
        out["needs_retry"] = True
        out["retry_count"] = retry_count + 1
        out["verify_retry"] = True
        out["query"] = reformulated or f"{state['original_query']} (Nigerian Constitution section text)"
        logger.info("verify: inadequate, retrying once with %r", out["query"])
    else:
        out["needs_retry"] = False
    return out


def _route_after_verify(state: AgentState) -> str:
    return "retrieve" if state.get("needs_retry") else END


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def build_agent_graph(service):
    """Compile the retrieval graph bound to a RagService instance."""
    sg = StateGraph(AgentState)
    sg.add_node("classify", lambda s: _classify_node(s, service))
    sg.add_node("retrieve", lambda s: _retrieve_node(s, service))
    sg.add_node("chain", lambda s: _chain_node(s, service))
    sg.add_node("verify", lambda s: _verify_node(s, service))

    sg.set_entry_point("classify")
    sg.add_edge("classify", "retrieve")
    sg.add_edge("retrieve", "chain")
    sg.add_edge("chain", "verify")
    sg.add_conditional_edges("verify", _route_after_verify, {"retrieve": "retrieve", END: END})
    return sg.compile()


def run_agent(graph, query: str) -> AgentState:
    """Invoke the compiled graph and return the final state."""
    initial: AgentState = {
        "query": query,
        "original_query": query,
        "retrieval_calls": 0,
        "retry_count": 0,
        "classify_ms": 0.0,
        "retrieve_ms": 0.0,
        "chain_ms": 0.0,
        "verify_ms": 0.0,
        "verify_retry": False,
        "chained_sections": [],
    }
    return graph.invoke(initial)
