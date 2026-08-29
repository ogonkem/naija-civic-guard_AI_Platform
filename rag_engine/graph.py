"""LangGraph retrieval agent: classify -> retrieve -> chain -> verify.

The retrieve and chain nodes call the standalone MCP server
(rag_engine/mcp_server.py) over a reused client session:

    retrieve  direct_lookup + a section number  -> MCP lookup_section
              interpretive                      -> MCP search_precedent (stub)
              anything else                     -> in-process hybrid retrieval
    chain     for each primary section          -> MCP find_related_sections

Per-tool-call {tool_name, tool_latency_ms, ok, error} is accumulated on the
state and copied onto the request's RequestMetrics (nested tool_calls column).
Generation is unchanged and still lives in RagService.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .sections import find_section_references

logger = logging.getLogger(__name__)

CLASSES = ("direct_lookup", "cross_reference", "interpretive")
MAX_CHAINED_SECTIONS = 2   # cap chain fan-out so latency stays bounded
MAX_VERIFY_RETRIES = 1     # never loop forever

_SECTION_IN_QUERY = re.compile(r"\bsections?\s+(\d{1,3})\b", re.IGNORECASE)


class AgentState(TypedDict, total=False):
    query: str
    original_query: str
    classification: str
    docs: list
    sources: list
    retrieval_calls: int
    chained_sections: list
    tool_calls: list           # [{tool_name, tool_latency_ms, ok, error}, ...]
    needs_retry: bool
    retry_count: int
    verify_retry: bool
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


def _section_number(text: str) -> int | None:
    m = _SECTION_IN_QUERY.search(text or "")
    return int(m.group(1)) if m else None


def _tool_entry(name, latency_ms, ok, error):
    return {"tool_name": name, "tool_latency_ms": round(latency_ms, 2),
            "ok": bool(ok), "error": error}


def _docs_from_lookup(payload, section_label) -> list:
    if not payload or not payload.get("found"):
        return []
    return [Document(page_content=c, metadata={"section": section_label})
            for c in payload.get("chunks", []) if c]


def _docs_from_related(payload) -> list:
    if not payload or not payload.get("found"):
        return []
    return [Document(page_content=r["text"], metadata={"section": r["section"]})
            for r in payload.get("related", []) if r.get("text")]


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
    mcp = getattr(service, "mcp", None)
    label = state.get("classification")
    query = state["query"]
    calls = list(state.get("tool_calls", []))
    docs: list = []

    num = _section_number(query)
    if mcp is not None and label == "direct_lookup" and num is not None:
        payload, lat, ok, err = mcp.lookup_section(num)
        calls.append(_tool_entry("lookup_section", lat, ok, err))
        if ok:
            docs = _docs_from_lookup(payload, f"Section {num}")

    if not docs:  # MCP tools do not do semantic search - fall back in-process
        docs = service.retrieve(query)

    if mcp is not None and label == "interpretive":
        _, lat, ok, err = mcp.search_precedent(state["original_query"])
        calls.append(_tool_entry("search_precedent", lat, ok, err))

    return {
        "docs": docs,
        "sources": _dedupe_sources(docs),
        "retrieval_calls": state.get("retrieval_calls", 0) + 1,
        "tool_calls": calls,
        "retrieve_ms": state.get("retrieve_ms", 0.0) + (time.perf_counter() - t0) * 1000.0,
    }


def _chain_node(state: AgentState, service) -> dict[str, Any]:
    t0 = time.perf_counter()
    mcp = getattr(service, "mcp", None)
    docs = list(state["docs"])
    have = set(state["sources"])
    calls = list(state.get("tool_calls", []))
    chained: list[str] = []
    extra_calls = 0

    if mcp is not None:
        for sid in list(state["sources"])[:MAX_CHAINED_SECTIONS]:
            payload, lat, ok, err = mcp.find_related_sections(sid)
            calls.append(_tool_entry("find_related_sections", lat, ok, err))
            extra_calls += 1
            for d in _docs_from_related(payload):
                s = d.metadata.get("section")
                if s and s not in have:
                    have.add(s)
                    docs.append(d)
                    chained.append(s)
        if extra_calls:
            logger.info("chain: %d find_related_sections MCP call(s), pulled in %s",
                        extra_calls, chained or "nothing new")
    else:
        # In-process fallback (no MCP client): regex over the retrieved text.
        refs = find_section_references("\n\n".join(d.page_content for d in docs),
                                      exclude=have)[:MAX_CHAINED_SECTIONS]
        for ref in refs:
            hits = service.retrieve(f"{ref} of the Constitution of Nigeria")
            extra_calls += 1
            picked = [d for d in hits if d.metadata.get("section") == ref] or hits[:1]
            if picked:
                docs.extend(picked)
                chained.append(ref)

    return {
        "docs": docs,
        "sources": _dedupe_sources(docs),
        "chained_sections": chained,
        "tool_calls": calls,
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
        "tool_calls": [],
    }
    return graph.invoke(initial)
