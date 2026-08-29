"""RagService: retrieval + generation for the Nigerian Constitution RAG.

Retrieval is a LangGraph agent (see rag_engine/graph.py):
    classify -> retrieve (hybrid: ChromaDB vector + BM25) -> chain -> verify
Generation (streaming the answer over the retrieved context) is unchanged and
still lives here. The public methods - query() and query_stream() - keep the
same return / stream shape they had before the agent was introduced.
"""

import logging
import os
import json
import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq

from .metrics import RequestMetrics
from .graph import build_agent_graph, run_agent, CLASSES
from .mcp_client import get_mcp_client
from .chroma import COLLECTION_NAME, get_chroma_client

logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class RagService:
    """Service class for handling RAG queries against the Nigerian Constitution."""

    RETRIEVAL_K = 5          # docs surfaced by the primary retrieve node
    _BM25_K = 3              # per-retriever k inside the hybrid ensemble
    _VECTOR_K = 3

    def __init__(self):
        # 1. Embeddings (must match the model used during ingestion). Public
        # model - no huggingface_hub.login().
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Vector store + hybrid retriever (ChromaDB semantic + BM25 keyword),
        # matching ingest.py's EnsembleRetriever design. BM25 isn't persisted,
        # so it's rebuilt here from the documents already in the Chroma store.
        # get_chroma_client() is a local persistent dir, or a ChromaDB server
        # container when CHROMA_HOST is set.
        self.vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        self._vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self._VECTOR_K}
        )
        self._hybrid = self._build_hybrid_retriever()

        # 3a. Generation LLM. LLM_PROVIDER selects it; the Prometheus split
        # (llm_provider_requests_total, generation_tokens_per_second) is on
        # ollama|openai, where "openai" = any OpenAI-compatible hosted API.
        self.llm, self.llm_provider = self._build_generation_llm()
        self.llm_model = (
            getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "") or ""
        )

        # 3b. Cheap/fast LLM used ONLY for the classify + verify nodes - never
        # for generation. Stays on Groq (fast/cheap); heuristic fallback covers
        # an outage. Override with CLASSIFY_LLM_MODEL.
        self.classify_llm = ChatGroq(
            model=os.getenv("CLASSIFY_LLM_MODEL", "allam-2-7b"),
            temperature=0,
            timeout=15,
            max_retries=1,
        )
        self.classify_model = (
            getattr(self.classify_llm, "model_name", None)
            or getattr(self.classify_llm, "model", "") or ""
        )

        # 4. Generation prompt.
        template = """You are a legal expert on the Nigerian Constitution.
        Use the following pieces of retrieved context to answer the question.
        If the answer isn't in the context, say you don't know—do not make up laws.

        Context: {context}
        Question: {question}

        Answer:"""
        self.QA_PROMPT = PromptTemplate.from_template(template)

        # 5. Persistent MCP client for the retrieve/chain tools (one subprocess
        # + one session for the whole process; calls reuse the open pipes).
        self.mcp = None
        try:
            self.mcp = get_mcp_client(env={
                "CHROMA_DB_PATH": os.path.abspath("chroma_db"),
                "MCP_LOG_LEVEL": "WARNING",
            })
            if not self.mcp.wait_ready(timeout=30):
                logger.warning("MCP client not ready; graph will use the in-process fallback")
            else:
                # First lookup_section pays a one-time ChromaDB init in the
                # subprocess (~0.3-0.5s); spend it at boot, not on request #1.
                self.mcp.lookup_section(1)
        except Exception as exc:  # never block startup on the MCP subprocess
            logger.warning("MCP client unavailable (%s); using in-process fallback", exc)
            self.mcp = None

        # 6. Compile the retrieval graph (bound to this service).
        self._graph = build_agent_graph(self)

        # 7. Warm up the embedding model so the first real request doesn't pay
        # the one-time torch / model-load cost.
        try:
            self.embeddings.embed_query("warmup")
        except Exception as exc:  # best-effort only
            logger.warning(f"Embedding warmup failed: {exc}")

    # ------------------------------------------------------------------ #
    # LLM selection                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_generation_llm():
        """Returns (llm, provider_label). LLM_PROVIDER: openai (default) | ollama."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "ollama":
            llm = ChatOllama(
                model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_URL),
                temperature=0,
            )
            return llm, "ollama"
        # "openai" == an OpenAI-compatible hosted API. Default target is Groq
        # (works with the existing GROQ_API_KEY); point OPENAI_BASE_URL /
        # OPENAI_API_KEY at real OpenAI to use that instead.
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", ""),
            temperature=0,
            timeout=30,
            max_retries=2,
            stream_usage=True,   # exact output-token count on the final chunk
        )
        return llm, "openai"

    # ------------------------------------------------------------------ #
    # Retrieval primitives used by the graph nodes                        #
    # ------------------------------------------------------------------ #
    def _build_hybrid_retriever(self):
        try:
            raw = self.vectorstore.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=c, metadata=m or {})
                for c, m in zip(raw["documents"], raw["metadatas"])
                if c
            ]
            if not docs:
                raise ValueError("Chroma collection is empty")
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = self._BM25_K
            hybrid = EnsembleRetriever(
                retrievers=[bm25, self._vector_retriever],
                weights=[0.4, 0.6],  # lean semantic, keep keyword recall
            )
            logger.info("Hybrid retriever ready (BM25 over %d chunks + Chroma vector).", len(docs))
            return hybrid
        except Exception as exc:
            logger.warning("Hybrid retriever unavailable, using vector-only: %s", exc)
            return None

    def retrieve(self, query: str):
        """Hybrid retrieval (BM25 + vector). Falls back to vector-only.

        This is the "existing hybrid retrieval, unchanged" that the graph's
        retrieve and chain nodes call.
        """
        if self._hybrid is not None:
            docs = self._hybrid.invoke(query)
        else:
            vec = self.embeddings.embed_query(query)
            docs = self.vectorstore.similarity_search_by_vector(vec, k=self.RETRIEVAL_K)
        return docs[: self.RETRIEVAL_K + 3]  # small headroom for the ensemble fusion

    # ------------------------------------------------------------------ #
    # Cheap-LLM nodes: classify + verify                                  #
    # ------------------------------------------------------------------ #
    def classify_query(self, query: str) -> str:
        prompt = (
            "Classify this question about the Nigerian Constitution into exactly one label.\n"
            "- direct_lookup: asks what a specific section or article says\n"
            "- cross_reference: asks which sections relate to a topic, or how sections connect\n"
            "- interpretive: asks for interpretation, implication, or real-world application\n\n"
            f"Question: {query}\n"
            "Answer with only the label."
        )
        try:
            resp = (self.classify_llm.invoke(prompt).content or "").strip().lower()
            for label in CLASSES:
                if label in resp or label.replace("_", " ") in resp:
                    return label
        except Exception as exc:
            logger.warning("classify LLM failed (%s); using heuristic", exc)
        return self._classify_heuristic(query)

    @staticmethod
    def _classify_heuristic(query: str) -> str:
        q = (query or "").lower()
        if any(w in q for w in ("relate", "related", "connect", "cross-reference",
                                "cross reference", "which sections", "what sections",
                                "list the sections")):
            return "cross_reference"
        if any(w in q for w in ("mean", "interpret", "imply", "implication", "apply",
                                "application", "can the government", "is it legal",
                                "does it allow", "how does")):
            return "interpretive"
        return "direct_lookup"

    def verify_retrieval(self, query: str, label: str, retrieved_text: str):
        """Lightweight self-check: does the retrieved text plausibly answer a
        question of this type? Returns (adequate: bool, reformulated: str|None).

        Deliberately a cheap heuristic - no LLM call, deterministic, ~0ms - so
        the one allowed retry only fires when retrieval genuinely came back thin.
        """
        import re as _re

        text = (retrieved_text or "")
        low = text.lower()

        if label == "cross_reference":
            distinct = len(_re.findall(r"section\s+\d+", low))
            adequate = distinct >= 2 and len(text) > 300
        elif label == "interpretive":
            adequate = len(text) > 400
        else:  # direct_lookup
            nums = _re.findall(r"\d{1,3}", query)
            has_section_text = "section" in low or bool(_re.search(r"\b\d{1,3}\.\s", text))
            num_present = (not nums) or any(f"section {n}" in low or f"{n}." in text for n in nums)
            adequate = len(text) > 200 and has_section_text and num_present

        if adequate:
            return True, None
        return False, f"{query} — full text of the relevant Nigerian Constitution section"

    # ------------------------------------------------------------------ #
    # Graph driver                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_state_to_metrics(state, metrics):
        metrics.classify_label = state.get("classification") or ""
        metrics.retrieval_calls = state.get("retrieval_calls")
        metrics.verify_retry = bool(state.get("verify_retry"))
        metrics.classify_ms = state.get("classify_ms")
        metrics.retrieve_ms = state.get("retrieve_ms")
        metrics.chain_ms = state.get("chain_ms")
        metrics.verify_ms = state.get("verify_ms")
        metrics.tool_calls = state.get("tool_calls") or []
        # Keep the Phase 2 field meaningful: retrieval_time_ms == retrieve node.
        metrics.retrieval_time_ms = state.get("retrieve_ms")

    @staticmethod
    def _agent_event(node, state, new_tools):
        """Compact per-node event for the streamed UI trace."""
        ev = {"type": "agent", "node": node,
              "ms": round(state.get(f"{node}_ms") or 0.0, 1)}
        tools = [[t["tool_name"], round(t["tool_latency_ms"]), t["ok"]] for t in new_tools]
        if node == "classify":
            ev["label"] = state.get("classification")
        elif node == "retrieve":
            ev["calls"] = state.get("retrieval_calls")
            ev["tools"] = tools
            ev["sources"] = (state.get("sources") or [])[:6]
        elif node == "chain":
            ev["chained"] = state.get("chained_sections") or []
            ev["tools"] = tools
        elif node == "verify":
            ev["retry"] = bool(state.get("needs_retry"))
        return ev

    def run_retrieval_agent(self, user_query: str, metrics: "RequestMetrics | None" = None):
        """Run classify -> retrieve -> chain -> verify; copy trace onto metrics."""
        state = run_agent(self._graph, user_query)
        if metrics is not None:
            self._apply_state_to_metrics(state, metrics)
        return state

    # ------------------------------------------------------------------ #
    # Public API (unchanged shape)                                        #
    # ------------------------------------------------------------------ #
    def query(self, user_query: str):
        """Retrieve (via the agent) and generate an answer (non-streaming)."""
        try:
            state = self.run_retrieval_agent(user_query)
            docs = state["docs"]

            context_text = "\n\n".join(doc.page_content for doc in docs)
            prompt = self.QA_PROMPT.format(context=context_text, question=user_query)
            answer = self.llm.invoke(prompt).content

            sources = [doc.metadata.get("section") for doc in docs]
            return {
                "answer": answer,
                "sources": list(set(sources)),
                "source_documents": docs,
                "retrieved_contexts": [doc.page_content for doc in docs],
                "classification": state.get("classification"),
            }
        except Exception as e:
            logger.error(f"RAG Query Error: {e}")
            return {"error": str(e)}

    def query_stream(self, user_query: str, metrics: "RequestMetrics | None" = None):
        """Stream, in order:
          * one ``{"type":"agent","node":...}`` line as EACH graph node finishes
            (classify / retrieve / chain / verify) - this is what the UI shows
            live as "what the agent is doing"
          * one ``{"type":"metadata",...}`` line
          * ``{"type":"token"}`` lines (the answer)
        Generation is unchanged.
        """
        try:
            # --- Agentic retrieval, streamed node-by-node ---
            initial = {
                "query": user_query, "original_query": user_query,
                "retrieval_calls": 0, "retry_count": 0,
                "classify_ms": 0.0, "retrieve_ms": 0.0, "chain_ms": 0.0, "verify_ms": 0.0,
                "verify_retry": False, "chained_sections": [], "tool_calls": [],
            }
            state = dict(initial)
            seen_tools = 0
            for step in self._graph.stream(initial):
                for node, update in step.items():
                    state.update(update)
                    all_tools = state.get("tool_calls") or []
                    new_tools, seen_tools = all_tools[seen_tools:], len(all_tools)
                    yield json.dumps(self._agent_event(node, state, new_tools)) + "\n"

            if metrics is not None:
                self._apply_state_to_metrics(state, metrics)

            docs = state["docs"]
            sources = state.get("sources") or list(
                {d.metadata.get("section") for d in docs}
            )
            retrieved_contexts = [d.page_content for d in docs]
            if metrics is not None:
                metrics.retrieved_section_ids = sources
                metrics.retrieved_contexts = retrieved_contexts

            yield json.dumps({
                "type": "metadata",
                "sources": sources,
                "retrieved_contexts": retrieved_contexts,
            }) + "\n"

            context_text = "\n\n".join(retrieved_contexts)
            formatted_prompt = self.QA_PROMPT.format(context=context_text, question=user_query)

            # --- Generation (streamed) - unchanged ---
            t0 = time.perf_counter()
            parts = []
            exact_output_tokens = None
            for chunk in self.llm.stream(formatted_prompt):
                parts.append(chunk.content)
                yield json.dumps({"type": "token", "text": chunk.content}) + "\n"
                usage = getattr(chunk, "usage_metadata", None) or {}
                if usage.get("output_tokens"):
                    exact_output_tokens = usage["output_tokens"]
            full_text = "".join(parts)
            if metrics is not None:
                metrics.generation_time_ms = (time.perf_counter() - t0) * 1000.0
                metrics.set_tokens_from_text(full_text, exact_count=exact_output_tokens)
                metrics.response_text = full_text

        except Exception as e:
            logger.error(f"RAG Stream Error: {e}")
            if metrics is not None:
                metrics.error = f"{type(e).__name__}: {e}"
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
