"""
This module defines the RagService class, which encapsulates the logic for querying the RAG system.
The RagService class initializes the necessary components for retrieval and generation, including:
1. Embeddings: Uses the same HuggingFace model as the ingest phase to ensure consistency.
2. Vector Store: Loads the existing ChromaDB vector store created during ingestion.
3. Language Model: Initializes the Ollama chat model for generating responses.  
4. Custom Prompt: Defines a system prompt to guide the LLM in providing accurate and context-aware answers based on the retrieved sections.
The query method takes a user query, retrieves relevant sections from the vector store, and generates a response using the LLM. It also extracts and returns the source sections for transparency.
Make sure to have the ChromaDB vector store set up and the Ollama model running before using this service.
"""

import logging
import os
import json
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from .metrics import RequestMetrics

logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class RagService:
    """Service class for handling RAG queries against the Nigerian Constitution."""

    # Number of chunks pulled from the vector store per query. A smaller prompt
    # means faster time-to-first-token and less chance of tripping Groq's
    # free-tier token-rate limit (whose retry/backoff is a big source of lag).
    RETRIEVAL_K = 5

    def __init__(self):
        # 1. Embeddings (must match the model used during ingestion).
        # all-MiniLM-L6-v2 is a public model, so we intentionally do NOT call
        # huggingface_hub.login() here - it added a blocking network round-trip
        # to every process start for no benefit.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Load the existing vector store and build the retriever ONCE, up
        # front. Previously the retriever (and a whole RetrievalQA chain) was
        # rebuilt on every request.
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.RETRIEVAL_K}
        )

        # 3. LLM. Groq is a fast hosted endpoint; Ollama kept for reference.
        # self.llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_URL)
        self.llm = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            timeout=30,
            max_retries=2,
        )

        # Provider / model labels recorded on every RequestMetrics row.
        if isinstance(self.llm, ChatGroq):
            self.llm_provider = "groq"
        elif isinstance(self.llm, ChatOllama):
            self.llm_provider = "ollama"
        else:
            self.llm_provider = self.llm.__class__.__name__.lower()
        self.llm_model = (
            getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "") or ""
        )

        # 4. Custom System Prompt for Legal Precision
        template = """You are a legal expert on the Nigerian Constitution.
        Use the following pieces of retrieved context to answer the question.
        If the answer isn't in the context, say you don't know—do not make up laws.

        Context: {context}
        Question: {question}

        Answer:"""
        self.QA_PROMPT = PromptTemplate.from_template(template)

        # 5. Warm up the embedding model so the first real user query doesn't
        # pay the one-time torch / model-load cost.
        try:
            self.embeddings.embed_query("warmup")
        except Exception as exc:  # best-effort only
            logger.warning(f"Embedding warmup failed: {exc}")

    def query(self, user_query: str):
        """Retrieve relevant sections and generate an answer (non-streaming).

        Kept for the evaluation suite; the live endpoint uses query_stream().
        """
        try:
            docs = self.retriever.invoke(user_query)

            context_text = "\n\n".join(doc.page_content for doc in docs)
            prompt = self.QA_PROMPT.format(context=context_text, question=user_query)
            answer = self.llm.invoke(prompt).content

            sources = [doc.metadata.get("section") for doc in docs]
            return {
                "answer": answer,
                "sources": list(set(sources)),
                "source_documents": docs,                                  # for the evaluator
                "retrieved_contexts": [doc.page_content for doc in docs],  # for the evaluator
            }
        except Exception as e:
            logger.error(f"RAG Query Error: {e}")
            return {"error": str(e)}
        
    def query_stream(self, user_query: str, metrics: "RequestMetrics | None" = None):
        """Stream the answer token-by-token, metadata JSON line first.

        When ``metrics`` is provided, per-stage timings and the output token
        count are recorded on it. This method never persists metrics - the
        caller does that once, in a finally block.
        """
        try:
            # --- Stage 1: embed the query --------------------------------------
            t0 = time.perf_counter()
            query_vec = self.embeddings.embed_query(user_query)
            if metrics is not None:
                metrics.embedding_time_ms = (time.perf_counter() - t0) * 1000.0

            # --- Stage 2: vector search (ChromaDB) ---------------------------
            t0 = time.perf_counter()
            docs = self.vectorstore.similarity_search_by_vector(query_vec, k=self.RETRIEVAL_K)
            if metrics is not None:
                metrics.retrieval_time_ms = (time.perf_counter() - t0) * 1000.0

            sources = list(set(doc.metadata.get("section") for doc in docs))
            retrieved_contexts = [doc.page_content for doc in docs]
            if metrics is not None:
                # Stashed for the async evaluator hand-off (not persisted here).
                metrics.retrieved_section_ids = sources
                metrics.retrieved_contexts = retrieved_contexts
            yield json.dumps({
                "type": "metadata",
                "sources": sources,
                "retrieved_contexts": retrieved_contexts,
            }) + "\n"

            context_text = "\n\n".join(doc.page_content for doc in docs)
            formatted_prompt = self.QA_PROMPT.format(context=context_text, question=user_query)

            # --- Stage 3: LLM generation (streamed) --------------------------
            t0 = time.perf_counter()
            parts = []
            exact_output_tokens = None
            for chunk in self.llm.stream(formatted_prompt):
                parts.append(chunk.content)
                yield json.dumps({"type": "token", "text": chunk.content}) + "\n"
                # Groq (and others) attach an exact usage count on the final
                # chunk; prefer it over the whitespace estimate when present.
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
        

