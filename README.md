# 🇳🇬 Naija Civic Guard: Nigerian Constitution RAG

Empowering citizens with AI-driven, verifiable insights into the Nigerian Constitution.

Naija Civic Guard is a Retrieval-Augmented Generation (RAG) system grounded in the
text of the Constitution of the Federal Republic of Nigeria. For each question it
retrieves the relevant constitutional sections and asks an LLM to answer **only**
from that retrieved context, returning the source section numbers alongside every
answer.

---

## 🏗️ Architecture

### 1. Ingestion — `ingest.py`

* **Load:** `PyPDFLoader` reads `constitution-of-the-federal-republic-of-nigeria.pdf`.
* **Section tagging:** a regex tags every chunk with its `Section N` number so answers can cite specific sections.
* **Chunking:** `RecursiveCharacterTextSplitter` — `chunk_size=800`, `chunk_overlap=150`, split on legal boundaries (`\nSection `, `\nPART `, …).
* **Noise filter:** short Preamble fragments are dropped.
* **Index:** chunks are embedded and written to a persistent **ChromaDB** store in `chroma_db/`.
* `ingest.py` also builds an in-memory BM25 + `EnsembleRetriever` for experimentation. **The running app does not use it** — serving is pure vector search (see below).

### 2. Retrieval + generation — `rag_engine/services.py`

* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU via `langchain-huggingface`. The model is public — no HuggingFace token required.
* **Vector store:** the persisted ChromaDB collection, queried as a retriever with **`k = 5`** (`RagService.RETRIEVAL_K`).
* **LLM:** **Groq** via `langchain-groq` (`ChatGroq`), model from `GROQ_LLM_MODEL`, `temperature=0`. Ollama is still present in the code as a commented alternative — swap in `ChatOllama` + `OLLAMA_BASE_URL` to use a local model instead.
* **Prompt:** a system prompt instructs the model to answer only from the retrieved context and to say it doesn't know otherwise.

### 3. API — `rag_engine/views.py`

`POST /api/chat/` with `{"query": "..."}` returns a **streamed** `application/x-ndjson`
body — one JSON object per line:

| line | shape |
|---|---|
| metadata (first) | `{"type": "metadata", "sources": [...], "retrieved_contexts": [...]}` |
| token (repeated) | `{"type": "token", "text": "..."}` |
| done (last) | `{"type": "done", "duration": <seconds>}` |
| error | `{"type": "error", "error": "..."}` |

The browser client (`static/js/chat.js` + `templates/chat.html`) renders tokens as
they arrive.

`RagService` is constructed once per process and **warmed at server boot**
(`RagEngineConfig.ready()`) so the first request doesn't pay the model-load cost.
Set `RAG_WARMUP=0` to skip the warm-up during fast dev restarts.

### 4. Request metrics — `rag_engine/metrics.py` + `RequestMetric` model

Every `POST /api/chat/` builds a `RequestMetrics` dataclass at the start of the
request, fills it in stage by stage (embedding → retrieval → generation), and
writes **one row** to the `rag_request_metrics` table from a `finally` block —
a single inline INSERT, so a row lands even if generation errors partway
through (the exception text is stored in `error`).

Columns: `request_id` (UUID, for joining async eval results back later),
`timestamp`, `query_text`, `provider`, `model`, `embedding_time_ms`,
`retrieval_time_ms`, `generation_time_ms`, `total_time_ms`, `tokens_generated`
(exact from the provider when available, else a whitespace-split estimate —
`tokens_generated_is_estimate` flags which), `tokens_per_second`, `error`.

Inspect it:
```
python manage.py dump_metrics -n 20      # table of the last N
python manage.py dump_metrics --errors   # only failed requests
python manage.py dump_metrics --csv      # CSV, all fields
```
Also registered in the Django admin (`/admin/`), and it's a plain table —
query it with SQL directly if you prefer.

### 5. Async per-request evaluation — Celery + Redis

After the response is fully streamed and the `RequestMetric` row is written,
the gateway hands the request off to a **Celery task** (`evaluate_request_task`)
— fire-and-forget, on a background thread, so the request path never touches
Redis. Payload: `{request_id, query, retrieved_context, retrieved_section_ids,
response_text}`.

The task (`rag_engine/tasks.py`) scores the request with `rag_engine/eval_core.py`
— the *same* helpers the offline `retrieval_eval.py` uses — and writes one row
to the **`eval_results`** table:

| field | always | notes |
|---|---|---|
| `request_id` | ✓ | joins back to `rag_request_metrics` |
| `keyword_coverage`, `keyword_source` | ✓ | vs. ground-truth keywords, or (for a real off-set query) vs. keywords derived from the query |
| `retrieved_section_ids`, `response_chars` | ✓ | |
| `matched_ground_truth` | ✓ | whether the query is in `evaluation_set.jsonl` |
| `hit`, `reciprocal_rank`, `target_section` | only with ground truth | stay `NULL` for real user queries |

`eval_results` is a **separate table** joined on `request_id` — the request
process and the eval worker never write the same row. If Redis or the worker
is down the enqueue fails softly (logged, `WARNING`) and the request is
unaffected; that request just never gets an `eval_results` row.

Run the worker (needs Redis — see setup):
```
celery -A civic_guard worker -Q eval -l info --pool=solo   # --pool=solo on Windows
```
Inspect the join:
```
python manage.py dump_eval -n 20        # requests + their eval results
python manage.py dump_eval --pending    # requests not yet evaluated
```

### 6. Offline evaluation — `retrieval_eval.py`

Runs `evaluation_set.jsonl` through `RagService.query()` (the non-streaming path)
and writes `eval_report.md` with Hit Rate, Mean Reciprocal Rank and keyword
coverage, plus a per-query breakdown. Shares its metric math with the async
task via `rag_engine/eval_core.py`.

---

## 📊 Benchmarks

From the most recent `retrieval_eval.py` run recorded in `eval_report.md`:

| Metric | Value |
|---|---|
| Hit Rate (target section retrieved) | 60% |
| Mean Reciprocal Rank | 0.45 |
| Avg keyword coverage | 84% |

> These figures were recorded with an earlier retrieval configuration (`k = 10`).
> Re-run `python retrieval_eval.py` to refresh them for the current `k = 5` setup.

---

## 🛠️ Stack

| Layer | Choice |
|---|---|
| Web framework | Django 6.x (Python 3.12) |
| API | Django REST Framework — streaming NDJSON endpoint |
| Orchestration | LangChain (`langchain-core`, `langchain-chroma`, `langchain-huggingface`, `langchain-groq`) |
| LLM | Groq API (`ChatGroq`); Ollama supported as a commented alternative |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB, persisted in `chroma_db/` |
| Observability | LangSmith (optional, via env) |
| Async eval | Celery 5 + Redis (broker **and** result backend) |
| Serving | Gunicorn + Nginx via Docker Compose |
| Static files | WhiteNoise |

---

## 🚀 Setup

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Configure `.env`** — copy the template and fill it in:
   ```
   cp .env.example .env
   ```
   ```
   # Django
   DJANGO_SECRET_KEY=<generate one, see below>
   DJANGO_DEBUG=true                     # set false in production
   DJANGO_ALLOWED_HOSTS=                 # comma-separated hostnames in production

   # LLM (required)
   GROQ_API_KEY=your_groq_key
   GROQ_LLM_MODEL=openai/gpt-oss-20b     # any model your Groq key can access

   # Observability (optional)
   LANGSMITH_TRACING=false
   LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=Naija-Civic-Guard

   # Async evaluation (Celery) — broker + result backend
   REDIS_URL=redis://localhost:6379/0
   # CELERY_TASK_ALWAYS_EAGER=1          # run eval inline, no Redis/worker (dev only)

   # Only needed if you switch the LLM to local Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   ```
   * Generate a secret key:
     `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`.
     If `DJANGO_SECRET_KEY` is unset the app falls back to an insecure dev key so
     `runserver` still works. When `DEBUG` is on and `DJANGO_ALLOWED_HOSTS` is
     empty, `localhost`/`127.0.0.1` are allowed automatically.
   * `GROQ_LLM_MODEL` must be a model your key is allowed to use. Llama models
     (e.g. `llama-3.1-8b-instant`) require a Groq key with Llama access;
     `openai/gpt-oss-20b` is the tested default. List what your key can access:
     `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`.
   * No `HF_API_TOKEN` is needed — the embedding model is public.

3. **Build the index** (writes `chroma_db/`; delete that folder first to rebuild):
   ```
   python ingest.py
   ```

4. **Run:**
   ```
   python manage.py migrate
   python manage.py runserver
   ```
   Then open <http://127.0.0.1:8000/>.

5. **Async evaluation worker** (optional — the app runs fine without it, you
   just get no `eval_results` rows):
   ```
   docker run -d --name ncg-redis -p 6379:6379 redis:7-alpine
   celery -A civic_guard worker -Q eval -l info --pool=solo
   ```

6. **Tests and evaluation:**
   ```
   python manage.py test rag_engine
   python manage.py dump_metrics      # per-request latency / throughput
   python manage.py dump_eval         # requests joined to async eval results
   python retrieval_eval.py           # offline batch eval -> eval_report.md
   ```

---

## 🐳 Docker

```
docker-compose up --build
```

Brings up Gunicorn (`web`, with worker threads for the streaming responses) behind
Nginx. Nginx proxies `/` with `proxy_buffering off` so the token stream is not held
back; `.env` is passed through via `env_file`.

---

## ⚖️ Disclaimer

Naija Civic Guard is an AI-powered educational tool. While it is built on the
official Nigerian Constitution, always verify legal findings against the Official
Gazette or a legal professional.
