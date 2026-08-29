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
* `ingest.py` also builds a BM25 + `EnsembleRetriever`; the serving path now uses the same hybrid design (BM25 rebuilt at startup from the persisted Chroma docs) — see below.

### 2. Retrieval agent + generation — `rag_engine/graph.py`, `rag_engine/services.py`

Retrieval is a **LangGraph agent** (`classify → retrieve → chain → verify`); the
generation step is unchanged. `query()` and `POST /api/chat/` keep their
previous return / stream shape.

| node | what it does |
|---|---|
| **classify** | one **cheap/fast** LLM call (`CLASSIFY_LLM_MODEL`, default `allam-2-7b` — *not* the generation model) labels the query `direct_lookup` / `cross_reference` / `interpretive`. Keyword heuristic fallback if the call fails. |
| **retrieve** | `direct_lookup` + a section number → MCP `lookup_section` (direct metadata fetch, no semantic search); `interpretive` → MCP `search_precedent` (stub); otherwise in-process hybrid retrieval (`EnsembleRetriever` over ChromaDB vector + BM25). |
| **chain** | for each primary section (cap 2) → MCP `find_related_sections`, which runs the shared section regex (`rag_engine/sections.py`) over that section's text and returns the cross-referenced sections. Falls back to in-process regex if the MCP client is down. |
| **verify** | cheap deterministic self-check (no LLM) — is the retrieved text substantive enough for this question type? If not, **one** retry with a reformulated query (hard cap, no loop). |

**MCP tool server** (`rag_engine/mcp_server.py`, official `mcp` SDK, stdio) exposes
`lookup_section(number)`, `find_related_sections(section_id)`, and
`search_precedent(query)` (stub — returns *"not yet implemented — case law
integration planned"*). It reads ChromaDB directly via the `chromadb` client
(no embeddings), so it starts fast. `RagService` holds **one** `McpToolClient`
(`rag_engine/mcp_client.py`) — one subprocess + one session for the process
lifetime; every tool call reuses the open pipes. If it can't start, the agent
uses the in-process fallback and requests still work.

* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, local CPU, public model (no token).
* **Generation LLM:** **Groq** `ChatGroq`, model from `GROQ_LLM_MODEL`, `temperature=0`. Ollama available as a commented alternative.
* **Prompt:** answer only from retrieved context, otherwise say so.

### 3. API — `rag_engine/views.py`

`POST /api/chat/` with `{"query": "..."}` returns a **streamed** `application/x-ndjson`
body — one JSON object per line:

| line | shape |
|---|---|
| metadata (first) | `{"type": "metadata", "sources": [...], "retrieved_contexts": [...]}` |
| token (repeated) | `{"type": "token", "text": "..."}` |
| done (last) | `{"type": "done", "duration": <s>, "timings_ms": {classify, retrieve, chain, verify, generation, total}, "agent": {classify_label, retrieval_calls, verify_retry}, ...}` |
| error | `{"type": "error", "error": "..."}` |

The browser client (`static/js/chat.js` + `templates/chat.html`) renders tokens as
they arrive.

`RagService` is constructed once per process and **warmed at server boot**
(`RagEngineConfig.ready()`) so the first request doesn't pay the model-load cost.
Set `RAG_WARMUP=0` to skip the warm-up during fast dev restarts.

#### Gateway (DRF) — API key + rate limit + audit log

`POST /api/chat/` sits behind a DRF gateway (all in `rag_engine/`):

* **Auth** — `ApiKeyAuthentication` (custom `BaseAuthentication`) checks the
  `X-API-Key` header against the `ApiKey` model (`key`, `owner`, `is_active`,
  `requests_per_minute`, `created_at`). Missing / unknown / inactive key →
  **401**. Create keys with `python manage.py create_api_key --owner "<name>" [--rpm N]`
  or in the Django admin — never by hand.
* **Rate limit** — `ApiKeyRateThrottle` (DRF `SimpleRateThrottle`) keyed on the
  API key, not a Django user. Default `api_key` rate is `60/min`
  (`API_KEY_DEFAULT_RATE`); an `ApiKey.requests_per_minute` overrides it for
  that key. Over the limit → **429**. (Throttle state is in Django's default
  LocMemCache — per-process; use a shared cache for multi-worker gunicorn.)
* **Audit log** — `AuditLogMiddleware` (plain Django middleware, last in the
  chain) writes one `RequestAuditLog` row per `/api/` request via the ORM:
  `api_key`, `api_key_owner`, `endpoint`, `method`, `status_code`, `timestamp`,
  and `request_id`. The `request_id` is the `X-Request-ID` response header
  `ChatView` sets from its `RequestMetrics`, so **audit log and metrics join on
  `request_id`**. 401/429 never reach the agent → those rows have a null
  `request_id` and no `RequestMetric`. Inspect via the admin or SQL.

### 4. Request metrics — `rag_engine/metrics.py` + `RequestMetric` model

Every `POST /api/chat/` builds a `RequestMetrics` dataclass at the start of the
request, fills it in stage by stage (embedding → retrieval → generation), and
writes **one row** to the `rag_request_metrics` table from a `finally` block —
a single inline INSERT, so a row lands even if generation errors partway
through (the exception text is stored in `error`).

Columns: `request_id` (UUID, for joining async eval results back later),
`timestamp`, `query_text`, `provider`, `model`, `retrieval_time_ms`,
`generation_time_ms`, `total_time_ms`, `tokens_generated`
(exact from the provider when available, else a whitespace-split estimate —
`tokens_generated_is_estimate` flags which), `tokens_per_second`, `error`, plus
the **retrieval-agent trace**: `classify_label`, `retrieval_calls` (> 1 once the
chain node fires a follow-up retrieval), `verify_retry`, per-node latency
`classify_ms` / `retrieve_ms` / `chain_ms` / `verify_ms`, and **`tool_calls`**
— a nested JSON list of every MCP tool call the retrieve/chain nodes made
(`{tool_name, tool_latency_ms, ok, error}`), on the same row.

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
| API | Django REST Framework — streaming NDJSON endpoint, behind an API-key + rate-limit + audit-log gateway |
| Orchestration | LangChain + **LangGraph** (retrieval agent) |
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
   GROQ_LLM_MODEL=openai/gpt-oss-20b     # generation model; any your Groq key can access
   CLASSIFY_LLM_MODEL=allam-2-7b         # cheap/fast model for the classify node only

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
   python manage.py create_api_key --owner "me"      # gateway needs a key
   python manage.py runserver
   ```
   Call the API with the key:
   ```
   curl -N -X POST localhost:8000/api/chat/ \
     -H "X-API-Key: <key>" -H "Content-Type: application/json" \
     -d '{"query":"What does Section 33 say?"}'
   ```
   (The browser chat page at `/` posts to the same endpoint and now needs the
   key wired into `static/js/chat.js` — it will 401 as-is.)

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
