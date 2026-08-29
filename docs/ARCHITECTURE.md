# Architecture & design notes

A deep walk through every concept in this system and the trade-offs behind each
choice. Written to be studied, not skimmed — each section is *what it is → how
it's used here → why / what you'd give up → what an interviewer might poke at*.

The [README](../README.md) is the elevator pitch; this is the whiteboard.

---

## 0. The system in one breath

A user asks a plain-English question about the Nigerian Constitution. The
**gateway** (Django + DRF) authenticates the caller, rate-limits them, and runs
a **LangGraph agent** that classifies the question, retrieves constitutional
text (hybrid vector + keyword search, plus **MCP tools** for direct section
lookup and cross-reference following), self-checks the retrieval, and streams a
grounded answer back token-by-token. Every request writes a **latency/throughput
row to Postgres synchronously** and emits **Prometheus** metrics; a **Celery
worker** scores retrieval quality asynchronously into a second Postgres table.
**dbt** models sit on top for trend analysis. The whole thing is eight Docker
services; **Ollama and hosted LLM APIs stay on the host**.

Request lifecycle (happy path):

```
client
  │  POST /api/chat/  {query}   + X-API-Key
  ▼
[PrometheusBeforeMiddleware] → [AuditLogMiddleware wraps the rest]
  ▼
DRF: ApiKeyAuthentication → IsAuthenticated → ApiKeyRateThrottle
  ▼
ChatView.post:  create RequestMetrics(request_id=uuid4)
  ▼
LangGraph:  classify ──► retrieve ──► chain ──► verify ──┐
              (cheap LLM)  (hybrid +   (MCP find_   (heuristic; │
                            MCP lookup) related)      1 retry)  │
              ◄───────────────── retry once if inadequate ◄─────┘
  ▼
generation: llm.stream(prompt over retrieved context)  → NDJSON tokens to client
  ▼
finally:  metrics.persist()  → INSERT rag_request_metrics (Postgres)
          record_request_metrics(metrics)  → Prometheus counters/histograms
          enqueue evaluate_request_task     → Redis   (fire-and-forget, threaded)
  ▼
AuditLogMiddleware:  INSERT request_audit_log  (api_key, endpoint, status, request_id)

... seconds later, off the critical path ...
Celery worker: evaluate_request_task → score → INSERT eval_results (Postgres)
                                              → Prometheus (worker's own :9540)
```

---

## 1. Retrieval-Augmented Generation (RAG)

**Concept.** Instead of trusting an LLM's parametric memory, you *retrieve*
relevant source text at query time and put it in the prompt: "answer using only
this context." The model becomes a reader/summariser over grounded evidence.

**Here.** The corpus is one PDF (the 1999 Constitution). `ingest.py` loads it,
tags each chunk with a section number, embeds the chunks, and stores them in
ChromaDB. At query time the agent retrieves ~5 chunks and the generation prompt
is literally *"Use the following pieces of retrieved context… if the answer
isn't in the context, say you don't know — do not make up laws."*

**Why RAG at all / trade-offs.**

| RAG | Fine-tuning | Long-context (dump the whole doc) |
|---|---|---|
| Fresh data without retraining; citations; smaller model works | Knowledge baked in, no retrieval infra, lower latency | Simple, no vector DB |
| Retrieval can miss (garbage in → garbage out); prompt-assembly complexity; more moving parts | Expensive, slow to update, still hallucinates, no provenance | Cost ∝ tokens; "lost in the middle"; 300-section constitution is borderline but a full statute corpus isn't |

**Interviewer pokes.** *"How do you know retrieval worked?"* → the eval pipeline
(§13). *"What if the answer spans many sections?"* → the `chain` node follows
cross-references; still bounded. *"Hallucination?"* → grounding reduces it, the
prompt instructs abstention, but it does **not** eliminate it — stated plainly
in the README limitations.

---

## 2. Chunking

**Concept.** Documents are split into passages small enough to embed meaningfully
and to fit several into a prompt, large enough to be self-contained.

**Here.** `RecursiveCharacterTextSplitter`, `chunk_size=800`,
`chunk_overlap=150`, splitting preferentially on legal boundaries
(`"\nSection "`, `"\nPART "`, then paragraph/line). A regex tags each chunk with
the `Section N` it falls under; short Preamble fragments are dropped as noise.

**Trade-offs.**
- **Small chunks** → precise vector matches, but a section's meaning gets
  fragmented and the model sees less continuous text. **Large chunks** → more
  context per hit, but embeddings blur (one vector for many ideas) and you fit
  fewer hits.
- **Overlap** buys continuity across a split at the cost of duplicate text in
  the index and in the prompt.
- **Structure-aware splitting** (on `Section`/`PART`) beats blind
  character-count splitting for legal text, but the constitution's PDF layout is
  messy, so the section tagging is coarse — roughly one section per page. This
  is the single biggest quality limitation (§13, README).

**What you'd do better.** Parse the real section hierarchy (regex on
"33. (1)…", "(a)…", subsection numbering), store `section`, `subsection`,
`chapter`, `part` as structured metadata, and let retrieval filter on it.

---

## 3. Embeddings & vector search

**Concept.** An embedding model maps text to a dense vector such that
semantically similar text lands nearby. Retrieval = nearest-neighbour search in
that space (cosine / L2), usually approximate (HNSW/IVF) for speed.

**Here.** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~90 MB, runs on
CPU in a few ms per query. ChromaDB stores the vectors + metadata and does the
ANN search.

**Trade-offs.**
- **MiniLM (small, local, free)** vs a large hosted embedding model
  (`text-embedding-3-large`, 3072-dim): the big model retrieves noticeably
  better on hard queries but adds a network hop, a cost per call, and a
  dependency. For a single well-known document, MiniLM is a defensible default;
  for a broad legal corpus you'd upgrade.
- **The ingest and the query side must use the *same* embedding model.** Change
  it and you must re-ingest — the index is only comparable within one model's
  space. (`rag_engine/chroma.py` + `ingest.py` share the client factory to keep
  them honest.)
- **Approximate NN** trades a little recall for a lot of latency. At 700 chunks
  it's irrelevant; at 70 M it's the whole ballgame.

**Interviewer pokes.** *"Cosine vs dot product vs Euclidean?"* → cosine for
normalised text embeddings (direction = meaning, magnitude = noise).
*"Curse of dimensionality?"* → why ANN structures and dimensionality choices
matter at scale.

---

## 4. Hybrid retrieval (BM25 + vector)

**Concept.** Vector search is great at *meaning* ("fair trial" ≈ "due
process") and bad at *exact tokens* ("Section 45", a defined term, a name).
BM25 (a TF-IDF-family lexical ranker) is the opposite. Combine them.

**Here.** `EnsembleRetriever` over a Chroma vector retriever **and** a BM25
retriever, weighted `[0.4 vector, 0.6-ish keyword]`. BM25 isn't persisted, so
`RagService.__init__` pulls every chunk out of Chroma once and builds the BM25
index in memory at boot.

**Trade-offs.**
- **Ensembling** (here: reciprocal-rank fusion of the two result lists) reliably
  beats either retriever alone, but you now maintain two indexes and tune a
  weight. The weight is corpus-dependent and there's no free lunch in picking
  it — we lean keyword because "Section N" queries dominate.
- **In-memory BM25** is fine for 700 chunks (rebuilt in ~100 ms at boot). For a
  large corpus you'd use a real lexical engine (Elasticsearch/OpenSearch/Tantivy)
  and lose the "rebuild on boot" simplicity.
- The MCP `lookup_section` tool (§7) is a *third* path — pure metadata filter,
  no ranking at all — used when the query names a section number, because then
  neither BM25 nor vector search is the right tool.

---

## 5. The agentic retrieval pipeline (LangGraph)

**Concept.** Instead of one `similarity_search`, model retrieval as a small
**state machine**: nodes transform a shared state dict, edges (some conditional)
decide what runs next. This lets retrieval *branch on the query* and *react to
its own output*.

**Here.** `rag_engine/graph.py` — a compiled `StateGraph`:

```
classify ─► retrieve ─► chain ─► verify ─(needs_retry?)─► retrieve   (once)
                                              └─(else)──► END
```

| node | does | why it's a separate node |
|---|---|---|
| **classify** | one cheap LLM call → `direct_lookup` / `cross_reference` / `interpretive`; heuristic fallback | downstream nodes behave differently per type; isolating it makes the choice observable and swappable |
| **retrieve** | `direct_lookup` + a section number → MCP `lookup_section` (skip semantic search entirely); `interpretive` → also MCP `search_precedent` (stub); else hybrid retrieval | the right retrieval *method* depends on the question shape |
| **chain** | scan retrieved text for references to *other* sections (shared regex) → MCP `find_related_sections` to pull those in (cap 2) | legal answers routinely say "subject to section 45" — the first hit is incomplete without the referenced section |
| **verify** | cheap deterministic check: is the retrieved text substantive for this question type? If not, reformulate and retry **once** | catches thin retrieval before wasting a generation call on it |

**Trade-offs.**
- **Agent vs single-shot.** You gain: targeted lookups, cross-reference
  following, a self-correction loop, and a legible trace (surfaced in the UI and
  in `rag_request_metrics`). You pay: +1 LLM call (classify) on every request,
  up to 2× retrieval work on a retry, more code, more failure modes. The
  measured tax is small (classify ~250–600 ms, retrieve/chain ~40–90 ms
  combined) because generation dominates.
- **Bounded loops.** `verify → retrieve` could ping-pong forever. `retry_count`
  is hard-capped at 1 in the routing function. *Always* cap agent loops — an
  unbounded self-correcting agent is a latency and cost incident waiting to
  happen.
- **Heuristic verify vs LLM verify.** An LLM "is this good enough?" call is more
  semantic but adds latency, cost, and non-determinism, and can itself be
  wrong. The heuristic (enough distinct sections? enough characters? the asked
  section number present?) is 0 ms, deterministic, and testable. It has false
  negatives (unnecessary retries) — acceptable because the retry is capped and
  cheap.
- **Classify with an LLM vs pure rules.** Rules are free and deterministic but
  brittle on phrasing. The cheap LLM generalises; the rule set is the fallback
  when it errors or times out. Best of both.

**Interviewer pokes.** *"Why LangGraph and not just if/else?"* → the state
machine gives you streaming per-node events, conditional retry routing, and a
place to add nodes (rerank, decompose) without rewriting control flow. *"What's
in the state?"* → query, original_query, classification, docs, sources,
retrieval_calls, tool_calls, per-node latency, retry_count. *"How do you stream
the agent's steps?"* → `graph.stream()` yields `{node: delta}` after each node;
the view emits a `{"type":"agent","node":…}` NDJSON line per node before the
answer tokens.

---

## 6. Model routing (cheap classifier, main generator)

**Concept.** Not every LLM call needs your best model. Route by task: a tiny
model for classification/routing/extraction, the capable model for the
user-facing generation.

**Here.** `classify_llm` = a small fast model (`CLASSIFY_LLM_MODEL`, default
`allam-2-7b` on Groq); `self.llm` = the generation model
(`LLM_PROVIDER` → `openai` or `ollama`). Classify also has a keyword-heuristic
fallback so an outage of the small model degrades gracefully.

**Trade-offs.**
- **Cost/latency**: a 7B classifier is ~10× cheaper and faster than a 70B
  generator, on a call that happens on *every* request. Over a day of traffic
  that's real money and real p50.
- **Quality risk**: a small model misclassifies more. Mitigations: a 3-way
  choice is easy; the heuristic backstops it; a wrong label mostly just picks a
  slightly worse retrieval path, not a wrong answer.
- **Operational**: two model configs, two failure modes, two sets of rate
  limits to watch. `llm_provider_requests_total{provider}` and
  `generation_tokens_per_second{provider}` exist so you can see each.

---

## 7. MCP — tools as a service

**Concept.** The **Model Context Protocol** is a JSON-RPC standard for exposing
"tools" (typed functions) to an LLM/agent over a transport (stdio for a local
subprocess, streamable-HTTP for a network service). It decouples *what a tool
does* from *who calls it*.

**Here.** `rag_engine/mcp_server.py` is a standalone `FastMCP` server exposing
three tools:

| tool | what | notes |
|---|---|---|
| `lookup_section(number)` | direct ChromaDB metadata fetch of a section's chunks | **bypasses semantic search** — the point of it |
| `find_related_sections(section_id)` | regex the section's own text for cross-refs, return those sections | the `chain` node's engine |
| `search_precedent(query)` | **stub** — returns *"not yet implemented — case law integration planned"* | a message, not an error, so callers carry on |

`rag_engine/mcp_client.py` holds **one** subprocess/session for the process
lifetime; every call reuses the open pipes. In compose it's a separate container
(`MCP_SERVER_URL=http://mcp:8100/mcp`, streamable-HTTP); locally the gateway
spawns it over stdio.

**Trade-offs.**
- **Why extract retrieval into MCP tools at all?** Independent testability
  (call the server directly), reusability (another agent/service can use the
  same tools), a clean seam for adding tools (a real `search_precedent`), and a
  process boundary (the tool server can be scaled/deployed/updated
  separately). You pay: a network hop per call, a new service to run, and a
  fallback path to maintain.
- **Connection reuse is non-negotiable.** A fresh stdio subprocess or HTTP
  session per call costs ~300–500 ms (ChromaDB client init in the subprocess).
  Reused, warm calls are ~10–30 ms. The `McpToolClient` keeps a persistent
  `ClientSession` on a background asyncio loop precisely for this. *This is a
  classic "N+1 connections" bug in disguise* — an interviewer will ask how you
  know it's reused (warm-call latency; subprocess count stays at 1).
- **Async SDK, sync callers.** The `mcp` SDK is async-only; LangGraph nodes are
  sync. Bridge: run the session on a dedicated event-loop thread,
  `run_coroutine_threadsafe` + `future.result(timeout)` from the sync side.
- **Graceful degradation.** If the MCP client can't reach the server, the graph
  falls back to the in-process regex/retrieval path and the request still
  answers; `tool_calls` records the failure. Never let a tool outage take down
  the request.
- **stdio vs streamable-HTTP.** stdio: zero network config, but only same-host,
  1:1 with the parent. HTTP: cross-container, many clients, load-balanceable,
  but you manage a port, readiness, and auth. Env-gated so local dev stays
  trivial.

---

## 8. Streaming responses

**Concept.** Send the answer as it's generated rather than buffering the whole
thing. The user sees the first token in ~1 s instead of staring at a spinner
for the full generation.

**Here.** `POST /api/chat/` returns `StreamingHttpResponse` of
newline-delimited JSON (`application/x-ndjson`): agent-step lines, then a
metadata line, then token lines, then a `done` line with the full timing
breakdown. `X-Accel-Buffering: no` + nginx `proxy_buffering off` so no proxy
holds it back.

**Trade-offs.**
- **NDJSON vs SSE vs WebSocket.** NDJSON over a plain POST: dead simple to
  produce (a generator that `yield`s), works through normal HTTP infra, easy to
  parse (`split('\n')`). SSE adds an event framing and auto-reconnect you don't
  need for a one-shot request. WebSocket is bidirectional overkill here.
- **A streaming response holds its server worker for the whole generation.**
  With sync gunicorn workers, 2 workers = 2 concurrent users max. Fix:
  `--threads` (the work is I/O-bound — waiting on the LLM — so threads
  parallelise fine). This project runs **1 worker + 8 threads** — one process
  keeps the `prometheus_client` registry consistent for `/metrics` (§12) while
  threads handle concurrency.
- **Errors mid-stream.** Once you've sent a `200` and some tokens you can't send
  a `500`. You emit an `{"type":"error"}` line and the client renders it. The
  `finally` block still runs, so the metrics row is still written (with `error`
  set) even on a partial failure — this is why persistence is in `finally`, not
  after a clean return.
- **Backpressure.** If the client reads slowly, the generator blocks on `yield`
  and the measured "generation time" absorbs network stalls. For localhost it's
  noise; over the internet you'd measure "time to last LLM token" separately.

---

## 9. The gateway: authentication, rate limiting, audit

### 9a. API-key authentication

**Concept.** A custom DRF `BaseAuthentication` class inspects a request header,
resolves it to a principal, and returns `(user, auth)` or raises.

**Here.** `ApiKeyAuthentication` reads `X-API-Key`, looks up an `ApiKey` row
(`key`, `owner`, `is_active`, `requests_per_minute`, `created_at`), and returns
a lightweight `ApiKeyUser` (so DRF's `IsAuthenticated` is satisfied without a
`django.contrib.auth` User) plus the `ApiKey` as `request.auth`. Missing /
unknown / inactive → `AuthenticationFailed`. `create_api_key` management command
+ admin registration so keys are never hand-edited.

**Trade-offs.**
- **401 vs 403.** DRF returns `403` for a failed auth *unless* the authenticator
  defines `authenticate_header()`. We define it → callers get a proper `401
  Unauthorized` (you're not authenticated) rather than `403 Forbidden` (you are,
  but you can't). Small thing, correct thing.
- **Static keys in a table** vs OAuth2 / JWT / mTLS: trivial to issue and
  revoke, no token-refresh dance, works for machine-to-machine. You give up:
  rotation workflow, per-endpoint scopes, expiry, and the key travels on every
  request (so: HTTPS only, and consider hashing keys at rest — here they're
  stored plaintext, which is a known simplification).
- **`ApiKeyUser` shim** vs mapping to real Django users: avoids dragging in the
  auth tables and sessions for a pure API. Cost: anything expecting a real
  `User` (admin actions attributed to a user, permissions framework) doesn't
  apply.

### 9b. Rate limiting

**Concept.** Cap requests per principal per window. Protects the backend and
enforces fair use / plan limits.

**Here.** `ApiKeyRateThrottle` extends DRF's `SimpleRateThrottle`, keyed on the
**API key** (not a Django user). Default `60/min` from settings; an
`ApiKey.requests_per_minute` overrides per key. Over the limit → `429`.

**Trade-offs.**
- **Algorithm.** DRF's `SimpleRateThrottle` is a **sliding-log** (keeps
  timestamps of recent requests, evicts old ones). Simple and fairly accurate;
  memory ∝ requests-in-window. Alternatives: **fixed window** (cheap, but
  allows 2× burst at the boundary), **token bucket** (smooth, allows controlled
  bursts, a bit more state), **leaky bucket**. For per-key API limits the log is
  fine.
- **Backing store.** DRF throttling uses the Django cache. This project uses the
  default **LocMemCache** → the counter is **per-process**. With one gunicorn
  worker that's exact; with N workers the effective limit is N× the configured
  rate. Correct fix for multi-worker: a shared cache (Redis). Called out
  explicitly in the README.
- **Keying on the key, not the user.** Cleaner for an API: the limit follows the
  credential, works before you've built a full user model, and per-key override
  is a single column.

### 9c. Audit logging

**Concept.** A durable record of *who called what and what happened*, at a layer
that sees every request including the ones the view rejects.

**Here.** `AuditLogMiddleware` is **plain Django middleware** (not DRF),
positioned last so it wraps the view and sees the final status. It writes one
`request_audit_log` row per `/api/` request via the ORM: `api_key` (FK, nullable
for a bad key), `api_key_owner`, `endpoint`, `method`, `status_code`,
`timestamp`, and `request_id`.

**Trade-offs.**
- **Middleware vs DRF layer.** DRF's auth/permission/throttle run *inside* the
  view machinery; a `401`/`429` is raised and handled there. Middleware sits
  *outside* it, so it records the 401s and 429s too — which is exactly what you
  want in an audit log. A DRF-level hook would miss the rejections.
- **The join key.** `request_id` is a `uuid4` minted at the top of `ChatView`
  and put on the `X-Request-ID` response header; the middleware reads it back
  off the response. Successful requests share it with their `rag_request_metrics`
  row; 401/429 requests never reach the agent so they have a null `request_id`
  and no metrics row. **This is lightweight distributed tracing** — one id
  correlates audit log ↔ metrics ↔ eval results across two processes and three
  tables.
- **Auditing must never break the response.** The write is wrapped in
  `try/except` and logs on failure. Same principle everywhere in this codebase:
  observability is best-effort; the user's request is not.

---

## 10. Correlation via `request_id`

**Concept.** In a system with multiple stores and an async worker, you need one
identifier that ties every artefact of a single request together. This is the
poor-man's trace id.

**Here.** `request_id` (uuid4) flows:
`ChatView` mints it → `RequestMetrics(request_id=…)` → `X-Request-ID` header →
`AuditLogMiddleware` copies it into `request_audit_log` → the eval task is
enqueued with it → `eval_results.request_id`. Every table has it indexed.

```sql
SELECT a.status_code, m.provider, m.total_time_ms, e.keyword_coverage
FROM request_audit_log a
JOIN rag_request_metrics m ON m.request_id = a.request_id
LEFT JOIN eval_results   e ON e.request_id = a.request_id
WHERE a.request_id = '…';
```

**Trade-off vs a real tracing system (OpenTelemetry).** A single correlation id
in your own tables is zero-dependency and queryable with SQL, but you don't get
spans, timing waterfalls, cross-service propagation, or a UI. For this scale it's
the right amount; at microservice scale you'd adopt OTel and propagate a
`traceparent`.

---

## 11. Observability: two layers, one source of truth

**Concept.** **Metrics** (Prometheus) answer "what's the p95 latency by
provider over the last hour, and alert me if it doubles" — pre-aggregated
time-series, cheap to keep forever, bad at "show me *that* request".
**Event logs / a table** (Postgres `rag_request_metrics`) answer "show me every
slow request yesterday with its full agent trace and query text" — high
cardinality, joinable, expensive to keep forever. You want both, and you must
not compute the numbers twice.

**Here.** One `RequestMetrics` dataclass is populated through the request
lifecycle (per-node latency, token count, provider, tool calls, error). In the
`finally` block it is used **once** to (a) `INSERT` the Postgres row and (b)
feed `record_request_metrics()` which `.observe()`s the Prometheus histograms /
`.inc()`s the counters. Latency is measured in one place
(`time.perf_counter()` around each graph node) and reused.

**The custom Prometheus series:**

| metric | type | why |
|---|---|---|
| `request_latency_seconds{stage}` | histogram | p50/p95/p99 by stage (embedding/retrieval/generation/total) |
| `generation_tokens_per_second{provider}` | histogram | throughput, split by provider |
| `mcp_tool_calls_total{tool_name}` | counter | is the agent using the tools; which |
| `agent_retries_total` | counter | how often verify fails — a retrieval-quality smoke signal |
| `llm_provider_requests_total{provider}` | counter | traffic split |
| `eval_keyword_coverage` | gauge | last eval-set query's coverage (worker) |
| `celery_eval_task_duration_seconds` | histogram | is async eval itself getting slow |
| `celery_queue_depth` | gauge | is eval falling behind generation |
| `celery_eval_task_failures_total` | counter | eval reliability |

**Trade-offs.**
- **Histogram vs summary.** Histograms have server-side aggregatable buckets
  (`histogram_quantile` across instances) and fixed cost; summaries compute
  quantiles client-side (can't aggregate) but are exact. For anything you'll
  slice by label or sum across replicas, histograms win. You must pick buckets
  up front — bad buckets = useless quantiles. (Ours span 50 ms → 60 s because
  ollama generation is genuinely tens of seconds.)
- **Counter vs gauge.** Counters only go up (rate() them); gauges go up and
  down (`queue_depth`, `keyword_coverage`). Using the wrong one breaks the
  PromQL.
- **Cardinality.** `mcp_tool_calls_total{tool_name}` is fine (3 values).
  `…{request_id}` would be a cardinality bomb — that dimension belongs in
  Postgres, not Prometheus. Knowing where each dimension lives is the whole
  skill.
- **`django-prometheus` vs hand-rolled `prometheus_client` + middleware.**
  `django-prometheus` gives the automatic request/response/db series and the
  `/metrics` view for free and is the Django-native integration. You still
  define custom business metrics with bare `prometheus_client` objects
  alongside it — which is what `metrics_prom.py` does.

---

## 12. The multiprocess metrics problem

**Concept.** `prometheus_client` keeps counts in an in-process registry.
`/metrics` serves *that process's* registry. If your app runs as N processes
(gunicorn prefork workers, Celery prefork children), a scrape hits one of them
and sees a fraction of the truth — or inconsistent numbers as the load balancer
rotates.

**Here, two instances of it and two fixes:**
1. **Gateway.** Run gunicorn with **1 worker + 8 threads**. Threads share the
   process and the registry; the work is I/O-bound so throughput is fine. The
   alternative — `PROMETHEUS_MULTIPROC_DIR` multiprocess mode — needs a shared
   tmpfs, a gunicorn `child_exit` hook to `mark_process_dead`, and it degrades
   gauges (only `livesum`/`max`/`min` aggregations survive). Not worth it here.
2. **Celery worker.** Prefork pool → tasks execute in **forked children** whose
   metric increments never reach the parent. First symptom: `eval ok` in the
   logs but `celery_eval_task_duration_seconds_count` stuck at `0` on
   `:9540/metrics`. Fix: `--pool=threads --concurrency=4` — tasks run in the
   main process, the registry is shared, and the `worker_ready` signal's
   `start_http_server(9540)` sees every observation. The worker exposes its
   **own** `/metrics` (separate registry, separate process) and Prometheus
   scrapes it as a second target.

**Interviewer pokes.** *"Why not just run more gunicorn workers?"* → then you're
back to the multiproc problem; pick multiproc mode *or* threads, and threads are
simpler when the workload is I/O-bound. *"How would this change at scale?"* →
each replica is a scrape target with its own `instance` label; Prometheus
aggregates across them with `sum()/histogram_quantile()`; you'd move throttle
state to Redis at that point too.

---

## 13. Asynchronous evaluation

**Concept.** Scoring an answer's quality is valuable but not something the user
should wait for. Decouple it: the request path enqueues a job and returns; a
worker consumes the queue and writes results later.

**Here.** After the response is fully streamed and the metrics row is written,
`_enqueue_eval()` submits `evaluate_request_task` to Redis via a **bounded
thread pool** (so the request thread never does broker I/O — up, slow, or
down). The Celery worker (own container, `-Q eval`) runs `evaluate_retrieval()`
— the *same* scoring code the offline `retrieval_eval.py` uses — and writes an
`eval_results` row keyed by `request_id`.

**Trade-offs & the reasoning:**
- **Two tables, joined — not one row updated by two processes.** The gateway
  owns `rag_request_metrics`; the worker owns `eval_results`. They never write
  the same row. This sidesteps write contention and the "which process's write
  wins" question entirely — a `LEFT JOIN` on `request_id` reassembles the full
  picture, and "no eval row yet" is a natural, correct state.
- **Fire-and-forget really means it.** The enqueue is on a background thread and
  wrapped in try/except with a short broker timeout and `retry=False`. If Redis
  is down, the request is completely unaffected — the eval row just never
  appears (`dump_eval --pending` surfaces those). Verified by stopping Redis
  mid-run: request latency unchanged, warning logged.
- **Delivery semantics.** Celery + Redis with `acks_late=True` is
  **at-least-once**: a worker crash mid-task re-delivers. So the task should be
  idempotent-ish — here a re-run just writes another `eval_results` row (or you
  add a unique constraint on `request_id`). At-least-once + a dedupe key beats
  at-most-once (lost evals) for this.
- **Queue depth as a health signal.** `celery_queue_depth` (polled via
  `redis.llen("eval")`) climbing steadily means eval is slower than generation.
  It doesn't affect user latency, but it tells you the eval data is going stale
  and you need more worker concurrency. This is the kind of metric that only
  matters *because* the system is decoupled.
- **Why Celery and not a Django management command on cron, or `asyncio`?** Cron
  batches (stale data, thundering herd); in-process `asyncio` doesn't survive a
  gateway restart and competes with request handling. A broker + worker gives
  durability, backpressure visibility, independent scaling, and retries.

---

## 14. Evaluation methodology

**Concept.** You can't improve retrieval you don't measure. Two modes: **offline**
(run a fixed question set through the system, compute aggregate scores — for
regression testing a change) and **online** (score live traffic — for drift and
real-world quality).

**Here.** `evaluation_set.jsonl` = `{query, target: "Section N", keywords: […]}`.
`eval_core.evaluate_retrieval()` computes:

| metric | definition | what it proxies |
|---|---|---|
| **hit rate** | was the `target` section among the retrieved section ids? (1/0) | **recall@k** — did we fetch the right passage at all |
| **MRR** | `1 / rank` of the target section (0 if absent) | how *highly* we ranked it — reciprocal rank, averaged |
| **keyword coverage** | fraction of expected legal terms present in the retrieved text | a **precision-ish** proxy — is the fetched text actually on-topic / does the answer have the right terminology |

For a real user query with no ground truth, `keyword coverage` still computes
(keywords derived from the query, stop-words removed); `hit`/`MRR` stay `null`.

**Trade-offs.**
- **Retrieval metrics vs answer metrics.** hit/MRR/coverage measure *retrieval*.
  They don't measure whether the *generated answer* is correct, complete, or
  faithful — that needs an LLM-judge or human eval, which is slower, costlier,
  and itself noisy. This project deliberately does the cheap, deterministic,
  retrieval-side eval and is honest that it's a floor, not a ceiling.
- **Keyword coverage is a weak precision proxy** — a chunk can contain the words
  and still be the wrong context. It's chosen because it's free and
  ground-truth-optional.
- **`k` matters.** hit rate is `recall@k`; shrink `k` and it drops, grow `k` and
  generation cost + "lost in the middle" rise. The eval numbers in the README
  were taken at a specific `k`; changing retrieval config invalidates the
  baseline (which is why `retrieval_eval.py` regenerates `eval_report.md`).
- **Offline set is tiny (~15 queries).** Fine for catching a regression,
  useless as an absolute quality claim. Real work: hundreds of queries,
  stratified by question type, refreshed.

---

## 15. LLM provider abstraction

**Concept.** Treat the generation model as swappable. Same interface
(`langchain` chat model: `.stream()`, `.invoke()`), different backends.

**Here.** `LLM_PROVIDER` selects:
- `openai` → `ChatOpenAI` pointed at any OpenAI-compatible endpoint (default:
  Groq's, using `GROQ_API_KEY`; set `OPENAI_BASE_URL` for real OpenAI)
- `ollama` → `ChatOllama` against the host's Ollama (`host.docker.internal`)

The Prometheus label is normalised to `ollama | openai` so dashboards split
cleanly.

**Trade-offs (hosted API vs self-hosted), from this deployment's own numbers:**

| | hosted (`openai`/Groq, gpt-oss-20b) | self-hosted (`ollama`, llama3.2, CPU) |
|---|---:|---|
| generation latency (avg) | **~1.2 s** | **~22 s** (first call +78 s cold load) |
| throughput | **~349 tok/s** | **~8 tok/s** |
| retrieval latency | ~41 ms | ~41 ms (identical — provider-independent) |
| cost model | per token, external | fixed infra, no per-call cost |
| data | leaves your network | stays in your network |
| control | vendor's model list, rate limits, deprecations | you own the model, the version, the capacity |
| ops | none | you run the GPU box, the model files, the scaling |

**Takeaway an interviewer wants:** everything except the generation step is
identical across providers, so the provider choice is a pure
*latency/throughput/cost* vs *privacy/control* trade on one node. The
abstraction cost is tiny (`_build_generation_llm()`), the optionality is
valuable. Also: **streaming token usage** — both `ChatGroq` and
`ChatOpenAI(stream_usage=True)` attach an exact `output_tokens` on the final
chunk; when absent, the code falls back to a whitespace-split estimate and flags
it (`tokens_generated_is_estimate`).

---

## 16. Data layer

**Concept.** Pick a database per the access pattern; keep the schema stable
across environments; let the ORM be the one writer contract.

**Here.** `DATABASES` is env-driven: **SQLite** when `POSTGRES_HOST` is unset
(local dev, tests — zero services), **Postgres** in compose. Both the
synchronous metrics writer and the async eval writer go through the Django ORM,
so pointing `default` at Postgres moves *both* with **no code change and the
same migrations**.

**Trade-offs.**
- **SQLite → Postgres transparently.** SQLite: single file, no server, perfect
  for dev and CI, but single-writer locking, weak concurrency, no real
  `jsonb`. Postgres: concurrent writers (gateway + worker), native `jsonb`
  (`tool_calls`, `retrieved_section_ids`), native `uuid`. The models are
  written to the common denominator so the swap is just settings.
- **`JSONField` for `tool_calls`** vs a child table (`request_tool_call` rows
  FK'd to the metric): the JSON column keeps "everything about one request in
  one row", is easy to write from the dataclass, and Postgres can still query
  into it (`jsonb_array_length`, `jsonb_array_elements`). A child table gives
  proper indexing/aggregation on tool-call fields but adds a join and a second
  insert per request. For a nested log that's mostly read whole, JSON wins.
- **Nullable columns as state.** `eval_results.hit` is `null` when there's no
  ground truth; `request_audit_log.request_id` is `null` for a rejected
  request. `null` = "this dimension doesn't apply here", which is more honest
  than a sentinel.
- **Two writers, never the same row** (see §13) — the design rule that keeps the
  two-process setup contention-free.

---

## 17. dbt — the analytics layer

**Concept.** dbt turns SQL `SELECT`s into version-controlled, tested,
dependency-ordered models in the warehouse. It's the "T" of ELT — transform
*after* load, in the database.

**Here.** `dbt/` (dbt-postgres) reads the two operational tables as `sources`
and builds three **views**:

| model | grain | answers |
|---|---|---|
| `daily_query_volume` | day | traffic, errors, retries, how many got evaluated |
| `avg_latency_by_provider_and_stage` | provider × stage | avg / p50 / p95 per pipeline stage — feeds the README Performance table |
| `eval_score_trend` | day × provider | hit rate / MRR / keyword coverage over time |

Run: `docker compose --profile analytics run --rm dbt build` (`--profile` so it's
opt-in, not part of `up`).

**Trade-offs.**
- **View vs table (`materialized`).** Views: always fresh, zero storage, but
  recompute on every read — fine here (small tables, ad-hoc reads). Tables:
  fast reads, but stale until the next `dbt run` and you pay storage.
  **Incremental** models (append only new rows) are the middle ground for large
  fact tables — not needed at this volume.
- **dbt vs a Prometheus recording rule vs application code.** Prometheus is for
  *operational* time-series and alerting (seconds-to-hours, low cardinality).
  dbt is for *analytical* questions over the full history with joins and
  arbitrary SQL ("eval score by provider by week, correlated with query type").
  Doing this aggregation in the Django app would scatter business logic and
  recompute on every request. dbt keeps it declarative, tested, and in the
  warehouse where analysts live.
- **`source` freshness / tests.** `schema.yml` asserts `not_null`, `unique`,
  `accepted_values` — cheap regression guards that run in `dbt build`. At scale
  you'd add `dbt source freshness` to catch a stalled pipeline.
- **Unpivoting** (`avg_latency_by_provider_and_stage`): the gateway stores one
  *column* per stage (`classify_ms`, `retrieve_ms`, …); the model `UNION ALL`s
  them into `(stage, latency_ms)` rows so `stage` is a queryable dimension.
  Wide-to-long is a standard analytics-engineering move.

---

## 18. Containerisation

**Concept.** One service per concern, declared in `docker-compose.yml`, each
with a healthcheck and explicit dependency ordering, so `docker compose up`
reproduces the whole system.

**Here.** 8 services: `gateway`, `worker` (same image, different command), `mcp`,
`chromadb`, `postgres`, `redis`, `prometheus`, `grafana`, plus an opt-in `dbt`.
`depends_on` with `condition: service_healthy` gates the gateway on Postgres +
Redis being *ready*, not just *started*. Prometheus config and the Grafana
datasource + dashboard are **provisioned from files**, not clicked together.

**Trade-offs & the potholes this project actually hit:**
- **Image size / build time.** The Linux `torch` wheel pulls **~2.5 GB of CUDA
  libraries** by default. The gateway does CPU embedding only, so the Dockerfile
  installs `torch==…+cpu` from the PyTorch CPU index *first*, then
  `-r requirements.txt` sees it satisfied and skips the CUDA deps → **812 MB
  image, ~4 min build** instead of a 5 GB image that never finished. Lesson:
  know what your base deps drag in.
- **Layer caching.** `COPY requirements.txt` + `pip install` is its own layer
  *before* `COPY . .`, so a code change rebuilds in seconds (pip layer cached).
  Ordering Dockerfile steps cheapest-changing → most-changing is the whole game.
- **`.dockerignore`.** Without it, `COPY . .` ships `.venv/` (multi-GB), `.git`,
  `chroma_db/`, `staticfiles/` — bloating the context upload and the image.
- **Pin your framework.** `Django>=5.2.14` (unpinned) resolved to a newer
  Django in the container that **removed `cc_delim_re`**, which the pinned DRF
  imports → `ImportError` on boot, works fine locally. Pinned to `==6.0.5`.
  Lesson: pin transitive-compat-sensitive deps; a lockfile would prevent this.
- **`healthcheck` + `depends_on: condition`.** `service_started` only means the
  container's PID 1 is alive; `service_healthy` waits for the healthcheck. The
  gateway's entrypoint *also* waits on the Postgres/Chroma TCP ports before
  migrating — belt and braces, because "container up" ≠ "Postgres accepting
  connections".
- **Networking gotcha.** Incremental `docker compose up <service>` after a
  partially-failed `up` left a container attached to **no network** (broke DNS:
  `Temporary failure in name resolution` for `postgres`). Fix: a clean
  `docker compose down` + a single `up`. `extra_hosts: host-gateway` also
  interfered with Docker Desktop's embedded DNS and was removed —
  `host.docker.internal` is provided natively there.
- **Windows reserved ports.** Publishing `55432` failed (`bind: forbidden`) —
  it's in a Hyper-V-reserved range. Postgres is simply not published to the
  host; use `docker compose exec postgres psql`.
- **Ollama stays on the host.** GPU passthrough into Docker is
  platform-specific pain; the pragmatic call is to leave the GPU workload on the
  host and reach it via `host.docker.internal`. Explicitly a "day one" trade.

---

## 19. Configuration, secrets, cold-start, static files

- **12-factor config.** Everything environment-specific is an env var with a
  sane fallback: `DJANGO_SECRET_KEY` (falls back to an obviously-insecure dev
  key), `DJANGO_DEBUG`, `DATABASES`, `CHROMA_HOST`, `MCP_SERVER_URL`,
  `LLM_PROVIDER`, `REDIS_URL`. `.env.example` is the authoritative list. The
  fallbacks mean `runserver` works with zero config; production overrides via
  the environment.
- **Secrets.** `.env` is gitignored (verified it's never been committed). The
  Django `SECRET_KEY` *was* hardcoded and committed — moved to
  `DJANGO_SECRET_KEY` and flagged for rotation. It signs sessions, password-reset
  tokens, and signed cookies; leaking it in a repo matters if it's ever reused
  in prod. Ideal: a secrets manager (Vault, SSM), not a file.
- **Cold start / boot warm-up.** `RagService.__init__` loads a
  sentence-transformers model (~torch import + weights), builds the BM25 index,
  and connects the MCP client — ~10–25 s. Done lazily via a process-wide
  singleton, and triggered at boot from `AppConfig.ready()` *only for real
  server processes* (not for `migrate`/`test`/`shell`), so the first user
  request doesn't eat the cold start and management commands stay fast.
  `RAG_WARMUP=0` skips it for rapid dev restarts. Trade-off: boot is slower;
  first request is fast; a `runserver` autoreload pays the cost each reload.
- **Static files.** WhiteNoise + `ManifestStaticFilesStorage`: `collectstatic`
  content-hashes every asset (`chat.ee8d1a74.js`) and writes a manifest;
  `{% static %}` resolves to the hashed name; WhiteNoise serves it with
  `Cache-Control: immutable, max-age=1y` — safe *because* the name changes when
  the content does. In `DEBUG=True` Django deliberately serves the unhashed name
  from `static/` via finders so you don't re-run `collectstatic` on every edit;
  the manifest path only engages in production. (`staticfiles/` is build output —
  gitignored, regenerated by the entrypoint.) The setting is `STORAGES`
  (Django 4.2+) — it was misnamed `STORAGE` and silently ignored, so none of
  this was actually happening until fixed.

---

## 20. Known limitations & what's next

Honest gaps (also in the README):

- **No jurisdiction handling** — one document, no federal/state distinction, no
  amendment awareness.
- **No document versioning** — no "as of" date, no diffing across amendments, no
  provenance beyond "which chunk".
- **Retrieval precision below a production bar for legal use** — coarse
  section tagging (≈one section per page), small eval set, hit rate not where
  you'd want it before anyone relies on an answer.
- **`search_precedent` is a stub** — no case-law integration.
- **Single-node** — one Postgres, one Redis, one gunicorn worker, LocMem
  throttle cache. Fine for a team, not multi-region.
- **Auth is a shared plaintext secret** — no rotation, no scopes, keys not
  hashed at rest.
- **Grafana/Prometheus are minimal** — one dashboard, self-scrape + two
  targets, no alerting rules.

Next moves, roughly in priority order: parse the real section/subsection
hierarchy and re-ingest; grow the eval set and add an LLM-judge for answer
faithfulness; move throttle state and the Prometheus registry to a shared store
so the gateway can scale horizontally; hash API keys and add rotation/scopes;
add Prometheus alert rules + a proper Grafana folder; wire a real
`search_precedent`.
