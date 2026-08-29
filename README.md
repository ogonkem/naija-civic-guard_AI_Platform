# Naija Civic Guard

**A retrieval platform for the Nigerian Constitution — with the operational
scaffolding a team needs to actually run it.**

Ask a question in plain English, get an answer grounded *only* in the
constitutional text, with the sections it came from. Under that sits a real
platform: an agentic retrieval pipeline, an API gateway with auth + rate
limiting, per-request metrics in Postgres, asynchronous evaluation, MCP tools,
Prometheus/Grafana, and dbt models over the logs — all wired together in one
`docker compose up`.

---

## Architecture

```mermaid
flowchart LR
    U[Client / browser] -->|X-API-Key| GW

    subgraph gateway [Gateway  Django + DRF]
      GW[API key auth + rate limit] --> AG
      AG[LangGraph agent<br/>classify → retrieve → chain → verify] --> GEN[Generation<br/>openai | ollama]
      GW -.audit + RequestMetrics.-> PG[(Postgres)]
      GW -->|/metrics| PROM
    end

    AG <-->|MCP streamable-http| MCP[MCP tool server<br/>lookup_section<br/>find_related_sections<br/>search_precedent]
    AG --> CH[(ChromaDB<br/>vector + BM25)]
    MCP --> CH

    GW -->|enqueue after response| RQ[(Redis)]
    RQ --> W[Celery worker<br/>evaluate_request_task]
    W -->|eval_results| PG
    W -->|/metrics :9540| PROM

    PROM[Prometheus] --> GRAF[Grafana<br/>provisioned dashboard]
    PG --> DBT[dbt models<br/>volume · latency · eval trend]

    GEN -.->|LLM_PROVIDER=ollama| OLL[Host Ollama<br/>outside Docker, GPU]
    GEN -.->|LLM_PROVIDER=openai| API[Groq / OpenAI API]
```

| Service | Role |
|---|---|
| **gateway** | Django + DRF. API-key auth (`X-API-Key` → `ApiKey` model), per-key rate limit (DRF throttle), audit-log middleware, the LangGraph retrieval agent, streaming NDJSON responses, `RequestMetric` written to Postgres per request, `/metrics` for Prometheus. |
| **mcp** | Standalone MCP server (`mcp` SDK, streamable-http). Three tools the agent's retrieve/chain nodes call: `lookup_section`, `find_related_sections`, `search_precedent` (stub). Talks to ChromaDB directly. |
| **chromadb** | Vector store. Populated on first boot from the constitution PDF (`AUTO_INGEST`). |
| **postgres** | Audit log + `rag_request_metrics` + `eval_results`. One DB, joined on `request_id`. |
| **redis** | Celery broker for the async evaluator. |
| **worker** | Celery worker running `evaluate_request_task` — scores each request (keyword coverage, hit rate, MRR) *after* the response is sent. Own `/metrics` on :9540. |
| **prometheus** | Scrapes `gateway:8000/metrics` + `worker:9540/metrics`. |
| **grafana** | Auto-provisioned datasource + dashboard (latency by stage, tokens/sec by provider, tool-call breakdown, eval coverage, async-eval health). |
| **dbt** (opt-in) | `dbt-postgres` models over the metrics tables (`--profile analytics`). |

**Request path:** `client → gateway (auth, rate limit) → LangGraph agent
(classify → retrieve/chain via MCP → verify) → generation (streamed) →
RequestMetric to Postgres → enqueue eval → response`. The eval runs later, in
the worker, off the user's critical path.

---

## Onboarding a new team

```bash
git clone https://github.com/ogonkem/naija-civic-guard_AI_Platform.git
cd naija-civic-guard_AI_Platform

cp .env.example .env
#   edit .env — at minimum:
#     GROQ_API_KEY=...            (the "openai" provider path; OpenAI-compatible)
#     DJANGO_SECRET_KEY=...       (python -c "from django.core.management.utils import get_random_secret_key as g; print(g())")
#   optional:
#     LLM_PROVIDER=ollama         (use a model on the host's Ollama instead)

docker compose up --build          # brings up all 8 services; first boot ingests the PDF

# mint an API key
docker compose exec gateway python manage.py create_api_key --owner "my-team"

# ask something
curl -N -X POST localhost:8000/api/chat/ \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query":"What does Section 45 say about restrictions on fundamental rights?"}'
```

Then:
- **Grafana** — <http://localhost:3000> (anon access, the *Naija Civic Guard – RAG* dashboard is pre-loaded)
- **Prometheus** — <http://localhost:9090>
- **Metrics** — <http://localhost:8000/metrics>, <http://localhost:9540/metrics>
- **dbt models** — `docker compose --profile analytics run --rm dbt build`
- **Browser chat** — <http://localhost:8000/> (auto-provisions its own key)

Everything not in Docker: **Ollama** (wants direct GPU access on the host) and
any **hosted LLM API** (Groq / OpenAI). Containers reach the host's Ollama via
`host.docker.internal`.

### Every env var

See `.env.example` — it is the authoritative list, grouped by concern
(Django, LLM provider, gateway, Postgres, Redis/Celery, ChromaDB, MCP,
observability). Values docker-compose injects for the container network are
marked "compose-set".

---

## What this closes

Starting from a notebook-grade RAG script, this repo adds:

- **A gateway, not a bare endpoint** — API-key identity, per-key rate limits, an audit log you can join to metrics by `request_id`.
- **Observability that isn't `print`** — per-request latency broken out by pipeline stage, throughput, agent trace (classify label, retrieval calls, retries, MCP tool calls), all in Postgres *and* Prometheus, computed once and reused.
- **Evaluation that doesn't block users** — every request is scored asynchronously (keyword coverage always; hit / MRR when the query is in the eval set), written to a separate table, with queue-depth and task-duration metrics so you can see eval falling behind.
- **Retrieval as an agent, not one `similarity_search`** — a classify → retrieve → chain → verify graph that does direct section lookups, follows cross-references, and self-checks with one bounded retry.
- **Tools as a service** — retrieval primitives extracted into an MCP server the agent calls over a reused connection, so they're independently testable and reusable.
- **A provider switch** — `openai` vs `ollama` selectable per deploy, with the latency/throughput split visible in Grafana.
- **Analytics over the logs** — dbt models for query volume, latency by provider/stage, and eval-score trend, ready for a warehouse.
- **One command to stand it all up** — `docker compose up`, Prometheus scraping and the Grafana dashboard provisioned automatically.

---

## Limitations (honest)

- **No jurisdiction handling.** It only knows the 1999 Constitution of the Federal Republic of Nigeria. It does not distinguish federal vs. state law, does not know about amendments beyond what's in the ingested PDF, and will confidently answer as if that one document is the whole of Nigerian law.
- **No document versioning.** One PDF, one Chroma collection. There's no notion of "as of" a date, no diffing across constitutional amendments, no provenance beyond "which chunk".
- **Retrieval precision is not at a production bar for legal use.** Section tagging is a coarse regex (roughly one section per PDF page), so `lookup_section` and the chain step can attach the wrong neighbouring section; the offline eval (`retrieval_eval.py`) shows hit rate well under what you'd want before anyone relies on an answer. Treat outputs as pointers to sections to read, not as legal advice.
- **The LLM can still be wrong.** Grounding reduces fabrication; it doesn't eliminate it. `search_precedent` is a stub — there is no case-law integration.
- **Single-node.** SQLite for local dev, one Postgres, one Redis, one gunicorn worker, LocMemCache throttling. Fine for a team; not a multi-region deployment.
- **Auth is a shared secret.** API keys in a table, no rotation workflow, no per-endpoint scopes, no OAuth.

---

## Performance

Real numbers from the `rag_request_metrics` Postgres table (this deployment's
own logs), not estimates. Generation provider is selected by `LLM_PROVIDER`;
there is **no "modal" provider in this project** — the two paths are `openai`
(Groq's OpenAI-compatible API, model `openai/gpt-oss-20b`) and `ollama` (host
Ollama, `llama3.2`).

**Latency by stage** (`analytics.avg_latency_by_provider_and_stage`, 16 requests
through the containerised stack — 10 `openai`, 6 `ollama`):

| stage | openai — avg / p95 | ollama — avg / p95 |
|---|---:|---:|
| classify (cheap LLM) | 361 / 575 ms | 565 / 677 ms |
| retrieval (MCP `lookup_section` + hybrid) | 41 / 58 ms | 41 / 57 ms |
| chain (MCP `find_related_sections`) | 39 / 56 ms | 37 / 57 ms |
| verify (heuristic) | 0.1 ms | 0.1 ms |
| **generation** | **1160 / 1406 ms** | **22164 / 64677 ms** |
| **total** | **1607 / 1926 ms** | **19869 / 62232 ms** |

**Throughput** (`generation_tokens_per_second`): `openai` ≈ **349 tok/s**
average; `ollama` (`llama3.2` on a CPU host) ≈ **8 tok/s** — and its first call
paid a ~78 s cold model load, which is what pulls the p95 up.

**Retrieval quality** (`analytics.eval_score_trend`, eval-set queries only) is
provider-independent, as expected — `openai` hit-rate 0.71 / MRR 0.64 / keyword
coverage 0.69; `ollama` 0.75 / 0.63 / 0.73.

**Takeaway:** everything except generation is identical across providers
(retrieval is ~41 ms either way); the provider choice is entirely a
latency/throughput-vs-privacy trade on the generation step. Async eval adds
**~26 ms of worker time** per request (`celery_eval_task_duration_seconds`) and
lands in Postgres a few seconds *after* the response — visible in Grafana as
the eval-coverage panel updating a step behind the latency panels.

_Reproduce: run queries under each `LLM_PROVIDER`, then
`docker compose --profile analytics run --rm dbt build` and read the
`analytics.*` views._

---

## Repo layout

| Path | What |
|---|---|
| `rag_engine/` | Django app: `views` (gateway), `graph` (LangGraph agent), `services` (LLM + retrieval), `mcp_server` / `mcp_client`, `metrics` / `metrics_prom`, `tasks` (Celery eval), `authentication` / `throttling` / `middleware`, `eval_core`, `sections` |
| `civic_guard/` | Django project settings, `celery.py`, URLs (`/metrics`) |
| `ingest.py` | PDF → chunk → tag → ChromaDB |
| `retrieval_eval.py` | Offline eval → `eval_report.md` |
| `dbt/` | `dbt-postgres` analytics models |
| `docker/` | `gateway.Dockerfile`, `mcp.Dockerfile`, `dbt.Dockerfile`, `prometheus.yml`, `grafana/` provisioning + dashboard |
| `docker-compose.yml` | The 8-service stack |

## Tests

```bash
python manage.py test rag_engine        # unit + integration (SQLite, no external services)
python retrieval_eval.py                # offline retrieval quality
```

## Disclaimer

An AI-powered educational tool. Verify every legal finding against the Official
Gazette or a qualified legal professional.
