# Manual test walkthrough

Exercise every part of the stack by hand. Commands assume a bash shell
(Git Bash on Windows is fine) run from the repo root. `psql` output is shown
with `-P pager=off` so it doesn't page.

---

## 0. Prerequisites

- Docker Desktop running
- `.env` present: `cp .env.example .env`, then set at least
  - `GROQ_API_KEY=...` (the `openai` provider path — Groq's OpenAI-compatible API)
  - `DJANGO_SECRET_KEY=...`
    (`python -c "from django.core.management.utils import get_random_secret_key as g; print(g())"`)
- For the Ollama test only: Ollama running on the host with a small model
  (`ollama pull llama3.2`)

---

## 1. Bring the stack up

```bash
docker compose up -d --build
```

First boot takes a few minutes: the gateway downloads the embedding model,
runs migrations against Postgres, ingests the constitution PDF into ChromaDB
(`AUTO_INGEST=1`), then starts gunicorn and warms the retrieval agent.

Watch it finish:

```bash
docker compose logs -f gateway
# wait for:  "RAG service warmed up on boot."  then Ctrl-C
```

---

## 2. Health-check every service

```bash
docker compose ps
# all 8 should be "running"; postgres and redis "(healthy)"

curl -s -o /dev/null -w "gateway   %{http_code}\n" http://localhost:8000/metrics
curl -s -o /dev/null -w "worker    %{http_code}\n" http://localhost:9540/metrics
curl -s -o /dev/null -w "chromadb  %{http_code}\n" http://localhost:8001/api/v2/heartbeat
curl -s -o /dev/null -w "mcp       %{http_code}\n" http://localhost:8100/mcp    # 400/406 is fine - it wants an MCP handshake, not a bare GET
curl -s -o /dev/null -w "prom      %{http_code}\n" http://localhost:9090/-/healthy
curl -s -o /dev/null -w "grafana   %{http_code}\n" http://localhost:3000/api/health
```

Expected: `200` for gateway, worker, chromadb, prom, grafana.

---

## 3. Create an API key

```bash
docker compose exec gateway python manage.py create_api_key --owner "manual-test"
```

Copy the `key   : ncg_...` value. Save it for the shell:

```bash
KEY=ncg_...paste-here...
```

List keys any time: `docker compose exec gateway python manage.py create_api_key --list`

---

## 4. Gateway auth — no key ⇒ 401

```bash
curl -s -o /dev/null -w "no key -> %{http_code}\n" \
  -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does Section 33 say?"}'
```

Expected: `401`.

```bash
curl -s -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' -H 'X-API-Key: bogus' \
  -d '{"query":"x"}'
```

Expected: `{"detail":"Invalid API key."}` (also 401).

---

## 5. Ask a real question (streamed answer)

```bash
curl -N -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"query":"What does Section 45 say about restrictions on fundamental rights?"}'
```

You get newline-delimited JSON:
- one `{"type":"metadata","sources":[...],...}` line
- many `{"type":"token","text":"..."}` lines (the answer, streamed)
- a final `{"type":"done","request_id":"...","timings_ms":{...},"agent":{...},"mcp_tool_calls":[...]}` line

**Grab the `request_id`** from the `done` line — you'll join on it next:

```bash
RID=$(curl -s -N -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"query":"What does Section 45 say about restrictions on fundamental rights?"}' \
  | grep '"type": "done"' \
  | python -c "import sys,json;print(json.loads(sys.stdin.read())['request_id'])")
echo "request_id = $RID"
```

The `done` line's `mcp_tool_calls` should show `lookup_section` then
`find_related_sections` — the agent's retrieve and chain nodes calling the MCP
server, each with its own `tool_latency_ms`.

---

## 6. The synchronous metrics row lands in Postgres immediately

```bash
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c "
SELECT request_id, provider, classify_label, retrieval_calls,
       jsonb_array_length(tool_calls) AS mcp_calls,
       round(retrieve_ms::numeric,0)  AS retrieve_ms,
       round(generation_time_ms::numeric,0) AS gen_ms,
       round(total_time_ms::numeric,0) AS total_ms
FROM rag_request_metrics WHERE request_id = '$RID';"
```

One row, populated. This was written inline, before the HTTP response finished.

---

## 7. The async evaluation lands a moment later

Wait ~5 seconds, then:

```bash
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c "
SELECT request_id, matched_ground_truth,
       round(keyword_coverage::numeric,2) AS coverage,
       keyword_source, hit
FROM eval_results WHERE request_id = '$RID';"

docker compose logs worker | grep "eval ok" | tail -3
```

The `eval_results` row is written by the **worker container** — proof the
Redis → Celery → Postgres chain works across containers. It's a separate table,
joined to `rag_request_metrics` by `request_id`, and it appears *after* the
response (async, off the user's critical path).

Join the two:

```bash
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c "
SELECT m.request_id, m.provider, m.total_time_ms::int AS req_ms,
       e.matched_ground_truth, round(e.keyword_coverage::numeric,2) AS cov
FROM rag_request_metrics m
JOIN eval_results e ON e.request_id = m.request_id
WHERE m.request_id = '$RID';"
```

---

## 8. The audit log

Every `/api/` request (including the 401s from step 4) is logged by Django
middleware:

```bash
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c "
SELECT to_char(timestamp,'HH24:MI:SS') AS t, method, endpoint, status_code,
       api_key_owner, request_id
FROM request_audit_log ORDER BY timestamp DESC LIMIT 8;"
```

The 200 rows carry a `request_id` that joins to `rag_request_metrics`; the 401s
have a null `request_id` and no metrics row.

---

## 9. Rate limiting ⇒ 429

Make a low-limit key and hammer it:

```bash
LIMITED=$(docker compose exec -T gateway python manage.py create_api_key --owner "load-test" --rpm 3 \
  | grep 'key   :' | awk '{print $3}')

for i in 1 2 3 4 5; do
  curl -s -o /dev/null -w "req $i -> %{http_code}\n" \
    -X POST http://localhost:8000/api/chat/ \
    -H 'Content-Type: application/json' -H "X-API-Key: $LIMITED" \
    -d '{"query":"section 1"}'
done
```

Expected: `200, 200, 200, 429, 429`. The 429s never reach the agent.

---

## 10. Prometheus + the /metrics endpoints

```bash
# custom request-path series (gateway):
curl -s http://localhost:8000/metrics | grep -E '^(request_latency_seconds_count|llm_provider_requests_total|mcp_tool_calls_total|agent_retries_total|generation_tokens_per_second_count)'

# worker series (its own registry):
curl -s http://localhost:9540/metrics | grep -E '^(eval_keyword_coverage|celery_eval_task_duration_seconds_count|celery_queue_depth|celery_eval_task_failures_total) '

# Prometheus scrape targets - all should be "up":
curl -s http://localhost:9090/api/v1/targets \
  | python -c "import sys,json;[print(t['labels']['job'], t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]"
```

---

## 11. Grafana dashboard

Open <http://localhost:3000/d/civic-guard> (anonymous access is on; no login).

The **Naija Civic Guard – RAG** dashboard is provisioned automatically. Set the
time range to *Last 15 minutes*, refresh every 5s. Run a few more queries
(step 5) and watch:

- **Request latency by stage** — `retrieval` flat and low, `generation` the bulk
- **Generation tokens/sec by provider** — one series per provider you've used
- **MCP tool-call breakdown** — `lookup_section` / `find_related_sections` / `search_precedent`
- **Eval keyword coverage** — updates a step *behind* the latency panels (that's the async eval lag; it's expected)
- **Celery eval queue depth / task duration** — near zero when eval keeps up

Panels can take 10–20 s to first render after the dashboard loads.

---

## 12. Switch the generation provider and see the split

Currently `LLM_PROVIDER` comes from `.env` / compose (default `openai`).
Switch to the host's Ollama and recreate just the gateway:

```bash
LLM_PROVIDER=ollama docker compose up -d gateway
# wait ~40s, then re-check /metrics:
docker compose logs gateway | grep "warmed up on boot"
```

Run the batch again (step 5, a few queries). Then:

```bash
curl -s http://localhost:8000/metrics | grep '^llm_provider_requests_total'
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c "
SELECT provider, count(*) AS n,
       round(avg(generation_time_ms)::numeric,0) AS gen_avg_ms,
       round(avg(tokens_per_second)::numeric,1)  AS tok_per_s
FROM rag_request_metrics WHERE coalesce(error,'')='' GROUP BY provider;"
```

You'll see two `provider` rows — `openai` fast (~1 s, hundreds of tok/s) and
`ollama` slow (tens of seconds, single-digit tok/s on CPU). Retrieval latency
is identical for both. Switch back with `LLM_PROVIDER=openai docker compose up -d gateway`.

---

## 13. The browser chat page

Open <http://localhost:8000/>. It provisions its own `browser-ui` API key
(embedded in the page), so you can just type a question and hit **Ask AI** —
the answer streams in with its source sections and generation time. A 429 or
401 shows a readable message instead of a blank error.

---

## 14. dbt analytics models

Build the three models over the Postgres logs:

```bash
docker compose --profile analytics run --rm dbt build
```

Expected: `Completed successfully` / `PASS=9`. Then read them:

```bash
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c \
  "SELECT * FROM analytics.avg_latency_by_provider_and_stage ORDER BY provider, stage;"
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c \
  "SELECT * FROM analytics.eval_score_trend;"
docker compose exec -T postgres psql -U civicguard -d civicguard -P pager=off -c \
  "SELECT * FROM analytics.daily_query_volume;"
```

---

## 15. MCP tool server directly (optional)

The MCP server is a real container on `:8100`. Call a tool over the same
transport the agent uses:

```bash
docker compose exec -T gateway python -c "
from rag_engine.mcp_client import McpToolClient
c = McpToolClient(url='http://mcp:8100/mcp'); c.wait_ready(20)
print('lookup_section(45)     ->', c.lookup_section(45)[0].get('found'), c.lookup_section(45)[1], 'ms')
print('find_related(45)       ->', c.find_related_sections('Section 45')[0].get('references'))
print('search_precedent(...)  ->', c.search_precedent('murder')[0])
c.close()"
```

`search_precedent` returns the stub message
(`not yet implemented — case law integration planned`), by design.

---

## 16. Tear down

```bash
docker compose down            # stop + remove containers, keep data volumes
docker compose down -v         # also wipe Postgres + ChromaDB volumes (fresh start next time)
```

---

## Troubleshooting

| symptom | fix |
|---|---|
| gateway restarts / `ImportError: cc_delim_re` | `requirements.txt` must pin `Django==6.0.5` (6.1 breaks DRF 3.17); rebuild: `docker compose build gateway worker` |
| retrieval returns nothing | ChromaDB wasn't ingested — `docker compose exec gateway python manage.py ingest` |
| `/api/chat/` hangs on the Ollama provider | host Ollama not running, or the model isn't pulled (`ollama pull llama3.2`) |
| Grafana panels blank | give them 10–20 s; confirm data with the Prometheus queries in step 10 |
| a published host port fails to bind (Windows) | that port is in a reserved range — change the `*_HOST_PORT` in `.env` |
