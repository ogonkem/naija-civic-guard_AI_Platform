# dbt analytics (dbt-postgres)

Three views over the gateway's Postgres logs (`rag_request_metrics`,
`eval_results`), materialised into schema `analytics`.

| model | grain | what it answers |
|---|---|---|
| `daily_query_volume` | day | how much traffic, how many errors / agent retries / evaluated |
| `avg_latency_by_provider_and_stage` | provider × stage | avg / p50 / p95 latency per pipeline stage, `openai` vs `ollama` — feeds the README Performance table |
| `eval_score_trend` | day × provider | retrieval quality over time on eval-set queries: `hit_rate` (recall@k proxy), `mrr`, `avg_keyword_coverage` (precision proxy) |

## Run

Inside the compose stack (Postgres reachable as `postgres`):

```bash
docker compose --profile analytics run --rm dbt build      # run + test
docker compose --profile analytics run --rm dbt run
docker compose exec postgres psql -U civicguard -d civicguard \
  -c "select * from analytics.avg_latency_by_provider_and_stage"
```

Locally: `pip install dbt-postgres`, export `POSTGRES_HOST/PORT/USER/PASSWORD/DB`,
then `dbt build` from this directory (`DBT_PROFILES_DIR=.`).

Connection comes from `profiles.yml`, which reads the same `POSTGRES_*` env
vars docker-compose sets.
