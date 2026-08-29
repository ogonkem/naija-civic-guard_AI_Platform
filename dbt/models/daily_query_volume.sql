-- Query volume per day: total, errors, and how many hit the async evaluator.
{{ config(materialized='view') }}

select
    date_trunc('day', m.timestamp)::date            as day,
    count(*)                                         as queries,
    count(*) filter (where coalesce(m.error, '') <> '')  as errored,
    count(*) filter (where m.verify_retry)           as with_agent_retry,
    count(distinct e.request_id)                     as evaluated,
    round(avg(m.total_time_ms)::numeric, 1)          as avg_total_ms,
    round(avg(m.tokens_generated)::numeric, 1)       as avg_tokens
from {{ source('gateway', 'rag_request_metrics') }} m
left join {{ source('gateway', 'eval_results') }} e
    on e.request_id = m.request_id
group by 1
order by 1
