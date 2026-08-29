-- Mean / p50 / p95 latency for each pipeline stage, split by LLM provider.
-- The gateway stores one column per stage; unpivot so stage is a dimension.
{{ config(materialized='view') }}

with unpivoted as (
    {% set stages = [
        ('classify',   'classify_ms'),
        ('retrieval',  'retrieve_ms'),
        ('chain',      'chain_ms'),
        ('verify',     'verify_ms'),
        ('generation', 'generation_time_ms'),
        ('total',      'total_time_ms'),
    ] %}
    {% for stage_name, col in stages %}
    select
        coalesce(nullif(provider, ''), 'unknown') as provider,
        '{{ stage_name }}'                        as stage,
        {{ col }}                                 as latency_ms
    from {{ source('gateway', 'rag_request_metrics') }}
    where {{ col }} is not null
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)

select
    provider,
    stage,
    count(*)                                                             as n,
    round(avg(latency_ms)::numeric, 1)                                   as avg_ms,
    round((percentile_cont(0.50) within group (order by latency_ms))::numeric, 1) as p50_ms,
    round((percentile_cont(0.95) within group (order by latency_ms))::numeric, 1) as p95_ms
from unpivoted
group by provider, stage
order by provider,
    array_position(array['classify','retrieval','chain','verify','generation','total'], stage)
