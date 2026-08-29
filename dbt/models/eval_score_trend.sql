-- Retrieval quality over time for eval-set queries (matched_ground_truth).
--   hit_rate  : fraction where the target section was retrieved  (recall@k proxy)
--   mrr       : mean reciprocal rank of the target section
--   avg_keyword_coverage : share of expected legal terms present in the context
--                          (precision proxy - the answer's terminology quality)
{{ config(materialized='view') }}

select
    date_trunc('day', m.timestamp)::date                 as day,
    coalesce(nullif(m.provider, ''), 'unknown')          as provider,
    count(*)                                             as eval_set_queries,
    round(avg((e.hit)::int)::numeric, 3)                 as hit_rate,
    round(avg(e.reciprocal_rank)::numeric, 3)            as mrr,
    round(avg(e.keyword_coverage)::numeric, 3)           as avg_keyword_coverage
from {{ source('gateway', 'eval_results') }} e
join {{ source('gateway', 'rag_request_metrics') }} m
    on m.request_id = e.request_id
where e.matched_ground_truth
group by 1, 2
order by 1, 2
