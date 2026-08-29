"""Pure, dependency-free evaluation helpers.

Shared by:
  * `retrieval_eval.py`      - the offline batch evaluator (CLI)
  * `rag_engine.tasks`       - the async per-request Celery evaluator

Nothing here touches Django, ChromaDB or the LLM: functions take already-
retrieved text / section ids and return numbers. That's what lets the same
logic run both offline against the eval set and online against live traffic.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# evaluation_set.jsonl lives at the repo root, next to manage.py.
EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "evaluation_set.jsonl"

# Tiny stop-word list for deriving keywords from a free-text user query when no
# ground-truth keyword list exists.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "from", "that",
    "this", "these", "those", "it", "its", "into", "about", "what", "which",
    "who", "whom", "whose", "how", "when", "where", "why", "does", "do", "did",
    "can", "could", "should", "would", "may", "might", "will", "shall", "there",
    "their", "they", "them", "you", "your", "his", "her", "hers", "our", "us",
    "if", "than", "then", "so", "such", "not", "no", "any", "all", "some",
}


def load_evaluation_set(path: Path | str = EVAL_SET_PATH) -> list[dict]:
    """Read evaluation_set.jsonl -> list of {query, target, keywords}."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_query(q: str) -> str:
    return " ".join((q or "").lower().split())


@lru_cache(maxsize=8)
def _ground_truth_index(path_str: str) -> dict[str, dict]:
    return {normalize_query(e["query"]): e for e in load_evaluation_set(path_str)}


def find_ground_truth(query: str, path: Path | str = EVAL_SET_PATH) -> dict | None:
    """Return the eval-set entry for `query`, or None for an off-set (real) query."""
    return _ground_truth_index(str(path)).get(normalize_query(query))


def keyword_coverage(keywords, retrieved_text: str) -> float | None:
    """Fraction of `keywords` that appear (case-insensitive substring) in the
    retrieved context. None when there are no keywords to check."""
    if not keywords:
        return None
    text = (retrieved_text or "").lower()
    hits = sum(1 for k in keywords if str(k).lower() in text)
    return hits / len(keywords)


def derive_keywords_from_query(query: str) -> list[str]:
    """Fallback keyword set for a query with no ground truth: content words
    from the query itself (>=3 chars, minus stop words)."""
    toks = re.findall(r"[a-z0-9]{3,}", (query or "").lower())
    seen, out = set(), []
    for t in toks:
        if t not in _STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def section_hit(retrieved_section_ids, target: str) -> bool:
    return target in (retrieved_section_ids or [])


def reciprocal_rank(retrieved_section_ids, target: str) -> float:
    for i, sid in enumerate(retrieved_section_ids or []):
        if sid == target:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    query: str,
    retrieved_context,
    retrieved_section_ids,
    response_text: str = "",
    eval_set_path: Path | str = EVAL_SET_PATH,
) -> dict:
    """Evaluate a single (already executed) RAG request.

    `retrieved_context` may be a list of chunk strings or one joined string.
    Returns a plain dict; hit / rank / target stay None when the query has no
    ground truth in the eval set. keyword_coverage is computed either way
    (against ground-truth keywords, or against keywords derived from the query).
    """
    if isinstance(retrieved_context, (list, tuple)):
        ctx = "\n\n".join(str(c) for c in retrieved_context)
    else:
        ctx = retrieved_context or ""

    section_ids = list(retrieved_section_ids or [])
    gt = find_ground_truth(query, eval_set_path)

    if gt is not None:
        keywords = list(gt.get("keywords", []))
        keyword_source = "ground_truth"
        target = gt.get("target")
        hit = section_hit(section_ids, target)
        rank = reciprocal_rank(section_ids, target)
    else:
        keywords = derive_keywords_from_query(query)
        keyword_source = "query"
        target = None
        hit = None
        rank = None

    return {
        "matched_ground_truth": gt is not None,
        "keyword_coverage": keyword_coverage(keywords, ctx),
        "keyword_source": keyword_source,
        "keywords_checked": keywords,
        "target_section": target,
        "hit": hit,
        "reciprocal_rank": rank,
        "retrieved_section_ids": section_ids,
        "response_chars": len(response_text or ""),
    }
