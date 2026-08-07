"""Transparent retrieval metrics: Recall@K, RR/MRR, nDCG@K.

Reused from Example 05 (standard IR definitions — not custom variants).
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int,
) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hit = {cid for cid in top if cid in relevant_ids}
    return len(hit) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int | None = None,
) -> float:
    """RR = 1 / rank of the first relevant result (1-based)."""
    if not relevant_ids:
        return 0.0
    ranking = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """MRR = mean of per-query reciprocal ranks."""
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _gain(relevance: int) -> float:
    """Graded gain: 2^rel - 1 (Järvelin & Kekäläinen)."""
    if relevance <= 0:
        return 0.0
    return float((1 << relevance) - 1)


def dcg_at_k(relevance_grades: list[int], *, k: int) -> float:
    """DCG@K = Σ_{i=1..K} (2^{rel_i} - 1) / log2(i + 1)."""
    if k <= 0:
        return 0.0
    total = 0.0
    for i in range(k):
        rel = relevance_grades[i] if i < len(relevance_grades) else 0
        total += _gain(rel) / math.log2(i + 2)
    return total


def idcg_at_k(all_grades: list[int], *, k: int) -> float:
    """Ideal DCG@K: DCG of the best possible ranking of known grades."""
    ideal = sorted((g for g in all_grades if g > 0), reverse=True)
    return dcg_at_k(ideal, k=k)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, int],
    *,
    k: int,
) -> tuple[float, float, float]:
    """nDCG@K with graded relevance. Returns ``(ndcg, dcg, idcg)``."""
    if k <= 0:
        return 0.0, 0.0, 0.0
    grades = [relevance.get(cid, 0) for cid in retrieved_ids[:k]]
    dcg = dcg_at_k(grades, k=k)
    idcg = idcg_at_k(list(relevance.values()), k=k)
    if idcg <= 0.0:
        return 0.0, dcg, idcg
    return dcg / idcg, dcg, idcg


def first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int | None = None,
) -> int | None:
    """1-based rank of the first relevant hit, or None if absent."""
    ranking = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None
