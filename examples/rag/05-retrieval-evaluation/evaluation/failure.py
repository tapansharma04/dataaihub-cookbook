"""Explanatory failure labels — summaries, not replacement metrics."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.dataset import EvalQuery
from evaluation.metrics import first_relevant_rank


@dataclass(frozen=True)
class FailureSummary:
    """Per-query retrieval outcome labels for inspection.

    Labels summarize binary presence/rank of relevant evidence. Graded
    nDCG remains the ranking-quality metric.
    """

    label: str  # GOOD | LATE | MISS | MIXED
    first_relevant_rank: int | None
    missed_ids: list[str]
    late_ids: list[str]
    good_ids: list[str]
    explanation: str


def classify_failure(
    retrieved_ids: list[str],
    case: EvalQuery,
    *,
    k: int,
    late_after_rank: int = 1,
) -> FailureSummary:
    """Classify how relevant evidence sits in the top-K ranking.

    Per relevant chunk (grade >= 1):
    - GOOD: present at rank <= late_after_rank
    - LATE: present in top-K but rank > late_after_rank
    - MISS: absent from top-K

    Query-level label:
    - GOOD: every relevant chunk is GOOD (or the only relevant is GOOD)
    - LATE: at least one LATE, none MISS
    - MISS: no relevant chunk in top-K
    - MIXED: some hit, some miss
    """
    relevant = sorted(case.relevant_ids())
    top = retrieved_ids[:k]
    rank_by_id = {cid: i for i, cid in enumerate(top, start=1)}

    good_ids: list[str] = []
    late_ids: list[str] = []
    missed_ids: list[str] = []
    for cid in relevant:
        rank = rank_by_id.get(cid)
        if rank is None:
            missed_ids.append(cid)
        elif rank <= late_after_rank:
            good_ids.append(cid)
        else:
            late_ids.append(cid)

    first_rank = first_relevant_rank(retrieved_ids, set(relevant), k=k)

    # Query-level label emphasizes the first relevant hit (MRR-aligned),
    # while good/late/missed ID lists preserve per-chunk detail for graded sets.
    if not relevant:
        label = "MISS"
        explanation = "No positive relevance judgments for this query."
    elif first_rank is None:
        label = "MISS"
        explanation = f"No relevant chunks in top-{k}. Missing: {missed_ids}."
    elif missed_ids:
        label = "MIXED"
        explanation = (
            f"Retrieved some relevant evidence but missed {missed_ids}; "
            f"first relevant at rank {first_rank}."
        )
    elif first_rank > late_after_rank:
        label = "LATE"
        explanation = (
            f"Relevant evidence retrieved but first hit at rank {first_rank} "
            f"(after rank {late_after_rank})."
        )
    else:
        label = "GOOD"
        explanation = (
            f"Relevant evidence at rank {first_rank} (within top {late_after_rank})."
        )

    return FailureSummary(
        label=label,
        first_relevant_rank=first_rank,
        missed_ids=missed_ids,
        late_ids=late_ids,
        good_ids=good_ids,
        explanation=explanation,
    )
