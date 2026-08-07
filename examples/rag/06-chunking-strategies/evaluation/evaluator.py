"""Score ranked retrieval against evidence-derived chunk relevance."""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.dataset import EvalQuery
from evaluation.evidence import (
    EvidenceUnit,
    build_chunk_relevance,
    evidence_coverage,
)
from evaluation.metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.chunking.base import Chunk


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    relevance_grade: int
    is_relevant: bool
    score: float
    text: str = ""
    section: str | None = None
    evidence_ids: tuple[str, ...] = ()
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class QueryEvalResult:
    query_id: str
    query: str
    strategy: str
    k: int
    retrieved: list[RankedHit]
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    dcg_at_k: float
    idcg_at_k: float
    ndcg_at_k: float
    evidence_coverage: float
    evidence_found: list[str]
    evidence_missed: list[str]
    chunk_relevance: dict[str, int]
    latency_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyEvalResult:
    strategy: str
    k: int
    query_count: int
    mean_recall_at_k: float
    mrr: float
    mean_ndcg_at_k: float
    mean_evidence_coverage: float
    per_query: list[QueryEvalResult]
    total_retrieval_ms: int = 0


def evaluate_ranking(
    case: EvalQuery,
    ranked: list[tuple[Chunk, float]],
    *,
    strategy: str,
    k: int,
    all_chunks: list[Chunk],
    units_by_id: dict[str, EvidenceUnit],
    binary_threshold: int = 1,
    latency_ms: dict[str, int] | None = None,
) -> QueryEvalResult:
    """Compare one ranked list using evidence-derived chunk grades."""
    chunk_relevance = build_chunk_relevance(
        all_chunks,
        evidence_grades=case.evidence_grades,
        units_by_id=units_by_id,
    )
    relevant_ids = {
        cid for cid, grade in chunk_relevance.items() if grade >= binary_threshold
    }
    retrieved_ids = [chunk.id for chunk, _ in ranked]

    hits: list[RankedHit] = []
    retrieved_chunks: list[Chunk] = []
    for rank, (chunk, score) in enumerate(ranked[:k], start=1):
        grade = chunk_relevance.get(chunk.id, 0)
        hits.append(
            RankedHit(
                chunk_id=chunk.id,
                rank=rank,
                relevance_grade=grade,
                is_relevant=grade >= binary_threshold,
                score=score,
                text=chunk.text,
                section=chunk.section,
                evidence_ids=chunk.evidence_ids,
                start=chunk.start,
                end=chunk.end,
            )
        )
        retrieved_chunks.append(chunk)

    recall = recall_at_k(retrieved_ids, relevant_ids, k=k)
    rr = reciprocal_rank(retrieved_ids, relevant_ids, k=k)
    first_rank = first_relevant_rank(retrieved_ids, relevant_ids, k=k)
    ndcg, dcg, idcg = ndcg_at_k(retrieved_ids, chunk_relevance, k=k)

    coverage = evidence_coverage(
        retrieved_chunks,
        required_evidence_ids=case.required_evidence_ids(),
        units_by_id=units_by_id,
    )

    return QueryEvalResult(
        query_id=case.id,
        query=case.query,
        strategy=strategy,
        k=k,
        retrieved=hits,
        recall_at_k=recall,
        reciprocal_rank=rr,
        first_relevant_rank=first_rank,
        dcg_at_k=dcg,
        idcg_at_k=idcg,
        ndcg_at_k=ndcg,
        evidence_coverage=float(coverage["coverage"]),
        evidence_found=list(coverage["found"]),  # type: ignore[arg-type]
        evidence_missed=list(coverage["missed"]),  # type: ignore[arg-type]
        chunk_relevance=chunk_relevance,
        latency_ms=dict(latency_ms or {}),
    )


def aggregate_strategy(
    strategy: str,
    k: int,
    per_query: list[QueryEvalResult],
) -> StrategyEvalResult:
    if not per_query:
        return StrategyEvalResult(
            strategy=strategy,
            k=k,
            query_count=0,
            mean_recall_at_k=0.0,
            mrr=0.0,
            mean_ndcg_at_k=0.0,
            mean_evidence_coverage=0.0,
            per_query=[],
            total_retrieval_ms=0,
        )
    mean_recall = sum(q.recall_at_k for q in per_query) / len(per_query)
    mrr = mean_reciprocal_rank([q.reciprocal_rank for q in per_query])
    mean_ndcg = sum(q.ndcg_at_k for q in per_query) / len(per_query)
    mean_cov = sum(q.evidence_coverage for q in per_query) / len(per_query)
    total_latency = sum(int(q.latency_ms.get("retrieval_ms", 0)) for q in per_query)
    return StrategyEvalResult(
        strategy=strategy,
        k=k,
        query_count=len(per_query),
        mean_recall_at_k=mean_recall,
        mrr=mrr,
        mean_ndcg_at_k=mean_ndcg,
        mean_evidence_coverage=mean_cov,
        per_query=per_query,
        total_retrieval_ms=total_latency,
    )
