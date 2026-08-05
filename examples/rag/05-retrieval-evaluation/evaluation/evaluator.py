"""Score ranked retrieval results against golden judgments."""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.dataset import EvalQuery
from evaluation.failure import FailureSummary, classify_failure
from evaluation.metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    relevance_grade: int
    is_relevant: bool
    text: str = ""
    title: str = ""


@dataclass(frozen=True)
class QueryEvalResult:
    query_id: str
    query: str
    pipeline: str
    k: int
    retrieved: list[RankedHit]
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    dcg_at_k: float
    idcg_at_k: float
    ndcg_at_k: float
    failure: FailureSummary
    latency_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEvalResult:
    pipeline: str
    k: int
    query_count: int
    mean_recall_at_k: float
    mrr: float
    mean_ndcg_at_k: float
    per_query: list[QueryEvalResult]
    total_latency_ms: int = 0


def _title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("# ").strip()
    return ""


def evaluate_ranking(
    case: EvalQuery,
    retrieved_ids: list[str],
    *,
    pipeline: str,
    k: int,
    chunk_text_by_id: dict[str, str] | None = None,
    latency_ms: dict[str, int] | None = None,
) -> QueryEvalResult:
    """Compare one ranked list to golden judgments and compute metrics."""
    texts = chunk_text_by_id or {}
    relevant = case.relevant_ids()
    hits: list[RankedHit] = []
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        grade = case.grade(chunk_id)
        text = texts.get(chunk_id, "")
        hits.append(
            RankedHit(
                chunk_id=chunk_id,
                rank=rank,
                relevance_grade=grade,
                is_relevant=grade >= 1,
                text=text,
                title=_title_from_text(text),
            )
        )

    recall = recall_at_k(retrieved_ids, relevant, k=k)
    rr = reciprocal_rank(retrieved_ids, relevant, k=k)
    first_rank = first_relevant_rank(retrieved_ids, relevant, k=k)
    ndcg, dcg, idcg = ndcg_at_k(retrieved_ids, case.relevance, k=k)
    failure = classify_failure(retrieved_ids, case, k=k)

    return QueryEvalResult(
        query_id=case.id,
        query=case.query,
        pipeline=pipeline,
        k=k,
        retrieved=hits,
        recall_at_k=recall,
        reciprocal_rank=rr,
        first_relevant_rank=first_rank,
        dcg_at_k=dcg,
        idcg_at_k=idcg,
        ndcg_at_k=ndcg,
        failure=failure,
        latency_ms=dict(latency_ms or {}),
    )


def aggregate_pipeline(
    pipeline: str,
    k: int,
    per_query: list[QueryEvalResult],
) -> PipelineEvalResult:
    if not per_query:
        return PipelineEvalResult(
            pipeline=pipeline,
            k=k,
            query_count=0,
            mean_recall_at_k=0.0,
            mrr=0.0,
            mean_ndcg_at_k=0.0,
            per_query=[],
            total_latency_ms=0,
        )
    mean_recall = sum(q.recall_at_k for q in per_query) / len(per_query)
    mrr = mean_reciprocal_rank([q.reciprocal_rank for q in per_query])
    mean_ndcg = sum(q.ndcg_at_k for q in per_query) / len(per_query)
    total_latency = sum(int(q.latency_ms.get("total_ms", 0)) for q in per_query)
    return PipelineEvalResult(
        pipeline=pipeline,
        k=k,
        query_count=len(per_query),
        mean_recall_at_k=mean_recall,
        mrr=mrr,
        mean_ndcg_at_k=mean_ndcg,
        per_query=per_query,
        total_latency_ms=total_latency,
    )
