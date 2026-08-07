"""Evaluator tests with synthetic chunks — no network."""

from __future__ import annotations

from evaluation.dataset import EvalQuery
from evaluation.evaluator import aggregate_strategy, evaluate_ranking
from evaluation.evidence import EvidenceUnit
from rag.chunking.base import Chunk


def _unit(eid: str, start: int, end: int) -> EvidenceUnit:
    return EvidenceUnit(
        id=eid,
        section="S",
        anchor="x" * (end - start),
        start=start,
        end=end,
    )


def _chunk(cid: str, start: int, end: int, strategy: str = "fixed") -> Chunk:
    return Chunk(
        id=cid,
        text="t" * (end - start),
        start=start,
        end=end,
        strategy=strategy,
        source="sample",
        evidence_ids=(),
    )


def test_evaluate_ranking_recall_and_rr():
    units = {
        "ev-a": _unit("ev-a", 0, 20),
        "ev-b": _unit("ev-b", 100, 120),
    }
    chunks = [
        _chunk("fixed-0000", 0, 30),  # full ev-a
        _chunk("fixed-0001", 40, 60),  # irrelevant
        _chunk("fixed-0002", 90, 130),  # full ev-b
    ]
    case = EvalQuery(
        id="q1",
        query="test",
        evidence_grades={"ev-a": 3, "ev-b": 2},
        rationale={"ev-a": "a", "ev-b": "b"},
    )
    ranked = [(chunks[0], 0.9), (chunks[1], 0.5), (chunks[2], 0.4)]
    result = evaluate_ranking(
        case,
        ranked,
        strategy="fixed",
        k=3,
        all_chunks=chunks,
        units_by_id=units,
    )
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.retrieved[0].relevance_grade == 3
    assert result.evidence_coverage == 1.0


def test_evaluate_no_relevant_retrieval():
    units = {"ev-a": _unit("ev-a", 0, 20)}
    chunks = [
        _chunk("fixed-0000", 0, 30),
        _chunk("fixed-0001", 50, 80),
    ]
    case = EvalQuery(
        id="q1",
        query="test",
        evidence_grades={"ev-a": 3},
        rationale={"ev-a": "a"},
    )
    ranked = [(chunks[1], 0.9)]  # miss
    result = evaluate_ranking(
        case,
        ranked,
        strategy="fixed",
        k=3,
        all_chunks=chunks,
        units_by_id=units,
    )
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.ndcg_at_k == 0.0
    assert result.first_relevant_rank is None
    assert result.evidence_coverage == 0.0


def test_aggregate_mrr():
    units = {"ev-a": _unit("ev-a", 0, 10)}
    chunks = [_chunk("fixed-0000", 0, 20), _chunk("fixed-0001", 50, 60)]
    case = EvalQuery(
        id="q",
        query="q",
        evidence_grades={"ev-a": 3},
        rationale={"ev-a": "a"},
    )
    r1 = evaluate_ranking(
        case,
        [(chunks[0], 1.0)],
        strategy="fixed",
        k=3,
        all_chunks=chunks,
        units_by_id=units,
        latency_ms={"retrieval_ms": 10},
    )
    r2 = evaluate_ranking(
        case,
        [(chunks[1], 1.0), (chunks[0], 0.5)],
        strategy="fixed",
        k=3,
        all_chunks=chunks,
        units_by_id=units,
        latency_ms={"retrieval_ms": 20},
    )
    agg = aggregate_strategy("fixed", 3, [r1, r2])
    assert agg.mrr == 0.75  # (1.0 + 0.5) / 2
    assert agg.total_retrieval_ms == 30
