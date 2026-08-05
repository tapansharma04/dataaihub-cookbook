"""Evaluator and failure-analysis tests — no network."""

from __future__ import annotations

import pytest

from config import get_settings
from evaluation.dataset import EvalQuery, load_eval_dataset
from evaluation.evaluator import aggregate_pipeline, evaluate_ranking
from evaluation.failure import classify_failure


def _case() -> EvalQuery:
    return EvalQuery(
        id="demo",
        query="demo question",
        relevance={"a": 3, "b": 1},
        rationale={"a": "primary", "b": "tangential"},
    )


def test_evaluate_ranking_matches_judgments():
    result = evaluate_ranking(
        _case(),
        ["x", "a", "b"],
        pipeline="dense",
        k=3,
        chunk_text_by_id={"a": "## Auth", "b": "## Other", "x": "## Noise"},
    )
    assert [h.chunk_id for h in result.retrieved] == ["x", "a", "b"]
    assert result.retrieved[0].relevance_grade == 0
    assert result.retrieved[0].is_relevant is False
    assert result.retrieved[1].relevance_grade == 3
    assert result.retrieved[1].is_relevant is True
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == pytest.approx(0.5)
    assert result.first_relevant_rank == 2
    assert result.ndcg_at_k > 0
    assert result.failure.label in {"LATE", "MIXED", "GOOD"}


def test_aggregate_across_queries():
    case = _case()
    q1 = evaluate_ranking(case, ["a", "x", "y"], pipeline="hybrid", k=3)
    q2 = evaluate_ranking(case, ["x", "y", "z"], pipeline="hybrid", k=3)
    agg = aggregate_pipeline("hybrid", 3, [q1, q2])
    assert agg.query_count == 2
    # q1 retrieves 1 of 2 relevant → Recall=0.5; q2 retrieves none → 0.0
    assert agg.mean_recall_at_k == pytest.approx((0.5 + 0.0) / 2)
    assert agg.mrr == pytest.approx((1.0 + 0.0) / 2)
    assert 0.0 <= agg.mean_ndcg_at_k <= 1.0


def test_stable_k_behavior():
    case = _case()
    at_1 = evaluate_ranking(case, ["x", "a", "b"], pipeline="dense", k=1)
    at_3 = evaluate_ranking(case, ["x", "a", "b"], pipeline="dense", k=3)
    assert at_1.recall_at_k == 0.0
    assert at_3.recall_at_k == 1.0
    assert len(at_1.retrieved) == 1
    assert len(at_3.retrieved) == 3


def test_failure_miss_late_good():
    case = _case()
    miss = classify_failure(["x", "y", "z"], case, k=3)
    assert miss.label == "MISS"
    assert miss.missed_ids == ["a", "b"]

    late = classify_failure(["x", "a", "b"], case, k=3)
    assert late.label in {"LATE", "MIXED"}
    assert late.first_relevant_rank == 2

    good = classify_failure(["a", "b", "x"], case, k=3)
    assert good.label == "GOOD"
    assert good.first_relevant_rank == 1


def test_load_eval_dataset_has_rationales():
    settings = get_settings()
    dataset = load_eval_dataset(settings.eval_path)
    assert 5 <= len(dataset.queries) <= 10
    for q in dataset.queries:
        assert q.relevance
        assert set(q.relevance) == set(q.rationale)
        assert all(grade >= 1 for grade in q.relevance.values())
