"""Metric unit tests — mirrored from Example 05, no network."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    dcg_at_k,
    first_relevant_rank,
    idcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_all_relevant_retrieved():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_partially_retrieved():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_none_retrieved():
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_rr_relevant_at_rank_1():
    assert reciprocal_rank(["a", "b"], {"a"}, k=3) == 1.0


def test_rr_no_relevant_result():
    assert reciprocal_rank(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_rr_outside_k_is_zero():
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_mrr_aggregate():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx((1.0 + 0.5) / 3)


def test_first_relevant_rank():
    assert first_relevant_rank(["x", "a"], {"a"}, k=3) == 2
    assert first_relevant_rank(["x", "y"], {"a"}, k=3) is None


def test_ndcg_ideal_ranking():
    ndcg, dcg, idcg = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2)
    assert dcg == pytest.approx(idcg)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_imperfect_ranking():
    ndcg_bad, _, idcg = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=2)
    ndcg_good, _, _ = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2)
    assert ndcg_bad < ndcg_good
    assert idcg > 0


def test_ndcg_no_relevant_is_zero():
    ndcg, dcg, idcg = ndcg_at_k(["x", "y"], {}, k=2)
    assert ndcg == 0.0
    assert idcg == 0.0
    assert dcg == 0.0


def test_dcg_matches_manual():
    # grades [3, 0] → (7)/log2(2) + 0 = 7
    assert dcg_at_k([3, 0], k=2) == pytest.approx(7.0)
    assert idcg_at_k([3, 1], k=2) == pytest.approx(7.0 + 1.0 / math.log2(3))
