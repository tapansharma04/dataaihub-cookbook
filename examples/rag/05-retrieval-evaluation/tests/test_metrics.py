"""Deterministic metric unit tests — no network, no models."""

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


def test_recall_multiple_relevant_and_duplicates_count_once():
    # Duplicate "a" in ranking still contributes one hit.
    assert recall_at_k(["a", "a", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)


def test_recall_respects_k_cutoff():
    assert recall_at_k(["x", "a", "b"], {"a", "b"}, k=1) == 0.0
    assert recall_at_k(["x", "a", "b"], {"a", "b"}, k=2) == 0.5


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_rr_relevant_at_rank_1():
    assert reciprocal_rank(["a", "b"], {"a"}, k=3) == 1.0


def test_rr_relevant_later():
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / 3)


def test_rr_no_relevant_result():
    assert reciprocal_rank(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_rr_outside_k_is_zero():
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_mrr_aggregate():
    rrs = [1.0, 0.5, 0.0]
    assert mean_reciprocal_rank(rrs) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_first_relevant_rank():
    assert first_relevant_rank(["x", "a"], {"a"}, k=3) == 2
    assert first_relevant_rank(["x", "y"], {"a"}, k=3) is None


def test_ndcg_ideal_ranking():
    # Perfect order of grades 3 then 1.
    ndcg, dcg, idcg = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2)
    assert dcg == pytest.approx(idcg)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_imperfect_ranking():
    ndcg_bad, _, idcg = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=2)
    ndcg_good, _, _ = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2)
    assert idcg > 0
    assert ndcg_bad < ndcg_good
    assert 0.0 < ndcg_bad < 1.0


def test_ndcg_irrelevant_results():
    ndcg, dcg, idcg = ndcg_at_k(["x", "y"], {"a": 3}, k=2)
    assert dcg == 0.0
    assert idcg > 0
    assert ndcg == 0.0


def test_ndcg_graded_gain_formula():
    # Manual DCG for grades [3, 0, 1]:
    # (2^3-1)/log2(2) + 0 + (2^1-1)/log2(4) = 7/1 + 1/2 = 7.5
    grades = [3, 0, 1]
    expected = 7.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    assert dcg_at_k(grades, k=3) == pytest.approx(expected)


def test_ndcg_zero_idcg_edge_case():
    ndcg, dcg, idcg = ndcg_at_k(["a", "b"], {}, k=2)
    assert idcg == 0.0
    assert ndcg == 0.0
    assert dcg == 0.0


def test_idcg_uses_best_possible_ordering():
    # IDCG should ignore retrieval order and sort grades desc.
    assert idcg_at_k([1, 3], k=2) == pytest.approx(dcg_at_k([3, 1], k=2))
