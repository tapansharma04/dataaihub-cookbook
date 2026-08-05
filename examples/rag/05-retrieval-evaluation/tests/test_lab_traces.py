"""Validate retrieval-evaluation lab traces (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

TRACES = Path(__file__).resolve().parents[1] / "lab_traces.json"
REPORT = Path(__file__).resolve().parents[1] / "evaluation_report.json"

REQUIRED_STEPS = {
    "evaluation-dataset",
    "relevance-judgments",
    "run-retrieval",
    "inspect-ranked-results",
    "calculate-recall-at-k",
    "calculate-rr-mrr",
    "calculate-ndcg-at-k",
    "compare-pipelines",
    "inspect-failures",
    "evaluation-summary",
}

EXPECTED_TRACE_IDS = {
    "discover-auth-idle-timeout",
    "ranking-sensitivity-profile-async",
    "regression-econn-42-remediation",
}


def test_lab_traces_exist_and_are_measured():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    assert len(traces) == 3
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        assert trace["labId"] == "retrieval-evaluation"
        assert trace["metricsProvenance"] == "measured"
        assert trace["cookbook"]["path"] == "examples/rag/05-retrieval-evaluation"
        assert trace["teachingClass"] in {
            "DISCOVER",
            "RANKING_SENSITIVITY",
            "REGRESSION",
        }
        assert "selectionNote" in trace
        assert trace["input"]["relevanceJudgments"]
        assert trace["input"]["rationale"]
        assert len(trace["pipelines"]) == 4


def test_lab_traces_include_required_stages():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    for trace in traces:
        step_ids = {step["id"] for step in trace["steps"]}
        assert REQUIRED_STEPS <= step_ids
        for pipeline in trace["pipelines"]:
            metrics = pipeline["metrics"]
            assert "recallAtK" in metrics
            assert "reciprocalRank" in metrics
            assert "dcgAtK" in metrics
            assert "idcgAtK" in metrics
            assert "ndcgAtK" in metrics
            assert "firstRelevantRank" in metrics
            assert metrics["provenance"] == "computed"
            for hit in pipeline["retrieved"]:
                assert "chunkId" in hit
                assert "rank" in hit
                assert "relevanceGrade" in hit
                assert "isRelevant" in hit


def test_per_query_metrics_match_retrieved_ids():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    for trace in traces:
        relevance = trace["input"]["relevanceJudgments"]
        relevant = {cid for cid, g in relevance.items() if g >= 1}
        k = trace["input"]["k"]
        for pipeline in trace["pipelines"]:
            ids = [h["chunkId"] for h in pipeline["retrieved"]]
            metrics = pipeline["metrics"]
            assert metrics["recallAtK"] == round(recall_at_k(ids, relevant, k=k), 6)
            assert metrics["reciprocalRank"] == round(
                reciprocal_rank(ids, relevant, k=k), 6
            )
            ndcg, dcg, idcg = ndcg_at_k(ids, relevance, k=k)
            assert metrics["ndcgAtK"] == round(ndcg, 6)
            assert metrics["dcgAtK"] == round(dcg, 6)
            assert metrics["idcgAtK"] == round(idcg, 6)


def test_aggregates_correspond_to_report():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    # Aggregates embedded in traces should match the full measured report.
    for trace in traces:
        assert trace["fullEvaluationAggregates"] == report["aggregates"]


def test_report_aggregates_match_per_query_means():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    k = report["k"]
    for agg in report["aggregates"]:
        name = agg["pipeline"]
        per = report["perQuery"][name]
        recalls = [row["recallAtK"] for row in per.values()]
        rrs = [row["reciprocalRank"] for row in per.values()]
        ndcgs = [row["ndcgAtK"] for row in per.values()]
        assert agg["queryCount"] == len(per)
        assert agg["meanRecallAtK"] == round(sum(recalls) / len(recalls), 6)
        assert agg["mrr"] == round(mean_reciprocal_rank(rrs), 6)
        assert agg["meanNdcgAtK"] == round(sum(ndcgs) / len(ndcgs), 6)
        assert agg["k"] == k


def test_discover_case_shows_miss_vs_hit():
    traces = {t["traceId"]: t for t in json.loads(TRACES.read_text(encoding="utf-8"))}
    trace = traces["discover-auth-idle-timeout"]
    assert trace["teachingClass"] == "DISCOVER"
    by_pipe = {p["pipeline"]: p for p in trace["pipelines"]}
    dense_ids = {h["chunkId"] for h in by_pipe["dense"]["retrieved"]}
    hybrid_ids = {h["chunkId"] for h in by_pipe["hybrid"]["retrieved"]}
    assert "sample-12" in dense_ids
    assert "sample-12" not in hybrid_ids
    assert (
        by_pipe["dense"]["metrics"]["recallAtK"]
        > by_pipe["hybrid"]["metrics"]["recallAtK"]
    )


def test_ranking_sensitivity_recall_flat_ndcg_moves():
    traces = {t["traceId"]: t for t in json.loads(TRACES.read_text(encoding="utf-8"))}
    trace = traces["ranking-sensitivity-profile-async"]
    assert trace["teachingClass"] == "RANKING_SENSITIVITY"
    recalls = {p["pipeline"]: p["metrics"]["recallAtK"] for p in trace["pipelines"]}
    ndcgs = {p["pipeline"]: p["metrics"]["ndcgAtK"] for p in trace["pipelines"]}
    assert len(set(recalls.values())) == 1
    assert len(set(ndcgs.values())) > 1
    # Measured: advanced reorder did not improve nDCG vs dense/hybrid.
    assert ndcgs["hybrid-reranked"] < ndcgs["dense"]
    assert ndcgs["hybrid-reranked"] < ndcgs["hybrid"]


def test_regression_case_advanced_not_better():
    traces = {t["traceId"]: t for t in json.loads(TRACES.read_text(encoding="utf-8"))}
    trace = traces["regression-econn-42-remediation"]
    assert trace["teachingClass"] == "REGRESSION"
    by_pipe = {p["pipeline"]: p["metrics"] for p in trace["pipelines"]}
    assert by_pipe["hybrid-reranked"]["recallAtK"] <= by_pipe["hybrid"]["recallAtK"]
    assert by_pipe["query-transform"]["ndcgAtK"] <= by_pipe["dense"]["ndcgAtK"]


def test_no_fabricated_chunk_ids():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for trace in traces:
        qid = trace["input"]["queryId"]
        for pipeline in trace["pipelines"]:
            name = pipeline["pipeline"]
            expected = report["perQuery"][name][qid]["retrievedIds"]
            got = [h["chunkId"] for h in pipeline["retrieved"]]
            assert got == expected
