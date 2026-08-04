"""Validate query-transformation trace shape (no network)."""

from __future__ import annotations

import json
from pathlib import Path

TRACES = Path(__file__).resolve().parents[1] / "lab_traces.json"

REQUIRED_STEPS = {
    "chunking",
    "embed-corpus",
    "bm25-index",
    "query-transform",
    "query-baseline",
    "prompt",
    "llm",
    "output",
}

EXPECTED_TRACE_IDS = {
    "transformation-bridges-idle-auth",
    "original-query-already-strong-econn42",
    "expansion-adds-platform-noise",
}


def _by_id(traces: list[dict]) -> dict[str, dict]:
    return {t["traceId"]: t for t in traces}


def test_lab_traces_exist_and_are_measured():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    assert len(traces) == 3
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        assert trace["labId"] == "query-transformation"
        assert trace["metricsProvenance"] == "measured"
        assert trace["metrics"]["provenance"] == "measured"
        assert trace["cookbook"]["path"] == "examples/rag/04-query-transformation"
        assert trace["teachingClass"] in {
            "TRANSFORM_HELPS",
            "REDUNDANT",
            "ADDS_NOISE",
        }


def test_lab_traces_include_multi_query_provenance():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    for trace in traces:
        step_ids = {step["id"] for step in trace["steps"]}
        assert REQUIRED_STEPS <= step_ids

        multi = next(s for s in trace["steps"] if s["id"] == "query-transform")
        baseline = next(s for s in trace["steps"] if s["id"] == "query-baseline")
        prompt = next(s for s in trace["steps"] if s["id"] == "prompt")

        assert multi["detail"]["path"] == "multi-query"
        assert baseline["detail"]["path"] == "original-only"
        assert "candidateCounts" in multi["detail"]
        assert "perQueryRetrievalLatencyMs" in multi["detail"]
        assert "rerankedContext" in multi["detail"]
        assert "foundBy" in multi["detail"]["mergedCandidates"][0]
        assert "perQueryProvenance" in multi["detail"]["mergedCandidates"][0]
        assert prompt["detail"]["contextSource"] == "reranked"
        assert prompt["detail"]["generationQuestion"] == trace["input"]["question"]
        assert prompt["detail"]["finalContextIds"] == [
            c["id"] for c in multi["detail"]["rerankedContext"]
        ]
        assert trace["input"]["rerankerQuery"] == trace["input"]["question"]
        assert trace["input"]["generationQuestion"] == trace["input"]["question"]
        assert trace["output"]["answer"]


def test_case_a_transform_discovers_useful_candidate():
    """TRANSFORM_HELPS: relevant chunk absent originally, discovered by Qi."""
    traces = _by_id(json.loads(TRACES.read_text(encoding="utf-8")))
    trace = traces["transformation-bridges-idle-auth"]
    assert trace["teachingClass"] == "TRANSFORM_HELPS"

    comparison = trace["comparison"]
    discovered = comparison["candidatesDiscoveredByTransform"]
    assert "sample-12" in discovered
    assert "sample-12" not in comparison["originalOnlyCandidateIds"]
    assert "sample-12" in comparison["multiQueryCandidateIds"]
    assert "sample-12" in comparison["multiQueryFinalContextIds"]
    assert "sample-12" not in comparison["originalOnlyFinalContextIds"]

    multi = next(s for s in trace["steps"] if s["id"] == "query-transform")
    hit = next(c for c in multi["detail"]["mergedCandidates"] if c["id"] == "sample-12")
    # Must be found by a transformed query, not only Q0.
    assert any(qid != "Q0" for qid in hit["foundBy"])
    assert "Q0" not in hit["foundBy"]


def test_case_b_is_redundant_with_duplicates():
    traces = _by_id(json.loads(TRACES.read_text(encoding="utf-8")))
    trace = traces["original-query-already-strong-econn42"]
    assert trace["teachingClass"] == "REDUNDANT"

    comparison = trace["comparison"]
    assert comparison["candidatesDiscoveredByTransform"] == []
    assert (
        comparison["originalOnlyFinalContextIds"]
        == comparison["multiQueryFinalContextIds"]
    )

    multi = next(s for s in trace["steps"] if s["id"] == "query-transform")
    counts = multi["detail"]["candidateCounts"]
    assert counts["duplicates"] > 0
    assert counts["beforeDedup"] > counts["afterDedup"]


def test_case_c_introduces_additional_candidates():
    traces = _by_id(json.loads(TRACES.read_text(encoding="utf-8")))
    trace = traces["expansion-adds-platform-noise"]
    assert trace["teachingClass"] == "ADDS_NOISE"

    discovered = trace["comparison"]["candidatesDiscoveredByTransform"]
    assert len(discovered) > 0
    # Noise must not be confused with "automatically becomes the whole answer".
    # At least one discovered id should be inspectable in the merged pool.
    multi = next(s for s in trace["steps"] if s["id"] == "query-transform")
    merged_ids = {c["id"] for c in multi["detail"]["mergedCandidates"]}
    assert set(discovered) <= merged_ids
