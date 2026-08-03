"""Validate committed lab_traces.json shape for DataAIHub Lab replay.

No network / no paid APIs — reads the measured export already in the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

TRACES = Path(__file__).resolve().parents[1] / "lab_traces.json"

REQUIRED_STEPS = {
    "chunking",
    "embed-corpus",
    "bm25-index",
    "embed-query",
    "retrieve-dense",
    "retrieve-lexical",
    "fuse-rrf",
    "prompt",
    "llm",
    "output",
}

EXPECTED_TRACE_IDS = {
    "dense-friendly-connectivity",
    "lexical-friendly-error-code",
    "hybrid-mixed-api-version",
}


def test_lab_traces_exist_and_are_measured():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    assert len(traces) == 3
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        assert trace["labId"] == "hybrid-rag"
        assert trace["metricsProvenance"] == "measured"
        assert trace["metrics"]["provenance"] == "measured"
        assert trace["cookbook"]["path"] == "examples/rag/02-hybrid-rag"


def test_lab_traces_include_retrieval_provenance_for_lab_ui():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    for trace in traces:
        step_ids = {step["id"] for step in trace["steps"]}
        assert REQUIRED_STEPS <= step_ids

        dense = next(s for s in trace["steps"] if s["id"] == "retrieve-dense")
        lexical = next(s for s in trace["steps"] if s["id"] == "retrieve-lexical")
        fusion = next(s for s in trace["steps"] if s["id"] == "fuse-rrf")

        assert dense["detail"]["similarity"] == "cosine"
        assert lexical["detail"]["algorithm"] == "bm25"
        assert fusion["detail"]["formula"] == "RRF(d) = Σ 1 / (k + rank(d))"
        assert fusion["detail"]["rrfK"] == 60
        assert "denseRanks" in fusion["detail"]["inputs"]
        assert "lexicalRanks" in fusion["detail"]["inputs"]

        for chunk in fusion["detail"]["chunks"]:
            assert "id" in chunk
            assert "rrfScore" in chunk
            assert "rank" in chunk
            assert "denseRank" in chunk
            assert "lexicalRank" in chunk

        assert trace["output"]["answer"]
        assert trace["input"]["question"]
