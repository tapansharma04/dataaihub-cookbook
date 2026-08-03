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
    "rerank",
    "prompt",
    "llm",
    "output",
}

EXPECTED_TRACE_IDS = {
    "clear-reordering-cert-vs-econn42",
    "ranking-agrees-handshake-checklist",
    "ambiguous-profile-switch",
}


def test_lab_traces_exist_and_are_measured():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    assert len(traces) == 3
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        assert trace["labId"] == "reranking"
        assert trace["metricsProvenance"] == "measured"
        assert trace["metrics"]["provenance"] == "measured"
        assert trace["cookbook"]["path"] == "examples/rag/03-reranking"


def test_lab_traces_include_rerank_provenance_for_lab_ui():
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    for trace in traces:
        step_ids = {step["id"] for step in trace["steps"]}
        assert REQUIRED_STEPS <= step_ids

        fusion = next(s for s in trace["steps"] if s["id"] == "fuse-rrf")
        rerank = next(s for s in trace["steps"] if s["id"] == "rerank")
        prompt = next(s for s in trace["steps"] if s["id"] == "prompt")

        assert fusion["detail"]["topK"] == trace["input"]["config"]["candidateK"]
        assert rerank["detail"]["model"]
        assert rerank["detail"]["candidateK"] == trace["input"]["config"]["candidateK"]
        assert (
            rerank["detail"]["finalContextK"]
            == trace["input"]["config"]["finalContextK"]
        )

        for chunk in rerank["detail"]["chunks"]:
            assert "id" in chunk
            assert "rerankerScore" in chunk
            assert "rank" in chunk
            assert "previousRank" in chunk
            assert "rrfScore" in chunk
            assert "movement" in chunk

        # Prompt must be built from reranked context.
        assert prompt["detail"]["contextSource"] == "reranked"
        assert trace["output"]["answer"]
        assert trace["input"]["question"]
