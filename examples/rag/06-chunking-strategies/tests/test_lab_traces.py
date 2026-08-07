"""Lab trace export shape tests — stubbed, no network."""

from __future__ import annotations

import json
from pathlib import Path

from config import ALL_STRATEGIES, EXAMPLE_ID, Settings
from evaluation.dataset import load_eval_dataset
from evaluation.metrics import recall_at_k
from experiment import run_experiment
from export_lab_traces import classify_example_cases
from rag.loader import load_document
from tests.test_experiment import _StubClient

ROOT = Path(__file__).resolve().parents[1]


def test_classify_example_cases_references_real_queries():
    text = load_document(ROOT / "data" / "sample.md")
    dataset = load_eval_dataset(ROOT / "data" / "eval_set.json", text)
    settings = Settings(openai_api_key="test", eval_k=3)
    result = run_experiment(
        _StubClient(),  # type: ignore[arg-type]
        text,
        dataset,
        settings,
        k=3,
    )
    per = {name: {} for name in ALL_STRATEGIES}
    for strat in result.evaluations:
        for q in strat.per_query:
            per[strat.strategy][q.query_id] = q

    cases = classify_example_cases(per, dataset)
    assert 1 <= len(cases) <= 3
    known = {q.id for q in dataset.queries}
    for spec in cases:
        assert spec["queryId"] in known
        assert spec["exampleClass"]
        assert "selectionNote" in spec
        assert "teaching" not in spec["exampleClass"].lower()


def test_metrics_match_evaluator_output():
    text = load_document(ROOT / "data" / "sample.md")
    dataset = load_eval_dataset(ROOT / "data" / "eval_set.json", text)
    settings = Settings(openai_api_key="test", eval_k=3)
    result = run_experiment(
        _StubClient(),  # type: ignore[arg-type]
        text,
        dataset,
        settings,
        k=3,
    )
    for strat in result.evaluations:
        for q in strat.per_query:
            relevant = {cid for cid, g in q.chunk_relevance.items() if g >= 1}
            ids = [h.chunk_id for h in q.retrieved]
            assert recall_at_k(ids, relevant, k=3) == q.recall_at_k


def test_committed_lab_traces_schema_if_present():
    """When lab_traces.json exists (after measured export), validate shape."""
    path = ROOT / "lab_traces.json"
    if not path.exists():
        return
    traces = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert traces
    for trace in traces:
        assert trace["labId"] == EXAMPLE_ID
        assert "exampleClass" in trace
        assert "teachingClass" not in trace
        assert "teaching" not in trace.get("exampleClass", "").lower()
        assert trace["corpus"]["text"]
        assert set(trace["strategies"]) == set(ALL_STRATEGIES)
        for name in ALL_STRATEGIES:
            assert trace["strategies"][name]["chunks"]
        for detail in trace["retrievalByStrategy"]:
            for hit in detail["retrieved"]:
                assert "text" in hit and hit["text"]
                assert "chunkId" in hit
                assert "start" in hit and "end" in hit
        assert len(trace["steps"]) >= 10
        assert trace["architecture"]["layout"] == "fan-out-compare"
        # Presentation is a projection of existing fields (optional on older traces).
        if "presentation" in trace:
            pres = trace["presentation"]
            assert "originalDocument" in pres
            assert "evidenceSpans" in pres
            assert "chunkBoundariesByStrategy" in pres
            assert "chunksContainingEvidenceByStrategy" in pres
            assert pres["originalDocument"]["text"] == trace["corpus"]["text"]
