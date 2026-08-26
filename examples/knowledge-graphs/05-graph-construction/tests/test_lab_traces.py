"""Trace schema, provenance, CoT, and export stability tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EXAMPLE_ID, Settings
from graph.builder import RdfGraphStore
from graph.cases import CASES, get_case
from graph.extractor import StructuredExtractor
from graph.runner import run_case
from graph.trace import build_trace

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"
LAB_TRACES_PATH = ROOT / "lab_traces.json"
LAB_TRACES_LLM_PATH = ROOT / "lab_traces_llm.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "ENTITY_EXTRACTION",
        "RELATIONSHIP_EXTRACTION",
        "ENTITY_LINKING",
        "INVALID_FACT",
    }
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "entity-extraction-alice",
        "relationship-extraction-alice-platform",
        "entity-linking-known-entities",
        "invalid-fact-unsupported-predicate",
    }
)
COT_FIELD_NAMES = frozenset(
    {
        "chainOfThought",
        "chain_of_thought",
        "reasoning",
        "hiddenReasoning",
        "hidden_reasoning",
        "internalReasoning",
        "internal_reasoning",
        "thoughtProcess",
        "thought_process",
        "thought",
        "thoughts",
        "privateReasoning",
        "private_reasoning",
        "cot",
        "scratchpad",
    }
)
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "totalMs",
        "total_ms",
        "modelMs",
        "model_ms",
        "modelLatencyMs",
        "model_latency_ms",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "extractionQuality",
        "graphQuality",
        "accuracy",
        "confidence",
        "intelligence",
        "benchmarkScore",
        "extraction_quality",
        "graph_quality",
        "benchmark_score",
    }
)


def _collect_cot_violations(obj: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in COT_FIELD_NAMES:
                violations.append(child_path)
            violations.extend(_collect_cot_violations(value, child_path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            violations.extend(_collect_cot_violations(item, f"{path}[{index}]"))
    return violations


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_volatile(value)
            for key, value in obj.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(item) for item in obj]
    return obj


def _build(trace_id: str, *, mode: str = "structured") -> dict[str, Any]:
    settings = Settings(graph_path=GRAPH_PATH, openai_api_key="")
    case = get_case(trace_id)
    store = RdfGraphStore.fresh(start=case.start_graph, seed_path=GRAPH_PATH)
    result = run_case(
        case,
        settings,
        mode=mode,  # type: ignore[arg-type]
        extractor=StructuredExtractor(),
        store=store,
    )
    return build_trace(case=case, result=result, settings=settings, store=store)


def _assert_structured_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert trace["executionMode"] == "structured"
    assert isinstance(trace["traceId"], str) and trace["traceId"]
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert "graph" in trace
    assert "proposal" in trace
    assert "validation" in trace
    assert "sequence" in trace and len(trace["sequence"]) >= 2
    assert "steps" in trace and len(trace["steps"]) >= 2
    assert "metrics" in trace
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["modelTurns"] == 0
    assert trace["architecture"]["layout"] == "graph-construction"
    assert trace["architecture"]["graphModel"] == "rdf"
    assert trace["architecture"]["executionEngine"] == "rdflib"
    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "source_loaded" in kinds
    assert "extraction_started" in kinds
    assert "termination" in kinds
    assert _collect_cot_violations(trace) == []


def test_all_four_structured_traces():
    traces = [_build(case.trace_id) for case in CASES]
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    assert {trace["exampleClass"] for trace in traces} == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_structured_trace_contract(trace)


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES) == 4
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_structured_trace_contract(trace)


def test_committed_structured_traces_match_regenerated_semantically():
    if not LAB_TRACES_PATH.exists():
        return
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    regenerated = [_build(case.trace_id) for case in CASES]
    committed_by_id = {trace["traceId"]: trace for trace in committed}
    for trace in regenerated:
        left = _strip_volatile(committed_by_id[trace["traceId"]])
        right = _strip_volatile(trace)
        assert left == right


def test_deterministic_structured_traces_stable_across_runs():
    first = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    second = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    assert first == second


def test_committed_llm_traces_are_separate_file():
    if not LAB_TRACES_LLM_PATH.exists():
        return
    if LAB_TRACES_PATH.exists():
        structured = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
        assert all(trace.get("executionMode") == "structured" for trace in structured)
    llm_traces = json.loads(LAB_TRACES_LLM_PATH.read_text(encoding="utf-8"))
    assert isinstance(llm_traces, list)
    assert len(llm_traces) == len(CASES) == 4
    assert {trace["traceId"] for trace in llm_traces} == EXPECTED_TRACE_IDS
    assert all(trace.get("executionMode") == "llm_assisted" for trace in llm_traces)


def test_no_chain_of_thought_in_lab_traces():
    for path in (LAB_TRACES_PATH, LAB_TRACES_LLM_PATH):
        if not path.exists():
            continue
        traces = json.loads(path.read_text(encoding="utf-8"))
        for trace in traces:
            violations = _collect_cot_violations(trace)
            assert violations == [], f"{trace['traceId']}: {violations}"


def test_no_hidden_reasoning_in_fresh_traces():
    for case in CASES:
        trace = _build(case.trace_id)
        assert _collect_cot_violations(trace) == []


def test_invalid_fact_trace_preserves_rejection():
    trace = _build("invalid-fact-unsupported-predicate")
    assert trace["graphUnchanged"] is True
    assert trace["triplesCreated"] == []
    rejected = trace["validation"]["rejectedRelationships"]
    assert rejected[0]["predicate"] == "supervises"
    assert rejected[0]["reason"] == "unsupported_predicate"
    assert trace["termination"] == "validation_rejected"
