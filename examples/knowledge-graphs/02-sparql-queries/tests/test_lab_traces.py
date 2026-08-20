"""Trace schema, provenance, CoT, and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EXAMPLE_ID, Settings
from sparql.cases import CASES, get_case
from sparql.graph import RdfGraphStore
from sparql.runner import run_case
from sparql.trace import build_signature_view, build_trace
from sparql.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "BASIC_SELECT",
        "MULTI_PATTERN_QUERY",
        "FILTER_QUERY",
        "NO_MATCH",
    }
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "basic-select-knowledge-platform-people",
        "multi-pattern-alice-technologies",
        "filter-platform-team-people",
        "no-match-quantum-platform",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {"completed", "no_match", "query_rejected", "row_limit", "query_failed"}
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "graphIntelligenceScore",
        "graphQualityScore",
        "answerConfidenceScore",
        "benchmarkScore",
        "queryQualityScore",
        "graph_intelligence_score",
        "graph_quality_score",
        "answer_confidence_score",
        "benchmark_score",
        "query_quality_score",
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
    }
)
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "executionMs",
        "execution_ms",
        "queryExecutionMs",
        "query_execution_ms",
        "totalMs",
        "total_ms",
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


def _build(trace_id: str) -> dict[str, Any]:
    settings = Settings(graph_path=GRAPH_PATH)
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = get_case(trace_id)
    result = run_case(case, store, settings)
    return build_trace(case=case, result=result, settings=settings, store=store)


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert isinstance(trace["traceId"], str) and trace["traceId"]
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert "graph" in trace
    assert "sequence" in trace and len(trace["sequence"]) >= 2
    assert "steps" in trace and len(trace["steps"]) >= 2
    assert "metrics" in trace
    assert set(trace["metrics"]) >= {
        "queryExecutionMs",
        "resultRows",
        "triplePatterns",
        "filterCount",
        "variables",
        "queryCase",
        "bindingsReturned",
        "terminationReason",
        "provenance",
    }
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["provenance"] == "measured"
    assert trace["input"]["query"] == trace["output"]["query"]

    termination_reason = trace["metrics"]["terminationReason"]
    assert termination_reason in SUPPORTED_TERMINATION_REASONS
    assert trace["output"]["terminationReason"] == termination_reason

    termination_events = [
        event for event in trace["sequence"] if event["kind"] == "termination"
    ]
    assert len(termination_events) >= 1
    assert termination_events[-1]["detail"]["reason"] == termination_reason

    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    assert "signatureFlow" in trace["presentation"]
    assert trace["architecture"]["layout"] == "sparql-queries"
    assert trace["architecture"]["graphModel"] == "rdf"
    assert trace["architecture"]["executionEngine"] == "rdflib"

    kinds = [event["kind"] for event in trace["sequence"]]
    assert "user_request" in kinds
    assert "query_started" in kinds
    assert "query_executed" in kinds
    assert "result_bindings" in kinds
    assert "termination" in kinds

    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})
        assert "note" not in event

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []


def test_all_four_cases_have_corresponding_traces():
    traces = [_build(case.trace_id) for case in CASES]
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    assert {trace["exampleClass"] for trace in traces} == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_trace_contract(trace)


def test_case_trace_alignment():
    traces = [_build(case.trace_id) for case in CASES]
    case_by_trace = {case.trace_id: case for case in CASES}
    for trace in traces:
        case = case_by_trace[trace["traceId"]]
        assert trace["exampleClass"] == case.example_class
        assert trace["input"]["question"] == case.question
        assert trace["input"]["queryName"] == case.query_name


def test_basic_select_trace_preserves_bindings():
    trace = _build("basic-select-knowledge-platform-people")
    _assert_trace_contract(trace)
    labels = sorted(row["person"]["label"] for row in trace["output"]["bindings"])
    assert labels == ["Alice", "Bob"]
    iris = {row["person"]["iri"] for row in trace["output"]["bindings"]}
    assert iris == {str(EX.alice), str(EX.bob)}


def test_multi_pattern_trace_preserves_join_bindings():
    trace = _build("multi-pattern-alice-technologies")
    _assert_trace_contract(trace)
    assert len(trace["output"]["bindings"]) == 1
    row = trace["output"]["bindings"][0]
    assert row["person"]["label"] == "Alice"
    assert row["project"]["label"] == "Knowledge Platform"
    assert row["technology"]["label"] == "PostgreSQL"
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert "PATTERN_1" in phases
    assert "JOIN" in phases
    assert "PATTERN_2" in phases


def test_filter_trace_preserves_sparql_filter():
    trace = _build("filter-platform-team-people")
    _assert_trace_contract(trace)
    assert 'FILTER(?team = "platform")' in trace["input"]["query"]
    labels = sorted(row["person"]["label"] for row in trace["output"]["bindings"])
    assert labels == ["Alice", "Bob"]
    filter_phases = [
        item
        for item in trace["presentation"]["signatureView"]
        if item["phase"] == "FILTER"
    ]
    assert len(filter_phases) == 1


def test_no_match_trace_shape():
    trace = _build("no-match-quantum-platform")
    _assert_trace_contract(trace)
    assert trace["output"]["matches"] == []
    assert trace["metrics"]["resultRows"] == 0
    assert trace["metrics"]["terminationReason"] == "no_match"
    phases = [item["phase"] for item in trace["presentation"]["signatureView"]]
    assert "ZERO_BINDINGS" in phases
    assert "NO_MATCH" in phases
    executed = next(
        event for event in trace["sequence"] if event["kind"] == "query_executed"
    )
    assert executed["detail"]["success"] is True


def test_query_preserved_in_trace():
    for case in CASES:
        trace = _build(case.trace_id)
        from sparql.queries import get_query

        assert trace["input"]["query"] == get_query(case.query_name).query
        started = next(
            event for event in trace["sequence"] if event["kind"] == "query_started"
        )
        assert started["detail"]["query"] == trace["input"]["query"]


def test_build_trace_separates_presentation_metadata():
    trace = _build("basic-select-knowledge-platform-people")
    assert "purpose" in trace["presentation"]
    for event in trace["sequence"]:
        assert "signatureFlow" not in event
        assert "signatureView" not in event


def test_signature_view_ends_with_termination():
    settings = Settings(graph_path=GRAPH_PATH)
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = get_case("basic-select-knowledge-platform-people")
    result = run_case(case, store, settings)
    view = build_signature_view(result, example_class=case.example_class)
    assert view[-1]["phase"] == "TERMINATION"


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES) == 4
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_trace_contract(trace)


def test_committed_traces_match_regenerated_semantically():
    if not LAB_TRACES_PATH.exists():
        return
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    regenerated = [_build(case.trace_id) for case in CASES]
    committed_by_id = {trace["traceId"]: trace for trace in committed}
    for trace in regenerated:
        left = _strip_volatile(committed_by_id[trace["traceId"]])
        right = _strip_volatile(trace)
        assert left == right


def test_no_chain_of_thought_in_lab_traces():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        violations = _collect_cot_violations(trace)
        assert violations == [], (
            f"{trace['traceId']} contains hidden-reasoning fields: {violations}"
        )


def test_trace_provenance_model():
    traces = [_build(case.trace_id) for case in CASES]
    for trace in traces:
        assert trace["provenance"]["model"] == "not_used"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert trace["input"]["config"]["modelDriver"] == "not_used"
