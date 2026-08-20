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
        "INSERT_DATA",
        "INSERT_WHERE",
        "DELETE_DATA",
        "UPDATE_AND_VERIFY",
    }
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "insert-data-billing-portal-redis",
        "insert-where-person-uses-technology",
        "delete-data-billing-portal-postgresql",
        "update-and-verify-billing-portal-technology",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {
        "completed",
        "update_rejected",
        "update_failed",
        "verification_failed",
        "row_limit",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "graphIntelligenceScore",
        "graphQualityScore",
        "updateQualityScore",
        "reasoningScore",
        "confidenceScore",
        "benchmarkScore",
        "graph_intelligence_score",
        "graph_quality_score",
        "update_quality_score",
        "reasoning_score",
        "confidence_score",
        "benchmark_score",
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
        "updateExecutionMs",
        "update_execution_ms",
        "verificationExecutionMs",
        "verification_execution_ms",
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
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    case = get_case(trace_id)
    result = run_case(case, settings, store=store)
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
        "updateExecutionMs",
        "verificationExecutionMs",
        "insertedTripleCount",
        "deletedTripleCount",
        "beforeTripleCount",
        "afterTripleCount",
        "verificationRows",
        "updateType",
        "verificationQueryCount",
        "terminationReason",
        "provenance",
    }
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["provenance"] == "measured"
    assert trace["input"]["updateQuery"] == trace["output"]["updateQuery"]
    assert trace["input"]["verificationQuery"] == trace["output"]["verificationQuery"]

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
    assert trace["architecture"]["layout"] == "sparql-updates"
    assert trace["architecture"]["graphModel"] == "rdf"
    assert trace["architecture"]["executionEngine"] == "rdflib"
    assert trace["input"]["config"]["freshGraphPerCase"] is True

    kinds = [event["kind"] for event in trace["sequence"]]
    assert "user_request" in kinds
    assert "update_started" in kinds
    assert "update_executed" in kinds
    assert "graph_state" in kinds
    assert "verification_started" in kinds
    assert "verification_result" in kinds
    assert "update_completed" in kinds
    assert "termination" in kinds

    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})
        assert "note" not in event

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []

    assert "before" in trace["output"]
    assert "after" in trace["output"]
    assert "insertedTriples" in trace["output"]
    assert "deletedTriples" in trace["output"]
    assert "verificationBindings" in trace["output"]


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
        assert trace["input"]["updateName"] == case.update_name


def test_insert_data_trace_preserves_mutation():
    trace = _build("insert-data-billing-portal-redis")
    _assert_trace_contract(trace)
    assert trace["metrics"]["insertedTripleCount"] == 1
    assert len(trace["output"]["insertedTriples"]) == 1
    assert trace["output"]["insertedTriples"][0]["object"]["iri"] == str(EX.redis)
    labels = sorted(
        row["technology"]["label"] for row in trace["output"]["verificationBindings"]
    )
    assert labels == ["PostgreSQL", "Redis"]
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert phases[:4] == ["QUESTION", "BEFORE", "INSERT", "AFTER"]
    assert "VERIFY" in phases


def test_insert_where_trace_preserves_derived_triples():
    trace = _build("insert-where-person-uses-technology")
    _assert_trace_contract(trace)
    assert trace["metrics"]["insertedTripleCount"] == 3
    assert len(trace["output"]["insertedTriples"]) == 3
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert "PATTERN" in phases
    assert "INSERT_DERIVED_TRIPLES" in phases


def test_delete_data_trace_preserves_removal():
    trace = _build("delete-data-billing-portal-postgresql")
    _assert_trace_contract(trace)
    assert trace["metrics"]["deletedTripleCount"] == 1
    assert trace["output"]["verificationBindings"] == []
    assert len(trace["output"]["after"]) == 0
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert "DELETE" in phases


def test_update_and_verify_trace_preserves_swap():
    trace = _build("update-and-verify-billing-portal-technology")
    _assert_trace_contract(trace)
    assert trace["metrics"]["deletedTripleCount"] == 1
    assert trace["metrics"]["insertedTripleCount"] == 1
    assert len(trace["output"]["verificationBindings"]) == 1
    assert trace["output"]["verificationBindings"][0]["technology"]["label"] == "Redis"
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert "DELETE_PLUS_INSERT" in phases
    assert trace["presentation"]["signatureFlow"] == (
        "BEFORE → DELETE + INSERT → AFTER → VERIFY"
    )


def test_update_query_preserved_in_trace():
    from sparql.queries import get_update

    for case in CASES:
        trace = _build(case.trace_id)
        predefined = get_update(case.update_name)
        assert trace["input"]["updateQuery"] == predefined.update_query
        assert trace["input"]["verificationQuery"] == predefined.verification_query
        started = next(
            event for event in trace["sequence"] if event["kind"] == "update_started"
        )
        assert started["detail"]["updateQuery"] == trace["input"]["updateQuery"]
        executed = next(
            event for event in trace["sequence"] if event["kind"] == "update_executed"
        )
        assert executed["detail"]["method"] == "Graph.update"


def test_build_trace_separates_presentation_metadata():
    trace = _build("insert-data-billing-portal-redis")
    assert "purpose" in trace["presentation"]
    for event in trace["sequence"]:
        assert "signatureFlow" not in event
        assert "signatureView" not in event


def test_signature_view_ends_with_termination():
    settings = Settings(graph_path=GRAPH_PATH)
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    case = get_case("insert-data-billing-portal-redis")
    result = run_case(case, settings, store=store)
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
