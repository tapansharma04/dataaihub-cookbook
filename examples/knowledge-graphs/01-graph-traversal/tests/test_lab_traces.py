"""Trace schema, provenance, CoT, and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EXAMPLE_ID, Settings
from graph.cases import CASES, get_case
from graph.store import GraphStore
from graph.trace import build_signature_view, build_trace
from graph.traversal import run_case
from graph.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "DIRECT_RELATIONSHIP",
        "MULTI_HOP_TRAVERSAL",
        "RELATIONSHIP_FILTER",
        "NO_PATH",
    }
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "direct-relationship-employs",
        "multi-hop-alice-technologies",
        "relationship-filter-project-people",
        "no-path-alice-uses",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {"completed", "no_path", "invalid_entity", "invalid_relationship", "depth_limit"}
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "graphIntelligenceScore",
        "graphQualityScore",
        "answerConfidenceScore",
        "benchmarkScore",
        "graph_intelligence_score",
        "graph_quality_score",
        "answer_confidence_score",
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
    store = GraphStore.from_path(GRAPH_PATH)
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
        "entitiesVisited",
        "relationshipsVisited",
        "traversalDepth",
        "matchedRelationships",
        "pathFound",
        "executionMs",
        "terminationReason",
        "maxDepth",
        "provenance",
    }
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["provenance"] == "measured"

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
    assert trace["architecture"]["layout"] == "graph-traversal"
    assert trace["architecture"]["graphModel"] == "rdf"
    signature_phases = [
        item["phase"] for item in trace["presentation"]["signatureView"]
    ]
    assert signature_phases[-1] == "TERMINATION"

    kinds = [event["kind"] for event in trace["sequence"]]
    assert "user_request" in kinds
    assert "traversal_started" in kinds
    assert "traversal_step" in kinds
    assert "result" in kinds
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
    assert len(traces) == len(CASES) == 4
    case_by_trace = {case.trace_id: case for case in CASES}
    for trace in traces:
        case = case_by_trace[trace["traceId"]]
        assert trace["exampleClass"] == case.example_class
        assert trace["input"]["question"] == case.question
        assert trace["input"]["startId"] == case.start_id


def test_direct_traversal_preserves_all_matched_entities():
    trace = _build("direct-relationship-employs")
    _assert_trace_contract(trace)
    answer_ids = [entity["id"] for entity in trace["output"]["answers"]]
    assert answer_ids == [str(EX.alice), str(EX.bob), str(EX.carol)]
    labels = {entity["id"]: entity["label"] for entity in trace["output"]["answers"]}
    assert labels == {
        str(EX.alice): "Alice",
        str(EX.bob): "Bob",
        str(EX.carol): "Carol",
    }
    matches = [
        event for event in trace["sequence"] if event["kind"] == "relationship_match"
    ]
    assert len(matches) == 3
    assert [event["detail"]["triple"]["object"]["id"] for event in matches] == [
        str(EX.alice),
        str(EX.bob),
        str(EX.carol),
    ]
    assert all(
        event["detail"]["triple"]["subject"]["id"] == str(EX.acmeAI)
        for event in matches
    )
    assert all(
        event["detail"]["triple"]["predicate"]["label"] == "employs"
        for event in matches
    )
    assert all(
        event["detail"]["triple"]["predicate"]["id"] == str(EX.employs)
        for event in matches
    )
    assert all(event["detail"]["direction"] == "outgoing" for event in matches)
    assert len(trace["output"]["paths"]) == 3
    assert all(path["depth"] == 1 for path in trace["output"]["paths"])


def test_direct_branching_metrics_count_distinct_visits_not_answers():
    trace = _build("direct-relationship-employs")
    metrics = trace["metrics"]
    answers = trace["output"]["answers"]
    assert metrics["traversalDepth"] == 1
    assert metrics["relationshipsVisited"] == 3
    assert metrics["matchedRelationships"] == metrics["relationshipsVisited"]
    assert metrics["entitiesVisited"] == 4
    assert len(answers) == 3
    assert metrics["entitiesVisited"] != len(answers)
    assert metrics["traversalDepth"] != metrics["entitiesVisited"]


def test_multi_hop_trace_preserves_path_evidence():
    trace = _build("multi-hop-alice-technologies")
    _assert_trace_contract(trace)
    path = trace["output"]["paths"][0]
    assert path["depth"] == 2
    assert [entity["id"] for entity in path["entities"]] == [
        str(EX.alice),
        str(EX.knowledgePlatform),
        str(EX.postgresql),
    ]
    assert [entity["label"] for entity in path["entities"]] == [
        "Alice",
        "Knowledge Platform",
        "PostgreSQL",
    ]
    assert path["relationships"] == [
        {
            "subject": str(EX.alice),
            "predicate": {"id": str(EX.worksOn), "label": "worksOn"},
            "object": str(EX.knowledgePlatform),
        },
        {
            "subject": str(EX.knowledgePlatform),
            "predicate": {"id": str(EX.uses), "label": "uses"},
            "object": str(EX.postgresql),
        },
    ]
    matches = [
        event for event in trace["sequence"] if event["kind"] == "relationship_match"
    ]
    assert [event["detail"]["direction"] for event in matches] == [
        "outgoing",
        "outgoing",
    ]
    assert matches[0]["detail"]["from"]["id"] == str(EX.alice)
    assert matches[0]["detail"]["to"]["id"] == str(EX.knowledgePlatform)
    assert matches[1]["detail"]["from"]["id"] == str(EX.knowledgePlatform)
    assert matches[1]["detail"]["to"]["id"] == str(EX.postgresql)
    view = trace["presentation"]["signatureView"]
    phases = [item["phase"] for item in view]
    assert phases == [
        "ENTITY",
        "RELATIONSHIP",
        "ENTITY",
        "RELATIONSHIP",
        "ENTITY",
        "RESULT",
        "TERMINATION",
    ]
    entities = [item["entity"]["id"] for item in view if item["phase"] == "ENTITY"]
    predicates = [
        item["predicate"]["label"] for item in view if item["phase"] == "RELATIONSHIP"
    ]
    assert entities == [
        str(EX.alice),
        str(EX.knowledgePlatform),
        str(EX.postgresql),
    ]
    assert predicates == ["worksOn", "uses"]


def test_incoming_relationship_preserves_stored_triple_and_walk_direction():
    trace = _build("relationship-filter-project-people")
    _assert_trace_contract(trace)
    assert trace["input"]["hops"] == [{"predicate": "worksOn", "direction": "incoming"}]
    matches = [
        event for event in trace["sequence"] if event["kind"] == "relationship_match"
    ]
    assert {event["detail"]["direction"] for event in matches} == {"incoming"}
    alice = next(
        event for event in matches if event["detail"]["to"]["id"] == str(EX.alice)
    )
    triple = alice["detail"]["triple"]
    assert triple["subject"]["id"] == str(EX.alice)
    assert triple["subject"]["label"] == "Alice"
    assert triple["predicate"]["label"] == "worksOn"
    assert triple["predicate"]["id"] == str(EX.worksOn)
    assert triple["object"]["id"] == str(EX.knowledgePlatform)
    assert triple["object"]["label"] == "Knowledge Platform"
    assert alice["detail"]["from"]["id"] == str(EX.knowledgePlatform)
    assert alice["detail"]["to"]["id"] == str(EX.alice)
    path = next(
        item
        for item in trace["output"]["paths"]
        if item["entities"][-1]["id"] == str(EX.alice)
    )
    assert path["relationships"][0] == {
        "subject": str(EX.alice),
        "predicate": {"id": str(EX.worksOn), "label": "worksOn"},
        "object": str(EX.knowledgePlatform),
    }


def test_no_path_trace_shape():
    trace = _build("no-path-alice-uses")
    _assert_trace_contract(trace)
    assert trace["input"]["hops"] == [{"predicate": "uses", "direction": "outgoing"}]
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "relationship_match" not in kinds
    result_event = next(
        event for event in trace["sequence"] if event["kind"] == "result"
    )
    assert result_event["detail"]["pathFound"] is False
    assert trace["output"]["paths"] == []
    assert trace["output"]["answers"] == []
    assert "worksOn" not in json.dumps(trace["output"])
    assert str(EX.postgresql) not in json.dumps(trace["output"])
    phases = [item["phase"] for item in trace["presentation"]["signatureView"]]
    assert phases == ["START_ENTITY", "SEARCH", "NO_PATH", "TERMINATION"]
    assert trace["presentation"]["signatureFlow"] == "START ENTITY → SEARCH → NO PATH"
    assert trace["metrics"]["pathFound"] is False
    assert trace["metrics"]["terminationReason"] == "no_path"
    assert trace["metrics"]["traversalDepth"] == 1
    assert trace["metrics"]["relationshipsVisited"] == 0
    assert trace["metrics"]["matchedRelationships"] == 0


def test_matched_relationships_equals_relationships_visited():
    for case in CASES:
        metrics = _build(case.trace_id)["metrics"]
        assert metrics["matchedRelationships"] == metrics["relationshipsVisited"]
        assert metrics["traversalDepth"] == len(get_case(case.trace_id).hops)


def test_build_trace_separates_presentation_metadata():
    trace = _build("direct-relationship-employs")
    _assert_trace_contract(trace)
    assert "purpose" in trace["presentation"]
    for event in trace["sequence"]:
        assert "signatureFlow" not in event
        assert "signatureView" not in event


def test_signature_view_ends_with_termination():
    settings = Settings(graph_path=GRAPH_PATH)
    store = GraphStore.from_path(GRAPH_PATH)
    case = get_case("direct-relationship-employs")
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


def test_no_hidden_reasoning_in_fresh_traces():
    for case in CASES:
        violations = _collect_cot_violations(_build(case.trace_id))
        assert violations == []


def test_trace_provenance_model():
    traces = [_build(case.trace_id) for case in CASES]
    for trace in traces:
        assert trace["provenance"]["model"] == "not_used"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert trace["input"]["config"]["modelDriver"] == "not_used"
