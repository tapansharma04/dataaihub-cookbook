"""Measured case tests — four teaching walks and their answers."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from graph.cases import CASES, get_case
from graph.store import GraphStore
from graph.trace import SIGNATURE_FLOWS, build_signature_view
from graph.traversal import run_case
from graph.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _run(trace_id: str):
    settings = Settings(graph_path=GRAPH_PATH)
    store = GraphStore.from_path(GRAPH_PATH)
    case = get_case(trace_id)
    return case, run_case(case, store, settings)


def test_four_measured_cases_exist():
    assert len(CASES) == 4
    assert [case.example_class for case in CASES] == [
        "DIRECT_RELATIONSHIP",
        "MULTI_HOP_TRAVERSAL",
        "RELATIONSHIP_FILTER",
        "NO_PATH",
    ]


def test_direct_relationship_case():
    _, result = _run("direct-relationship-employs")
    assert result.metrics.path_found is True
    assert result.metrics.termination_reason == "completed"
    assert [entity.id for entity in result.answers] == [
        str(EX.alice),
        str(EX.bob),
        str(EX.carol),
    ]
    kinds = [event.kind for event in result.sequence]
    assert kinds[0] == "user_request"
    assert "traversal_started" in kinds
    assert "traversal_step" in kinds
    assert "relationship_match" in kinds
    assert kinds[-2] == "result"
    assert kinds[-1] == "termination"


def test_multi_hop_case():
    _, result = _run("multi-hop-alice-technologies")
    assert result.metrics.path_found is True
    assert result.metrics.traversal_depth == 2
    assert [entity.id for entity in result.answers] == [str(EX.postgresql)]
    steps = [event for event in result.sequence if event.kind == "traversal_step"]
    assert [event.detail["predicate"]["label"] for event in steps] == [
        "worksOn",
        "uses",
    ]
    assert [event.detail["predicate"]["id"] for event in steps] == [
        str(EX.worksOn),
        str(EX.uses),
    ]


def test_relationship_filter_case_uses_incoming_direction():
    _, result = _run("relationship-filter-project-people")
    assert result.hops[0].direction == "incoming"
    assert [entity.id for entity in result.answers] == [str(EX.alice), str(EX.bob)]
    matches = [event for event in result.sequence if event.kind == "relationship_match"]
    assert all(event.detail["direction"] == "incoming" for event in matches)
    assert all(
        event.detail["triple"]["predicate"]["label"] == "worksOn" for event in matches
    )
    assert all(
        event.detail["triple"]["predicate"]["id"] == str(EX.worksOn)
        for event in matches
    )


def test_no_path_case_does_not_infer_project_technology():
    _, result = _run("no-path-alice-uses")
    assert result.metrics.path_found is False
    assert result.metrics.termination_reason == "no_path"
    assert result.answers == []
    kinds = [event.kind for event in result.sequence]
    assert kinds == [
        "user_request",
        "graph_lookup",
        "traversal_started",
        "traversal_step",
        "traversal_completed",
        "result",
        "termination",
    ]
    result_event = result.sequence[-2]
    assert result_event.detail["pathFound"] is False
    assert result.output["answers"] == []


def test_case_signature_flows():
    for case in CASES:
        _, result = _run(case.trace_id)
        view = build_signature_view(result, example_class=case.example_class)
        phases = [item["phase"] for item in view]
        if case.example_class == "NO_PATH":
            assert phases == ["START_ENTITY", "SEARCH", "NO_PATH", "TERMINATION"]
        else:
            assert phases[0] == "ENTITY"
            assert "RELATIONSHIP" in phases
            assert phases[-2] == "RESULT"
            assert phases[-1] == "TERMINATION"
        assert SIGNATURE_FLOWS[case.example_class]
