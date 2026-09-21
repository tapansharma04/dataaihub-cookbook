"""Trace presentation, exact signatures, and no-CoT tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_orchestration
from agent.trace import (
    SIGNATURE_FLOWS,
    SIGNATURE_OMITTED_PHASES,
    build_signature_view,
    build_trace,
    signature_flow_from_view,
    signature_phases_from_view,
)
from config import EXAMPLE_ID, Settings

DATA = Path(__file__).resolve().parents[1] / "data"

COT_FIELD_NAMES = frozenset(
    {
        "chainOfThought",
        "chain_of_thought",
        "cot",
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
        "scratchpad",
    }
)

EXPECTED_SIGNATURE_PHASES = {
    "BASIC_ORCHESTRATION": [
        "TASK",
        "PLAN",
        "READY",
        "READY",
        "AGENT",
        "AGENT",
        "READY",
        "AGENT",
        "READY",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ],
    "DEPENDENCY_CHAIN": [
        "TASK",
        "PLAN",
        "READY",
        "AGENT",
        "READY",
        "AGENT",
        "READY",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ],
    "CONDITIONAL_BRANCH": [
        "TASK",
        "PLAN",
        "READY",
        "AGENT",
        "READY",
        "AGENT",
        "CONDITION",
        "READY",
        "CONDITION",
        "SKIP",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ],
    "ORCHESTRATION_FAILURE": [
        "TASK",
        "PLAN",
        "READY",
        "AGENT",
        "FAILURE",
        "SKIP",
        "SKIP",
        "TERMINATION",
    ],
}

FORBIDDEN_PHASES = frozenset({"DELEGATE", "HANDOFF", "HANDOFF_REJECTED", "PASS_RESULT"})


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


def _build(trace_id: str) -> dict[str, Any]:
    settings = Settings(openai_api_key="", data_dir=DATA)
    case = get_case(trace_id)
    result = run_orchestration(case, catalog=Catalog(DATA))
    return build_trace(case=case, result=result, settings=settings)


def test_signature_flows_match_spec():
    for example_class, phases in EXPECTED_SIGNATURE_PHASES.items():
        assert SIGNATURE_FLOWS[example_class] == " → ".join(phases)


def test_signature_flows_match_runtime_signature_view():
    settings = Settings(openai_api_key="", data_dir=DATA)
    for case in CASES:
        result = run_orchestration(case, catalog=Catalog(DATA))
        view = build_signature_view(result.sequence)
        phases = signature_phases_from_view(view)
        assert phases == EXPECTED_SIGNATURE_PHASES[case.example_class]
        assert signature_flow_from_view(view) == SIGNATURE_FLOWS[case.example_class]
        trace = build_trace(case=case, result=result, settings=settings)
        assert (
            trace["presentation"]["signatureFlow"]
            == SIGNATURE_FLOWS[case.example_class]
        )
        assert [
            item["phase"]
            for item in trace["presentation"]["signatureView"]
            if item["phase"] not in SIGNATURE_OMITTED_PHASES
        ] == phases
        assert [event.kind for event in result.sequence].count("termination") == 1


def test_basic_signature_shows_independent_ready_then_dependent_ready():
    case = get_case("payments-incident-basic-orchestration")
    result = run_orchestration(case, catalog=Catalog(DATA))
    view = [
        item
        for item in build_signature_view(result.sequence)
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert [item["phase"] for item in view] == EXPECTED_SIGNATURE_PHASES[
        "BASIC_ORCHESTRATION"
    ]
    assert view[2]["phase"] == "READY"
    assert view[2]["nodeId"] == "status"
    assert view[3]["phase"] == "READY"
    assert view[3]["nodeId"] == "docs"
    assert view[4]["agentId"] == "status_agent"
    assert view[5]["agentId"] == "docs_agent"
    assert view[6]["phase"] == "READY"
    assert view[6]["nodeId"] == "analysis"
    assert view[6]["dependencyIds"] == ["status", "docs"]
    assert view[8]["phase"] == "READY"
    assert view[8]["nodeId"] == "decision"


def test_chain_signature_interleaves_ready_and_agent():
    case = get_case("payments-status-dependency-chain")
    result = run_orchestration(case, catalog=Catalog(DATA))
    view = [
        item
        for item in build_signature_view(result.sequence)
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    phases = [item["phase"] for item in view]
    assert phases == EXPECTED_SIGNATURE_PHASES["DEPENDENCY_CHAIN"]
    agents = [item for item in view if item["phase"] == "AGENT"]
    assert [item["agentId"] for item in agents] == [
        "status_agent",
        "analysis_agent",
        "decision_agent",
    ]
    readies = [item for item in view if item["phase"] == "READY"]
    assert [item["nodeId"] for item in readies] == ["status", "analysis", "decision"]


def test_conditional_signature_includes_condition_and_skip():
    trace = _build("payments-incident-conditional-branch")
    phases = signature_phases_from_view(trace["presentation"]["signatureView"])
    assert phases == EXPECTED_SIGNATURE_PHASES["CONDITIONAL_BRANCH"]
    assert phases.index("CONDITION") < phases.index("SKIP")
    view = [
        item
        for item in trace["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    conditions = [item for item in view if item["phase"] == "CONDITION"]
    assert conditions[0]["nodeId"] == "remediation"
    assert conditions[0]["selected"] is True
    assert conditions[1]["nodeId"] == "no_action"
    assert conditions[1]["selected"] is False
    skip = next(item for item in view if item["phase"] == "SKIP")
    assert skip["nodeId"] == "no_action"
    assert skip["reason"] == "condition_not_selected"
    agents = [item for item in view if item["phase"] == "AGENT"]
    assert [item["agentId"] for item in agents] == [
        "status_agent",
        "analysis_agent",
        "remediation_agent",
    ]


def test_failure_signature_skips_dependents_without_complete():
    trace = _build("unknown-service-orchestration-failure")
    phases = signature_phases_from_view(trace["presentation"]["signatureView"])
    assert phases == EXPECTED_SIGNATURE_PHASES["ORCHESTRATION_FAILURE"]
    assert phases.index("FAILURE") < phases.index("SKIP")
    assert phases.index("SKIP") < phases.index("TERMINATION")
    assert "COMPLETE" not in phases
    assert trace["output"]["ok"] is False
    assert trace["output"]["terminationReason"] == "workflow_failed"
    view = [
        item
        for item in trace["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    skips = [item for item in view if item["phase"] == "SKIP"]
    assert [item["nodeId"] for item in skips] == ["analysis", "decision"]
    assert skips[0]["reason"] == "dependency_failed"
    assert skips[1]["reason"] == "dependency_skipped"
    assert _collect_cot_violations(trace) == []
    assert trace["labId"] == EXAMPLE_ID


def test_no_delegate_or_handoff_phase_in_any_signature():
    for case in CASES:
        result = run_orchestration(case, catalog=Catalog(DATA))
        phases = signature_phases_from_view(build_signature_view(result.sequence))
        assert FORBIDDEN_PHASES.isdisjoint(phases)
        kinds = [event.kind for event in result.sequence]
        assert "delegation" not in kinds
        assert "handoff_requested" not in kinds
        assert "ownership_transferred" not in kinds
