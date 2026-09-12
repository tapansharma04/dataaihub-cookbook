"""Trace presentation and no-CoT tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_collaboration
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
    result = run_collaboration(case, catalog=Catalog(DATA))
    return build_trace(case=case, result=result, settings=settings)


EXPECTED_SIGNATURE_PHASES = {
    "BASIC_COLLABORATION": [
        "TASK",
        "DELEGATE",
        "AGENT",
        "DELEGATE",
        "AGENT",
        "AGGREGATE",
        "TERMINATION",
    ],
    "SEQUENTIAL_COLLABORATION": [
        "TASK",
        "DELEGATE",
        "AGENT",
        "DELEGATE",
        "PASS_RESULT",
        "AGENT",
        "AGGREGATE",
        "TERMINATION",
    ],
    "AGENT_FAILURE": [
        "TASK",
        "DELEGATE",
        "AGENT",
        "DELEGATE",
        "AGENT",
        "FAILURE",
        "AGGREGATE",
        "TERMINATION",
    ],
    "COLLABORATION_TERMINATION": [
        "TASK",
        "DELEGATE",
        "AGENT",
        "SKIP",
        "AGGREGATE",
        "TERMINATION",
    ],
}


def test_signature_flows_match_spec():
    for example_class, phases in EXPECTED_SIGNATURE_PHASES.items():
        assert SIGNATURE_FLOWS[example_class] == " → ".join(phases)


def test_signature_flows_match_runtime_signature_view():
    settings = Settings(openai_api_key="", data_dir=DATA)
    for case in CASES:
        result = run_collaboration(case, catalog=Catalog(DATA))
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


def test_sequential_signature_pass_result_before_analysis_agent():
    settings = Settings(openai_api_key="", data_dir=DATA)
    case = get_case("sequential-status-then-analysis")
    result = run_collaboration(case, catalog=Catalog(DATA))
    view = [
        item
        for item in build_signature_view(result.sequence)
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    phases = [item["phase"] for item in view]
    assert phases == EXPECTED_SIGNATURE_PHASES["SEQUENTIAL_COLLABORATION"]
    assert view[2]["phase"] == "AGENT"
    assert view[2]["agentId"] == "status_agent"
    assert view[4]["phase"] == "PASS_RESULT"
    assert view[4]["to"] == "analysis_agent"
    assert view[5]["phase"] == "AGENT"
    assert view[5]["agentId"] == "analysis_agent"
    pass_msg = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "agent_message" and event.detail.get("parentResultId")
    )
    analysis_result = next(
        i
        for i, event in enumerate(result.sequence)
        if (
            event.kind == "agent_result"
            and event.detail.get("agentId") == "analysis_agent"
        )
    )
    assert pass_msg < analysis_result
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert _collect_cot_violations(trace) == []
    committed_phases = [
        item["phase"]
        for item in trace["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert committed_phases == phases


def test_failure_signature_includes_failure_then_aggregate_then_termination():
    trace = _build("docs-agent-failure")
    phases = signature_phases_from_view(trace["presentation"]["signatureView"])
    assert phases == EXPECTED_SIGNATURE_PHASES["AGENT_FAILURE"]
    assert phases.index("FAILURE") < phases.index("AGGREGATE")
    assert phases.index("AGGREGATE") < phases.index("TERMINATION")
    assert phases.count("TERMINATION") == 1
    assert trace["presentation"]["signatureFlow"] == SIGNATURE_FLOWS["AGENT_FAILURE"]
    assert trace["output"]["ok"] is False
    assert trace["output"]["terminationReason"] == "agent_failed"
    assert "analysis_agent" not in trace["state"]["invokedAgentIds"]
    assert _collect_cot_violations(trace) == []
