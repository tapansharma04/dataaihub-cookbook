"""Trace presentation, exact signatures, and no-CoT tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_handoffs
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
    "BASIC_HANDOFF": [
        "TASK",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ],
    "MULTI_HOP_HANDOFF": [
        "TASK",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ],
    "INVALID_HANDOFF": [
        "TASK",
        "AGENT",
        "HANDOFF",
        "HANDOFF_REJECTED",
        "TERMINATION",
    ],
    "HANDOFF_FAILURE": [
        "TASK",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "FAILURE",
        "TERMINATION",
    ],
}


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
    result = run_handoffs(case, catalog=Catalog(DATA))
    return build_trace(case=case, result=result, settings=settings)


def test_signature_flows_match_spec():
    for example_class, phases in EXPECTED_SIGNATURE_PHASES.items():
        assert SIGNATURE_FLOWS[example_class] == " → ".join(phases)


def test_signature_flows_match_runtime_signature_view():
    settings = Settings(openai_api_key="", data_dir=DATA)
    for case in CASES:
        result = run_handoffs(case, catalog=Catalog(DATA))
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


def test_basic_signature_handoff_then_billing_then_complete():
    case = get_case("invoice-duplicate-basic-handoff")
    result = run_handoffs(case, catalog=Catalog(DATA))
    view = [
        item
        for item in build_signature_view(result.sequence)
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert [item["phase"] for item in view] == EXPECTED_SIGNATURE_PHASES[
        "BASIC_HANDOFF"
    ]
    assert view[1]["phase"] == "AGENT"
    assert view[1]["agentId"] == "triage_agent"
    assert view[2]["phase"] == "HANDOFF"
    assert view[2]["from"] == "triage_agent"
    assert view[2]["to"] == "billing_agent"
    assert view[3]["phase"] == "AGENT"
    assert view[3]["agentId"] == "billing_agent"
    assert view[4]["phase"] == "COMPLETE"
    assert view[4]["currentOwner"] == "billing_agent"


def test_multi_hop_signature_makes_both_transfers_explicit():
    case = get_case("refund-blocked-multi-hop")
    result = run_handoffs(case, catalog=Catalog(DATA))
    view = [
        item
        for item in build_signature_view(result.sequence)
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    phases = [item["phase"] for item in view]
    assert phases == EXPECTED_SIGNATURE_PHASES["MULTI_HOP_HANDOFF"]
    handoffs = [item for item in view if item["phase"] == "HANDOFF"]
    assert [(item["from"], item["to"]) for item in handoffs] == [
        ("triage_agent", "billing_agent"),
        ("billing_agent", "technical_agent"),
    ]
    agents = [item for item in view if item["phase"] == "AGENT"]
    assert [item["agentId"] for item in agents] == [
        "triage_agent",
        "billing_agent",
        "technical_agent",
    ]
    assert "DELEGATE" not in phases


def test_invalid_signature_rejects_without_second_agent():
    trace = _build("legal-target-invalid-handoff")
    phases = signature_phases_from_view(trace["presentation"]["signatureView"])
    assert phases == EXPECTED_SIGNATURE_PHASES["INVALID_HANDOFF"]
    assert phases.index("HANDOFF") < phases.index("HANDOFF_REJECTED")
    assert phases.count("AGENT") == 1
    assert "COMPLETE" not in phases
    view = [
        item
        for item in trace["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert view[1]["agentId"] == "triage_agent"
    assert view[2]["to"] == "legal_agent"
    assert view[3]["code"] == "unknown_target"
    assert view[3]["currentOwner"] == "triage_agent"
    assert trace["output"]["ok"] is False
    assert trace["output"]["terminationReason"] == "invalid_handoff"


def test_failure_signature_includes_failure_then_termination():
    trace = _build("unknown-system-handoff-failure")
    phases = signature_phases_from_view(trace["presentation"]["signatureView"])
    assert phases == EXPECTED_SIGNATURE_PHASES["HANDOFF_FAILURE"]
    assert phases.index("FAILURE") < phases.index("TERMINATION")
    assert "COMPLETE" not in phases
    assert trace["output"]["ok"] is False
    assert trace["output"]["terminationReason"] == "agent_failed"
    assert _collect_cot_violations(trace) == []
    assert trace["labId"] == EXAMPLE_ID


def test_no_delegate_phase_in_any_signature():
    for case in CASES:
        result = run_handoffs(case, catalog=Catalog(DATA))
        phases = signature_phases_from_view(build_signature_view(result.sequence))
        assert "DELEGATE" not in phases
        assert "AGGREGATE" not in phases
