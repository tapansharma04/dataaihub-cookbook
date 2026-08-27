"""Trace presentation and no-CoT tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from client.cases import get_case
from client.runner import run_case
from client.schemas import SequenceEvent
from client.trace import SIGNATURE_FLOWS, build_signature_view, build_trace
from config import EXAMPLE_ID, Settings
from server.app import build_server

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

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


def test_signature_flows_match_spec():
    assert SIGNATURE_FLOWS["DISCOVERY"] == "INITIALIZE → RESOURCES"
    assert (
        SIGNATURE_FLOWS["SINGLE_RESOURCE_READ"]
        == "INITIALIZE → DISCOVER → READ → CONTENT"
    )
    assert (
        SIGNATURE_FLOWS["MULTI_RESOURCE_READ"]
        == "INITIALIZE → DISCOVER → READ → READ → CONTENT"
    )
    assert (
        SIGNATURE_FLOWS["INVALID_RESOURCE"] == "INITIALIZE → DISCOVER → READ → REJECTED"
    )


def test_build_trace_provenance_and_presentation():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy")
    case = get_case("discovery")
    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["metricsProvenance"] == "measured"
    assert trace["presentation"]["signatureFlow"] == "INITIALIZE → RESOURCES"
    assert trace["metrics"]["modelTurns"] == 0
    assert trace["metrics"]["toolCalls"] == 0
    assert "resources" in trace
    assert len(trace["resources"]) == 3


def test_single_resource_signature_and_no_cot():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy")
    case = get_case("single-resource-read-knowledge-platform")
    result = run_case(case, settings=settings, server=build_server(DATA))
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "READ" in phases
    assert "CONTENT" in phases
    trace = build_trace(case=case, result=result, settings=settings)
    assert _collect_cot_violations(trace) == []
    assert trace["presentation"]["signatureFlow"] == (
        "INITIALIZE → DISCOVER → READ → CONTENT"
    )


def test_multi_resource_signature_has_two_reads():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy")
    case = get_case("multi-resource-read-services")
    result = run_case(case, settings=settings, server=build_server(DATA))
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert phases.count("READ") == 2
    assert "CONTENT" in phases
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["presentation"]["signatureFlow"] == (
        "INITIALIZE → DISCOVER → READ → READ → CONTENT"
    )


def test_protocol_error_event_still_supported():
    sequence = [
        SequenceEvent(
            kind="resource_read_request",
            detail={"method": "resources/read", "uri": "acme://x"},
        ),
        SequenceEvent(
            kind="error",
            detail={
                "stage": "transport",
                "message": "connection closed before response",
            },
        ),
    ]
    phases = [v["phase"] for v in build_signature_view(sequence)]
    assert phases == ["READ", "ERROR"]


def test_invalid_arguments_trace_has_no_redundant_error_event():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy")
    case = get_case("invalid-resource-uri")
    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "error" not in kinds
    response = next(
        event
        for event in trace["sequence"]
        if event["kind"] == "resource_read_response"
    )
    assert response["detail"]["isError"] is True
