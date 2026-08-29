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
    assert SIGNATURE_FLOWS["PROMPT_DISCOVERY"] == "INITIALIZE → DISCOVER → PROMPTS"
    assert (
        SIGNATURE_FLOWS["SINGLE_PROMPT_GET"]
        == "INITIALIZE → DISCOVER → PROMPTS → GET → MESSAGES"
    )
    assert (
        SIGNATURE_FLOWS["PROMPT_WITH_ARGUMENTS"]
        == "INITIALIZE → DISCOVER → PROMPTS → GET → ARGUMENTS → MESSAGES"
    )
    assert SIGNATURE_FLOWS["INVALID_PROMPT"] == "INITIALIZE → DISCOVER → GET → REJECTED"


def test_build_trace_provenance_and_presentation():
    settings = Settings(mcp_client_mode="legacy")
    case = get_case("prompt-discovery")
    result = run_case(case, settings=settings, server=build_server())
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["metricsProvenance"] == "measured"
    assert trace["presentation"]["signatureFlow"] == "INITIALIZE → DISCOVER → PROMPTS"
    assert trace["metrics"]["modelTurns"] == 0
    assert trace["metrics"]["toolCalls"] == 0
    assert trace["metrics"]["resourcesRead"] == 0
    assert "prompts" in trace
    assert len(trace["prompts"]) == 3


def test_single_prompt_signature_and_no_cot():
    settings = Settings(mcp_client_mode="legacy")
    case = get_case("single-prompt-get-summarize")
    result = run_case(case, settings=settings, server=build_server())
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "GET" in phases
    assert "MESSAGES" in phases
    assert "ARGUMENTS" not in phases
    trace = build_trace(case=case, result=result, settings=settings)
    assert _collect_cot_violations(trace) == []
    assert trace["presentation"]["signatureFlow"] == (
        "INITIALIZE → DISCOVER → PROMPTS → GET → MESSAGES"
    )


def test_multi_argument_signature_includes_arguments_phase():
    settings = Settings(mcp_client_mode="legacy")
    case = get_case("prompt-with-arguments-investigate")
    result = run_case(case, settings=settings, server=build_server())
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "ARGUMENTS" in phases
    assert "MESSAGES" in phases
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["presentation"]["signatureFlow"] == (
        "INITIALIZE → DISCOVER → PROMPTS → GET → ARGUMENTS → MESSAGES"
    )


def test_protocol_error_event_still_supported():
    sequence = [
        SequenceEvent(
            kind="prompt_get_request",
            detail={"method": "prompts/get", "name": "x"},
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
    assert phases == ["GET", "ERROR"]


def test_invalid_prompt_trace_has_no_redundant_error_event():
    settings = Settings(mcp_client_mode="legacy")
    case = get_case("invalid-prompt-name")
    result = run_case(case, settings=settings, server=build_server())
    trace = build_trace(case=case, result=result, settings=settings)
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "error" not in kinds
    response = next(
        event for event in trace["sequence"] if event["kind"] == "prompt_get_response"
    )
    assert response["detail"]["isError"] is True
