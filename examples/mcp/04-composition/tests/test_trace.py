"""Trace presentation and no-CoT tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from client.cases import get_case
from client.runner import run_case
from client.trace import SIGNATURE_FLOWS, build_signature_view, build_trace
from config import EXAMPLE_ID, Settings
from server.app import build_server

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


def test_signature_flows_match_spec():
    assert SIGNATURE_FLOWS["RESOURCE_TO_SAMPLING"] == (
        "INITIALIZE → RESOURCE → CONTEXT → SAMPLING → RESULT"
    )
    assert SIGNATURE_FLOWS["PROMPT_TO_SAMPLING"] == (
        "INITIALIZE → PROMPT → ARGUMENTS → SAMPLING → RESULT"
    )
    assert SIGNATURE_FLOWS["TOOL_RESOURCE_PROMPT_COMPOSITION"] == (
        "INITIALIZE → TOOL → RESOURCE → PROMPT → SAMPLING → RESULT"
    )
    assert SIGNATURE_FLOWS["SAMPLING_FAILURE"] == (
        "INITIALIZE → RESOURCE → SAMPLING → REJECTED"
    )


def test_build_trace_provenance_mock():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")
    case = get_case("resource-to-sampling")
    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert trace["provenance"] == {
        "model": "mock",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["metricsProvenance"] == "measured"
    assert _collect_cot_violations(trace) == []


def test_failure_trace_provenance_not_used():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")
    case = get_case("sampling-failure")
    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["provenance"]["model"] == "not_used"
    assert trace["metrics"]["successfulSamplings"] == 0
    assert _collect_cot_violations(trace) == []


def test_composition_signature_includes_all_primitives():
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")
    case = get_case("tool-resource-prompt-composition")
    result = run_case(case, settings=settings, server=build_server(DATA))
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "TOOL" in phases
    assert "RESOURCE" in phases
    assert "PROMPT" in phases
    assert "SAMPLING" in phases
    assert "RESULT" in phases
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["presentation"]["signatureFlow"] == (
        "INITIALIZE → TOOL → RESOURCE → PROMPT → SAMPLING → RESULT"
    )
