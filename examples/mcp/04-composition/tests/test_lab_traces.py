"""Validate committed lab_traces.json and semantic regeneration stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from client.cases import CASES, get_case
from client.runner import run_case
from client.trace import build_trace
from config import EXAMPLE_ID, Settings
from server.app import build_server

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_TRACE_IDS = frozenset(
    {
        "resource-to-sampling",
        "prompt-to-sampling",
        "tool-resource-prompt-composition",
        "sampling-failure",
    }
)
EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "RESOURCE_TO_SAMPLING",
        "PROMPT_TO_SAMPLING",
        "TOOL_RESOURCE_PROMPT_COMPOSITION",
        "SAMPLING_FAILURE",
    }
)
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
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "totalMs",
        "total_ms",
        "initializeMs",
        "initialize_ms",
        "discoveryMs",
        "discovery_ms",
        "resourceReadMs",
        "resource_read_ms",
        "promptGetMs",
        "prompt_get_ms",
        "toolCallMs",
        "tool_call_ms",
        "samplingMs",
        "sampling_ms",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "promptQualityScore",
        "relevanceScore",
        "intelligenceScore",
        "accuracyScore",
        "benchmarkScore",
        "confidenceScore",
        "tokenCount",
        "promptTokens",
        "completionTokens",
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
    settings = Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")
    case = get_case(trace_id)
    result = run_case(case, settings=settings, server=build_server(DATA))
    return build_trace(case=case, result=result, settings=settings)


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert trace["traceId"] in EXPECTED_TRACE_IDS
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"]["tools"] == "measured"
    assert trace["provenance"]["metrics"] == "measured"
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert "sequence" in trace and trace["sequence"]
    assert "presentation" in trace
    assert "signatureFlow" in trace["presentation"]
    assert _collect_cot_violations(trace) == []
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "sampling_request" in kinds
    assert "sampling_response" in kinds


def test_committed_lab_traces_schema():
    assert LAB_TRACES_PATH.exists()
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES)
    ids = {trace["traceId"] for trace in traces}
    assert ids == EXPECTED_TRACE_IDS
    classes = {trace["exampleClass"] for trace in traces}
    assert classes == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_trace_contract(trace)


def test_committed_sampling_failure_semantics():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    failure = next(t for t in traces if t["traceId"] == "sampling-failure")
    response = next(
        event for event in failure["sequence"] if event["kind"] == "sampling_response"
    )
    assert response["detail"]["isError"] is True
    assert "result" not in response["detail"]
    assert failure["metrics"]["failedSamplings"] == 1
    assert failure["metrics"]["successfulSamplings"] == 0
    assert failure["metrics"]["modelTurns"] == 0
    assert failure["provenance"]["model"] == "not_used"


def test_committed_mock_success_provenance():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    success = next(t for t in traces if t["traceId"] == "resource-to-sampling")
    assert success["provenance"]["model"] == "mock"
    assert success["metrics"]["successfulSamplings"] == 1


def test_no_chain_of_thought_in_lab_traces():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        assert _collect_cot_violations(trace) == []


def test_semantic_regeneration_matches_committed():
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    committed_by_id = {trace["traceId"]: trace for trace in committed}
    for case in CASES:
        regenerated = _build(case.trace_id)
        left = _strip_volatile(committed_by_id[case.trace_id])
        right = _strip_volatile(regenerated)
        assert left == right


def test_semantic_regeneration_is_stable_across_runs():
    first = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    second = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    assert first == second
