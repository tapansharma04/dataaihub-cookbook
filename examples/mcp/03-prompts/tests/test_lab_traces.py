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
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_TRACE_IDS = frozenset(
    {
        "prompt-discovery",
        "single-prompt-get-summarize",
        "prompt-with-arguments-investigate",
        "invalid-prompt-name",
    }
)
EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "PROMPT_DISCOVERY",
        "SINGLE_PROMPT_GET",
        "PROMPT_WITH_ARGUMENTS",
        "INVALID_PROMPT",
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
        "promptGetMs",
        "prompt_get_ms",
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
    settings = Settings(mcp_client_mode="legacy")
    case = get_case(trace_id)
    result = run_case(case, settings=settings, server=build_server())
    return build_trace(case=case, result=result, settings=settings)


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert trace["traceId"] in EXPECTED_TRACE_IDS
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert set(trace["metrics"]) >= {
        "promptsDiscovered",
        "promptsRequested",
        "successfulGets",
        "failedGets",
        "messageCount",
        "messageBytes",
        "modelTurns",
        "toolCalls",
        "resourcesRead",
        "totalMs",
        "terminationReason",
        "provenance",
    }
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["modelTurns"] == 0
    assert trace["metrics"]["toolCalls"] == 0
    assert trace["metrics"]["resourcesRead"] == 0
    assert "sequence" in trace and trace["sequence"]
    assert "prompts" in trace and len(trace["prompts"]) == 3
    assert "presentation" in trace
    assert "signatureFlow" in trace["presentation"]
    assert _collect_cot_violations(trace) == []


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


def test_committed_invalid_prompt_semantics():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    invalid = next(t for t in traces if t["traceId"] == "invalid-prompt-name")
    kinds = [event["kind"] for event in invalid["sequence"]]
    assert "prompts_list_response" in kinds
    assert "prompt_get_request" in kinds
    assert "prompt_get_response" in kinds
    assert "error" not in kinds
    response = next(
        event for event in invalid["sequence"] if event["kind"] == "prompt_get_response"
    )
    assert response["detail"]["isError"] is True
    assert response["detail"]["name"] == "does-not-exist"
    assert "messages" not in response["detail"]
    assert invalid["metrics"]["failedGets"] == 1
    assert invalid["metrics"]["successfulGets"] == 0
    assert invalid["metrics"]["messageCount"] == 0


def test_discovery_trace_has_no_get_event():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    discovery = next(t for t in traces if t["traceId"] == "prompt-discovery")
    kinds = [event["kind"] for event in discovery["sequence"]]
    assert "prompt_get_request" not in kinds
    assert "prompt_get_response" not in kinds


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
