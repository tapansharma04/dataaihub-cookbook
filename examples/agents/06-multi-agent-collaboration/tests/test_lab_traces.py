"""Validate committed lab_traces.json and semantic regeneration stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_collaboration
from agent.trace import SIGNATURE_FLOWS, SIGNATURE_OMITTED_PHASES, build_trace
from config import EXAMPLE_ID, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_TRACE_IDS = frozenset(
    {
        "independent-status-and-docs",
        "sequential-status-then-analysis",
        "docs-agent-failure",
        "operational-short-circuit",
    }
)
EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "BASIC_COLLABORATION",
        "SEQUENTIAL_COLLABORATION",
        "AGENT_FAILURE",
        "COLLABORATION_TERMINATION",
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
        "coordinatorMs",
        "coordinator_ms",
        "agentMs",
        "agent_ms",
        "synthesisMs",
        "synthesis_ms",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "qualityScore",
        "collaborationScore",
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
    settings = Settings(openai_api_key="", data_dir=DATA)
    case = get_case(trace_id)
    result = run_collaboration(case, catalog=Catalog(DATA))
    return build_trace(case=case, result=result, settings=settings)


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert trace["traceId"] in EXPECTED_TRACE_IDS
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"]["tools"] == "measured"
    assert trace["provenance"]["metrics"] == "measured"
    assert trace["provenance"]["model"] == "mock"
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["sequence"]
    assert trace["steps"]
    assert trace["state"]
    assert "presentation" in trace
    assert _collect_cot_violations(trace) == []
    kinds = [event["kind"] for event in trace["sequence"]]
    assert kinds[0] == "task_created"
    assert kinds[-1] == "termination"
    assert "collaborationScore" not in trace["metrics"]


def test_committed_lab_traces_schema():
    assert LAB_TRACES_PATH.exists()
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert len(traces) == len(CASES)
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_trace_contract(trace)
        assert (
            trace["presentation"]["signatureFlow"]
            == SIGNATURE_FLOWS[trace["exampleClass"]]
        )
        kinds = [event["kind"] for event in trace["sequence"]]
        assert kinds.count("termination") == 1
        assert kinds[-1] == "termination"


def test_committed_sequential_preserves_data_flow():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    sequential = next(
        trace
        for trace in traces
        if trace["traceId"] == "sequential-status-then-analysis"
    )
    status = sequential["state"]["results"][0]
    analysis = sequential["state"]["results"][1]
    assert analysis["payload"]["basedOn"] == status["payload"]
    phases = [
        item["phase"]
        for item in sequential["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert phases == [
        "TASK",
        "DELEGATE",
        "AGENT",
        "DELEGATE",
        "PASS_RESULT",
        "AGENT",
        "AGGREGATE",
        "TERMINATION",
    ]
    view = [
        item
        for item in sequential["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert view[2]["agentId"] == "status_agent"
    assert view[4]["phase"] == "PASS_RESULT"
    assert view[4]["to"] == "analysis_agent"
    assert view[5]["agentId"] == "analysis_agent"
    assert (
        sequential["presentation"]["signatureFlow"]
        == SIGNATURE_FLOWS["SEQUENTIAL_COLLABORATION"]
    )


def test_committed_failure_is_not_success():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    failure = next(
        trace for trace in traces if trace["traceId"] == "docs-agent-failure"
    )
    assert failure["output"]["ok"] is False
    assert failure["metrics"]["terminationReason"] == "agent_failed"
    assert failure["metrics"]["failedAgentResults"] == 1
    kinds = [event["kind"] for event in failure["sequence"]]
    assert kinds.count("termination") == 1
    assert kinds.index("failure") < kinds.index("aggregation")
    assert kinds.index("aggregation") < kinds.index("termination")
    assert failure["presentation"]["signatureFlow"].endswith(
        "FAILURE → AGGREGATE → TERMINATION"
    )


def test_no_chain_of_thought_in_lab_traces():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        assert _collect_cot_violations(trace) == []


def test_semantic_regeneration_matches_committed():
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    committed_by_id = {trace["traceId"]: trace for trace in committed}
    for case in CASES:
        regenerated = _build(case.trace_id)
        assert _strip_volatile(committed_by_id[case.trace_id]) == _strip_volatile(
            regenerated
        )


def test_semantic_regeneration_is_stable_across_runs():
    first = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    second = [_strip_volatile(_build(case.trace_id)) for case in CASES]
    assert first == second
