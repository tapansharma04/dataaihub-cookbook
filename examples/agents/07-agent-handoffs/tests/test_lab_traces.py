"""Validate committed lab_traces.json and semantic regeneration stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_handoffs
from agent.trace import SIGNATURE_FLOWS, SIGNATURE_OMITTED_PHASES, build_trace
from config import EXAMPLE_ID, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_TRACE_IDS = frozenset(
    {
        "invoice-duplicate-basic-handoff",
        "refund-blocked-multi-hop",
        "legal-target-invalid-handoff",
        "unknown-system-handoff-failure",
    }
)
EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "BASIC_HANDOFF",
        "MULTI_HOP_HANDOFF",
        "INVALID_HANDOFF",
        "HANDOFF_FAILURE",
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
        "agentMs",
        "agent_ms",
        "runtimeMs",
        "runtime_ms",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "qualityScore",
        "handoffScore",
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
    result = run_handoffs(case, catalog=Catalog(DATA))
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
    assert "qualityScore" not in trace["metrics"]
    data_dir = trace["input"]["config"]["dataDir"]
    assert data_dir == "data"
    assert not Path(data_dir).is_absolute()


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


def test_committed_basic_preserves_ownership_transfer():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    basic = next(
        trace
        for trace in traces
        if trace["traceId"] == "invoice-duplicate-basic-handoff"
    )
    assert basic["state"]["currentOwner"] == "billing_agent"
    assert basic["state"]["previousOwner"] == "triage_agent"
    phases = [
        item["phase"]
        for item in basic["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert phases == [
        "TASK",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ]
    assert basic["presentation"]["signatureFlow"] == SIGNATURE_FLOWS["BASIC_HANDOFF"]


def test_committed_multi_hop_preserves_data_flow_and_ownership():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    multi = next(
        trace for trace in traces if trace["traceId"] == "refund-blocked-multi-hop"
    )
    billing = multi["state"]["results"][1]
    technical = multi["state"]["results"][2]
    assert technical["payload"]["basedOn"] == billing["handoff"]["payload"]
    assert "ticket_id" not in technical["payload"]
    assert "ticket_id" not in technical["payload"]["basedOn"]
    assert "ticket_id" not in billing["handoff"]["payload"]
    assert "ticket_id" not in billing["payload"]
    assert "ticket_id" not in multi["state"]["results"][0]["payload"]
    assert multi["state"]["currentOwner"] == "technical_agent"
    transfers = [
        event for event in multi["sequence"] if event["kind"] == "ownership_transferred"
    ]
    assert [
        (event["detail"]["from"], event["detail"]["to"]) for event in transfers
    ] == [
        ("triage_agent", "billing_agent"),
        ("billing_agent", "technical_agent"),
    ]
    phases = [
        item["phase"]
        for item in multi["presentation"]["signatureView"]
        if item["phase"] not in SIGNATURE_OMITTED_PHASES
    ]
    assert phases == [
        "TASK",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "HANDOFF",
        "AGENT",
        "COMPLETE",
        "TERMINATION",
    ]


def test_committed_invalid_is_rejected_not_rerouted():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    invalid = next(
        trace for trace in traces if trace["traceId"] == "legal-target-invalid-handoff"
    )
    assert invalid["output"]["ok"] is False
    assert invalid["metrics"]["terminationReason"] == "invalid_handoff"
    assert invalid["metrics"]["handoffsRejected"] == 1
    assert invalid["metrics"]["ownershipTransfers"] == 0
    assert invalid["state"]["currentOwner"] == "triage_agent"
    kinds = [event["kind"] for event in invalid["sequence"]]
    assert "handoff_rejected" in kinds
    assert "handoff_accepted" not in kinds
    assert (
        invalid["presentation"]["signatureFlow"] == SIGNATURE_FLOWS["INVALID_HANDOFF"]
    )


def test_committed_failure_is_not_success():
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    failure = next(
        trace
        for trace in traces
        if trace["traceId"] == "unknown-system-handoff-failure"
    )
    assert failure["output"]["ok"] is False
    assert failure["metrics"]["terminationReason"] == "agent_failed"
    assert failure["metrics"]["failedAgentResults"] == 1
    kinds = [event["kind"] for event in failure["sequence"]]
    assert kinds.count("termination") == 1
    assert kinds.index("failure") < kinds.index("termination")
    assert "completed" not in kinds
    assert failure["presentation"]["signatureFlow"].endswith("FAILURE → TERMINATION")


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
