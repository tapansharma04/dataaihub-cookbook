"""Trace schema, provenance, CoT, and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case, scripted_client_for
from agent.loop import run_memory_loop
from agent.memory import FixedClock, MemoryStore
from agent.source import AuthoritativeStore
from agent.trace import build_signature_view, build_trace
from config import EXAMPLE_ID, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_EXAMPLE_CLASSES = frozenset({"NO_MEMORY", "STORE", "RECALL", "STALE_MEMORY"})
EXPECTED_TRACE_IDS = frozenset(
    {
        "no-memory-notification-preference",
        "store-email-notification-preference",
        "recall-email-notification-preference",
        "stale-memory-notification-preference",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {"final_answer", "max_turns", "invalid_action", "error"}
)
COT_FIELD_NAMES = frozenset(
    {
        "chainOfThought",
        "chain_of_thought",
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
    }
)
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "totalMs",
        "modelMs",
        "toolMs",
        "total_ms",
        "model_ms",
        "tool_ms",
    }
)


def _known_scopes() -> set[str]:
    users = json.loads((DATA / "users.json").read_text(encoding="utf-8"))
    return set(users)


def _collect_cot_violations(obj: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in COT_FIELD_NAMES:
                violations.append(child_path)
            violations.extend(_collect_cot_violations(value, child_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(_collect_cot_violations(item, f"{path}[{i}]"))
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


def _result(trace_id: str):
    case = get_case(trace_id)
    max_turns = case.max_turns if case.max_turns is not None else 6
    result = run_memory_loop(
        interactions=[item.request for item in case.interactions],
        scope=case.scope,
        model=scripted_client_for(case),
        memory_store=MemoryStore(known_scopes=_known_scopes(), clock=FixedClock()),
        authoritative=AuthoritativeStore.from_data_dir(DATA),
        max_turns=max_turns,
    )
    return case, result, max_turns


def _build(trace_id: str) -> dict[str, Any]:
    settings = Settings(openai_api_key="", data_dir=DATA)
    case, result, max_turns = _result(trace_id)
    return build_trace(case=case, result=result, settings=settings, max_turns=max_turns)


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert isinstance(trace["traceId"], str) and trace["traceId"]
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"] == {
        "model": "case-harness",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert "tools" in trace and trace["tools"]
    assert "sequence" in trace and len(trace["sequence"]) >= 2
    assert "steps" in trace and len(trace["steps"]) >= 2
    assert "state" in trace
    assert "metrics" in trace
    assert set(trace["metrics"]) >= {
        "totalMs",
        "modelMs",
        "toolMs",
        "modelTurns",
        "toolCalls",
        "successfulToolCalls",
        "failedToolCalls",
        "terminationReason",
        "maxTurns",
        "memoryWrites",
        "memoryReads",
        "memoryHits",
        "memoryMisses",
        "memoryScope",
        "memoryVersion",
        "staleMemoryDetected",
        "provenance",
    }
    assert trace["metrics"]["provenance"] == "measured"
    assert "memoryQuality" not in trace["metrics"]
    assert "memoryScore" not in trace["metrics"]
    assert "memoryAccuracy" not in trace["metrics"]

    termination_reason = trace["metrics"]["terminationReason"]
    assert termination_reason in SUPPORTED_TERMINATION_REASONS
    assert trace["output"]["terminationReason"] == termination_reason
    assert trace["state"]["terminationReason"] == termination_reason

    termination_steps = [s for s in trace["steps"] if s["type"] == "termination"]
    assert len(termination_steps) >= 1
    assert termination_steps[-1]["detail"]["reason"] == termination_reason
    termination_events = [e for e in trace["sequence"] if e["kind"] == "termination"]
    assert len(termination_events) >= 1
    assert termination_events[-1]["detail"]["reason"] == termination_reason

    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    assert trace["architecture"]["layout"] == "agent-memory"
    signature_phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert signature_phases[-1] == "TERMINATION"

    kinds = [e["kind"] for e in trace["sequence"]]
    assert "user_request" in kinds

    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})
        assert "interactionId" in event

    for step in trace["steps"]:
        if step["type"] == "llm":
            assert step["metrics"]["provenance"] == "case-harness"
        else:
            assert step["metrics"]["provenance"] == "measured"

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []


def test_all_four_cases_have_corresponding_traces():
    traces = [_build(case.trace_id) for case in CASES]
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    assert {t["exampleClass"] for t in traces} == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_trace_contract(trace)


def test_case_trace_alignment():
    traces = [_build(case.trace_id) for case in CASES]
    assert len(traces) == len(CASES) == 4
    case_by_trace = {c.trace_id: c for c in CASES}
    for trace in traces:
        case = case_by_trace[trace["traceId"]]
        assert trace["exampleClass"] == case.example_class


def test_no_memory_trace_represents_miss():
    trace = _build("no-memory-notification-preference")
    _assert_trace_contract(trace)
    kinds = [e["kind"] for e in trace["sequence"]]
    assert "memory_not_found" in kinds
    assert "memory_stored" not in kinds
    phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert "MEMORY_MISS" in phases
    assert "missNote" in trace["presentation"]
    assert trace["metrics"]["memoryMisses"] == 1


def test_store_trace_represents_write():
    trace = _build("store-email-notification-preference")
    kinds = [e["kind"] for e in trace["sequence"]]
    assert "memory_write_requested" in kinds
    assert "memory_stored" in kinds
    phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert "MEMORY_WRITE" in phases
    stored = next(e for e in trace["sequence"] if e["kind"] == "memory_stored")
    assert stored["detail"]["source"] == "user"
    assert stored["detail"]["scope"] == "u-1001"


def test_recall_trace_preserves_interaction_boundaries():
    trace = _build("recall-email-notification-preference")
    _assert_trace_contract(trace)
    ids = [e["interactionId"] for e in trace["sequence"] if e["interactionId"]]
    assert "interaction-1" in ids
    assert "interaction-2" in ids
    stored = next(e for e in trace["sequence"] if e["kind"] == "memory_stored")
    retrieved = next(e for e in trace["sequence"] if e["kind"] == "memory_retrieved")
    assert stored["interactionId"] == "interaction-1"
    assert retrieved["interactionId"] == "interaction-2"
    phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert "INTERACTION_BOUNDARY" in phases
    assert "MEMORY_WRITE" in phases
    assert "MEMORY_RETRIEVE" in phases
    assert "MEMORY_USED" in phases
    second_request = trace["input"]["interactions"][1]["request"]
    assert "email" not in second_request.lower()
    assert "email" in trace["output"]["answer"].lower()


def test_stale_memory_trace_is_lab_ready():
    trace = _build("stale-memory-notification-preference")
    _assert_trace_contract(trace)
    retrieved = next(e for e in trace["sequence"] if e["kind"] == "memory_retrieved")
    observation = next(e for e in trace["sequence"] if e["kind"] == "observation")
    assert retrieved["detail"]["record"]["value"]["channel"] == "email"
    assert observation["detail"]["current"]["channel"] == "sms"
    assert observation["detail"]["staleMemoryDetected"] is True
    assert trace["metrics"]["staleMemoryDetected"] is True
    phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert "MEMORY_RETRIEVE" in phases
    assert "CURRENT_SOURCE" in phases
    assert "MEMORY_STALE" in phases
    assert "staleNote" in trace["presentation"]
    assert "sms" in trace["output"]["answer"].lower()


def test_build_trace_separates_presentation_metadata():
    trace = _build("store-email-notification-preference")
    _assert_trace_contract(trace)
    assert "stageNotes" in trace["presentation"]
    assert "securityNote" in trace["presentation"]
    for event in trace["sequence"]:
        assert "note" not in event


def test_signature_view_ends_with_termination():
    _, result, _ = _result("store-email-notification-preference")
    view = build_signature_view(result.sequence)
    phases = [v["phase"] for v in view]
    assert phases[0] == "USER_REQUEST"
    assert "MEMORY_WRITE" in phases
    assert phases[-1] == "TERMINATION"


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES) == 4
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_trace_contract(trace)


def test_committed_traces_match_regenerated_semantically():
    if not LAB_TRACES_PATH.exists():
        return
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    regenerated = [_build(case.trace_id) for case in CASES]
    committed_by_id = {t["traceId"]: t for t in committed}
    for trace in regenerated:
        left = _strip_volatile(committed_by_id[trace["traceId"]])
        right = _strip_volatile(trace)
        assert left == right


def test_no_chain_of_thought_in_lab_traces():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        violations = _collect_cot_violations(trace)
        assert violations == [], (
            f"{trace['traceId']} contains hidden-reasoning fields: {violations}"
        )


def test_no_hidden_reasoning_in_fresh_traces():
    for case in CASES:
        violations = _collect_cot_violations(_build(case.trace_id))
        assert violations == []


def test_trace_provenance_model():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        assert trace["provenance"]["model"] == "case-harness"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert trace["input"]["config"]["modelDriver"] == "case-harness"
