"""Trace schema and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_agent_loop
from agent.tools import build_registry
from agent.trace import build_signature_view, build_trace
from config import EXAMPLE_ID, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {"SIMPLE_LOOP", "TERMINATION", "MAX_TURNS", "INVALID_ACTION"}
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
    }
)


def _result(trace_id: str):
    case = get_case(trace_id)
    registry = build_registry(DATA)
    max_turns = case.max_turns if case.max_turns is not None else 6
    return (
        case,
        run_agent_loop(
            request=case.request,
            model=scripted_client_for(case),
            registry=registry,
            executor=ToolExecutor(registry),
            max_turns=max_turns,
        ),
        max_turns,
    )


def _collect_cot_violations(obj: Any, path: str = "") -> list[str]:
    """Return dotted paths where hidden-reasoning field names appear."""
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


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    """Validate a single exported trace against the Agent Loop lab contract."""
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
        "provenance",
    }
    assert trace["metrics"]["provenance"] == "measured"

    termination_reason = trace["metrics"]["terminationReason"]
    assert termination_reason in SUPPORTED_TERMINATION_REASONS
    assert trace["output"]["terminationReason"] == termination_reason
    assert trace["state"]["terminationReason"] == termination_reason

    # Every trace must end with an explicit termination step/event.
    termination_steps = [s for s in trace["steps"] if s["type"] == "termination"]
    assert len(termination_steps) >= 1
    assert termination_steps[-1]["detail"]["reason"] == termination_reason
    termination_events = [e for e in trace["sequence"] if e["kind"] == "termination"]
    assert len(termination_events) >= 1
    assert termination_events[-1]["detail"]["reason"] == termination_reason

    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    assert trace["architecture"]["layout"] == "agent-loop"
    signature_phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert signature_phases[-1] == "TERMINATION"

    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})

    for step in trace["steps"]:
        if step["type"] == "llm":
            assert step["metrics"]["provenance"] == "case-harness"
        elif step["type"] in {"tool_call", "observation"}:
            assert step["metrics"]["provenance"] == "measured"

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []


def test_build_trace_separates_teaching_metadata():
    settings = Settings(openai_api_key="", data_dir=DATA)
    case, result, max_turns = _result("simple-loop-payments-docs")
    trace = build_trace(
        case=case, result=result, settings=settings, max_turns=max_turns
    )
    _assert_trace_contract(trace)
    assert trace["exampleClass"] == "SIMPLE_LOOP"
    assert trace["metrics"]["terminationReason"] == "final_answer"
    assert trace["metrics"]["maxTurns"] == max_turns
    assert "stageNotes" in trace["presentation"]
    assert "securityNote" in trace["presentation"]


def test_signature_view_shows_loop_and_termination():
    _, result, _ = _result("simple-loop-payments-docs")
    view = build_signature_view(result.sequence)
    phases = [v["phase"] for v in view]
    assert phases.count("MODEL_DECISION") == 3
    assert phases.count("TOOL_CALL") == 2
    assert phases.count("OBSERVATION") == 2
    assert phases.count("LOOP") == 2
    assert "FINAL_ANSWER" in phases
    assert phases[-1] == "TERMINATION"
    loop = next(v for v in view if v["phase"] == "LOOP")
    assert "observation" in loop["note"].lower() or "feeds" in loop["note"].lower()


def test_signature_view_max_turns_note():
    _, result, _ = _result("max-turns-safety-boundary")
    view = build_signature_view(result.sequence)
    term = next(v for v in view if v["phase"] == "TERMINATION")
    assert term["reason"] == "max_turns"
    assert "indefinitely" in term["note"].lower() or "limit" in term["note"].lower()


def test_invalid_action_trace_metadata():
    settings = Settings(openai_api_key="", data_dir=DATA)
    case, result, max_turns = _result("invalid-action-rejected")
    trace = build_trace(
        case=case, result=result, settings=settings, max_turns=max_turns
    )
    _assert_trace_contract(trace)
    assert trace["exampleClass"] == "INVALID_ACTION"
    assert trace["metrics"]["terminationReason"] == "invalid_action"
    assert "invalidActionNote" in trace["presentation"]
    assert "case harness" in trace["presentation"]["invalidActionNote"].lower() or (
        "unrecognized" in trace["presentation"]["invalidActionNote"].lower()
    )


def test_case_trace_alignment():
    """Each measured case maps to exactly one exported trace."""
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert len(traces) == len(CASES) == 4

    trace_ids = [t["traceId"] for t in traces]
    assert len(trace_ids) == len(set(trace_ids)), "trace IDs must be unique"

    case_by_trace = {c.trace_id: c for c in CASES}
    assert set(trace_ids) == set(case_by_trace)

    for trace in traces:
        case = case_by_trace[trace["traceId"]]
        assert trace["exampleClass"] == case.example_class

    classes = {t["exampleClass"] for t in traces}
    assert classes == EXPECTED_EXAMPLE_CLASSES


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    for trace in traces:
        _assert_trace_contract(trace)


def test_trace_termination_in_every_measured_trace():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        reason = trace["metrics"]["terminationReason"]
        assert reason in SUPPORTED_TERMINATION_REASONS
        assert trace["sequence"][-1]["kind"] in {"termination", "final_answer"}
        # Termination event must appear in sequence for all cases.
        term_events = [e for e in trace["sequence"] if e["kind"] == "termination"]
        assert term_events, f"{trace['traceId']} missing termination event"
        assert term_events[-1]["detail"]["reason"] == reason


def test_no_chain_of_thought_in_lab_traces():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        violations = _collect_cot_violations(trace)
        assert violations == [], (
            f"{trace['traceId']} contains hidden-reasoning fields: {violations}"
        )


def test_trace_provenance_model():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        assert trace["provenance"]["model"] == "case-harness"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert trace["input"]["config"]["modelDriver"] == "case-harness"
