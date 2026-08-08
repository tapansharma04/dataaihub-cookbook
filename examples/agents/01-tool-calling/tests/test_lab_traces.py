"""Trace schema and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import CASES, get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_tool_calling_loop
from agent.tools import build_registry
from agent.trace import build_signature_view, build_trace
from config import EXAMPLE_ID, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _result(trace_id: str):
    case = get_case(trace_id)
    registry = build_registry(DATA)
    return case, run_tool_calling_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=ToolExecutor(registry),
    )


def test_build_trace_separates_teaching_metadata():
    settings = Settings(openai_api_key="", data_dir=DATA)
    case, result = _result("single-tool-service-status")
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert trace["exampleClass"] == "SINGLE_TOOL"
    assert "teaching" not in trace["exampleClass"].lower()
    assert trace["metricsProvenance"] == "measured"
    assert trace["metrics"]["provenance"] == "measured"
    assert trace["provenance"] == {
        "model": "case-harness",
        "tools": "measured",
        "metrics": "measured",
    }
    assert "selectionNote" in trace
    assert "presentation" in trace
    # Measured sequence must not embed README prose fields.
    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})
    # Model turns are harness-driven; tool steps are measured.
    for step in trace["steps"]:
        if step["type"] == "llm":
            assert step["metrics"]["provenance"] == "case-harness"
        elif step["type"] in {"tool_call", "observation"}:
            assert step["metrics"]["provenance"] == "measured"


def test_signature_view_direct_answer_marks_skipped_tool():
    _, result = _result("direct-answer")
    view = build_signature_view(result.sequence)
    phases = [v["phase"] for v in view]
    assert "THINK_DECIDE" in phases
    assert "FINAL_ANSWER" in phases
    skipped = next(v for v in view if v["phase"] == "TOOL_CALL")
    assert skipped.get("skipped") is True


def test_signature_view_multi_step_explicit_sequence():
    _, result = _result("multi-step-user-and-payments")
    view = build_signature_view(result.sequence)
    phases = [v["phase"] for v in view]
    assert phases.count("TOOL_CALL") == 2
    assert phases.count("OBSERVATION") == 2
    assert "NEXT_ACTION" in phases
    assert phases[-1] == "FINAL_ANSWER" or "FINAL_ANSWER" in phases


def test_committed_lab_traces_schema_if_present():
    path = ROOT / "lab_traces.json"
    if not path.exists():
        return
    traces = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES)
    ids = {t["traceId"] for t in traces}
    assert ids == {c.trace_id for c in CASES}
    for trace in traces:
        assert trace["labId"] == EXAMPLE_ID
        assert trace["metricsProvenance"] == "measured"
        assert trace["provenance"]["model"] == "case-harness"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert "exampleClass" in trace
        assert "tools" in trace and trace["tools"]
        assert "sequence" in trace and trace["sequence"]
        assert "steps" in trace and len(trace["steps"]) >= 2
        assert "metrics" in trace
        assert set(trace["metrics"]) >= {
            "totalMs",
            "modelMs",
            "toolMs",
            "modelTurns",
            "toolCalls",
            "successfulToolCalls",
            "failedToolCalls",
            "provenance",
        }
        assert "presentation" in trace
        assert "signatureView" in trace["presentation"]
        # No fabricated CoT field.
        for event in trace["sequence"]:
            assert "chainOfThought" not in event.get("detail", {})
            assert "reasoning" not in event.get("detail", {})
    classes = {t["exampleClass"] for t in traces}
    assert classes == {
        "DIRECT_ANSWER",
        "SINGLE_TOOL",
        "MULTI_STEP",
        "ERROR_RECOVERY",
    }


def test_error_recovery_trace_preserves_failure():
    settings = Settings(openai_api_key="", data_dir=DATA)
    case, result = _result("recovery-invalid-service-name")
    trace = build_trace(case=case, result=result, settings=settings)
    assert case.example_class == "ERROR_RECOVERY"
    assert trace["exampleClass"] == "ERROR_RECOVERY"
    assert trace["metrics"]["failedToolCalls"] == 1
    assert trace["metrics"]["successfulToolCalls"] == 1
    obs = [e for e in trace["sequence"] if e["kind"] == "observation"]
    assert obs[0]["detail"]["ok"] is False
    assert obs[1]["detail"]["ok"] is True
    note = trace["presentation"]["errorRecoveryNote"]
    assert "case harness" in note.lower()
    assert "error" in note.lower()
