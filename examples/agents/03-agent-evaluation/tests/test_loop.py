"""Loop sequencing tests for measured evaluation cases — no paid APIs."""

from __future__ import annotations

from pathlib import Path

from agent.cases import get_case
from agent.run import run_measured_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _run(trace_id: str):
    return run_measured_case(get_case(trace_id), data_dir=DATA)


def test_task_success_sequence():
    result = _run("task-success-payments-docs")
    kinds = [e.kind for e in result.sequence]
    assert kinds == [
        "user_request",
        "model_decision",
        "tool_call",
        "observation",
        "model_decision",
        "tool_call",
        "observation",
        "model_decision",
        "final_answer",
        "termination",
    ]
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == ["get_service_status", "search_documentation"]
    assert result.metrics.termination_reason == "final_answer"
    assert "degraded" in result.answer.lower()


def test_partial_success_has_extra_tool_call():
    result = _run("partial-success-extra-profile")
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == [
        "get_service_status",
        "search_documentation",
        "get_user_profile",
    ]
    assert result.metrics.tool_calls == 3
    assert result.metrics.successful_tool_calls == 3
    assert result.metrics.termination_reason == "final_answer"


def test_recovery_sequence_preserves_error_then_success():
    result = _run("tool-error-recovery-payments")
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == [
        "get_service_status",
        "get_service_status",
        "search_documentation",
    ]
    args = [e.detail["arguments"] for e in result.sequence if e.kind == "tool_call"]
    assert args[0]["service"] == "payments-api"
    assert args[1]["service"] == "payments"
    observations = [e for e in result.sequence if e.kind == "observation"]
    assert observations[0].detail["ok"] is False
    assert observations[1].detail["ok"] is True
    assert observations[2].detail["ok"] is True
    assert result.metrics.failed_tool_calls == 1
    assert result.metrics.termination_reason == "final_answer"


def test_goal_miss_terminates_normally_with_an_answer():
    result = _run("goal-miss-wrong-answer")
    assert result.metrics.termination_reason == "final_answer"
    assert result.sequence[-1].kind == "termination"
    assert result.sequence[-1].detail["reason"] == "final_answer"
    assert "operational" in result.answer.lower()
    assert "pay-2041" not in result.answer.lower()
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == ["get_service_status", "search_documentation"]
