"""Loop sequencing tests — scripted model, real tools, no paid APIs."""

from __future__ import annotations

from pathlib import Path

from agent.cases import get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_tool_calling_loop
from agent.model import ScriptedModelClient, ScriptedTurn
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _run(case_id: str):
    case = get_case(case_id)
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    return run_tool_calling_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
    )


def test_direct_answer_has_no_tool_calls():
    result = _run("direct-answer")
    assert result.metrics.tool_calls == 0
    assert result.metrics.model_turns == 1
    assert result.answer
    kinds = [e.kind for e in result.sequence]
    assert kinds == ["user_request", "model_turn", "final_answer"]
    assert result.sequence[1].detail["decision"] == "final_answer"


def test_single_tool_sequence():
    result = _run("single-tool-service-status")
    assert result.metrics.tool_calls == 1
    assert result.metrics.successful_tool_calls == 1
    assert result.metrics.failed_tool_calls == 0
    assert result.metrics.model_turns == 2
    kinds = [e.kind for e in result.sequence]
    assert kinds == [
        "user_request",
        "model_turn",
        "tool_call",
        "observation",
        "model_turn",
        "final_answer",
    ]
    assert "operational" in result.answer.lower() or "billing" in result.answer.lower()


def test_multi_step_sequence():
    result = _run("multi-step-user-and-payments")
    assert result.metrics.tool_calls == 2
    assert result.metrics.successful_tool_calls == 2
    assert result.metrics.model_turns == 3
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == ["get_user_profile", "get_service_status"]
    assert "enterprise" in result.answer.lower()
    assert "degraded" in result.answer.lower() or "pay-2041" in result.answer.lower()


def test_recovery_shows_failure_then_success():
    result = _run("recovery-invalid-service-name")
    assert result.metrics.tool_calls == 2
    assert result.metrics.failed_tool_calls == 1
    assert result.metrics.successful_tool_calls == 1
    observations = [e for e in result.sequence if e.kind == "observation"]
    assert observations[0].detail["ok"] is False
    assert observations[0].detail["error"]["code"] == "unknown_service"
    assert observations[1].detail["ok"] is True
    assert "billing" in result.answer.lower()


def test_metrics_are_measured_non_negative():
    result = _run("single-tool-service-status")
    m = result.metrics
    assert m.provenance == "measured"
    assert m.total_ms >= 0
    assert m.model_ms >= 0
    assert m.tool_ms >= 0


def test_unauthorized_tool_surfaces_in_observation():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry, caller_roles={"viewer"})
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "get_user_profile",
                        "arguments": {"user_id": "u-1001"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content="I am not authorized to read that profile.",
                finish_reason="stop",
            ),
        ]
    )
    result = run_tool_calling_loop(
        request="Show profile for u-1001",
        model=model,
        registry=registry,
        executor=executor,
    )
    assert result.metrics.failed_tool_calls == 1
    obs = next(e for e in result.sequence if e.kind == "observation")
    assert obs.detail["error"]["code"] == "unauthorized"
