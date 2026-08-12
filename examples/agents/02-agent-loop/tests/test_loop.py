"""Loop sequencing tests — scripted model, real tools, no paid APIs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.cases import get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import classify_decision, run_agent_loop
from agent.model import ScriptedModelClient, ScriptedTurn
from agent.schemas import ModelTurn
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _run(case_id: str):
    case = get_case(case_id)
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    max_turns = case.max_turns if case.max_turns is not None else 6
    return run_agent_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
        max_turns=max_turns,
    )


def test_simple_loop_multiple_iterations():
    result = _run("simple-loop-payments-docs")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.model_turns == 3
    assert result.metrics.tool_calls == 2
    assert result.metrics.successful_tool_calls == 2
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
    assert result.state["terminationReason"] == "final_answer"
    assert len(result.state["observations"]) == 2
    assert "degraded" in result.answer.lower() or "pay-2041" in result.answer.lower()


def test_termination_after_first_useful_result():
    result = _run("termination-after-status")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.model_turns == 2
    assert result.metrics.tool_calls == 1
    kinds = [e.kind for e in result.sequence]
    assert kinds == [
        "user_request",
        "model_decision",
        "tool_call",
        "observation",
        "model_decision",
        "final_answer",
        "termination",
    ]
    term = next(e for e in result.sequence if e.kind == "termination")
    assert term.detail["reason"] == "final_answer"


def test_max_turns_safety_boundary():
    case = get_case("max-turns-safety-boundary")
    # Harness would propose a fourth tool turn; runtime must stop at max_turns=3.
    assert case.max_turns == 3
    assert len(case.turns) > case.max_turns

    result = _run("max-turns-safety-boundary")
    assert result.metrics.termination_reason == "max_turns"
    assert result.metrics.max_turns == 3
    assert result.metrics.model_turns == 3
    assert result.metrics.tool_calls == 3
    assert result.state["currentTurn"] == 3
    assert result.state["terminationReason"] == "max_turns"
    term = next(e for e in result.sequence if e.kind == "termination")
    assert term.detail["reason"] == "max_turns"
    assert term.detail["maxTurns"] == 3
    assert term.detail["currentTurn"] == 3
    # Fourth harness turn must not execute — no tool from the unused scripted turn.
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == [
        "get_service_status",
        "search_documentation",
        "get_service_status",
    ]
    assert "auth" not in names
    assert result.sequence[-1].kind == "termination"
    # Runtime stops before a fourth model decision / tool execution cycle.
    model_decisions = [e for e in result.sequence if e.kind == "model_decision"]
    assert len(model_decisions) == 3
    assert all(e.detail["decision"] == "tool_call" for e in model_decisions)
    assert "max_turns" in result.answer.lower() or "turn limit" in result.answer.lower()


def test_max_turns_does_not_execute_after_limit():
    """Prove ToolExecutor is never invoked once the turn budget is exhausted."""
    case = get_case("max-turns-safety-boundary")
    registry = build_registry(DATA)
    real_executor = ToolExecutor(registry)
    with patch.object(real_executor, "execute", wraps=real_executor.execute) as spy:
        run_agent_loop(
            request=case.request,
            model=scripted_client_for(case),
            registry=registry,
            executor=real_executor,
            max_turns=case.max_turns or 3,
        )
        assert spy.call_count == 3


def test_invalid_action_terminates_safely():
    case = get_case("invalid-action-rejected")
    # Harness emits an unsupported action kind ("continue"), not tool_call/final_answer.
    assert case.turns[0].decision == "continue"

    result = _run("invalid-action-rejected")
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.tool_calls == 0
    assert result.metrics.model_turns == 1
    assert result.state["terminationReason"] == "invalid_action"
    assert len(result.state["toolCalls"]) == 0
    assert len(result.state["observations"]) == 0
    kinds = [e.kind for e in result.sequence]
    assert kinds == [
        "user_request",
        "model_decision",
        "error",
        "termination",
    ]
    assert "tool_call" not in kinds
    assert "observation" not in kinds
    decision_event = result.sequence[1]
    assert decision_event.detail["decision"] == "invalid_action"
    error_event = next(e for e in result.sequence if e.kind == "error")
    assert error_event.detail["code"] == "invalid_action"
    term = next(e for e in result.sequence if e.kind == "termination")
    assert term.detail["reason"] == "invalid_action"


def test_invalid_action_never_reaches_executor():
    """Unsupported actions terminate at the runtime boundary, not in ToolExecutor."""
    case = get_case("invalid-action-rejected")
    registry = build_registry(DATA)
    executor = MagicMock(spec=ToolExecutor)
    result = run_agent_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
        max_turns=6,
    )
    executor.execute.assert_not_called()
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.tool_calls == 0


def test_turn_counting_in_state():
    result = _run("simple-loop-payments-docs")
    assert result.state["currentTurn"] == 3
    assert len(result.state["decisions"]) == 3
    assert result.metrics.model_turns == result.state["currentTurn"]


def test_metrics_are_measured_non_negative():
    result = _run("simple-loop-payments-docs")
    m = result.metrics
    assert m.provenance == "measured"
    assert m.total_ms >= 0
    assert m.model_ms >= 0
    assert m.tool_ms >= 0
    assert m.max_turns >= m.model_turns or m.termination_reason == "final_answer"


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
    result = run_agent_loop(
        request="Show profile for u-1001",
        model=model,
        registry=registry,
        executor=executor,
    )
    assert result.metrics.failed_tool_calls == 1
    obs = next(e for e in result.sequence if e.kind == "observation")
    assert obs.detail["error"]["code"] == "unauthorized"
    assert result.metrics.termination_reason == "final_answer"


def test_classify_decision_invalid_without_tools():
    turn = ModelTurn(content="keep going", tool_calls=[], decision="invalid_action")
    assert classify_decision(turn) == "invalid_action"
