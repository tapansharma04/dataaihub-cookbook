"""Agent state initialization and transition tests."""

from __future__ import annotations

from pathlib import Path

from agent.state import RecordedDecision, initial_state
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STEPS = [
    {
        "id": "step-1",
        "description": "Check billing",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "billing"},
    },
    {
        "id": "step-2",
        "description": "Summarize",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]


def test_initial_state_setup():
    state = initial_state(
        request="Check billing",
        max_turns=4,
        system_prompt="You are a helper.",
    )
    assert state.request == "Check billing"
    assert state.current_turn == 0
    assert state.max_turns == 4
    assert state.termination_reason is None
    assert state.terminated is False
    assert state.plan is None
    assert state.final_answer is None
    assert len(state.messages) == 2


def test_install_plan_is_application_owned():
    state = initial_state(request="r", max_turns=4, system_prompt="s")
    plan = state.install_plan(STEPS, registry=build_registry(DATA))
    assert plan.status == "pending"
    assert all(step.status == "pending" for step in plan.steps)
    started = state.start_current_step()
    assert started.status == "in_progress"


def test_terminate_is_idempotent():
    state = initial_state(request="x", max_turns=3, system_prompt="s")
    state.terminate("final_answer", answer="done")
    state.terminate("plan_failed", answer="should not overwrite")
    assert state.termination_reason == "final_answer"
    assert state.final_answer == "done"
    assert state.terminated is True


def test_public_dict_is_serializable_without_cot():
    state = initial_state(request="r", max_turns=2, system_prompt="s")
    state.begin_turn()
    state.record_decision(
        RecordedDecision(turn=1, decision="final_answer", content="ok")
    )
    state.terminate("final_answer", answer="ok")
    payload = state.to_public_dict()
    assert payload["currentTurn"] == 1
    assert payload["maxTurns"] == 2
    assert payload["terminationReason"] == "final_answer"
    assert "chainOfThought" not in payload
    assert "reasoning" not in payload
    assert "messages" not in payload
    assert "thought" not in payload
