"""Agent state initialization and transition tests."""

from __future__ import annotations

from agent.state import AgentState, RecordedDecision, RecordedObservation, initial_state


def test_initial_state_setup():
    state = initial_state(
        request="Check payments",
        max_turns=4,
        system_prompt="You are a helper.",
    )
    assert state.request == "Check payments"
    assert state.current_turn == 0
    assert state.max_turns == 4
    assert state.termination_reason is None
    assert state.terminated is False
    assert state.final_answer is None
    assert len(state.messages) == 2
    assert state.messages[0]["role"] == "system"
    assert state.messages[1]["role"] == "user"


def test_begin_turn_increments():
    state = AgentState(request="x", max_turns=3)
    assert state.begin_turn() == 1
    assert state.begin_turn() == 2
    assert state.current_turn == 2


def test_record_decision_and_observation():
    state = AgentState(request="x", max_turns=3)
    state.begin_turn()
    state.record_decision(RecordedDecision(turn=1, decision="tool_call", tool_calls=[]))
    state.record_observation(
        RecordedObservation(
            turn=1,
            call_id="c1",
            name="get_service_status",
            ok=True,
            result={"ok": True},
        )
    )
    assert len(state.decisions) == 1
    assert len(state.observations) == 1
    assert state.observations[0].ok is True


def test_terminate_is_idempotent():
    state = AgentState(request="x", max_turns=3)
    state.terminate("final_answer", answer="done")
    state.terminate("max_turns", answer="should not overwrite")
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
    assert payload["finalAnswer"] == "ok"
    assert "chainOfThought" not in payload
    assert "reasoning" not in payload
    assert "messages" not in payload  # conversation kept out of public dump
