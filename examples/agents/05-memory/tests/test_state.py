"""Agent state initialization and transition tests."""

from __future__ import annotations

from agent.memory import FixedClock, MemoryStore
from agent.state import RecordedDecision, initial_state

KNOWN = {"u-1001", "u-1002", "u-1003"}


def test_initial_state_setup():
    state = initial_state(scope="u-1001", max_turns=4)
    assert state.scope == "u-1001"
    assert state.current_turn == 0
    assert state.max_turns == 4
    assert state.termination_reason is None
    assert state.terminated is False
    assert state.final_answer is None
    assert state.interactions == []


def test_begin_interaction_resets_messages():
    state = initial_state(scope="u-1001", max_turns=4)
    state.begin_interaction("interaction-1", "Store email.", "system")
    state.messages.append({"role": "assistant", "content": "stored"})
    state.begin_interaction("interaction-2", "How should I be notified?", "system")
    assert state.current_interaction_id == "interaction-2"
    assert len(state.interactions) == 2
    assert state.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "How should I be notified?"},
    ]


def test_terminate_is_idempotent():
    state = initial_state(scope="u-1001", max_turns=3)
    state.begin_interaction("interaction-1", "x", "s")
    state.terminate("final_answer", answer="done")
    state.terminate("invalid_action", answer="should not overwrite")
    assert state.termination_reason == "final_answer"
    assert state.final_answer == "done"
    assert state.terminated is True


def test_public_dict_is_serializable_without_cot():
    state = initial_state(scope="u-1001", max_turns=2)
    state.begin_interaction("interaction-1", "r", "s")
    state.begin_turn()
    state.record_decision(
        RecordedDecision(
            turn=1,
            interaction_id="interaction-1",
            decision="final_answer",
            content="ok",
        )
    )
    store = MemoryStore(known_scopes=KNOWN, clock=FixedClock())
    record = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    state.record_write(record)
    state.terminate("final_answer", answer="ok")
    payload = state.to_public_dict()
    assert payload["currentTurn"] == 1
    assert payload["scope"] == "u-1001"
    assert payload["terminationReason"] == "final_answer"
    assert "chainOfThought" not in payload
    assert "reasoning" not in payload
    assert "messages" not in payload
    assert "thought" not in payload
    assert payload["memoryWrites"][0]["source"] == "user"
