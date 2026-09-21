"""Node state machine tests."""

from __future__ import annotations

import pytest

from agent.state import ALLOWED_TRANSITIONS, InvalidTransitionError, NodeRecord


def test_valid_success_path():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    assert record.state == "PENDING"
    record.transition("READY", reason="no_dependencies")
    record.transition("RUNNING", reason="scheduled")
    record.transition("COMPLETED", reason="agent_ok")
    assert record.state == "COMPLETED"
    assert [item.to_state for item in record.transitions] == [
        "READY",
        "RUNNING",
        "COMPLETED",
    ]


def test_valid_failure_path():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    record.transition("READY")
    record.transition("RUNNING")
    record.transition("FAILED", reason="agent_failed")
    assert record.state == "FAILED"


def test_pending_may_skip():
    record = NodeRecord(node_id="decision", agent_id="decision_agent")
    record.transition("SKIPPED", reason="dependency_failed")
    assert record.state == "SKIPPED"


def test_pending_cannot_run():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    with pytest.raises(InvalidTransitionError, match="PENDING → RUNNING"):
        record.transition("RUNNING")


def test_pending_cannot_complete():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    with pytest.raises(InvalidTransitionError, match="PENDING → COMPLETED"):
        record.transition("COMPLETED")


def test_completed_cannot_run_again():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    record.transition("READY")
    record.transition("RUNNING")
    record.transition("COMPLETED")
    with pytest.raises(InvalidTransitionError, match="COMPLETED → RUNNING"):
        record.transition("RUNNING")


def test_skipped_cannot_execute():
    record = NodeRecord(node_id="decision", agent_id="decision_agent")
    record.transition("SKIPPED", reason="dependency_failed")
    with pytest.raises(InvalidTransitionError, match="SKIPPED → RUNNING"):
        record.transition("RUNNING")
    with pytest.raises(InvalidTransitionError, match="SKIPPED → READY"):
        record.transition("READY")


def test_failed_cannot_become_completed():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    record.transition("READY")
    record.transition("RUNNING")
    record.transition("FAILED")
    with pytest.raises(InvalidTransitionError, match="FAILED → COMPLETED"):
        record.transition("COMPLETED")


def test_ready_cannot_skip():
    record = NodeRecord(node_id="status", agent_id="status_agent")
    record.transition("READY")
    with pytest.raises(InvalidTransitionError, match="READY → SKIPPED"):
        record.transition("SKIPPED")


def test_allowed_transition_table_is_explicit():
    assert ALLOWED_TRANSITIONS["PENDING"] == frozenset({"READY", "SKIPPED"})
    assert ALLOWED_TRANSITIONS["READY"] == frozenset({"RUNNING"})
    assert ALLOWED_TRANSITIONS["RUNNING"] == frozenset({"COMPLETED", "FAILED"})
    assert ALLOWED_TRANSITIONS["COMPLETED"] == frozenset()
    assert ALLOWED_TRANSITIONS["FAILED"] == frozenset()
    assert ALLOWED_TRANSITIONS["SKIPPED"] == frozenset()
