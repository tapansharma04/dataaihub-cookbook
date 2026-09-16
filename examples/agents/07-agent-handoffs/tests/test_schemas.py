"""Handoff contract and schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import AgentResult, HandoffRejection, HandoffRequest


def test_handoff_request_requires_source_target_and_reason():
    request = HandoffRequest(
        from_agent="triage_agent",
        to_agent="billing_agent",
        reason="Classified as billing.",
        payload={"invoice_id": "INV-1001"},
    )
    assert request.from_agent == "triage_agent"
    assert request.to_agent == "billing_agent"
    assert request.reason
    assert request.payload["invoice_id"] == "INV-1001"


def test_handoff_result_requires_handoff_object():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            agent_id="triage_agent",
            role="triage",
            kind="handoff",
            ok=True,
        )


def test_complete_result_cannot_include_handoff():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            agent_id="billing_agent",
            role="billing",
            kind="complete",
            ok=True,
            handoff=HandoffRequest(
                from_agent="billing_agent",
                to_agent="technical_agent",
                reason="should not be here",
            ),
        )


def test_failure_result_requires_ok_false():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            agent_id="technical_agent",
            role="technical",
            kind="failure",
            ok=True,
            error={"code": "unknown_system"},
        )


def test_complete_result_requires_ok_true():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            agent_id="billing_agent",
            role="billing",
            kind="complete",
            ok=False,
            payload={"summary": "nope"},
        )


def test_valid_complete_and_failure_results():
    complete = AgentResult(
        result_id="result-1",
        agent_id="billing_agent",
        role="billing",
        kind="complete",
        ok=True,
        payload={"summary": "refund"},
    )
    failure = AgentResult(
        result_id="result-2",
        agent_id="technical_agent",
        role="technical",
        kind="failure",
        ok=False,
        error={"code": "unknown_system"},
    )
    assert complete.handoff is None
    assert failure.ok is False


def test_handoff_rejection_preserves_provenance():
    rejection = HandoffRejection(
        handoff_id="handoff-1",
        from_agent="triage_agent",
        to_agent="legal_agent",
        code="unknown_target",
        message="Unknown handoff target 'legal_agent'.",
        parent_result_id="result-1",
    )
    assert rejection.from_agent == "triage_agent"
    assert rejection.to_agent == "legal_agent"
    assert rejection.parent_result_id == "result-1"


def test_max_handoffs_is_a_rejection_code():
    rejection = HandoffRejection(
        handoff_id="handoff-2",
        from_agent="billing_agent",
        to_agent="technical_agent",
        code="max_handoffs",
        message="Handoff budget exhausted.",
        parent_result_id="result-2",
    )
    assert rejection.code == "max_handoffs"
