"""Orchestration schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import AgentResult, NodeCondition, OrchestrationNode


def test_failure_result_requires_error():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            node_id="status",
            agent_id="status_agent",
            role="status",
            ok=False,
        )


def test_success_result_cannot_include_error():
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result-1",
            node_id="status",
            agent_id="status_agent",
            role="status",
            ok=True,
            error={"code": "nope"},
        )


def test_valid_success_and_failure_results():
    ok = AgentResult(
        result_id="result-1",
        node_id="status",
        agent_id="status_agent",
        role="status",
        ok=True,
        payload={"service": {"service": "payments"}},
        upstream_result_ids=[],
    )
    failed = AgentResult(
        result_id="result-2",
        node_id="status",
        agent_id="status_agent",
        role="status",
        ok=False,
        error={"code": "unknown_service"},
    )
    assert ok.error is None
    assert failed.ok is False


def test_node_condition_is_explicit():
    condition = NodeCondition(source_node_id="analysis", field="incident", equals=True)
    node = OrchestrationNode(
        node_id="remediation",
        agent_id="remediation_agent",
        dependencies=["analysis"],
        condition=condition,
    )
    assert node.condition is not None
    assert node.condition.equals is True
