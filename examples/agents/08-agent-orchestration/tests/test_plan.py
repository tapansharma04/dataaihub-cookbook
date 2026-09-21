"""Plan validation tests."""

from __future__ import annotations

import pytest

from agent.cases import get_case
from agent.plan import PlanValidationError, build_plan, validate_plan
from agent.schemas import NodeCondition, OrchestrationNode, OrchestrationPlan


def _plan(*nodes: OrchestrationNode) -> OrchestrationPlan:
    return OrchestrationPlan(plan_id="plan-test", nodes=list(nodes))


def test_measured_cases_have_valid_plans():
    for trace_id in (
        "payments-incident-basic-orchestration",
        "payments-status-dependency-chain",
        "payments-incident-conditional-branch",
        "unknown-service-orchestration-failure",
    ):
        plan = build_plan(get_case(trace_id))
        validate_plan(plan)
        assert plan.plan_id == f"plan-{trace_id}"


def test_basic_plan_dependencies():
    plan = build_plan(get_case("payments-incident-basic-orchestration"))
    by_id = {node.node_id: node for node in plan.nodes}
    assert by_id["status"].dependencies == []
    assert by_id["docs"].dependencies == []
    assert by_id["analysis"].dependencies == ["status", "docs"]
    assert by_id["decision"].dependencies == ["analysis"]


def test_duplicate_node_id_is_invalid():
    plan = _plan(
        OrchestrationNode(node_id="status", agent_id="status_agent"),
        OrchestrationNode(node_id="status", agent_id="docs_agent"),
    )
    with pytest.raises(PlanValidationError, match="duplicate node_id"):
        validate_plan(plan)


def test_unknown_dependency_is_invalid():
    plan = _plan(
        OrchestrationNode(
            node_id="analysis",
            agent_id="analysis_agent",
            dependencies=["status"],
        )
    )
    with pytest.raises(PlanValidationError, match="unknown node"):
        validate_plan(plan)


def test_self_dependency_is_invalid():
    plan = _plan(
        OrchestrationNode(
            node_id="status",
            agent_id="status_agent",
            dependencies=["status"],
        )
    )
    with pytest.raises(PlanValidationError, match="depends on itself"):
        validate_plan(plan)


def test_cycle_is_invalid():
    plan = _plan(
        OrchestrationNode(
            node_id="a",
            agent_id="status_agent",
            dependencies=["b"],
        ),
        OrchestrationNode(
            node_id="b",
            agent_id="docs_agent",
            dependencies=["a"],
        ),
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(plan)


def test_condition_source_must_be_a_dependency():
    plan = _plan(
        OrchestrationNode(node_id="status", agent_id="status_agent"),
        OrchestrationNode(
            node_id="remediation",
            agent_id="remediation_agent",
            condition=NodeCondition(
                source_node_id="status",
                field="incident",
                equals=True,
            ),
        ),
    )
    with pytest.raises(PlanValidationError, match="must also be a dependency"):
        validate_plan(plan)


def test_unknown_condition_source_is_invalid():
    plan = _plan(
        OrchestrationNode(node_id="status", agent_id="status_agent"),
        OrchestrationNode(
            node_id="remediation",
            agent_id="remediation_agent",
            dependencies=["status"],
            condition=NodeCondition(
                source_node_id="missing",
                field="incident",
                equals=True,
            ),
        ),
    )
    with pytest.raises(PlanValidationError, match="condition source"):
        validate_plan(plan)


def test_empty_plan_is_invalid():
    with pytest.raises(PlanValidationError, match="no nodes"):
        validate_plan(OrchestrationPlan(plan_id="plan-empty", nodes=[]))


def test_empty_agent_id_is_invalid():
    plan = _plan(OrchestrationNode(node_id="status", agent_id="  "))
    with pytest.raises(PlanValidationError, match="empty agent_id"):
        validate_plan(plan)
