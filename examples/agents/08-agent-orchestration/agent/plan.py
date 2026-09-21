"""Orchestration plan construction and validation.

The plan is a runtime artifact: nodes, dependencies, and conditions are
inspectable before any agent executes.
"""

from __future__ import annotations

from agent.cases import MeasuredCase
from agent.schemas import NodeCondition, OrchestrationNode, OrchestrationPlan


class PlanValidationError(ValueError):
    """Raised when a workflow plan is structurally invalid."""


def validate_plan(plan: OrchestrationPlan) -> None:
    if not plan.plan_id.strip():
        raise PlanValidationError("plan_id is empty")
    ids = plan.node_ids()
    if not ids:
        raise PlanValidationError("plan has no nodes")
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        raise PlanValidationError(f"duplicate node_id: {', '.join(duplicates)}")
    by_id = {node.node_id: node for node in plan.nodes}
    for node in plan.nodes:
        if not node.node_id.strip():
            raise PlanValidationError("node_id is empty")
        if not node.agent_id.strip():
            raise PlanValidationError(f"node '{node.node_id}' has an empty agent_id")
        for dep in node.dependencies:
            if dep == node.node_id:
                raise PlanValidationError(f"node '{node.node_id}' depends on itself")
            if dep not in by_id:
                raise PlanValidationError(
                    f"node '{node.node_id}' depends on unknown node '{dep}'"
                )
        if node.condition is not None:
            source = node.condition.source_node_id
            if source not in by_id:
                raise PlanValidationError(
                    f"node '{node.node_id}' condition source '{source}' is unknown"
                )
            if source not in node.dependencies:
                raise PlanValidationError(
                    f"node '{node.node_id}' condition source '{source}' "
                    "must also be a dependency"
                )
            if not node.condition.field.strip():
                raise PlanValidationError(
                    f"node '{node.node_id}' condition field is empty"
                )
    _assert_acyclic(by_id)


def _assert_acyclic(by_id: dict[str, OrchestrationNode]) -> None:
    white, gray, black = 0, 1, 2
    color = {node_id: white for node_id in by_id}

    def visit(node_id: str) -> None:
        color[node_id] = gray
        for dep in by_id[node_id].dependencies:
            if color[dep] == gray:
                raise PlanValidationError(
                    f"cycle detected involving '{node_id}' and '{dep}'"
                )
            if color[dep] == white:
                visit(dep)
        color[node_id] = black

    for node_id in by_id:
        if color[node_id] == white:
            visit(node_id)


def build_plan(case: MeasuredCase) -> OrchestrationPlan:
    nodes = [
        OrchestrationNode(
            node_id=spec.node_id,
            agent_id=spec.agent_id,
            dependencies=list(spec.dependencies),
            condition=(
                NodeCondition(
                    source_node_id=spec.condition.source_node_id,
                    field=spec.condition.field,
                    equals=spec.condition.equals,
                )
                if spec.condition is not None
                else None
            ),
            assignment=dict(spec.assignment),
        )
        for spec in case.nodes
    ]
    plan = OrchestrationPlan(plan_id=f"plan-{case.trace_id}", nodes=nodes)
    validate_plan(plan)
    return plan
