"""Explicit node state machine and application-owned orchestration state.

The runtime is the only writer. Node states are not collapsed into a
boolean. Invalid transitions are rejected.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import (
    AgentResult,
    NodeState,
    NodeTransition,
    OrchestrationPlan,
    ReadyReason,
    SkipReason,
    TerminationReason,
)

ALLOWED_TRANSITIONS: dict[NodeState, frozenset[NodeState]] = {
    "PENDING": frozenset({"READY", "SKIPPED"}),
    "READY": frozenset({"RUNNING"}),
    "RUNNING": frozenset({"COMPLETED", "FAILED"}),
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
    "SKIPPED": frozenset(),
}


class InvalidTransitionError(ValueError):
    def __init__(
        self,
        node_id: str,
        from_state: NodeState,
        to_state: NodeState,
    ) -> None:
        self.node_id = node_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition for '{node_id}': {from_state} → {to_state}"
        )


class NodeRecord(BaseModel):
    """Runtime record for one plan node, including state history."""

    node_id: str
    agent_id: str
    state: NodeState = "PENDING"
    dependency_ids: list[str] = Field(default_factory=list)
    upstream_result_ids: list[str] = Field(default_factory=list)
    result: AgentResult | None = None
    skip_reason: SkipReason | None = None
    ready_reason: ReadyReason | None = None
    condition_matched: bool | None = None
    transitions: list[NodeTransition] = Field(default_factory=list)

    def transition(self, to_state: NodeState, *, reason: str | None = None) -> None:
        if to_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransitionError(self.node_id, self.state, to_state)
        previous = self.state
        self.state = to_state
        self.transitions.append(
            NodeTransition(from_state=previous, to_state=to_state, reason=reason)
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "nodeId": self.node_id,
            "agentId": self.agent_id,
            "state": self.state,
            "dependencyIds": list(self.dependency_ids),
            "upstreamResultIds": list(self.upstream_result_ids),
            "result": (
                self.result.model_dump(mode="json") if self.result is not None else None
            ),
            "skipReason": self.skip_reason,
            "readyReason": self.ready_reason,
            "conditionMatched": self.condition_matched,
            "transitions": [item.model_dump(mode="json") for item in self.transitions],
        }


class OrchestrationState(BaseModel):
    task_id: str
    request: str
    service: str | None = None
    plan: OrchestrationPlan | None = None
    nodes: dict[str, NodeRecord] = Field(default_factory=dict)
    results: list[AgentResult] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not None

    def record(self, node_id: str) -> NodeRecord:
        return self.nodes[node_id]

    def result_for_node(self, node_id: str) -> AgentResult | None:
        record = self.nodes.get(node_id)
        if record is None:
            return None
        return record.result

    def ready_node_ids(self) -> list[str]:
        if self.plan is None:
            return []
        return [
            node.node_id
            for node in self.plan.nodes
            if self.nodes[node.node_id].state == "READY"
        ]

    def blocked_nodes(self) -> list[dict[str, Any]]:
        if self.plan is None:
            return []
        blocked: list[dict[str, Any]] = []
        for node in self.plan.nodes:
            record = self.nodes[node.node_id]
            if record.state != "PENDING":
                continue
            waiting_on = [
                dep
                for dep in node.dependencies
                if self.nodes[dep].state not in {"COMPLETED", "FAILED", "SKIPPED"}
            ]
            blocked.append({"nodeId": node.node_id, "waitingOn": waiting_on})
        return blocked

    def initialize_plan(self, plan: OrchestrationPlan) -> None:
        self.plan = plan
        self.nodes = {
            node.node_id: NodeRecord(
                node_id=node.node_id,
                agent_id=node.agent_id,
                dependency_ids=list(node.dependencies),
            )
            for node in plan.nodes
        }

    def record_result(
        self,
        result: AgentResult,
        *,
        bound_node_id: str | None = None,
    ) -> None:
        bound = bound_node_id or result.node_id
        self.results.append(result)
        record = self.nodes.get(bound)
        if record is not None:
            record.result = result
            record.upstream_result_ids = list(result.upstream_result_ids)
        self.execution_order.append(bound)

    def node_ids_in(self, *states: NodeState) -> list[str]:
        return [
            node_id for node_id, record in self.nodes.items() if record.state in states
        ]

    def terminate(
        self,
        reason: TerminationReason,
        *,
        answer: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if self.termination_reason is not None:
            return
        self.termination_reason = reason
        if answer is not None:
            self.final_answer = answer
        if error is not None:
            self.errors.append(error)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "request": self.request,
            "service": self.service,
            "plan": self.plan.model_dump(mode="json")
            if self.plan is not None
            else None,
            "nodes": {
                node_id: record.to_public_dict()
                for node_id, record in self.nodes.items()
            },
            "nodeStates": {
                node_id: record.state for node_id, record in self.nodes.items()
            },
            "results": [item.model_dump(mode="json") for item in self.results],
            "executionOrder": list(self.execution_order),
            "invokedNodeIds": list(self.execution_order),
            "invokedAgentIds": [item.agent_id for item in self.results],
            "skippedNodeIds": [
                node_id
                for node_id, record in self.nodes.items()
                if record.state == "SKIPPED"
            ],
            "blockedNodes": self.blocked_nodes(),
            "finalAnswer": self.final_answer,
            "terminationReason": self.termination_reason,
            "errors": list(self.errors),
        }
