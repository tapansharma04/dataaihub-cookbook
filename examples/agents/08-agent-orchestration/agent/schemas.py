"""Typed schemas for application-owned agent orchestration.

The orchestrator owns a workflow plan and decides WHEN each node may
execute from dependencies, readiness, conditions, and node state.
It is not coordinator delegation and not ownership transfer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TerminationReason = Literal[
    "completed",
    "workflow_failed",
    "invalid_task",
    "invalid_plan",
    "error",
]
AgentRole = Literal[
    "status",
    "docs",
    "analysis",
    "decision",
    "remediation",
    "no_action",
]
NodeState = Literal[
    "PENDING",
    "READY",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED",
]
SkipReason = Literal[
    "condition_not_selected",
    "dependency_failed",
    "dependency_skipped",
]
ReadyReason = Literal["no_dependencies", "dependencies_completed"]


class NodeCondition(BaseModel):
    """Predicate evaluated against an upstream node's actual result."""

    source_node_id: str
    field: str
    equals: Any


class OrchestrationNode(BaseModel):
    """One workflow node. Dependencies and conditions are explicit."""

    node_id: str
    agent_id: str
    dependencies: list[str] = Field(default_factory=list)
    condition: NodeCondition | None = None
    assignment: dict[str, Any] = Field(default_factory=dict)


class OrchestrationPlan(BaseModel):
    """Inspectable workflow artifact owned by the runtime."""

    plan_id: str
    nodes: list[OrchestrationNode] = Field(default_factory=list)

    def node(self, node_id: str) -> OrchestrationNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(f"Unknown node '{node_id}'")

    def node_ids(self) -> list[str]:
        return [item.node_id for item in self.nodes]


class AgentResult(BaseModel):
    """Observable output of one node execution. Agents do not write workflow state."""

    result_id: str
    node_id: str
    agent_id: str
    role: AgentRole
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    upstream_result_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _failure_requires_error(self) -> AgentResult:
        if not self.ok and self.error is None:
            raise ValueError("ok=False requires an error object")
        if self.ok and self.error is not None:
            raise ValueError("ok=True cannot include an error object")
        return self


class NodeContext(BaseModel):
    """Inbound work for one node.

    Downstream agents must consume upstream_results. They do not reload
    another agent's catalog or private state.
    """

    task_id: str
    node_id: str
    agent_id: str
    assignment: dict[str, Any] = Field(default_factory=dict)
    upstream_results: dict[str, AgentResult] = Field(default_factory=dict)
    upstream_result_ids: list[str] = Field(default_factory=list)


class NodeTransition(BaseModel):
    from_state: NodeState
    to_state: NodeState
    reason: str | None = None


class SequenceEvent(BaseModel):
    """Ordered observable orchestration event. No hidden reasoning."""

    kind: Literal[
        "task_created",
        "plan_created",
        "node_ready",
        "node_started",
        "node_result",
        "node_completed",
        "node_failed",
        "node_skipped",
        "condition_evaluated",
        "completed",
        "termination",
        "error",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class OrchestrationMetrics(BaseModel):
    total_ms: int
    runtime_ms: int
    agent_ms: int
    nodes_ready: int
    nodes_executed: int
    nodes_completed: int
    nodes_failed: int
    nodes_skipped: int
    termination_reason: TerminationReason
    max_nodes: int
    provenance: Literal["measured"] = "measured"


class OrchestrationRunResult(BaseModel):
    case_id: str
    example_class: str
    request: str
    answer: str
    ok: bool
    model: str
    model_driver: str
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: OrchestrationMetrics
    state: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
