"""Typed schemas for multi-agent collaboration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TerminationReason = Literal[
    "aggregated",
    "agent_failed",
    "no_further_work",
    "max_delegations",
    "invalid_task",
    "error",
]
AgentRole = Literal["status", "docs", "analysis"]
StepKind = Literal["delegate", "delegate_if_incident", "aggregate"]


class AgentMessage(BaseModel):
    """Explicit message from the coordinator to one specialist."""

    message_id: str
    from_agent: str = "coordinator"
    to_agent: str
    task_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_result_id: str | None = None


class AgentResult(BaseModel):
    """Observable specialist output. Agents do not write TaskState."""

    result_id: str
    agent_id: str
    role: AgentRole
    ok: bool
    in_reply_to: str
    payload: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class Delegation(BaseModel):
    delegation_id: str
    agent_id: str
    role: AgentRole | None = None
    assignment: dict[str, Any] = Field(default_factory=dict)
    parent_result_id: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class SequenceEvent(BaseModel):
    """Ordered observable collaboration event. No hidden reasoning."""

    kind: Literal[
        "task_created",
        "coordinator_decision",
        "delegation",
        "agent_selected",
        "agent_message",
        "agent_result",
        "state_updated",
        "aggregation",
        "failure",
        "termination",
        "error",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class CollaborationMetrics(BaseModel):
    total_ms: int
    coordinator_ms: int
    agent_ms: int
    synthesis_ms: int
    agents_invoked: int
    successful_agent_results: int
    failed_agent_results: int
    delegations: int
    skipped_delegations: int
    messages: int
    termination_reason: TerminationReason
    max_delegations: int
    provenance: Literal["measured"] = "measured"


class CollaborationRunResult(BaseModel):
    case_id: str
    example_class: str
    request: str
    answer: str
    ok: bool
    model: str
    model_driver: str
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: CollaborationMetrics
    state: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
