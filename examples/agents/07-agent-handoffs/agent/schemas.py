"""Typed schemas for application-owned agent handoffs.

A handoff is an ownership transfer. It is not a coordinator delegation:
the active agent requests that another agent become the current owner.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TerminationReason = Literal[
    "completed",
    "invalid_handoff",
    "agent_failed",
    "max_handoffs",
    "invalid_task",
    "error",
]
AgentRole = Literal["triage", "billing", "technical"]
AgentOutcomeKind = Literal["handoff", "complete", "failure"]
HandoffRejectionCode = Literal[
    "unknown_target",
    "self_handoff",
    "empty_target",
    "not_current_owner",
    "max_handoffs",
]
EnteredVia = Literal["initial", "handoff"]


class HandoffRequest(BaseModel):
    """Explicit request to transfer ownership to another agent."""

    handoff_id: str = ""
    from_agent: str
    to_agent: str
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_result_id: str | None = None


class HandoffRejection(BaseModel):
    """Runtime refusal to transfer ownership. The current owner does not change."""

    handoff_id: str
    from_agent: str
    to_agent: str
    code: HandoffRejectionCode
    message: str
    parent_result_id: str | None = None


class AgentResult(BaseModel):
    """Observable output of the currently active owner.

    kind distinguishes a normal completion, a handoff request, and a failure.
    """

    result_id: str
    agent_id: str
    role: AgentRole
    kind: AgentOutcomeKind
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    handoff: HandoffRequest | None = None
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> AgentResult:
        if self.kind == "handoff" and self.handoff is None:
            raise ValueError("kind='handoff' requires a HandoffRequest")
        if self.kind != "handoff" and self.handoff is not None:
            raise ValueError("HandoffRequest is only valid when kind='handoff'")
        if self.kind == "failure" and self.ok:
            raise ValueError("kind='failure' requires ok=False")
        if self.kind == "complete" and not self.ok:
            raise ValueError("kind='complete' requires ok=True")
        if self.kind == "handoff" and not self.ok:
            raise ValueError("kind='handoff' requires ok=True")
        return self


class AgentContext(BaseModel):
    """Inbound work for the currently active owner.

    Downstream agents must consume inbound_handoff. They do not reload
    another agent's catalog or private state.
    """

    task_id: str
    request: str | None = None
    ticket_id: str | None = None
    current_owner: str
    inbound_handoff: HandoffRequest | None = None


class OwnershipRecord(BaseModel):
    agent_id: str
    entered_via: EnteredVia
    handoff_id: str | None = None
    parent_result_id: str | None = None


class SequenceEvent(BaseModel):
    """Ordered observable handoff-runtime event. No hidden reasoning."""

    kind: Literal[
        "task_created",
        "agent_activated",
        "agent_result",
        "handoff_requested",
        "handoff_accepted",
        "handoff_rejected",
        "ownership_transferred",
        "completed",
        "failure",
        "termination",
        "error",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class HandoffMetrics(BaseModel):
    total_ms: int
    agent_ms: int
    runtime_ms: int
    agents_activated: int
    handoffs_requested: int
    handoffs_accepted: int
    handoffs_rejected: int
    ownership_transfers: int
    successful_agent_results: int
    failed_agent_results: int
    termination_reason: TerminationReason
    max_handoffs: int
    provenance: Literal["measured"] = "measured"


class HandoffRunResult(BaseModel):
    case_id: str
    example_class: str
    request: str
    answer: str
    ok: bool
    model: str
    model_driver: str
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: HandoffMetrics
    state: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
