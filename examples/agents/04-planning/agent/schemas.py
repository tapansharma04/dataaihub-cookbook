"""Typed schemas for agent planning — plan state, decisions, tools, traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TerminationReason = Literal[
    "final_answer",
    "max_turns",
    "error",
    "invalid_action",
    "plan_failed",
]
DecisionKind = Literal[
    "create_plan",
    "revise_plan",
    "final_answer",
    "invalid_action",
]
ActionKind = Literal["tool_call", "finalize"]
StepIntent = Literal[
    "status_check",
    "docs_lookup",
    "remediation",
    "required_docs",
    "summarize",
]
StepStatus = Literal["pending", "in_progress", "completed", "skipped", "failed"]
PlanStatus = Literal["pending", "in_progress", "completed", "failed", "superseded"]
ObservationEffectKind = Literal["continue", "revise", "block"]


class ToolParameterProperty(BaseModel):
    type: str
    description: str


class ToolParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, ToolParameterProperty]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolDefinition(BaseModel):
    """Application-owned tool schema exposed for plan step validation."""

    name: str
    description: str
    parameters: ToolParameters


class ToolCallRequest(BaseModel):
    """A single runtime-executed tool invocation (observable, not CoT)."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallResult(BaseModel):
    """Structured tool execution outcome."""

    call_id: str
    name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_ms: int
    validated_arguments: dict[str, Any] | None = None


class ModelToolCall(BaseModel):
    id: str
    name: str
    arguments_json: str


class ModelTurn(BaseModel):
    """One observable model response turn.

    Planning turns propose a plan, a revision, or a final answer. Data tools
    are not selected turn-by-turn; the runtime executes validated plan steps.
    """

    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    decision: DecisionKind | None = None
    proposed_steps: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class SequenceEvent(BaseModel):
    """Ordered observable event in the planning runtime."""

    kind: Literal[
        "user_request",
        "model_decision",
        "plan_created",
        "plan_step_started",
        "tool_call",
        "observation",
        "plan_step_completed",
        "plan_revised",
        "final_answer",
        "termination",
        "error",
    ]
    turn: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class AgentRunMetrics(BaseModel):
    total_ms: int
    model_ms: int
    tool_ms: int
    model_turns: int
    tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    termination_reason: TerminationReason
    max_turns: int
    plan_steps: int
    completed_steps: int
    skipped_steps: int
    failed_steps: int
    plan_revisions: int
    plan_version: int
    plan_status: PlanStatus | None = None
    provenance: Literal["measured"] = "measured"


class AgentRunResult(BaseModel):
    request: str
    answer: str
    model: str
    model_driver: str
    tool_definitions: list[ToolDefinition]
    sequence: list[SequenceEvent]
    metrics: AgentRunMetrics
    state: dict[str, Any]
    errors: list[dict[str, Any]] = Field(default_factory=list)
