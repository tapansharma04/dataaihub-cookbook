"""Typed schemas for the agent loop — state, decisions, tools, traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TerminationReason = Literal["final_answer", "max_turns", "error", "invalid_action"]
DecisionKind = Literal["tool_call", "final_answer", "invalid_action"]


class ToolParameterProperty(BaseModel):
    type: str
    description: str


class ToolParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, ToolParameterProperty]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolDefinition(BaseModel):
    """Application-owned tool schema exposed to the model."""

    name: str
    description: str
    parameters: ToolParameters


class ToolCallRequest(BaseModel):
    """A single model-requested tool invocation (observable, not CoT)."""

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

    `decision` is the runtime-classified next action. Live OpenAI clients infer
    tool_call vs final_answer from tool_calls; the case harness may emit an
    explicit invalid decision to teach runtime rejection.
    """

    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    decision: DecisionKind | None = None
    finish_reason: str | None = None
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class SequenceEvent(BaseModel):
    """Ordered observable event in the agent loop."""

    kind: Literal[
        "user_request",
        "model_decision",
        "tool_call",
        "observation",
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
