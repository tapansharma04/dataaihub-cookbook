"""Typed schemas for agent memory — records, decisions, traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TerminationReason = Literal[
    "final_answer",
    "max_turns",
    "error",
    "invalid_action",
]
DecisionKind = Literal[
    "store_memory",
    "retrieve_memory",
    "final_answer",
    "invalid_action",
]
MemorySource = Literal["user", "system", "tool", "application"]
FreshnessResolution = Literal[
    "memory_used",
    "memory_miss",
    "current_source_preferred",
    "memory_matches_current",
]


class MemoryOperationDefinition(BaseModel):
    """Application-owned memory operation schema the model may propose."""

    name: str
    description: str
    parameters: dict[str, Any]


class ModelToolCall(BaseModel):
    id: str
    name: str
    arguments_json: str


class ModelTurn(BaseModel):
    """One observable model response turn.

    Memory turns propose a store, a retrieve, or a final answer. The
    application validates and performs memory operations.
    """

    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    decision: DecisionKind | None = None
    memory_write: dict[str, Any] | None = None
    memory_read: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class SequenceEvent(BaseModel):
    """Ordered observable event in the memory runtime."""

    kind: Literal[
        "user_request",
        "model_decision",
        "memory_write_requested",
        "memory_stored",
        "memory_retrieval_requested",
        "memory_retrieved",
        "memory_not_found",
        "observation",
        "final_answer",
        "termination",
        "error",
    ]
    turn: int | None = None
    interaction_id: str | None = None
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
    memory_writes: int
    memory_reads: int
    memory_hits: int
    memory_misses: int
    memory_scope: str
    memory_version: int | None = None
    stale_memory_detected: bool = False
    provenance: Literal["measured"] = "measured"


class AgentRunResult(BaseModel):
    request: str
    answer: str
    model: str
    model_driver: str
    memory_operations: list[MemoryOperationDefinition]
    sequence: list[SequenceEvent]
    metrics: AgentRunMetrics
    state: dict[str, Any]
    errors: list[dict[str, Any]] = Field(default_factory=list)
