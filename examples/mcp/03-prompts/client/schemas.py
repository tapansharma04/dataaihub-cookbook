"""Typed schemas for MCP prompt protocol runs and Lab traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PromptArgumentMeta(BaseModel):
    name: str
    description: str | None = None
    required: bool = False


class DiscoveredPrompt(BaseModel):
    name: str
    description: str | None = None
    arguments: list[PromptArgumentMeta] = Field(default_factory=list)


class SequenceEvent(BaseModel):
    """Ordered observable MCP protocol event."""

    kind: Literal[
        "initialize_request",
        "initialize_response",
        "prompts_list_request",
        "prompts_list_response",
        "prompt_get_request",
        "prompt_get_response",
        "error",
        "termination",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class McpRunMetrics(BaseModel):
    total_ms: int
    initialize_ms: int
    discovery_ms: int
    prompt_get_ms: int
    prompts_discovered: int
    prompts_requested: int
    successful_gets: int
    failed_gets: int
    message_count: int
    message_bytes: int
    model_turns: int = 0
    tool_calls: int = 0
    resources_read: int = 0
    termination_reason: str
    provenance: Literal["measured"] = "measured"


class McpRunResult(BaseModel):
    case_id: str
    example_class: str
    transport: str
    protocol_version: str | None = None
    server_name: str | None = None
    discovered_prompts: list[DiscoveredPrompt] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: McpRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
