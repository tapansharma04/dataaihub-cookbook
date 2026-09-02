"""Typed schemas for MCP composition protocol runs and Lab traces."""

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


class DiscoveredResource(BaseModel):
    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None


class DiscoveredTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class SequenceEvent(BaseModel):
    """Ordered observable MCP protocol event."""

    kind: Literal[
        "initialize_request",
        "initialize_response",
        "resources_list_request",
        "resources_list_response",
        "resource_read_request",
        "resource_read_response",
        "prompts_list_request",
        "prompts_list_response",
        "prompt_get_request",
        "prompt_get_response",
        "tools_list_request",
        "tools_list_response",
        "tool_call_request",
        "tool_call_response",
        "sampling_request",
        "sampling_response",
        "error",
        "termination",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class McpRunMetrics(BaseModel):
    total_ms: int
    initialize_ms: int
    discovery_ms: int
    resource_read_ms: int
    prompt_get_ms: int
    tool_call_ms: int
    sampling_ms: int
    resources_discovered: int
    resources_read: int
    prompts_discovered: int
    prompts_requested: int
    tools_discovered: int
    tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    sampling_requests: int
    successful_samplings: int
    failed_samplings: int
    model_turns: int
    termination_reason: str
    provenance: Literal["measured"] = "measured"


class McpRunResult(BaseModel):
    case_id: str
    example_class: str
    transport: str
    protocol_version: str | None = None
    server_name: str | None = None
    sampling_mode: str
    discovered_resources: list[DiscoveredResource] = Field(default_factory=list)
    discovered_prompts: list[DiscoveredPrompt] = Field(default_factory=list)
    discovered_tools: list[DiscoveredTool] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: McpRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
