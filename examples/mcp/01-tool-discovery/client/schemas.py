"""Typed schemas for MCP protocol runs and Lab traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DiscoveredTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class SequenceEvent(BaseModel):
    """Ordered observable MCP protocol event."""

    kind: Literal[
        "initialize_request",
        "initialize_response",
        "tools_list_request",
        "tools_list_response",
        "tool_call_request",
        "tool_call_response",
        "error",
        "termination",
    ]
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class McpRunMetrics(BaseModel):
    total_ms: int
    initialize_ms: int
    discovery_ms: int
    tool_call_ms: int
    tool_calls: int
    tools_discovered: int
    successful_tool_calls: int
    failed_tool_calls: int
    provenance: Literal["measured"] = "measured"


class McpRunResult(BaseModel):
    case_id: str
    example_class: str
    transport: str
    protocol_version: str | None = None
    server_name: str | None = None
    discovered_tools: list[DiscoveredTool] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: McpRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
