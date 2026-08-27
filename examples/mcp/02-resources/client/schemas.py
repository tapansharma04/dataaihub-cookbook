"""Typed schemas for MCP resource protocol runs and Lab traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DiscoveredResource(BaseModel):
    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None


class SequenceEvent(BaseModel):
    """Ordered observable MCP protocol event."""

    kind: Literal[
        "initialize_request",
        "initialize_response",
        "resources_list_request",
        "resources_list_response",
        "resource_read_request",
        "resource_read_response",
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
    resources_discovered: int
    resources_read: int
    successful_reads: int
    failed_reads: int
    resource_bytes: int
    model_turns: int = 0
    tool_calls: int = 0
    termination_reason: str
    provenance: Literal["measured"] = "measured"


class McpRunResult(BaseModel):
    case_id: str
    example_class: str
    transport: str
    protocol_version: str | None = None
    server_name: str | None = None
    discovered_resources: list[DiscoveredResource] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: McpRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
