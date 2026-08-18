"""Build Lab-oriented traces from measured MCP protocol runs."""

from __future__ import annotations

import time
from typing import Any

from client.cases import MeasuredCase
from client.runner import TRANSPORT_LABEL, _tools_payload
from client.schemas import McpRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable protocol events into the Lab teaching view."""
    view: list[dict[str, Any]] = []
    for event in sequence:
        if event.kind == "initialize_request":
            view.append({"phase": "INITIALIZE", "detail": event.detail})
        elif event.kind == "initialize_response":
            view.append(
                {
                    "phase": "INITIALIZED",
                    "protocolVersion": event.detail.get("protocolVersion"),
                    "serverInfo": event.detail.get("serverInfo"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "tools_list_request":
            view.append({"phase": "DISCOVER", "method": "tools/list"})
        elif event.kind == "tools_list_response":
            view.append(
                {
                    "phase": "TOOLS",
                    "toolCount": event.detail.get("toolCount"),
                    "tools": [t["name"] for t in event.detail.get("tools", [])],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "tool_call_request":
            view.append(
                {
                    "phase": "INVOKE",
                    "tool": event.detail.get("name"),
                    "arguments": event.detail.get("arguments"),
                }
            )
        elif event.kind == "tool_call_response":
            is_error = event.detail.get("isError")
            phase = "REJECTED" if is_error else "RESULT"
            view.append(
                {
                    "phase": phase,
                    "tool": event.detail.get("name"),
                    "isError": is_error,
                    "result": event.detail.get("result"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "error":
            # Reserved for protocol/transport failures outside a normal tools/call
            # response. Tool-level validation failures use tool_call_response with
            # isError=true; presentation derives REJECTED from that response.
            view.append(
                {
                    "phase": "ERROR",
                    "detail": event.detail,
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "termination":
            view.append({"phase": "TERMINATION", "detail": event.detail})
    return view


def sequence_to_steps(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        step_type = {
            "initialize_request": "protocol",
            "initialize_response": "protocol",
            "tools_list_request": "protocol",
            "tools_list_response": "protocol",
            "tool_call_request": "protocol",
            "tool_call_response": "protocol",
            "error": "error",
            "termination": "protocol",
        }.get(event.kind, event.kind)
        title = {
            "initialize_request": "Initialize request",
            "initialize_response": "Initialize response",
            "tools_list_request": "Tools list request",
            "tools_list_response": "Tools list response",
            "tool_call_request": "Tool call request",
            "tool_call_response": "Tool call response",
            "error": "Protocol error",
            "termination": "Session termination",
        }.get(event.kind, event.kind)
        status = "ok"
        if event.kind == "error":
            status = "error"
        if event.kind == "tool_call_response" and event.detail.get("isError"):
            status = "error"
        steps.append(
            {
                "id": f"step-{i}-{event.kind}",
                "type": step_type,
                "title": title,
                "status": status,
                "detail": event.detail,
                "metrics": {
                    "latencyMs": event.latency_ms,
                    "provenance": "measured",
                },
            }
        )
    return steps


def build_trace(
    *,
    case: MeasuredCase,
    result: McpRunResult,
    settings: Settings,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "initializeMs": metrics["initialize_ms"],
        "discoveryMs": metrics["discovery_ms"],
        "toolCallMs": metrics["tool_call_ms"],
        "toolCalls": metrics["tool_calls"],
        "toolsDiscovered": metrics["tools_discovered"],
        "successfulToolCalls": metrics["successful_tool_calls"],
        "failedToolCalls": metrics["failed_tool_calls"],
        "provenance": metrics["provenance"],
    }

    sequence_payload = [
        {
            "kind": e.kind,
            "detail": e.detail,
            "latencyMs": e.latency_ms,
        }
        for e in result.sequence
    ]

    signature = build_signature_view(result.sequence)
    if case.example_class == "INVALID_ARGUMENTS":
        signature_note = "INITIALIZE → DISCOVER → INVOKE → REJECTED"
    elif case.example_class == "DISCOVERY":
        signature_note = "INITIALIZE → DISCOVER → TOOLS"
    else:
        signature_note = "INITIALIZE → DISCOVER → INVOKE → RESULT"

    provenance = {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }

    presentation: dict[str, Any] = {
        "purpose": (
            "Frontend-friendly projection of observable MCP protocol events. "
            "Not a new measurement."
        ),
        "signatureView": signature,
        "signatureFlow": signature_note,
    }

    return {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": "guided",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metricsProvenance": "measured",
        "provenance": provenance,
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "mcp-tool-discovery",
            "stages": [
                "initialize",
                "tools-list",
                "tool-contract",
                "tools-call",
                "structured-result",
            ],
        },
        "input": {
            "case": case.trace_id,
            "config": {
                "transport": TRANSPORT_LABEL,
                "protocolVersion": result.protocol_version,
                "clientName": settings.client_name,
                "clientVersion": settings.client_version,
                "mcpClientMode": settings.mcp_client_mode,
            },
        },
        "tools": _tools_payload(result.discovered_tools),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "output": result.output,
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["mcp"],
        "relatedContent": ["mcp", "tool-discovery"],
        "cookbook": {"path": "examples/mcp/01-tool-discovery"},
    }
