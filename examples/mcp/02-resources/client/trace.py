"""Build Lab-oriented traces from measured MCP resource protocol runs."""

from __future__ import annotations

import time
from typing import Any

from client.cases import MeasuredCase
from client.runner import TRANSPORT_LABEL, _resources_payload
from client.schemas import McpRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

SIGNATURE_FLOWS = {
    "DISCOVERY": "INITIALIZE → RESOURCES",
    "SINGLE_RESOURCE_READ": "INITIALIZE → DISCOVER → READ → CONTENT",
    "MULTI_RESOURCE_READ": "INITIALIZE → DISCOVER → READ → READ → CONTENT",
    "INVALID_RESOURCE": "INITIALIZE → DISCOVER → READ → REJECTED",
}


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable protocol events into the Lab presentation view."""
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
        elif event.kind == "resources_list_request":
            view.append({"phase": "DISCOVER", "method": "resources/list"})
        elif event.kind == "resources_list_response":
            view.append(
                {
                    "phase": "RESOURCES",
                    "resourceCount": event.detail.get("resourceCount"),
                    "uris": [r["uri"] for r in event.detail.get("resources", [])],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "resource_read_request":
            view.append(
                {
                    "phase": "READ",
                    "uri": event.detail.get("uri"),
                }
            )
        elif event.kind == "resource_read_response":
            is_error = bool(event.detail.get("isError"))
            phase = "REJECTED" if is_error else "CONTENT"
            entry: dict[str, Any] = {
                "phase": phase,
                "uri": event.detail.get("uri"),
                "isError": is_error,
                "latencyMs": event.latency_ms,
            }
            if is_error:
                entry["error"] = event.detail.get("error")
            else:
                entry["contents"] = event.detail.get("contents")
            view.append(entry)
        elif event.kind == "error":
            # Reserved for protocol/transport failures outside a normal
            # resources/read response representation.
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
            "resources_list_request": "protocol",
            "resources_list_response": "protocol",
            "resource_read_request": "protocol",
            "resource_read_response": "protocol",
            "error": "error",
            "termination": "protocol",
        }.get(event.kind, event.kind)
        title = {
            "initialize_request": "Initialize request",
            "initialize_response": "Initialize response",
            "resources_list_request": "Resources list request",
            "resources_list_response": "Resources list response",
            "resource_read_request": "Resource read request",
            "resource_read_response": "Resource read response",
            "error": "Protocol error",
            "termination": "Session termination",
        }.get(event.kind, event.kind)
        status = "ok"
        if event.kind == "error":
            status = "error"
        if event.kind == "resource_read_response" and event.detail.get("isError"):
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
        "resourceReadMs": metrics["resource_read_ms"],
        "resourcesDiscovered": metrics["resources_discovered"],
        "resourcesRead": metrics["resources_read"],
        "successfulReads": metrics["successful_reads"],
        "failedReads": metrics["failed_reads"],
        "resourceBytes": metrics["resource_bytes"],
        "modelTurns": metrics["model_turns"],
        "toolCalls": metrics["tool_calls"],
        "terminationReason": metrics["termination_reason"],
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
    signature_note = SIGNATURE_FLOWS.get(
        case.example_class,
        "INITIALIZE → DISCOVER → READ → CONTENT",
    )

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
            "layout": "mcp-resources",
            "stages": [
                "initialize",
                "resources-list",
                "resource-identity",
                "resources-read",
                "resource-content",
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
            "requestedUris": list(case.resource_uris),
        },
        "resources": _resources_payload(result.discovered_resources),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "output": result.output,
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["mcp"],
        "relatedContent": ["mcp", "resources"],
        "cookbook": {"path": "examples/mcp/02-resources"},
    }
