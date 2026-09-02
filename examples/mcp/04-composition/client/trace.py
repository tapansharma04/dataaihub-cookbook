"""Build Lab-oriented traces from measured MCP composition runs."""

from __future__ import annotations

import time
from typing import Any

from client.cases import MeasuredCase
from client.runner import (
    TRANSPORT_LABEL,
    prompts_payload,
    resources_payload,
    tools_payload,
)
from client.schemas import McpRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

SIGNATURE_FLOWS = {
    "RESOURCE_TO_SAMPLING": ("INITIALIZE → RESOURCE → CONTEXT → SAMPLING → RESULT"),
    "PROMPT_TO_SAMPLING": ("INITIALIZE → PROMPT → ARGUMENTS → SAMPLING → RESULT"),
    "TOOL_RESOURCE_PROMPT_COMPOSITION": (
        "INITIALIZE → TOOL → RESOURCE → PROMPT → SAMPLING → RESULT"
    ),
    "SAMPLING_FAILURE": ("INITIALIZE → RESOURCE → SAMPLING → REJECTED"),
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
                    "uris": [item["uri"] for item in event.detail.get("resources", [])],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "resource_read_request":
            view.append({"phase": "RESOURCE", "uri": event.detail.get("uri")})
        elif event.kind == "resource_read_response":
            view.append(
                {
                    "phase": "CONTEXT",
                    "uri": event.detail.get("uri"),
                    "isError": bool(event.detail.get("isError")),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "prompts_list_request":
            view.append({"phase": "DISCOVER", "method": "prompts/list"})
        elif event.kind == "prompts_list_response":
            view.append(
                {
                    "phase": "PROMPTS",
                    "promptCount": event.detail.get("promptCount"),
                    "names": [item["name"] for item in event.detail.get("prompts", [])],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "prompt_get_request":
            view.append({"phase": "PROMPT", "name": event.detail.get("name")})
            arguments = event.detail.get("arguments") or {}
            if arguments:
                view.append(
                    {
                        "phase": "ARGUMENTS",
                        "name": event.detail.get("name"),
                        "arguments": arguments,
                    }
                )
        elif event.kind == "prompt_get_response":
            view.append(
                {
                    "phase": "MESSAGES",
                    "name": event.detail.get("name"),
                    "isError": bool(event.detail.get("isError")),
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
                    "names": [item["name"] for item in event.detail.get("tools", [])],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "tool_call_request":
            view.append(
                {
                    "phase": "TOOL",
                    "name": event.detail.get("name"),
                    "arguments": event.detail.get("arguments"),
                }
            )
        elif event.kind == "tool_call_response":
            is_error = bool(event.detail.get("isError"))
            view.append(
                {
                    "phase": "REJECTED" if is_error else "TOOL_RESULT",
                    "name": event.detail.get("name"),
                    "isError": is_error,
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "sampling_request":
            view.append(
                {
                    "phase": "SAMPLING",
                    "method": event.detail.get("method"),
                    "boundary": event.detail.get("boundary"),
                }
            )
        elif event.kind == "sampling_response":
            is_error = bool(event.detail.get("isError"))
            entry: dict[str, Any] = {
                "phase": "REJECTED" if is_error else "RESULT",
                "isError": is_error,
                "boundary": event.detail.get("boundary"),
                "latencyMs": event.latency_ms,
            }
            if is_error:
                entry["error"] = event.detail.get("error")
            else:
                result = event.detail.get("result") or {}
                entry["model"] = result.get("model")
            view.append(entry)
        elif event.kind == "error":
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
    titles = {
        "initialize_request": "Initialize request",
        "initialize_response": "Initialize response",
        "resources_list_request": "Resources list request",
        "resources_list_response": "Resources list response",
        "resource_read_request": "Resource read request",
        "resource_read_response": "Resource read response",
        "prompts_list_request": "Prompts list request",
        "prompts_list_response": "Prompts list response",
        "prompt_get_request": "Prompt get request",
        "prompt_get_response": "Prompt get response",
        "tools_list_request": "Tools list request",
        "tools_list_response": "Tools list response",
        "tool_call_request": "Tool call request",
        "tool_call_response": "Tool call response",
        "sampling_request": "Sampling request",
        "sampling_response": "Sampling response",
        "error": "Protocol error",
        "termination": "Session termination",
    }
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "error":
            status = "error"
        if event.kind in {"sampling_response", "tool_call_response"}:
            if event.detail.get("isError"):
                status = "error"
        step_type = "error" if status == "error" else "protocol"
        if event.kind in {"sampling_request", "sampling_response"}:
            step_type = "sampling" if status == "ok" else "error"
        steps.append(
            {
                "id": f"step-{i}-{event.kind}",
                "type": step_type,
                "title": titles.get(event.kind, event.kind),
                "status": status,
                "detail": event.detail,
                "metrics": {
                    "latencyMs": event.latency_ms,
                    "provenance": "measured",
                },
            }
        )
    return steps


def _provenance(result: McpRunResult) -> dict[str, str]:
    if result.sampling_mode == "live":
        model = "not_used"
        for event in result.sequence:
            if event.kind == "sampling_response" and not event.detail.get("isError"):
                payload = event.detail.get("result") or {}
                if payload.get("model"):
                    model = str(payload["model"])
                    break
        return {"model": model, "tools": "measured", "metrics": "measured"}
    if result.metrics.successful_samplings > 0:
        return {"model": "mock", "tools": "measured", "metrics": "measured"}
    return {"model": "not_used", "tools": "measured", "metrics": "measured"}


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
        "promptGetMs": metrics["prompt_get_ms"],
        "toolCallMs": metrics["tool_call_ms"],
        "samplingMs": metrics["sampling_ms"],
        "resourcesDiscovered": metrics["resources_discovered"],
        "resourcesRead": metrics["resources_read"],
        "promptsDiscovered": metrics["prompts_discovered"],
        "promptsRequested": metrics["prompts_requested"],
        "toolsDiscovered": metrics["tools_discovered"],
        "toolCalls": metrics["tool_calls"],
        "successfulToolCalls": metrics["successful_tool_calls"],
        "failedToolCalls": metrics["failed_tool_calls"],
        "samplingRequests": metrics["sampling_requests"],
        "successfulSamplings": metrics["successful_samplings"],
        "failedSamplings": metrics["failed_samplings"],
        "modelTurns": metrics["model_turns"],
        "terminationReason": metrics["termination_reason"],
        "provenance": metrics["provenance"],
    }

    sequence_payload = [
        {
            "kind": event.kind,
            "detail": event.detail,
            "latencyMs": event.latency_ms,
        }
        for event in result.sequence
    ]

    return {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": "guided",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metricsProvenance": "measured",
        "provenance": _provenance(result),
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "mcp-composition",
            "stages": [
                "initialize",
                "mcp-primitives",
                "sampling-request",
                "client-model-boundary",
                "sampling-response",
                "application-result",
            ],
            "samplingMode": result.sampling_mode,
        },
        "input": {
            "case": case.trace_id,
            "config": {
                "transport": TRANSPORT_LABEL,
                "protocolVersion": result.protocol_version,
                "clientName": settings.client_name,
                "clientVersion": settings.client_version,
                "mcpClientMode": settings.mcp_client_mode,
                "samplingMode": result.sampling_mode,
            },
        },
        "resources": resources_payload(result.discovered_resources),
        "prompts": prompts_payload(result.discovered_prompts),
        "tools": tools_payload(result.discovered_tools),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "output": result.output,
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": {
            "purpose": (
                "Frontend-friendly projection of observable MCP protocol events. "
                "Not a new measurement."
            ),
            "signatureView": build_signature_view(result.sequence),
            "signatureFlow": SIGNATURE_FLOWS.get(
                case.example_class,
                "INITIALIZE → PRIMITIVES → SAMPLING → RESULT",
            ),
        },
        "relatedEntities": ["mcp"],
        "relatedContent": ["mcp", "composition", "sampling"],
        "cookbook": {"path": "examples/mcp/04-composition"},
    }
