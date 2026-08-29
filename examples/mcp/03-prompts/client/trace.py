"""Build Lab-oriented traces from measured MCP prompt protocol runs."""

from __future__ import annotations

import time
from typing import Any

from client.cases import MeasuredCase
from client.runner import TRANSPORT_LABEL, _prompts_payload
from client.schemas import McpRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

SIGNATURE_FLOWS = {
    "PROMPT_DISCOVERY": "INITIALIZE → DISCOVER → PROMPTS",
    "SINGLE_PROMPT_GET": "INITIALIZE → DISCOVER → PROMPTS → GET → MESSAGES",
    "PROMPT_WITH_ARGUMENTS": (
        "INITIALIZE → DISCOVER → PROMPTS → GET → ARGUMENTS → MESSAGES"
    ),
    "INVALID_PROMPT": "INITIALIZE → DISCOVER → GET → REJECTED",
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
        elif event.kind == "prompts_list_request":
            view.append({"phase": "DISCOVER", "method": "prompts/list"})
        elif event.kind == "prompts_list_response":
            view.append(
                {
                    "phase": "PROMPTS",
                    "promptCount": event.detail.get("promptCount"),
                    "names": [
                        prompt["name"] for prompt in event.detail.get("prompts", [])
                    ],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "prompt_get_request":
            view.append(
                {
                    "phase": "GET",
                    "name": event.detail.get("name"),
                }
            )
            arguments = event.detail.get("arguments") or {}
            if len(arguments) > 1:
                view.append(
                    {
                        "phase": "ARGUMENTS",
                        "name": event.detail.get("name"),
                        "arguments": arguments,
                    }
                )
        elif event.kind == "prompt_get_response":
            is_error = bool(event.detail.get("isError"))
            phase = "REJECTED" if is_error else "MESSAGES"
            entry: dict[str, Any] = {
                "phase": phase,
                "name": event.detail.get("name"),
                "isError": is_error,
                "latencyMs": event.latency_ms,
            }
            if is_error:
                entry["error"] = event.detail.get("error")
            else:
                entry["messages"] = event.detail.get("messages")
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
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        step_type = {
            "initialize_request": "protocol",
            "initialize_response": "protocol",
            "prompts_list_request": "protocol",
            "prompts_list_response": "protocol",
            "prompt_get_request": "protocol",
            "prompt_get_response": "protocol",
            "error": "error",
            "termination": "protocol",
        }.get(event.kind, event.kind)
        title = {
            "initialize_request": "Initialize request",
            "initialize_response": "Initialize response",
            "prompts_list_request": "Prompts list request",
            "prompts_list_response": "Prompts list response",
            "prompt_get_request": "Prompt get request",
            "prompt_get_response": "Prompt get response",
            "error": "Protocol error",
            "termination": "Session termination",
        }.get(event.kind, event.kind)
        status = "ok"
        if event.kind == "error":
            status = "error"
        if event.kind == "prompt_get_response" and event.detail.get("isError"):
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
        "promptGetMs": metrics["prompt_get_ms"],
        "promptsDiscovered": metrics["prompts_discovered"],
        "promptsRequested": metrics["prompts_requested"],
        "successfulGets": metrics["successful_gets"],
        "failedGets": metrics["failed_gets"],
        "messageCount": metrics["message_count"],
        "messageBytes": metrics["message_bytes"],
        "modelTurns": metrics["model_turns"],
        "toolCalls": metrics["tool_calls"],
        "resourcesRead": metrics["resources_read"],
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
        "INITIALIZE → DISCOVER → PROMPTS → GET → MESSAGES",
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
            "layout": "mcp-prompts",
            "stages": [
                "initialize",
                "prompts-list",
                "prompt-identity",
                "prompts-get",
                "prompt-messages",
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
            "requestedPrompt": case.prompt_name,
            "requestedArguments": case.prompt_arguments or {},
        },
        "prompts": _prompts_payload(result.discovered_prompts),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "output": result.output,
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["mcp"],
        "relatedContent": ["mcp", "prompts"],
        "cookbook": {"path": "examples/mcp/03-prompts"},
    }
