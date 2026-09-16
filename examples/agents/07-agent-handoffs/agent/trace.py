"""Build recorded traces from measured handoff runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.cases import MeasuredCase
from agent.schemas import HandoffRunResult, SequenceEvent
from config import EXAMPLE_ID, ROOT, Settings

# Presentation compression: omit activation bookkeeping and the accept
# acknowledgement. HANDOFF is the request. Ownership transfer is visible
# on that request plus the following AGENT (new owner).
SIGNATURE_OMITTED_PHASES = frozenset(
    {
        "AGENT_ACTIVATED",
        "HANDOFF_ACCEPTED",
        "OWNERSHIP_TRANSFERRED",
    }
)

SIGNATURE_FLOWS = {
    "BASIC_HANDOFF": ("TASK → AGENT → HANDOFF → AGENT → COMPLETE → TERMINATION"),
    "MULTI_HOP_HANDOFF": (
        "TASK → AGENT → HANDOFF → AGENT → HANDOFF → AGENT → COMPLETE → TERMINATION"
    ),
    "INVALID_HANDOFF": ("TASK → AGENT → HANDOFF → HANDOFF_REJECTED → TERMINATION"),
    "HANDOFF_FAILURE": (
        "TASK → AGENT → HANDOFF → AGENT → HANDOFF → AGENT → FAILURE → TERMINATION"
    ),
}


def _portable_data_dir(settings: Settings) -> str:
    data_dir = Path(settings.data_dir).resolve()
    try:
        return str(data_dir.relative_to(ROOT.resolve()))
    except ValueError:
        return data_dir.name


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for event in sequence:
        if event.kind == "task_created":
            view.append(
                {
                    "phase": "TASK",
                    "taskId": event.detail.get("taskId"),
                    "request": event.detail.get("request"),
                }
            )
        elif event.kind == "agent_activated":
            view.append(
                {
                    "phase": "AGENT_ACTIVATED",
                    "agentId": event.detail.get("agentId"),
                    "enteredVia": event.detail.get("enteredVia"),
                    "currentOwner": event.detail.get("currentOwner"),
                }
            )
        elif event.kind == "agent_result":
            view.append(
                {
                    "phase": "AGENT",
                    "agentId": event.detail.get("agentId"),
                    "role": event.detail.get("role"),
                    "kind": event.detail.get("kind"),
                    "ok": event.detail.get("ok"),
                    "resultId": event.detail.get("resultId"),
                    "currentOwner": event.detail.get("currentOwner"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "handoff_requested":
            view.append(
                {
                    "phase": "HANDOFF",
                    "handoffId": event.detail.get("handoffId"),
                    "from": event.detail.get("from"),
                    "to": event.detail.get("to"),
                    "parentResultId": event.detail.get("parentResultId"),
                    "currentOwner": event.detail.get("currentOwner"),
                }
            )
        elif event.kind == "handoff_accepted":
            view.append(
                {
                    "phase": "HANDOFF_ACCEPTED",
                    "handoffId": event.detail.get("handoffId"),
                    "from": event.detail.get("from"),
                    "to": event.detail.get("to"),
                }
            )
        elif event.kind == "handoff_rejected":
            view.append(
                {
                    "phase": "HANDOFF_REJECTED",
                    "handoffId": event.detail.get("handoffId"),
                    "from": event.detail.get("from"),
                    "to": event.detail.get("to"),
                    "code": event.detail.get("code"),
                    "currentOwner": event.detail.get("currentOwner"),
                }
            )
        elif event.kind == "ownership_transferred":
            view.append(
                {
                    "phase": "OWNERSHIP_TRANSFERRED",
                    "from": event.detail.get("from"),
                    "to": event.detail.get("to"),
                    "currentOwner": event.detail.get("currentOwner"),
                    "previousOwner": event.detail.get("previousOwner"),
                }
            )
        elif event.kind == "completed":
            view.append(
                {
                    "phase": "COMPLETE",
                    "agentId": event.detail.get("agentId"),
                    "currentOwner": event.detail.get("currentOwner"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "failure":
            view.append(
                {
                    "phase": "FAILURE",
                    "agentId": event.detail.get("agentId"),
                    "error": event.detail.get("error"),
                    "currentOwner": event.detail.get("currentOwner"),
                }
            )
        elif event.kind == "error":
            view.append({"phase": "ERROR", "detail": event.detail})
        elif event.kind == "termination":
            view.append({"phase": "TERMINATION", "detail": event.detail})
    return view


def signature_phases_from_view(view: list[dict[str, Any]]) -> list[str]:
    return [
        item["phase"]
        for item in view
        if item.get("phase") not in SIGNATURE_OMITTED_PHASES
    ]


def signature_flow_from_view(view: list[dict[str, Any]]) -> str:
    return " → ".join(signature_phases_from_view(view))


def sequence_to_steps(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    titles = {
        "task_created": "Task created",
        "agent_activated": "Agent activated",
        "agent_result": "Agent result",
        "handoff_requested": "Handoff requested",
        "handoff_accepted": "Handoff accepted",
        "handoff_rejected": "Handoff rejected",
        "ownership_transferred": "Ownership transferred",
        "completed": "Completed",
        "failure": "Agent failure",
        "termination": "Termination",
        "error": "Error",
    }
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind in {"failure", "error", "handoff_rejected"}:
            status = "error"
        if event.kind == "agent_result" and event.detail.get("ok") is False:
            status = "error"
        step_type = "protocol"
        if event.kind in {"agent_activated", "agent_result"}:
            step_type = "agent"
        if event.kind in {
            "handoff_requested",
            "handoff_accepted",
            "ownership_transferred",
        }:
            step_type = "handoff"
        if event.kind in {"failure", "error", "handoff_rejected"} or status == "error":
            step_type = "error"
        if event.kind == "termination":
            step_type = "termination"
        if event.kind == "completed":
            step_type = "completion"
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


def build_trace(
    *,
    case: MeasuredCase,
    result: HandoffRunResult,
    settings: Settings,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump()
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "agentMs": metrics["agent_ms"],
        "runtimeMs": metrics["runtime_ms"],
        "agentsActivated": metrics["agents_activated"],
        "handoffsRequested": metrics["handoffs_requested"],
        "handoffsAccepted": metrics["handoffs_accepted"],
        "handoffsRejected": metrics["handoffs_rejected"],
        "ownershipTransfers": metrics["ownership_transfers"],
        "successfulAgentResults": metrics["successful_agent_results"],
        "failedAgentResults": metrics["failed_agent_results"],
        "terminationReason": metrics["termination_reason"],
        "maxHandoffs": metrics["max_handoffs"],
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
        "provenance": {
            "model": result.model_driver,
            "tools": "measured",
            "metrics": "measured",
        },
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "agent-handoffs",
            "stages": [
                "task",
                "activate",
                "handoff",
                "ownership-transfer",
                "complete-or-fail",
                "termination",
            ],
        },
        "input": {
            "request": case.request,
            "ticketId": case.ticket_id,
            "config": {
                "maxHandoffs": result.metrics.max_handoffs,
                "modelDriver": result.model_driver,
                "synthesizerModel": result.model,
                "clientName": "dataaihub-cookbook-agent-handoffs",
                "dataDir": _portable_data_dir(settings),
            },
        },
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "state": result.state,
        "output": {
            "answer": result.answer,
            "ok": result.ok,
            "terminationReason": result.metrics.termination_reason,
            "currentOwner": result.state.get("currentOwner"),
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": {
            "purpose": (
                "Frontend-friendly projection of observable handoff events. "
                "Not a new measurement."
            ),
            "signatureView": build_signature_view(result.sequence),
            "signatureFlow": SIGNATURE_FLOWS.get(
                case.example_class,
                "TASK → AGENT → HANDOFF → AGENT → TERMINATION",
            ),
        },
        "relatedEntities": ["agents"],
        "relatedContent": ["agents", "handoffs", "ownership-transfer"],
        "cookbook": {"path": "examples/agents/07-agent-handoffs"},
    }
