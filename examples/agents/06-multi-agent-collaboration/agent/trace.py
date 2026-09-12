"""Build Lab-oriented traces from measured collaboration runs."""

from __future__ import annotations

import time
from typing import Any

from agent.cases import MeasuredCase
from agent.schemas import CollaborationRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

# Presentation compression: omit the coordinator's AGGREGATE_DECISION.
# AGENT is the specialist result (after handle), not agent_selected.
# PASS_RESULT is the inbound message that carries a parent result, emitted
# before that specialist runs. Failure still includes AGGREGATE because the
# runtime synthesizes a brief from actual TaskState results before terminating.
SIGNATURE_OMITTED_PHASES = frozenset({"AGGREGATE_DECISION"})

SIGNATURE_FLOWS = {
    "BASIC_COLLABORATION": (
        "TASK → DELEGATE → AGENT → DELEGATE → AGENT → AGGREGATE → TERMINATION"
    ),
    "SEQUENTIAL_COLLABORATION": (
        "TASK → DELEGATE → AGENT → DELEGATE → PASS_RESULT → AGENT → "
        "AGGREGATE → TERMINATION"
    ),
    "AGENT_FAILURE": (
        "TASK → DELEGATE → AGENT → DELEGATE → AGENT → FAILURE → AGGREGATE → TERMINATION"
    ),
    "COLLABORATION_TERMINATION": (
        "TASK → DELEGATE → AGENT → SKIP → AGGREGATE → TERMINATION"
    ),
}


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
        elif event.kind == "coordinator_decision":
            decision = event.detail.get("decision")
            if decision == "skip":
                view.append(
                    {
                        "phase": "SKIP",
                        "agentId": event.detail.get("agentId"),
                        "reason": event.detail.get("reason"),
                    }
                )
            elif decision == "delegate":
                view.append(
                    {
                        "phase": "DELEGATE",
                        "agentId": event.detail.get("agentId"),
                    }
                )
            elif decision == "aggregate":
                view.append({"phase": "AGGREGATE_DECISION"})
        elif event.kind == "agent_message":
            if event.detail.get("parentResultId"):
                view.append(
                    {
                        "phase": "PASS_RESULT",
                        "to": event.detail.get("to"),
                        "parentResultId": event.detail.get("parentResultId"),
                    }
                )
        elif event.kind == "agent_result":
            view.append(
                {
                    "phase": "AGENT",
                    "agentId": event.detail.get("agentId"),
                    "role": event.detail.get("role"),
                    "ok": event.detail.get("ok"),
                    "resultId": event.detail.get("resultId"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "failure":
            view.append(
                {
                    "phase": "FAILURE",
                    "agentId": event.detail.get("agentId"),
                    "error": event.detail.get("error"),
                }
            )
        elif event.kind == "aggregation":
            view.append(
                {
                    "phase": "AGGREGATE",
                    "agentIds": event.detail.get("agentIds"),
                    "latencyMs": event.latency_ms,
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
        "coordinator_decision": "Coordinator decision",
        "delegation": "Delegation",
        "agent_selected": "Agent selected",
        "agent_message": "Agent message",
        "agent_result": "Agent result",
        "state_updated": "Shared state updated",
        "aggregation": "Coordinator aggregation",
        "failure": "Agent failure",
        "termination": "Termination",
        "error": "Error",
    }
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind in {"failure", "error"}:
            status = "error"
        if event.kind == "agent_result" and event.detail.get("ok") is False:
            status = "error"
        step_type = "protocol"
        if event.kind in {"agent_message", "agent_result"}:
            step_type = "agent"
        if event.kind in {"failure", "error"} or status == "error":
            step_type = "error"
        if event.kind == "termination":
            step_type = "termination"
        if event.kind == "aggregation":
            step_type = "aggregation"
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
    result: CollaborationRunResult,
    settings: Settings,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump()
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "coordinatorMs": metrics["coordinator_ms"],
        "agentMs": metrics["agent_ms"],
        "synthesisMs": metrics["synthesis_ms"],
        "agentsInvoked": metrics["agents_invoked"],
        "successfulAgentResults": metrics["successful_agent_results"],
        "failedAgentResults": metrics["failed_agent_results"],
        "delegations": metrics["delegations"],
        "skippedDelegations": metrics["skipped_delegations"],
        "messages": metrics["messages"],
        "terminationReason": metrics["termination_reason"],
        "maxDelegations": metrics["max_delegations"],
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
            "layout": "multi-agent-collaboration",
            "stages": [
                "task",
                "delegate",
                "specialist",
                "shared-state",
                "aggregate",
                "termination",
            ],
        },
        "input": {
            "request": case.request,
            "service": case.service,
            "config": {
                "maxDelegations": result.metrics.max_delegations,
                "modelDriver": result.model_driver,
                "synthesizerModel": result.model,
                "clientName": "dataaihub-cookbook-multi-agent",
                "dataDir": str(settings.data_dir),
            },
        },
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "state": result.state,
        "output": {
            "answer": result.answer,
            "ok": result.ok,
            "terminationReason": result.metrics.termination_reason,
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": {
            "purpose": (
                "Frontend-friendly projection of observable collaboration "
                "events. Not a new measurement."
            ),
            "signatureView": build_signature_view(result.sequence),
            "signatureFlow": SIGNATURE_FLOWS.get(
                case.example_class,
                "TASK → DELEGATE → AGENT → AGGREGATE → TERMINATION",
            ),
        },
        "relatedEntities": ["agents"],
        "relatedContent": ["agents", "multi-agent", "collaboration"],
        "cookbook": {"path": "examples/agents/06-multi-agent-collaboration"},
    }
