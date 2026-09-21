"""Build recorded traces from measured orchestration runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.cases import MeasuredCase
from agent.schemas import OrchestrationRunResult, SequenceEvent
from config import EXAMPLE_ID, ROOT, Settings

# Presentation compression: omit per-node start/complete bookkeeping.
# READY is the scheduling decision. AGENT is the node result.
SIGNATURE_OMITTED_PHASES = frozenset({"NODE_STARTED", "NODE_COMPLETED"})

SIGNATURE_FLOWS = {
    "BASIC_ORCHESTRATION": (
        "TASK → PLAN → READY → READY → AGENT → AGENT → READY → AGENT → "
        "READY → AGENT → COMPLETE → TERMINATION"
    ),
    "DEPENDENCY_CHAIN": (
        "TASK → PLAN → READY → AGENT → READY → AGENT → READY → AGENT → "
        "COMPLETE → TERMINATION"
    ),
    "CONDITIONAL_BRANCH": (
        "TASK → PLAN → READY → AGENT → READY → AGENT → CONDITION → READY → "
        "CONDITION → SKIP → AGENT → COMPLETE → TERMINATION"
    ),
    "ORCHESTRATION_FAILURE": (
        "TASK → PLAN → READY → AGENT → FAILURE → SKIP → SKIP → TERMINATION"
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
        elif event.kind == "plan_created":
            view.append(
                {
                    "phase": "PLAN",
                    "planId": event.detail.get("planId"),
                    "nodeIds": event.detail.get("nodeIds"),
                }
            )
        elif event.kind == "node_ready":
            view.append(
                {
                    "phase": "READY",
                    "nodeId": event.detail.get("nodeId"),
                    "agentId": event.detail.get("agentId"),
                    "reason": event.detail.get("reason"),
                    "dependencyIds": event.detail.get("dependencyIds"),
                    "upstreamResultIds": event.detail.get("upstreamResultIds"),
                }
            )
        elif event.kind == "node_started":
            view.append(
                {
                    "phase": "NODE_STARTED",
                    "nodeId": event.detail.get("nodeId"),
                    "agentId": event.detail.get("agentId"),
                }
            )
        elif event.kind == "node_result":
            view.append(
                {
                    "phase": "AGENT",
                    "nodeId": event.detail.get("nodeId"),
                    "agentId": event.detail.get("agentId"),
                    "role": event.detail.get("role"),
                    "ok": event.detail.get("ok"),
                    "resultId": event.detail.get("resultId"),
                    "upstreamResultIds": event.detail.get("upstreamResultIds"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "node_completed":
            view.append(
                {
                    "phase": "NODE_COMPLETED",
                    "nodeId": event.detail.get("nodeId"),
                    "resultId": event.detail.get("resultId"),
                }
            )
        elif event.kind == "node_failed":
            view.append(
                {
                    "phase": "FAILURE",
                    "nodeId": event.detail.get("nodeId"),
                    "agentId": event.detail.get("agentId"),
                    "error": event.detail.get("error"),
                    "resultId": event.detail.get("resultId"),
                }
            )
        elif event.kind == "node_skipped":
            view.append(
                {
                    "phase": "SKIP",
                    "nodeId": event.detail.get("nodeId"),
                    "agentId": event.detail.get("agentId"),
                    "reason": event.detail.get("reason"),
                }
            )
        elif event.kind == "condition_evaluated":
            view.append(
                {
                    "phase": "CONDITION",
                    "nodeId": event.detail.get("nodeId"),
                    "sourceNodeId": event.detail.get("sourceNodeId"),
                    "field": event.detail.get("field"),
                    "expected": event.detail.get("expected"),
                    "actual": event.detail.get("actual"),
                    "selected": event.detail.get("selected"),
                    "sourceResultId": event.detail.get("sourceResultId"),
                }
            )
        elif event.kind == "completed":
            view.append(
                {
                    "phase": "COMPLETE",
                    "terminalNodeIds": event.detail.get("terminalNodeIds"),
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
        "plan_created": "Plan created",
        "node_ready": "Node ready",
        "node_started": "Node started",
        "node_result": "Node result",
        "node_completed": "Node completed",
        "node_failed": "Node failed",
        "node_skipped": "Node skipped",
        "condition_evaluated": "Condition evaluated",
        "completed": "Completed",
        "termination": "Termination",
        "error": "Error",
    }
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind in {"node_failed", "error"}:
            status = "error"
        if event.kind == "node_result" and event.detail.get("ok") is False:
            status = "error"
        if event.kind == "node_skipped":
            status = "skipped"
        step_type = "protocol"
        if event.kind in {"node_started", "node_result", "node_completed"}:
            step_type = "agent"
        if event.kind in {"node_ready", "node_skipped", "condition_evaluated"}:
            step_type = "orchestration"
        if event.kind == "plan_created":
            step_type = "plan"
        if event.kind in {"node_failed", "error"} or status == "error":
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
    result: OrchestrationRunResult,
    settings: Settings,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump()
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "runtimeMs": metrics["runtime_ms"],
        "agentMs": metrics["agent_ms"],
        "nodesReady": metrics["nodes_ready"],
        "nodesExecuted": metrics["nodes_executed"],
        "nodesCompleted": metrics["nodes_completed"],
        "nodesFailed": metrics["nodes_failed"],
        "nodesSkipped": metrics["nodes_skipped"],
        "terminationReason": metrics["termination_reason"],
        "maxNodes": metrics["max_nodes"],
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
    view = build_signature_view(result.sequence)
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
            "layout": "agent-orchestration",
            "stages": [
                "task",
                "plan",
                "ready",
                "execute",
                "condition",
                "complete-or-fail",
                "termination",
            ],
        },
        "input": {
            "request": case.request,
            "service": case.service,
            "config": {
                "maxNodes": result.metrics.max_nodes,
                "modelDriver": result.model_driver,
                "synthesizerModel": result.model,
                "clientName": "dataaihub-cookbook-agent-orchestration",
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
            "nodeStates": result.state.get("nodeStates"),
            "executionOrder": result.state.get("executionOrder"),
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": {
            "purpose": (
                "Frontend-friendly projection of observable orchestration "
                "events. Not a new measurement."
            ),
            "signatureView": view,
            "signatureFlow": signature_flow_from_view(view),
            "logicalParallelism": (
                "READY means independently schedulable. Independent nodes "
                "become READY in the same wave before either executes. "
                "This example does not use OS threads."
            ),
        },
        "relatedEntities": ["agents"],
        "relatedContent": ["agents", "orchestration", "workflow-runtime"],
        "cookbook": {"path": "examples/agents/08-agent-orchestration"},
    }
