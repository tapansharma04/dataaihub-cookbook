"""Build Lab-oriented traces from measured AgentRunResult objects.

Separates measured fields from presentation metadata.
Presentation is derived from the measured sequence; it does not rewrite it.
"""

from __future__ import annotations

import time
from typing import Any

from agent.cases import MeasuredCase
from agent.loop import tool_definitions_payload
from agent.schemas import AgentRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

STAGE_NOTES = {
    "PLAN_CREATED": (
        "The application accepts a validated multi-step plan before execution."
    ),
    "STEP_ACTIVE": "The runtime marks the current plan step in_progress.",
    "STEP_COMPLETED": "The runtime records step completion against the plan.",
    "PLAN_REVISED": (
        "An observation invalidated remaining steps; plan vN+1 supersedes vN. "
        "The original plan remains in the trace."
    ),
    "PLAN_FAILED": (
        "A blocking observation prevented the remaining plan from completing. "
        "The plan is failed, not marked completed."
    ),
    "FINAL_PLAN": "The current plan version after execution stopped.",
    "FINAL_ANSWER": "The runtime records the observable final answer.",
    "TERMINATION": (
        "The runtime stops when the plan completes, fails, or a safety "
        "boundary is reached."
    ),
    "SECURITY": (
        "The model proposes plans, revisions, and answers; the application "
        "validates plan structure, allowed tools, state transitions, "
        "execution, and termination."
    ),
}


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable events into the Lab signature view."""
    view: list[dict[str, Any]] = []
    current_plan: dict[str, Any] | None = None

    for event in sequence:
        if event.kind == "user_request":
            view.append(
                {
                    "phase": "USER_REQUEST",
                    "request": event.detail.get("request"),
                }
            )
        elif event.kind == "model_decision":
            view.append(
                {
                    "phase": "MODEL_DECISION",
                    "turn": event.turn,
                    "decision": event.detail.get("decision"),
                    "content": event.detail.get("content"),
                    "proposedSteps": event.detail.get("proposedSteps") or [],
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "plan_created":
            current_plan = {
                "version": event.detail.get("version"),
                "status": event.detail.get("status"),
                "steps": event.detail.get("steps"),
            }
            view.append(
                {
                    "phase": "PLAN_CREATED",
                    "turn": event.turn,
                    "planId": event.detail.get("planId"),
                    "version": event.detail.get("version"),
                    "steps": event.detail.get("steps"),
                    "note": STAGE_NOTES["PLAN_CREATED"],
                }
            )
        elif event.kind == "plan_step_started":
            view.append(
                {
                    "phase": "STEP_ACTIVE",
                    "turn": event.turn,
                    "stepId": event.detail.get("stepId"),
                    "description": event.detail.get("description"),
                    "status": event.detail.get("status"),
                    "planVersion": event.detail.get("planVersion"),
                    "progress": event.detail.get("progress"),
                    "note": STAGE_NOTES["STEP_ACTIVE"],
                }
            )
        elif event.kind == "tool_call":
            view.append(
                {
                    "phase": "TOOL_CALL",
                    "turn": event.turn,
                    "callId": event.detail.get("callId"),
                    "stepId": event.detail.get("stepId"),
                    "name": event.detail.get("name"),
                    "arguments": event.detail.get("arguments"),
                    "validatedArguments": event.detail.get("validatedArguments"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "observation":
            view.append(
                {
                    "phase": "OBSERVATION",
                    "turn": event.turn,
                    "callId": event.detail.get("callId"),
                    "stepId": event.detail.get("stepId"),
                    "name": event.detail.get("name"),
                    "ok": event.detail.get("ok"),
                    "result": event.detail.get("result"),
                    "error": event.detail.get("error"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "plan_step_completed":
            view.append(
                {
                    "phase": "STEP_COMPLETED",
                    "turn": event.turn,
                    "stepId": event.detail.get("stepId"),
                    "status": event.detail.get("status"),
                    "planVersion": event.detail.get("planVersion"),
                    "progress": event.detail.get("progress"),
                    "note": STAGE_NOTES["STEP_COMPLETED"],
                }
            )
        elif event.kind == "plan_revised":
            current_plan = event.detail.get("revisedPlan")
            view.append(
                {
                    "phase": "PLAN_REVISED",
                    "turn": event.turn,
                    "fromVersion": event.detail.get("fromVersion"),
                    "toVersion": event.detail.get("toVersion"),
                    "reason": event.detail.get("reason"),
                    "originalPlan": event.detail.get("originalPlan"),
                    "revisedPlan": event.detail.get("revisedPlan"),
                    "completedStepIds": event.detail.get("completedStepIds"),
                    "skippedStepIds": event.detail.get("skippedStepIds"),
                    "addedStepIds": event.detail.get("addedStepIds"),
                    "observation": event.detail.get("observation"),
                    "supersedes": True,
                    "note": STAGE_NOTES["PLAN_REVISED"],
                }
            )
        elif event.kind == "final_answer":
            plan_status = event.detail.get("planStatus")
            if plan_status == "failed":
                view.append(
                    {
                        "phase": "PLAN_FAILED",
                        "turn": event.turn,
                        "planStatus": plan_status,
                        "planVersion": event.detail.get("planVersion"),
                        "note": STAGE_NOTES["PLAN_FAILED"],
                    }
                )
            view.append(
                {
                    "phase": "FINAL_PLAN",
                    "turn": event.turn,
                    "plan": current_plan,
                    "planStatus": plan_status,
                    "planVersion": event.detail.get("planVersion"),
                    "note": STAGE_NOTES["FINAL_PLAN"],
                }
            )
            view.append(
                {
                    "phase": "FINAL_ANSWER",
                    "turn": event.turn,
                    "answer": event.detail.get("answer"),
                }
            )
        elif event.kind == "termination":
            reason = event.detail.get("reason")
            phase_note = STAGE_NOTES["TERMINATION"]
            if reason == "plan_failed":
                phase_note = STAGE_NOTES["PLAN_FAILED"]
            view.append(
                {
                    "phase": "TERMINATION",
                    "turn": event.turn,
                    "reason": reason,
                    "planStatus": event.detail.get("planStatus"),
                    "planVersion": event.detail.get("planVersion"),
                    "progress": event.detail.get("progress"),
                    "detail": event.detail,
                    "note": phase_note,
                }
            )
        elif event.kind == "error":
            view.append(
                {
                    "phase": "ERROR",
                    "detail": event.detail,
                }
            )

    return view


def sequence_to_steps(
    sequence: list[SequenceEvent],
    *,
    model_driver: str,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    titles = {
        "user_request": "User request",
        "model_decision": "Model decision",
        "plan_created": "Plan created",
        "plan_step_started": "Plan step started",
        "tool_call": "Tool call",
        "observation": "Observation",
        "plan_step_completed": "Plan step completed",
        "plan_revised": "Plan revised",
        "final_answer": "Final answer",
        "termination": "Termination",
        "error": "Error",
    }
    types = {
        "user_request": "input",
        "model_decision": "llm",
        "plan_created": "plan_created",
        "plan_step_started": "plan_step_started",
        "tool_call": "tool_call",
        "observation": "observation",
        "plan_step_completed": "plan_step_completed",
        "plan_revised": "plan_revised",
        "final_answer": "output",
        "termination": "termination",
        "error": "error",
    }
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "observation" and event.detail.get("ok") is False:
            status = "error"
        if event.kind == "error":
            status = "error"
        if (
            event.kind == "plan_step_completed"
            and event.detail.get("status") == "failed"
        ):
            status = "error"
        if event.kind == "termination" and event.detail.get("reason") in {
            "max_turns",
            "invalid_action",
            "error",
            "plan_failed",
        }:
            status = "stopped"

        if event.kind == "model_decision":
            step_provenance = model_driver
        elif event.kind in {"tool_call", "observation"}:
            step_provenance = "measured"
        else:
            step_provenance = "measured"

        steps.append(
            {
                "id": f"step-{i}-{event.kind}",
                "type": types.get(event.kind, event.kind),
                "title": titles.get(event.kind, event.kind),
                "status": status,
                "detail": event.detail,
                "metrics": {
                    "latencyMs": event.latency_ms,
                    "provenance": step_provenance,
                },
            }
        )
    return steps


def _plan_comparison(result: AgentRunResult) -> dict[str, Any] | None:
    plans = result.state.get("plans") or []
    revisions = result.state.get("revisions") or []
    if not revisions or len(plans) < 2:
        return None
    original = plans[0]
    current = plans[-1]
    return {
        "originalPlan": original,
        "revisedPlan": current,
        "fromVersion": revisions[-1].get("fromVersion"),
        "toVersion": revisions[-1].get("toVersion"),
        "reason": revisions[-1].get("reason"),
        "supersedes": True,
    }


def build_trace(
    *,
    case: MeasuredCase,
    result: AgentRunResult,
    settings: Settings,
    max_turns: int,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump()
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "modelMs": metrics["model_ms"],
        "toolMs": metrics["tool_ms"],
        "modelTurns": metrics["model_turns"],
        "toolCalls": metrics["tool_calls"],
        "successfulToolCalls": metrics["successful_tool_calls"],
        "failedToolCalls": metrics["failed_tool_calls"],
        "terminationReason": metrics["termination_reason"],
        "maxTurns": metrics["max_turns"],
        "planSteps": metrics["plan_steps"],
        "completedSteps": metrics["completed_steps"],
        "skippedSteps": metrics["skipped_steps"],
        "failedSteps": metrics["failed_steps"],
        "planRevisions": metrics["plan_revisions"],
        "planVersion": metrics["plan_version"],
        "planStatus": metrics["plan_status"],
        "provenance": metrics["provenance"],
    }

    sequence_payload = [
        {
            "kind": e.kind,
            "turn": e.turn,
            "detail": e.detail,
            "latencyMs": e.latency_ms,
        }
        for e in result.sequence
    ]

    signature = build_signature_view(result.sequence)
    comparison = _plan_comparison(result)

    provenance = {
        "model": result.model_driver,
        "tools": "measured",
        "metrics": "measured",
    }

    presentation: dict[str, Any] = {
        "purpose": (
            "Frontend-friendly projection of observable planning events for "
            "PLAN_CREATED → STEP_ACTIVE → STEP_COMPLETED → PLAN_REVISED → "
            "FINAL_PLAN. Not a new measurement. Does not score plan quality."
        ),
        "signatureView": signature,
        "stageNotes": STAGE_NOTES,
        "securityNote": STAGE_NOTES["SECURITY"],
        "currentPlan": result.state.get("plan"),
        "planHistory": result.state.get("planHistory") or [],
        "plans": result.state.get("plans") or [],
    }
    if comparison is not None:
        presentation["planComparison"] = comparison
        presentation["revisionNote"] = STAGE_NOTES["PLAN_REVISED"]
    if case.example_class == "PLAN_FAILURE":
        presentation["failureNote"] = STAGE_NOTES["PLAN_FAILED"]
    if case.example_class == "PLAN_EXECUTION":
        presentation["progressNote"] = (
            "Step events include pending, in_progress, completed, and remaining."
        )

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
            "layout": "agent-planning",
            "stages": [
                "user-request",
                "plan-created",
                "plan-step-started",
                "tool-call",
                "observation",
                "plan-step-completed",
                "plan-revised",
                "final-answer",
                "termination",
            ],
        },
        "input": {
            "request": case.request,
            "config": {
                "chatModel": result.model,
                "modelDriver": result.model_driver,
                "maxTurns": max_turns,
                "maxToolCallsPerTurn": settings.max_tool_calls_per_turn,
                "toolTimeoutMs": settings.tool_timeout_ms,
            },
        },
        "state": result.state,
        "tools": tool_definitions_payload(result.tool_definitions),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(
            result.sequence,
            model_driver=result.model_driver,
        ),
        "output": {
            "answer": result.answer,
            "terminationReason": result.metrics.termination_reason,
            "planStatus": result.metrics.plan_status,
            "planVersion": result.metrics.plan_version,
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["openai"] if result.model_driver != "case-harness" else [],
        "relatedContent": ["agents", "agent-planning", "agent-loop", "tool-calling"],
        "cookbook": {"path": "examples/agents/04-planning"},
    }
