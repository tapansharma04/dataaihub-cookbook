"""Build Lab-oriented traces from measured AgentRunResult objects.

Separates measured fields from presentation/teaching metadata.
"""

from __future__ import annotations

import time
from typing import Any

from agent.cases import MeasuredCase
from agent.loop import tool_definitions_payload
from agent.schemas import AgentRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

STAGE_NOTES = {
    "MODEL_DECISION": "The model proposes the next observable action.",
    "TOOL_CALL": "The application validates and executes the requested tool.",
    "OBSERVATION": "The result becomes state available to the next model decision.",
    "LOOP": "The runtime feeds the observation back into the next decision.",
    "TERMINATION": (
        "The runtime stops when the task is complete or a safety boundary is reached."
    ),
    "MAX_TURNS": "A hard turn limit prevents an agent from running indefinitely.",
    "INVALID_ACTION": (
        "Unrecognized model actions are rejected by the application, not trusted."
    ),
    "SECURITY": (
        "The model proposes tools and arguments; the application enforces "
        "validation, allowlists, authorization, timeouts, and turn limits."
    ),
}


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable events into the Lab signature teaching view."""
    view: list[dict[str, Any]] = []
    saw_tool_call = False

    for event in sequence:
        if event.kind == "user_request":
            view.append(
                {
                    "phase": "USER_REQUEST",
                    "request": event.detail.get("request"),
                }
            )
        elif event.kind == "model_decision":
            decision = event.detail.get("decision")
            view.append(
                {
                    "phase": "MODEL_DECISION",
                    "turn": event.turn,
                    "decision": decision,
                    "content": event.detail.get("content"),
                    "toolCalls": event.detail.get("toolCalls") or [],
                    "latencyMs": event.latency_ms,
                    "note": STAGE_NOTES["MODEL_DECISION"],
                }
            )
        elif event.kind == "tool_call":
            saw_tool_call = True
            view.append(
                {
                    "phase": "TOOL_CALL",
                    "turn": event.turn,
                    "callId": event.detail.get("callId"),
                    "name": event.detail.get("name"),
                    "arguments": event.detail.get("arguments"),
                    "validatedArguments": event.detail.get("validatedArguments"),
                    "latencyMs": event.latency_ms,
                    "note": STAGE_NOTES["TOOL_CALL"],
                }
            )
        elif event.kind == "observation":
            view.append(
                {
                    "phase": "OBSERVATION",
                    "turn": event.turn,
                    "callId": event.detail.get("callId"),
                    "name": event.detail.get("name"),
                    "ok": event.detail.get("ok"),
                    "result": event.detail.get("result"),
                    "error": event.detail.get("error"),
                    "latencyMs": event.latency_ms,
                    "note": STAGE_NOTES["OBSERVATION"],
                }
            )
            view.append(
                {
                    "phase": "LOOP",
                    "turn": event.turn,
                    "fromCallId": event.detail.get("callId"),
                    "awaiting": "model_decision",
                    "note": STAGE_NOTES["LOOP"],
                }
            )
        elif event.kind == "final_answer":
            view.append(
                {
                    "phase": "FINAL_ANSWER",
                    "turn": event.turn,
                    "answer": event.detail.get("answer"),
                    "toolCallsUsed": saw_tool_call,
                }
            )
        elif event.kind == "termination":
            reason = event.detail.get("reason")
            phase_note = STAGE_NOTES["TERMINATION"]
            if reason == "max_turns":
                phase_note = STAGE_NOTES["MAX_TURNS"]
            elif reason == "invalid_action":
                phase_note = STAGE_NOTES["INVALID_ACTION"]
            view.append(
                {
                    "phase": "TERMINATION",
                    "turn": event.turn,
                    "reason": reason,
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
    for i, event in enumerate(sequence, start=1):
        step_type = {
            "user_request": "input",
            "model_decision": "llm",
            "tool_call": "tool_call",
            "observation": "observation",
            "final_answer": "output",
            "termination": "termination",
            "error": "error",
        }.get(event.kind, event.kind)
        title = {
            "user_request": "User request",
            "model_decision": "Model decision",
            "tool_call": "Tool call",
            "observation": "Observation",
            "final_answer": "Final answer",
            "termination": "Termination",
            "error": "Error",
        }.get(event.kind, event.kind)
        status = "ok"
        if event.kind == "observation" and event.detail.get("ok") is False:
            status = "error"
        if event.kind == "error":
            status = "error"
        if event.kind == "termination" and event.detail.get("reason") in {
            "max_turns",
            "invalid_action",
            "error",
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
                "type": step_type,
                "title": title,
                "status": status,
                "detail": event.detail,
                "metrics": {
                    "latencyMs": event.latency_ms,
                    "provenance": step_provenance,
                },
            }
        )
    return steps


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

    provenance = {
        "model": result.model_driver,
        "tools": "measured",
        "metrics": "measured",
    }

    presentation: dict[str, Any] = {
        "purpose": (
            "Frontend-friendly projection of observable agent-loop events for "
            "MODEL_DECISION → TOOL_CALL → OBSERVATION → LOOP → TERMINATION. "
            "Not a new measurement."
        ),
        "signatureView": signature,
        "stageNotes": STAGE_NOTES,
        "securityNote": STAGE_NOTES["SECURITY"],
    }
    if case.example_class == "MAX_TURNS":
        presentation["maxTurnsNote"] = STAGE_NOTES["MAX_TURNS"]
    if case.example_class == "INVALID_ACTION":
        presentation["invalidActionNote"] = (
            "The harness emitted an unrecognized action; the runtime terminated "
            "with invalid_action. This is not tool-level error recovery."
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
            "layout": "agent-loop",
            "stages": [
                "user-request",
                "model-decision",
                "tool-call",
                "observation",
                "loop",
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
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["openai"] if result.model_driver != "case-harness" else [],
        "relatedContent": ["agents", "agent-loop", "tool-calling"],
        "cookbook": {"path": "examples/agents/02-agent-loop"},
    }
