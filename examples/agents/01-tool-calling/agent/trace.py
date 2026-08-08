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


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable events into the Lab signature teaching view.

    Phases: THINK_DECIDE → TOOL_CALL → OBSERVATION → NEXT_ACTION → FINAL_ANSWER
    Absence of tool calls is explicit for direct-answer runs.
    """
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
        elif event.kind == "model_turn":
            decision = event.detail.get("decision")
            view.append(
                {
                    "phase": "THINK_DECIDE",
                    "turn": event.turn,
                    "decision": decision,
                    "content": event.detail.get("content"),
                    "toolCalls": event.detail.get("toolCalls") or [],
                    "latencyMs": event.latency_ms,
                    "note": (
                        "Observable model output only — no hidden chain-of-thought."
                    ),
                }
            )
            if decision == "final_answer":
                view.append(
                    {
                        "phase": "FINAL_ANSWER",
                        "turn": event.turn,
                        "answer": event.detail.get("content") or "",
                        "toolCallsUsed": saw_tool_call,
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
                }
            )
            view.append(
                {
                    "phase": "NEXT_ACTION",
                    "turn": event.turn,
                    "fromCallId": event.detail.get("callId"),
                    "awaiting": "model_turn",
                }
            )
        elif event.kind == "final_answer":
            if not any(v.get("phase") == "FINAL_ANSWER" for v in view):
                view.append(
                    {
                        "phase": "FINAL_ANSWER",
                        "turn": event.turn,
                        "answer": event.detail.get("answer"),
                        "toolCallsUsed": saw_tool_call,
                    }
                )
        elif event.kind == "error":
            view.append(
                {
                    "phase": "ERROR",
                    "detail": event.detail,
                }
            )

    # Direct-answer runs: make the absent tool call explicit.
    if not saw_tool_call:
        # Insert after the first THINK_DECIDE when present.
        insert_at = next(
            (i + 1 for i, v in enumerate(view) if v.get("phase") == "THINK_DECIDE"),
            len(view),
        )
        view.insert(
            insert_at,
            {
                "phase": "TOOL_CALL",
                "skipped": True,
                "reason": "Model answered without selecting a tool",
            },
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
            "model_turn": "llm",
            "tool_call": "tool_call",
            "observation": "observation",
            "final_answer": "output",
            "error": "error",
        }.get(event.kind, event.kind)
        title = {
            "user_request": "User request",
            "model_turn": "Model decide",
            "tool_call": "Tool call",
            "observation": "Tool observation",
            "final_answer": "Final answer",
            "error": "Error",
        }.get(event.kind, event.kind)
        status = "ok"
        if event.kind == "observation" and event.detail.get("ok") is False:
            status = "error"
        if event.kind == "error":
            status = "error"
        # Model turns follow the driver (case-harness ≠ live LLM measurement).
        # Tool calls / observations are real executor timings.
        if event.kind == "model_turn":
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
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    # camelCase for Lab JSON consistency with other examples
    metrics_out = {
        "totalMs": metrics["total_ms"],
        "modelMs": metrics["model_ms"],
        "toolMs": metrics["tool_ms"],
        "modelTurns": metrics["model_turns"],
        "toolCalls": metrics["tool_calls"],
        "successfulToolCalls": metrics["successful_tool_calls"],
        "failedToolCalls": metrics["failed_tool_calls"],
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

    # Explicit source split: harness model turns vs real tool execution vs
    # recorded run metrics. Keeps metricsProvenance for Lab/RAG convention.
    provenance = {
        "model": result.model_driver,
        "tools": "measured",
        "metrics": "measured",
    }

    presentation: dict[str, Any] = {
        "purpose": (
            "Frontend-friendly projection of observable model/tool events "
            "for the THINK/DECIDE → TOOL → OBSERVE → NEXT → ANSWER view. "
            "Not a new measurement."
        ),
        "signatureView": signature,
    }
    if case.example_class == "ERROR_RECOVERY":
        presentation["errorRecoveryNote"] = (
            "The trace demonstrates an error → corrected tool-call loop; the "
            "correction is supplied by the reproducible case harness."
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
            "layout": "tool-calling-loop",
            "stages": [
                "user-request",
                "model-decide",
                "tool-call",
                "observation",
                "next-action",
                "final-answer",
            ],
        },
        "input": {
            "request": case.request,
            "config": {
                "chatModel": result.model,
                "modelDriver": result.model_driver,
                "maxModelTurns": settings.max_model_turns,
                "maxToolCallsPerTurn": settings.max_tool_calls_per_turn,
                "toolTimeoutMs": settings.tool_timeout_ms,
            },
        },
        "tools": tool_definitions_payload(result.tool_definitions),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(
            result.sequence,
            model_driver=result.model_driver,
        ),
        "output": {"answer": result.answer},
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["openai"] if result.model_driver != "case-harness" else [],
        "relatedContent": ["agents", "tool-calling"],
        "cookbook": {"path": "examples/agents/01-tool-calling"},
    }
