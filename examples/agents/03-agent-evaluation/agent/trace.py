"""Build Lab-oriented traces from measured runs plus computed evaluation.

Separates measured execution fields from teaching/presentation metadata.
Evaluation is attached as a sibling of the measured trace; it does not
rewrite sequence events, metrics, or tool observations.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from agent.cases import MeasuredCase
from agent.loop import tool_definitions_payload
from agent.schemas import AgentRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings
from evaluation.evaluator import evaluate_run
from evaluation.schemas import EvaluationResult

STAGE_NOTES = {
    "MODEL_DECISION": "The model proposes the next observable action.",
    "TOOL_CALL": "The application validates and executes the requested tool.",
    "OBSERVATION": "The result becomes state available to the next model decision.",
    "FINAL_ANSWER": "The model produces an observable final answer.",
    "TERMINATION": (
        "The runtime stops when the task is complete or a safety boundary is reached."
    ),
    "EVALUATION": (
        "A separate evaluator scores outcome and trajectory against explicit "
        "case criteria. Evaluation does not change the measured run."
    ),
    "SECURITY": (
        "The model proposes tools and arguments; the application enforces "
        "validation, allowlists, authorization, timeouts, and turn limits."
    ),
}


def build_signature_view(
    sequence: list[SequenceEvent],
    evaluation: EvaluationResult | None = None,
) -> list[dict[str, Any]]:
    """Project observable events into the Lab signature teaching view."""
    view: list[dict[str, Any]] = []

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
                    "toolCalls": event.detail.get("toolCalls") or [],
                    "latencyMs": event.latency_ms,
                    "note": STAGE_NOTES["MODEL_DECISION"],
                }
            )
        elif event.kind == "tool_call":
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
        elif event.kind == "final_answer":
            view.append(
                {
                    "phase": "FINAL_ANSWER",
                    "turn": event.turn,
                    "answer": event.detail.get("answer"),
                    "note": STAGE_NOTES["FINAL_ANSWER"],
                }
            )
        elif event.kind == "termination":
            view.append(
                {
                    "phase": "TERMINATION",
                    "turn": event.turn,
                    "reason": event.detail.get("reason"),
                    "detail": event.detail,
                    "note": STAGE_NOTES["TERMINATION"],
                }
            )
        elif event.kind == "error":
            view.append(
                {
                    "phase": "ERROR",
                    "detail": event.detail,
                }
            )

    if evaluation is not None:
        view.append(
            {
                "phase": "EVALUATION",
                "taskSuccess": evaluation.task_success,
                "finalAnswerCorrect": evaluation.final_answer_correct,
                "trajectorySuccess": evaluation.trajectory_success,
                "note": STAGE_NOTES["EVALUATION"],
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
    evaluation: EvaluationResult | None = None,
) -> dict[str, Any]:
    evaluated = evaluation or evaluate_run(result, case.criteria, case_id=case.trace_id)

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
            "detail": copy.deepcopy(e.detail),
            "latencyMs": e.latency_ms,
        }
        for e in result.sequence
    ]

    signature = build_signature_view(result.sequence, evaluated)

    provenance = {
        "model": result.model_driver,
        "tools": "measured",
        "metrics": "measured",
    }

    presentation: dict[str, Any] = {
        "purpose": (
            "Frontend-friendly projection of observable agent events plus "
            "computed evaluation. Evaluation is not a new measurement of the "
            "run and is not mixed into operational metrics."
        ),
        "signatureView": signature,
        "stageNotes": STAGE_NOTES,
        "securityNote": STAGE_NOTES["SECURITY"],
        "evaluationNote": STAGE_NOTES["EVALUATION"],
    }

    return {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": "guided",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metricsProvenance": "measured",
        "evaluationProvenance": "computed",
        "provenance": provenance,
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "agent-evaluation",
            "stages": [
                "user-request",
                "model-decision",
                "tool-call",
                "observation",
                "final-answer",
                "termination",
                "evaluation",
            ],
        },
        "input": {
            "request": case.request,
            "criteria": case.criteria.to_public_dict(),
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
        "evaluation": evaluated.to_public_dict(),
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["openai"] if result.model_driver != "case-harness" else [],
        "relatedContent": [
            "agents",
            "agent-evaluation",
            "agent-loop",
            "tool-calling",
        ],
        "cookbook": {"path": "examples/agents/03-agent-evaluation"},
    }
