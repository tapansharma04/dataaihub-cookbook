"""Build Lab-oriented traces from measured AgentRunResult objects.

Separates measured fields from presentation metadata.
Presentation is derived from the measured sequence; it does not rewrite it.
"""

from __future__ import annotations

import time
from typing import Any

from agent.cases import MeasuredCase
from agent.schemas import AgentRunResult, SequenceEvent
from config import EXAMPLE_ID, Settings

STAGE_NOTES = {
    "MEMORY_WRITE": (
        "The application validates and stores a memory record. Information "
        "does not persist automatically just because the model saw it."
    ),
    "MEMORY_RETRIEVE": (
        "The runtime retrieves a scoped memory record for a later interaction."
    ),
    "MEMORY_MISS": (
        "No matching record exists for this scope and key. A miss is a "
        "normal observable state, not a memory system failure."
    ),
    "MEMORY_STALE": (
        "Stored memory exists but differs from the current authoritative "
        "source. The current source is preferred."
    ),
    "CURRENT_SOURCE": (
        "The current authoritative preference, distinct from stored memory."
    ),
    "MEMORY_USED": (
        "The later answer uses information that originated in stored memory."
    ),
    "INTERACTION_BOUNDARY": (
        "A later interaction starts. Conversation history is not carried "
        "forward; recalled information comes from the memory store."
    ),
    "FINAL_ANSWER": "The runtime records the observable final answer.",
    "TERMINATION": (
        "The runtime stops when the session completes or a safety boundary is reached."
    ),
    "SECURITY": (
        "The model proposes store or retrieve operations; the application "
        "validates scope, key, value, provenance, and performs the write "
        "or read. Memory from one scope never leaks into another."
    ),
}


def build_signature_view(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    """Project observable events into the Lab signature view."""
    view: list[dict[str, Any]] = []
    seen_user_request = False

    for event in sequence:
        iid = event.interaction_id or event.detail.get("interactionId")
        if event.kind == "user_request":
            if seen_user_request:
                view.append(
                    {
                        "phase": "INTERACTION_BOUNDARY",
                        "interactionId": iid,
                        "note": STAGE_NOTES["INTERACTION_BOUNDARY"],
                    }
                )
            seen_user_request = True
            view.append(
                {
                    "phase": "USER_REQUEST",
                    "interactionId": iid,
                    "request": event.detail.get("request"),
                    "scope": event.detail.get("scope"),
                }
            )
        elif event.kind == "model_decision":
            view.append(
                {
                    "phase": "MODEL_DECISION",
                    "turn": event.turn,
                    "interactionId": iid,
                    "decision": event.detail.get("decision"),
                    "content": event.detail.get("content"),
                    "latencyMs": event.latency_ms,
                }
            )
        elif event.kind == "memory_write_requested":
            view.append(
                {
                    "phase": "MEMORY_WRITE_REQUESTED",
                    "turn": event.turn,
                    "interactionId": iid,
                    "scope": event.detail.get("scope"),
                    "key": event.detail.get("key"),
                    "value": event.detail.get("value"),
                    "source": event.detail.get("source"),
                }
            )
        elif event.kind == "memory_stored":
            view.append(
                {
                    "phase": "MEMORY_WRITE",
                    "turn": event.turn,
                    "interactionId": iid,
                    "id": event.detail.get("id"),
                    "scope": event.detail.get("scope"),
                    "key": event.detail.get("key"),
                    "value": event.detail.get("value"),
                    "source": event.detail.get("source"),
                    "version": event.detail.get("version"),
                    "createdAt": event.detail.get("createdAt"),
                    "updatedAt": event.detail.get("updatedAt"),
                    "note": STAGE_NOTES["MEMORY_WRITE"],
                }
            )
        elif event.kind == "memory_retrieval_requested":
            view.append(
                {
                    "phase": "MEMORY_RETRIEVE_REQUESTED",
                    "turn": event.turn,
                    "interactionId": iid,
                    "scope": event.detail.get("scope"),
                    "key": event.detail.get("key"),
                }
            )
        elif event.kind == "memory_retrieved":
            record = event.detail.get("record") or {}
            view.append(
                {
                    "phase": "MEMORY_RETRIEVE",
                    "turn": event.turn,
                    "interactionId": iid,
                    "scope": event.detail.get("scope"),
                    "key": event.detail.get("key"),
                    "record": record,
                    "note": STAGE_NOTES["MEMORY_RETRIEVE"],
                }
            )
            view.append(
                {
                    "phase": "MEMORY_USED",
                    "turn": event.turn,
                    "interactionId": iid,
                    "record": record,
                    "note": STAGE_NOTES["MEMORY_USED"],
                }
            )
        elif event.kind == "memory_not_found":
            view.append(
                {
                    "phase": "MEMORY_MISS",
                    "turn": event.turn,
                    "interactionId": iid,
                    "scope": event.detail.get("scope"),
                    "key": event.detail.get("key"),
                    "found": False,
                    "note": STAGE_NOTES["MEMORY_MISS"],
                }
            )
        elif event.kind == "observation":
            freshness = event.detail.get("freshness") or {}
            stale = bool(event.detail.get("staleMemoryDetected"))
            view.append(
                {
                    "phase": "CURRENT_SOURCE",
                    "turn": event.turn,
                    "interactionId": iid,
                    "current": event.detail.get("current"),
                    "freshness": freshness,
                    "note": STAGE_NOTES["CURRENT_SOURCE"],
                }
            )
            if stale:
                view.append(
                    {
                        "phase": "MEMORY_STALE",
                        "turn": event.turn,
                        "interactionId": iid,
                        "freshness": freshness,
                        "resolution": event.detail.get("resolution"),
                        "note": STAGE_NOTES["MEMORY_STALE"],
                    }
                )
        elif event.kind == "final_answer":
            view.append(
                {
                    "phase": "FINAL_ANSWER",
                    "turn": event.turn,
                    "interactionId": iid,
                    "answer": event.detail.get("answer"),
                    "note": STAGE_NOTES["FINAL_ANSWER"],
                }
            )
        elif event.kind == "termination":
            view.append(
                {
                    "phase": "TERMINATION",
                    "turn": event.turn,
                    "interactionId": iid,
                    "reason": event.detail.get("reason"),
                    "detail": event.detail,
                    "note": STAGE_NOTES["TERMINATION"],
                }
            )
        elif event.kind == "error":
            view.append(
                {
                    "phase": "ERROR",
                    "interactionId": iid,
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
        "memory_write_requested": "Memory write requested",
        "memory_stored": "Memory stored",
        "memory_retrieval_requested": "Memory retrieval requested",
        "memory_retrieved": "Memory retrieved",
        "memory_not_found": "Memory not found",
        "observation": "Observation",
        "final_answer": "Final answer",
        "termination": "Termination",
        "error": "Error",
    }
    types = {
        "user_request": "input",
        "model_decision": "llm",
        "memory_write_requested": "memory_write_requested",
        "memory_stored": "memory_stored",
        "memory_retrieval_requested": "memory_retrieval_requested",
        "memory_retrieved": "memory_retrieved",
        "memory_not_found": "memory_not_found",
        "observation": "observation",
        "final_answer": "output",
        "termination": "termination",
        "error": "error",
    }
    for i, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "memory_not_found":
            status = "ok"
        if event.kind == "error":
            status = "error"
        if event.kind == "termination" and event.detail.get("reason") in {
            "max_turns",
            "invalid_action",
            "error",
        }:
            status = "stopped"
        if event.kind == "observation" and event.detail.get("staleMemoryDetected"):
            status = "ok"

        if event.kind == "model_decision":
            step_provenance = model_driver
        else:
            step_provenance = "measured"

        steps.append(
            {
                "id": f"step-{i}-{event.kind}",
                "type": types.get(event.kind, event.kind),
                "title": titles.get(event.kind, event.kind),
                "status": status,
                "detail": {
                    **event.detail,
                    "interactionId": event.interaction_id
                    or event.detail.get("interactionId"),
                },
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
        "memoryWrites": metrics["memory_writes"],
        "memoryReads": metrics["memory_reads"],
        "memoryHits": metrics["memory_hits"],
        "memoryMisses": metrics["memory_misses"],
        "memoryScope": metrics["memory_scope"],
        "memoryVersion": metrics["memory_version"],
        "staleMemoryDetected": metrics["stale_memory_detected"],
        "provenance": metrics["provenance"],
    }

    sequence_payload = [
        {
            "kind": e.kind,
            "turn": e.turn,
            "interactionId": e.interaction_id,
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
            "Frontend-friendly projection of observable memory events for "
            "STORE → RETRIEVE → USE, including MEMORY_MISS and MEMORY_STALE. "
            "Not a new measurement. Does not score memory quality."
        ),
        "signatureView": signature,
        "stageNotes": STAGE_NOTES,
        "securityNote": STAGE_NOTES["SECURITY"],
        "memoryRecords": result.state.get("memoryRecords") or [],
        "interactions": result.state.get("interactions") or [],
        "freshness": result.state.get("freshness") or [],
    }
    if case.example_class == "RECALL":
        presentation["recallNote"] = STAGE_NOTES["MEMORY_USED"]
    if case.example_class == "NO_MEMORY":
        presentation["missNote"] = STAGE_NOTES["MEMORY_MISS"]
    if case.example_class == "STALE_MEMORY":
        presentation["staleNote"] = STAGE_NOTES["MEMORY_STALE"]

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
            "layout": "agent-memory",
            "stages": [
                "user-request",
                "memory-write-requested",
                "memory-stored",
                "memory-retrieval-requested",
                "memory-retrieved",
                "memory-not-found",
                "observation",
                "final-answer",
                "termination",
            ],
        },
        "input": {
            "scope": case.scope,
            "request": case.interactions[-1].request,
            "interactions": [
                {
                    "id": f"interaction-{index}",
                    "request": item.request,
                }
                for index, item in enumerate(case.interactions, start=1)
            ],
            "config": {
                "chatModel": result.model,
                "modelDriver": result.model_driver,
                "maxTurns": max_turns,
                "toolTimeoutMs": settings.tool_timeout_ms,
            },
        },
        "state": result.state,
        "tools": [op.model_dump() for op in result.memory_operations],
        "sequence": sequence_payload,
        "steps": sequence_to_steps(
            result.sequence,
            model_driver=result.model_driver,
        ),
        "output": {
            "answer": result.answer,
            "terminationReason": result.metrics.termination_reason,
            "memoryScope": result.metrics.memory_scope,
            "memoryVersion": result.metrics.memory_version,
            "staleMemoryDetected": result.metrics.stale_memory_detected,
        },
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["openai"] if result.model_driver != "case-harness" else [],
        "relatedContent": [
            "agents",
            "agent-memory",
            "agent-planning",
            "agent-loop",
            "tool-calling",
        ],
        "cookbook": {"path": "examples/agents/05-memory"},
    }
