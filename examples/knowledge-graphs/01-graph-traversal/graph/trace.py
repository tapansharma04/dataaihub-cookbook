"""Build Lab-oriented traces from measured graph-traversal runs."""

from __future__ import annotations

import time
from typing import Any

from config import EXAMPLE_ID, Settings
from graph.cases import MeasuredCase
from graph.model import GraphRunResult, SequenceEvent
from graph.store import GraphStore, labeled_predicate

SIGNATURE_FLOWS = {
    "DIRECT_RELATIONSHIP": "ENTITY → RELATIONSHIP → ENTITY",
    "MULTI_HOP_TRAVERSAL": "ENTITY → RELATIONSHIP → ENTITY → RELATIONSHIP → ENTITY",
    "RELATIONSHIP_FILTER": "ENTITY ← RELATIONSHIP ← ENTITY",
    "NO_PATH": "START ENTITY → SEARCH → NO PATH",
}


def build_signature_view(
    result: GraphRunResult,
    *,
    example_class: str,
) -> list[dict[str, Any]]:
    """Project observable traversal into a future Lab teaching view.

    Presentation only. Not a measurement.
    """
    view: list[dict[str, Any]] = []
    start = result.start.public() if result.start is not None else None

    if example_class == "NO_PATH" or result.metrics.termination_reason == "no_path":
        hop = result.hops[0] if result.hops else None
        view.append({"phase": "START_ENTITY", "entity": start})
        view.append(
            {
                "phase": "SEARCH",
                "predicate": labeled_predicate(hop.predicate) if hop else None,
                "direction": hop.direction if hop else None,
            }
        )
        view.append({"phase": "NO_PATH", "pathFound": False})
        view.append(
            {
                "phase": "TERMINATION",
                "reason": result.metrics.termination_reason,
            }
        )
        return view

    if result.paths:
        path = result.paths[0]
        for index, entity in enumerate(path.entities):
            view.append({"phase": "ENTITY", "entity": entity.public()})
            if index < len(path.relationships):
                triple = path.relationships[index]
                hop = result.hops[index] if index < len(result.hops) else None
                view.append(
                    {
                        "phase": "RELATIONSHIP",
                        "predicate": labeled_predicate(triple.predicate),
                        "direction": hop.direction if hop else "outgoing",
                        "triple": {
                            "subject": triple.subject,
                            "predicate": labeled_predicate(triple.predicate),
                            "object": triple.object,
                        },
                    }
                )
    view.append(
        {
            "phase": "RESULT",
            "pathFound": result.metrics.path_found,
            "answerIds": [entity.id for entity in result.answers],
        }
    )
    view.append(
        {
            "phase": "TERMINATION",
            "reason": result.metrics.termination_reason,
        }
    )
    return view


def sequence_to_steps(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    titles = {
        "user_request": "User request",
        "graph_lookup": "Graph lookup",
        "traversal_started": "Traversal started",
        "traversal_step": "Traversal step",
        "relationship_match": "Relationship match",
        "traversal_completed": "Traversal completed",
        "result": "Result",
        "termination": "Termination",
    }
    steps: list[dict[str, Any]] = []
    for index, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "result" and event.detail.get("pathFound") is False:
            status = "empty"
        if event.kind == "termination" and event.detail.get("reason") not in {
            "completed",
            "no_path",
        }:
            status = "error"
        steps.append(
            {
                "id": f"step-{index}-{event.kind}",
                "type": event.kind,
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
    result: GraphRunResult,
    settings: Settings,
    store: GraphStore,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    # matchedRelationships is a presentation-friendly alias of relationshipsVisited.
    metrics_out = {
        "entitiesVisited": metrics["entities_visited"],
        "relationshipsVisited": metrics["relationships_visited"],
        "traversalDepth": metrics["traversal_depth"],
        "matchedRelationships": metrics["matched_relationships"],
        "pathFound": metrics["path_found"],
        "executionMs": metrics["execution_ms"],
        "terminationReason": metrics["termination_reason"],
        "maxDepth": metrics["max_depth"],
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

    presentation = {
        "purpose": (
            "Frontend-friendly projection of observable graph operations. "
            "Not a new measurement."
        ),
        "signatureFlow": SIGNATURE_FLOWS[case.example_class],
        "signatureView": build_signature_view(result, example_class=case.example_class),
    }

    return {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": "guided",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metricsProvenance": "measured",
        "provenance": {
            "model": "not_used",
            "tools": "measured",
            "metrics": "measured",
        },
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "graph-traversal",
            "graphModel": "rdf",
            "stages": [
                "entity",
                "relationship",
                "rdf-graph",
                "traversal",
                "evidence",
            ],
        },
        "input": {
            "case": case.trace_id,
            "question": case.question,
            "startId": case.start_id,
            "hops": [hop.model_dump() for hop in case.hops],
            "config": {
                "graphPath": "data/graph.ttl",
                "graphFormat": "turtle",
                "namespace": "https://dataaihub.co/example/kg/",
                "maxTraversalDepth": settings.max_traversal_depth,
                "modelDriver": "not_used",
            },
        },
        "graph": store.snapshot(),
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "output": result.output,
        "errors": result.errors,
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["knowledge-graph"],
        "relatedContent": ["knowledge-graph", "graph-traversal"],
        "cookbook": {"path": "examples/knowledge-graphs/01-graph-traversal"},
    }
