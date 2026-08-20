"""Build Lab-oriented traces from measured SPARQL UPDATE runs."""

from __future__ import annotations

import time
from typing import Any

from config import EXAMPLE_ID, Settings
from sparql.cases import MeasuredCase
from sparql.graph import RdfGraphStore
from sparql.model import UpdateRunResult
from sparql.queries import get_update

SIGNATURE_FLOWS = {
    "INSERT_DATA": "BEFORE → INSERT → AFTER → VERIFY",
    "INSERT_WHERE": "PATTERN → INSERT DERIVED TRIPLES → AFTER → VERIFY",
    "DELETE_DATA": "BEFORE → DELETE → AFTER → VERIFY",
    "UPDATE_AND_VERIFY": "BEFORE → DELETE + INSERT → AFTER → VERIFY",
}


def build_signature_view(
    result: UpdateRunResult,
    *,
    example_class: str,
) -> list[dict[str, Any]]:
    """Project observable UPDATE execution into a future Lab teaching view."""
    view: list[dict[str, Any]] = []
    view.append({"phase": "QUESTION", "text": result.question})
    view.append(
        {
            "phase": "BEFORE",
            "triples": [t.public() for t in result.before_state],
            "tripleCount": len(result.before_state),
        }
    )

    if example_class == "INSERT_DATA":
        view.append(
            {
                "phase": "INSERT",
                "updateQuery": result.update_query,
                "insertedTriples": [t.public() for t in result.inserted_triples],
            }
        )
    elif example_class == "INSERT_WHERE":
        view.append(
            {
                "phase": "PATTERN",
                "patterns": [
                    "?person ex:worksOn ?project .",
                    "?project ex:uses ?technology .",
                ],
            }
        )
        view.append(
            {
                "phase": "INSERT_DERIVED_TRIPLES",
                "updateQuery": result.update_query,
                "insertedTriples": [t.public() for t in result.inserted_triples],
            }
        )
    elif example_class == "DELETE_DATA":
        view.append(
            {
                "phase": "DELETE",
                "updateQuery": result.update_query,
                "deletedTriples": [t.public() for t in result.deleted_triples],
            }
        )
    elif example_class == "UPDATE_AND_VERIFY":
        view.append(
            {
                "phase": "DELETE_PLUS_INSERT",
                "updateQuery": result.update_query,
                "deletedTriples": [t.public() for t in result.deleted_triples],
                "insertedTriples": [t.public() for t in result.inserted_triples],
            }
        )

    view.append(
        {
            "phase": "AFTER",
            "triples": [t.public() for t in result.after_state],
            "tripleCount": len(result.after_state),
        }
    )
    view.append(
        {
            "phase": "VERIFY",
            "verificationQuery": result.verification_query,
            "rowCount": result.metrics.verification_rows,
            "bindings": [row.public() for row in result.verification_bindings],
        }
    )
    view.append(
        {
            "phase": "TERMINATION",
            "reason": result.metrics.termination_reason,
        }
    )
    return view


def sequence_to_steps(sequence: list[Any]) -> list[dict[str, Any]]:
    titles = {
        "user_request": "User request",
        "update_started": "Update started",
        "update_executed": "Update executed",
        "graph_state": "Graph state",
        "verification_started": "Verification started",
        "verification_result": "Verification result",
        "update_completed": "Update completed",
        "termination": "Termination",
    }
    steps: list[dict[str, Any]] = []
    for index, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "verification_result" and event.detail.get("rowCount") == 0:
            # Zero rows can be the expected DELETE_DATA verification outcome.
            status = "empty"
        if event.kind == "termination" and event.detail.get("reason") in {
            "update_failed",
            "verification_failed",
            "update_rejected",
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
    result: UpdateRunResult,
    settings: Settings,
    store: RdfGraphStore,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    metrics_out = {
        "updateExecutionMs": metrics["update_execution_ms"],
        "verificationExecutionMs": metrics["verification_execution_ms"],
        "insertedTripleCount": metrics["inserted_triple_count"],
        "deletedTripleCount": metrics["deleted_triple_count"],
        "beforeTripleCount": metrics["before_triple_count"],
        "afterTripleCount": metrics["after_triple_count"],
        "verificationRows": metrics["verification_rows"],
        "updateType": metrics["update_type"],
        "verificationQueryCount": metrics["verification_query_count"],
        "terminationReason": metrics["termination_reason"],
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

    predefined = get_update(case.update_name)
    presentation = {
        "purpose": (
            "Frontend-friendly projection of observable SPARQL UPDATE operations. "
            "Not a new measurement."
        ),
        "signatureFlow": SIGNATURE_FLOWS[case.example_class],
        "signatureView": build_signature_view(result, example_class=case.example_class),
        "teachingPoint": predefined.teaching_point,
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
            "layout": "sparql-updates",
            "graphModel": "rdf",
            "executionEngine": "rdflib",
            "stages": [
                "before",
                "sparql-update",
                "after",
                "verification",
            ],
        },
        "input": {
            "case": case.trace_id,
            "question": case.question,
            "updateName": result.update_name,
            "updateQuery": result.update_query,
            "verificationQuery": result.verification_query,
            "prefixes": result.prefixes,
            "config": {
                "graphPath": "data/graph.ttl",
                "graphFormat": "turtle",
                "namespace": "https://dataaihub.co/example/kg/",
                "maxResultRows": settings.max_result_rows,
                "modelDriver": "not_used",
                "freshGraphPerCase": True,
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
        "relatedContent": ["knowledge-graph", "sparql-updates"],
        "cookbook": {"path": "examples/knowledge-graphs/03-sparql-updates"},
    }
