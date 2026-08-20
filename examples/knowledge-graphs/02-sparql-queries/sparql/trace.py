"""Build Lab-oriented traces from measured SPARQL query runs."""

from __future__ import annotations

import time
from typing import Any

from config import EXAMPLE_ID, Settings
from sparql.cases import MeasuredCase
from sparql.graph import RdfGraphStore
from sparql.model import QueryRunResult

SIGNATURE_FLOWS = {
    "BASIC_SELECT": "QUESTION → QUERY PATTERN → BINDINGS → RESULT",
    "MULTI_PATTERN_QUERY": "PATTERN 1 → JOIN → PATTERN 2 → BINDINGS → RESULT",
    "FILTER_QUERY": "PATTERNS → FILTER → BINDINGS → RESULT",
    "NO_MATCH": "QUERY → EXECUTED → 0 BINDINGS → NO MATCH",
}


def build_signature_view(
    result: QueryRunResult,
    *,
    example_class: str,
) -> list[dict[str, Any]]:
    """Project observable SPARQL execution into a future Lab teaching view."""
    view: list[dict[str, Any]] = []
    view.append({"phase": "QUESTION", "text": result.question})

    if example_class == "BASIC_SELECT":
        view.append(
            {
                "phase": "QUERY_PATTERN",
                "patterns": result.patterns,
            }
        )
    elif example_class == "MULTI_PATTERN_QUERY":
        if len(result.patterns) >= 2:
            view.append({"phase": "PATTERN_1", "pattern": result.patterns[0]})
            view.append({"phase": "JOIN", "variable": "project"})
            view.append({"phase": "PATTERN_2", "pattern": result.patterns[2]})
    elif example_class == "FILTER_QUERY":
        view.append({"phase": "PATTERNS", "patterns": result.patterns[:4]})
        view.append({"phase": "FILTER", "expression": 'FILTER(?team = "platform")'})
    elif example_class == "NO_MATCH":
        view.append({"phase": "QUERY", "queryName": result.query_name})

    if result.metrics.result_rows == 0:
        view.append({"phase": "EXECUTED", "success": True})
        view.append({"phase": "ZERO_BINDINGS", "rowCount": 0})
        view.append({"phase": "NO_MATCH", "matches": []})
    else:
        view.append(
            {
                "phase": "BINDINGS",
                "rowCount": result.metrics.result_rows,
                "bindings": result.matches,
            }
        )
        view.append(
            {
                "phase": "RESULT",
                "matches": result.matches,
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
        "query_started": "Query started",
        "query_executed": "Query executed",
        "result_bindings": "Result bindings",
        "query_completed": "Query completed",
        "termination": "Termination",
    }
    steps: list[dict[str, Any]] = []
    for index, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "result_bindings" and event.detail.get("rowCount") == 0:
            status = "empty"
        if event.kind == "termination" and event.detail.get("reason") == "query_failed":
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
    result: QueryRunResult,
    settings: Settings,
    store: RdfGraphStore,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    metrics_out = {
        "queryExecutionMs": metrics["query_execution_ms"],
        "resultRows": metrics["result_rows"],
        "triplePatterns": metrics["triple_patterns"],
        "filterCount": metrics["filter_count"],
        "variables": metrics["variables"],
        "queryCase": metrics["query_case"],
        "bindingsReturned": metrics["bindings_returned"],
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

    presentation = {
        "purpose": (
            "Frontend-friendly projection of observable SPARQL operations. "
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
            "layout": "sparql-queries",
            "graphModel": "rdf",
            "executionEngine": "rdflib",
            "stages": [
                "question",
                "sparql-query",
                "triple-pattern",
                "bindings",
                "result",
            ],
        },
        "input": {
            "case": case.trace_id,
            "question": case.question,
            "queryName": result.query_name,
            "query": result.query,
            "prefixes": result.prefixes,
            "patterns": result.patterns,
            "config": {
                "graphPath": "data/graph.ttl",
                "graphFormat": "turtle",
                "namespace": "https://dataaihub.co/example/kg/",
                "maxResultRows": settings.max_result_rows,
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
        "relatedContent": ["knowledge-graph", "sparql-queries"],
        "cookbook": {"path": "examples/knowledge-graphs/02-sparql-queries"},
    }
