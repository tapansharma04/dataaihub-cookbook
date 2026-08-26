"""Build Lab-oriented traces from measured graph-construction runs."""

from __future__ import annotations

from typing import Any

from config import EXAMPLE_ID, FIXED_RECORDED_AT, Settings
from graph.builder import RdfGraphStore
from graph.cases import MeasuredCase
from graph.model import ConstructionResult

SIGNATURE_FLOWS = {
    "ENTITY_EXTRACTION": (
        "SOURCE → EXTRACTION → VALIDATION → ENTITY RESOLUTION → RDF GRAPH"
    ),
    "RELATIONSHIP_EXTRACTION": (
        "SOURCE → EXTRACTION → VALIDATION → RDF TRIPLE → RDF GRAPH"
    ),
    "ENTITY_LINKING": (
        "SOURCE → EXTRACTION → LABEL → STABLE IRI → RDF TRIPLE → RDF GRAPH"
    ),
    "INVALID_FACT": ("SOURCE → EXTRACTION → VALIDATION REJECTED → GRAPH UNCHANGED"),
}

LLM_SIGNATURE_PREFIX = "SOURCE → LLM EXTRACTION"


def build_signature_view(
    result: ConstructionResult,
    *,
    example_class: str,
) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    view.append({"phase": "SOURCE", "text": result.source_text})

    view.append(
        {
            "phase": "EXTRACTION",
            "mode": result.mode,
            "entities": [e.model_dump() for e in result.proposal.entities],
            "relationships": [r.model_dump() for r in result.proposal.relationships],
        }
    )

    if example_class == "ENTITY_LINKING" and result.validation.resolved_entities:
        view.append(
            {
                "phase": "ENTITY_LINKING",
                "links": [
                    {"label": entity.label, "iri": entity.iri}
                    for entity in result.validation.resolved_entities
                ],
            }
        )
    elif result.validation.resolved_entities:
        view.append(
            {
                "phase": "ENTITY_RESOLUTION",
                "entities": [
                    {
                        "label": entity.label,
                        "iri": entity.iri,
                        "entityType": entity.entity_type,
                    }
                    for entity in result.validation.resolved_entities
                ],
            }
        )

    if result.validation.rejected_relationships:
        view.append(
            {
                "phase": "VALIDATION_REJECTED",
                "rejected": [
                    r.model_dump() for r in result.validation.rejected_relationships
                ],
            }
        )
    elif result.validation.accepted_relationships:
        view.append(
            {
                "phase": "VALIDATION",
                "accepted": [
                    r.model_dump() for r in result.validation.accepted_relationships
                ],
            }
        )
    elif example_class == "ENTITY_EXTRACTION":
        view.append(
            {
                "phase": "VALIDATION",
                "resolvedCount": len(result.validation.resolved_entities),
            }
        )

    if result.triples_created:
        view.append(
            {
                "phase": "RDF_GRAPH",
                "triples": [t.model_dump() for t in result.triples_created],
                "tripleCount": result.graph_after_count,
            }
        )
    elif example_class == "INVALID_FACT":
        view.append(
            {
                "phase": "GRAPH_UNCHANGED",
                "tripleCount": result.graph_after_count,
                "before": result.graph_before_count,
                "after": result.graph_after_count,
            }
        )

    view.append({"phase": "TERMINATION", "reason": result.termination})
    return view


def sequence_to_steps(sequence: list[Any]) -> list[dict[str, Any]]:
    titles = {
        "source_loaded": "Source loaded",
        "extraction_started": "Extraction started",
        "entity_proposed": "Entity proposed",
        "relationship_proposed": "Relationship proposed",
        "validation_started": "Validation started",
        "validation_passed": "Validation passed",
        "validation_rejected": "Validation rejected",
        "entity_resolved": "Entity resolved",
        "triple_created": "Triple created",
        "graph_committed": "Graph committed",
        "graph_verified": "Graph verified",
        "result": "Result",
        "termination": "Termination",
    }
    steps: list[dict[str, Any]] = []
    for index, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "validation_rejected":
            status = "rejected"
        elif event.kind == "termination" and event.detail.get("reason") != "completed":
            status = "empty"
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
    result: ConstructionResult,
    settings: Settings,
    store: RdfGraphStore,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    metrics = result.metrics
    metrics_out = {
        "sourceCharacters": metrics.source_characters,
        "entitiesProposed": metrics.entities_proposed,
        "entitiesResolved": metrics.entities_resolved,
        "relationshipsProposed": metrics.relationships_proposed,
        "relationshipsAccepted": metrics.relationships_accepted,
        "relationshipsRejected": metrics.relationships_rejected,
        "triplesCreated": metrics.triples_created,
        "triplesRejected": metrics.triples_rejected,
        "graphTripleCount": metrics.graph_triple_count,
        "validationErrors": metrics.validation_errors,
        "modelTurns": metrics.model_turns,
        "totalMs": metrics.total_ms,
        "modelMs": metrics.model_ms,
        "terminationReason": metrics.termination_reason,
    }

    signature_flow = SIGNATURE_FLOWS[case.example_class]
    if result.mode == "llm_assisted":
        signature_flow = signature_flow.replace(
            "SOURCE → EXTRACTION",
            LLM_SIGNATURE_PREFIX,
        )

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
            "Frontend-friendly projection of observable graph-construction "
            "operations. Not a new measurement."
        ),
        "signatureFlow": signature_flow,
        "signatureView": build_signature_view(result, example_class=case.example_class),
    }

    architecture_stages = [
        "source",
        "extraction",
        "validation",
        "entity-resolution",
        "rdf-mapping",
        "graph-commit",
    ]
    if result.mode == "llm_assisted":
        architecture_stages[1] = "llm-extraction"

    timestamp = recorded_at
    if timestamp is None:
        if result.mode == "structured":
            timestamp = FIXED_RECORDED_AT
        else:
            import time

            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    trace: dict[str, Any] = {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": result.mode,
        "recordedAt": timestamp,
        "metricsProvenance": "measured",
        "provenance": result.provenance,
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "graph-construction",
            "graphModel": "rdf",
            "executionEngine": "rdflib",
            "stages": architecture_stages,
        },
        "input": {
            "case": case.trace_id,
            "mode": result.mode,
            "sourceText": case.source_text,
            "config": {
                "graphPath": "data/graph.ttl",
                "graphFormat": "turtle",
                "namespace": "https://dataaihub.co/example/kg/",
                "startGraph": case.start_graph,
                "entityResolution": "registry",
            },
        },
        "proposal": result.proposal.model_dump(),
        "validation": {
            "resolvedEntities": [
                e.model_dump() for e in result.validation.resolved_entities
            ],
            "unresolvedLabels": result.validation.unresolved_labels,
            "acceptedRelationships": [
                r.model_dump() for r in result.validation.accepted_relationships
            ],
            "rejectedRelationships": [
                r.model_dump() for r in result.validation.rejected_relationships
            ],
            "entityTypeErrors": result.validation.entity_type_errors,
        },
        "triplesCreated": [t.model_dump() for t in result.triples_created],
        "graph": store.snapshot(),
        "graphBeforeCount": result.graph_before_count,
        "graphAfterCount": result.graph_after_count,
        "graphUnchanged": result.graph_unchanged,
        "termination": result.termination,
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["knowledge-graph"],
        "relatedContent": ["knowledge-graph", "graph-construction"],
        "cookbook": {"path": "examples/knowledge-graphs/05-graph-construction"},
        "errors": result.errors,
    }

    if result.mode == "llm_assisted" and result.metrics.model_turns:
        trace["model"] = result.model
        trace["provider"] = result.provider
        trace["modelLatencyMs"] = result.model_latency_ms

    return trace
