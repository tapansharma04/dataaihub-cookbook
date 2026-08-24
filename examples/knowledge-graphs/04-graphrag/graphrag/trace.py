"""Build Lab-oriented traces from measured GraphRAG runs."""

from __future__ import annotations

import time
from typing import Any

from config import EXAMPLE_ID, Settings
from graphrag.cases import MeasuredCase
from graphrag.graph import RdfGraphStore
from graphrag.model import GraphRunResult

SIGNATURE_FLOWS = {
    "ENTITY_RETRIEVAL": "QUESTION → ENTITY → RELATIONSHIPS → SUBGRAPH → ANSWER",
    "MULTI_HOP_RETRIEVAL": "QUESTION → ENTITY → HOP 1 → HOP 2 → SUBGRAPH → ANSWER",
    "RELATIONSHIP_GROUNDED_ANSWER": "QUESTION → GRAPH FACTS → CONTEXT → ANSWER",
    "NO_RELEVANT_SUBGRAPH": (
        "QUESTION → ENTITY → SEARCH → NO RELEVANT SUBGRAPH → NO ANSWER"
    ),
}

LLM_SIGNATURE_SUFFIX = " → CONTEXT → LLM → ANSWER"


def build_signature_view(
    result: GraphRunResult,
    *,
    example_class: str,
) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    view.append({"phase": "QUESTION", "text": result.user_request})

    if result.resolved_entities:
        view.append(
            {
                "phase": "ENTITY",
                "entities": [
                    entity.model_dump() for entity in result.resolved_entities
                ],
            }
        )

    if example_class == "MULTI_HOP_RETRIEVAL" and result.paths:
        # Group path steps by measured hop position (not by path index).
        # Multiple edges at the same hop share one HOP_N phase.
        hop_count = max((len(path.steps) for path in result.paths), default=0)
        for hop in range(1, hop_count + 1):
            steps_at_hop: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for path in result.paths:
                if len(path.steps) < hop:
                    continue
                step = path.steps[hop - 1]
                key = (step.subject.iri, step.predicate.iri, step.object.iri)
                if key in seen:
                    continue
                seen.add(key)
                steps_at_hop.append(step.model_dump())
            if steps_at_hop:
                view.append({"phase": f"HOP_{hop}", "steps": steps_at_hop})

    if example_class == "RELATIONSHIP_GROUNDED_ANSWER" and result.subgraph:
        view.append(
            {
                "phase": "GRAPH_FACTS",
                "facts": [fact.public() for fact in result.subgraph],
            }
        )

    if example_class == "ENTITY_RETRIEVAL" and result.subgraph:
        view.append(
            {
                "phase": "RELATIONSHIPS",
                "facts": [fact.public() for fact in result.subgraph],
            }
        )

    if result.subgraph:
        view.append(
            {
                "phase": "SUBGRAPH",
                "tripleCount": len(result.subgraph),
                "facts": [fact.public() for fact in result.subgraph],
            }
        )
        if result.context:
            view.append({"phase": "CONTEXT", "facts": result.context})
    elif example_class == "NO_RELEVANT_SUBGRAPH":
        view.append(
            {"phase": "SEARCH", "hopsUsed": result.retrieval.get("hopsUsed", 0)}
        )
        view.append({"phase": "NO_RELEVANT_SUBGRAPH", "tripleCount": 0})

    if result.mode == "graphrag_llm" and result.metrics.model_turns:
        view.append({"phase": "LLM", "model": result.model})

    if result.answer:
        view.append({"phase": "ANSWER", "text": result.answer})
    elif example_class == "NO_RELEVANT_SUBGRAPH":
        view.append({"phase": "NO_ANSWER", "text": result.answer})

    view.append({"phase": "TERMINATION", "reason": result.termination})
    return view


def sequence_to_steps(sequence: list[Any]) -> list[dict[str, Any]]:
    titles = {
        "user_request": "User request",
        "entity_resolution": "Entity resolution",
        "retrieval_started": "Retrieval started",
        "retrieval_step": "Retrieval step",
        "subgraph_retrieved": "Subgraph retrieved",
        "context_assembled": "Context assembled",
        "model_request": "Model request",
        "model_response": "Model response",
        "final_answer": "Final answer",
        "termination": "Termination",
    }
    steps: list[dict[str, Any]] = []
    for index, event in enumerate(sequence, start=1):
        status = "ok"
        if event.kind == "termination" and event.detail.get("reason") != "completed":
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
    result: GraphRunResult,
    settings: Settings,
    store: RdfGraphStore,
) -> dict[str, Any]:
    metrics = result.metrics.model_dump(by_alias=True)
    metrics_out = {
        "entityCandidates": metrics["entity_candidates"],
        "resolvedEntityCount": metrics["resolved_entity_count"],
        "retrievalHops": metrics["retrieval_hops"],
        "entitiesRetrieved": metrics["entities_retrieved"],
        "relationshipsRetrieved": metrics["relationships_retrieved"],
        "subgraphTripleCount": metrics["subgraph_triple_count"],
        "contextFactCount": metrics["context_fact_count"],
        "retrievalExecutionMs": metrics["retrieval_execution_ms"],
        "contextAssemblyMs": metrics["context_assembly_ms"],
        "answerGenerationMs": metrics["answer_generation_ms"],
        "totalMs": metrics["total_ms"],
        "modelTurns": metrics["model_turns"],
        "terminationReason": metrics["termination_reason"],
    }

    signature_flow = SIGNATURE_FLOWS[case.example_class]
    if result.mode == "graphrag_llm":
        signature_flow = signature_flow.replace(" → ANSWER", LLM_SIGNATURE_SUFFIX)

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
            "Frontend-friendly projection of observable GraphRAG operations. "
            "Not a new measurement."
        ),
        "signatureFlow": signature_flow,
        "signatureView": build_signature_view(result, example_class=case.example_class),
    }

    architecture_stages = [
        "question",
        "entity-resolution",
        "graph-retrieval",
        "subgraph",
        "context",
    ]
    if result.mode == "graphrag_llm":
        architecture_stages.append("llm")
    architecture_stages.append("answer")

    trace: dict[str, Any] = {
        "labId": EXAMPLE_ID,
        "traceId": case.trace_id,
        "executionMode": result.mode,
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metricsProvenance": "measured",
        "provenance": result.provenance,
        "exampleClass": case.example_class,
        "selectionNote": case.selection_note,
        "architecture": {
            "layout": "graphrag",
            "graphModel": "rdf",
            "executionEngine": "rdflib",
            "stages": architecture_stages,
        },
        "input": {
            "case": case.trace_id,
            "mode": result.mode,
            "question": case.question,
            "config": {
                "graphPath": "data/graph.ttl",
                "graphFormat": "turtle",
                "namespace": "https://dataaihub.co/example/kg/",
                "maxHops": settings.max_hops,
                "entityResolution": "label-based",
            },
        },
        "graph": store.snapshot(),
        "resolvedEntities": [
            entity.model_dump() for entity in result.resolved_entities
        ],
        "retrieval": result.retrieval,
        "subgraph": [fact.public() for fact in result.subgraph],
        "paths": [
            {
                "steps": [
                    {
                        "subject": step.subject.model_dump(),
                        "predicate": step.predicate.model_dump(),
                        "object": step.object.model_dump(),
                    }
                    for step in path.steps
                ]
            }
            for path in result.paths
        ],
        "context": result.context,
        "answer": result.answer,
        "termination": result.termination,
        "sequence": sequence_payload,
        "steps": sequence_to_steps(result.sequence),
        "metrics": metrics_out,
        "presentation": presentation,
        "relatedEntities": ["knowledge-graph"],
        "relatedContent": ["knowledge-graph", "graphrag"],
        "cookbook": {"path": "examples/knowledge-graphs/04-graphrag"},
        "errors": result.errors,
    }

    if result.mode == "graphrag_llm":
        trace["model"] = result.model
        trace["provider"] = result.provider
        trace["modelLatencyMs"] = result.model_latency_ms

    return trace
