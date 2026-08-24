"""Execute GraphRAG cases in graph-grounded or LLM modes."""

from __future__ import annotations

import time
from typing import Any, Literal

from config import Settings
from graphrag.answer import INSUFFICIENT_EVIDENCE, generate_deterministic_answer
from graphrag.cases import MeasuredCase
from graphrag.context import assemble_context
from graphrag.entity_resolution import resolve_entities
from graphrag.graph import RdfGraphStore
from graphrag.llm import LLMClient, build_user_prompt
from graphrag.model import (
    GraphFact,
    GraphRunMetrics,
    GraphRunResult,
    Mode,
    RetrievalPath,
    SequenceEvent,
)
from graphrag.retrieval import retrieve_subgraph, validate_retrieval_config

ModeArg = Literal["graph_grounded", "graphrag_llm"]


def _public_facts(facts: list[GraphFact]) -> list[dict[str, dict[str, str]]]:
    return [fact.public() for fact in facts]


def _public_paths(paths: list[RetrievalPath]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for path in paths:
        payload.append(
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
        )
    return payload


def run_case(
    case: MeasuredCase,
    settings: Settings,
    *,
    mode: ModeArg = "graph_grounded",
    store: RdfGraphStore | None = None,
    llm_client: LLMClient | None = None,
) -> GraphRunResult:
    """Run one measured GraphRAG case."""
    validate_retrieval_config(case, settings_max_hops=settings.max_hops)
    total_started = time.perf_counter()

    if store is None:
        store = RdfGraphStore.from_path(settings.graph_path)

    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    termination = "completed"
    answer = ""
    context: list[str] = []
    retrieval_payload: dict[str, Any] = {}
    subgraph: list[GraphFact] = []
    paths: list[RetrievalPath] = []
    resolved, candidate_count = resolve_entities(case.question, store)
    context_assembly_ms = 0
    answer_generation_ms = 0
    model_latency_ms: int | None = None
    model_name: str | None = None
    provider: str | None = None
    model_turns = 0

    sequence.append(
        SequenceEvent(
            kind="user_request",
            detail={
                "caseId": case.trace_id,
                "mode": mode,
                "question": case.question,
            },
        )
    )

    sequence.append(
        SequenceEvent(
            kind="entity_resolution",
            detail={
                "method": "label-based",
                "entityCandidates": candidate_count,
                "resolvedEntities": [entity.model_dump() for entity in resolved],
            },
        )
    )

    if not resolved:
        termination = "no_entity_match"
        answer = INSUFFICIENT_EVIDENCE
        sequence.append(
            SequenceEvent(
                kind="termination",
                detail={"reason": termination},
            )
        )
        metrics = _build_metrics(
            candidate_count=candidate_count,
            resolved_count=0,
            retrieval_hops=0,
            entities_retrieved=0,
            relationships_retrieved=0,
            subgraph_count=0,
            context_count=0,
            retrieval_ms=0,
            context_ms=0,
            answer_ms=0,
            total_started=total_started,
            model_turns=0,
            termination=termination,
        )
        return GraphRunResult(
            case_id=case.trace_id,
            mode=mode,
            example_class=case.example_class,
            user_request=case.question,
            resolved_entities=resolved,
            retrieval=retrieval_payload,
            subgraph=subgraph,
            paths=paths,
            context=context,
            answer=answer,
            termination=termination,
            sequence=sequence,
            metrics=metrics,
            provenance=_provenance(mode, model_name),
            model=model_name,
            provider=provider,
            errors=errors,
        )

    sequence.append(
        SequenceEvent(
            kind="retrieval_started",
            detail={
                "maxHops": case.max_hops,
                "traversalSteps": [
                    {"predicate": step.predicate, "direction": step.direction}
                    for step in case.traversal_steps
                ],
                "seedEntities": [entity.iri for entity in resolved],
            },
        )
    )

    retrieval = retrieve_subgraph(store, resolved, case)
    subgraph = retrieval.facts
    paths = retrieval.paths

    for step_detail in retrieval.steps:
        sequence.append(
            SequenceEvent(
                kind="retrieval_step",
                detail=step_detail,
            )
        )

    entity_iris = sorted(
        {iri for fact in subgraph for iri in (fact.subject.iri, fact.object.iri)}
    )

    retrieval_payload = {
        "maxHops": case.max_hops,
        "hopsUsed": retrieval.hops_used,
        "entitiesRetrieved": entity_iris,
        "relationshipsRetrieved": len(subgraph),
        "steps": retrieval.steps,
        "executionMs": retrieval.execution_ms,
    }

    if not subgraph:
        termination = "no_relevant_subgraph"
        answer = INSUFFICIENT_EVIDENCE
        sequence.append(
            SequenceEvent(
                kind="subgraph_retrieved",
                detail={
                    "tripleCount": 0,
                    "facts": [],
                    "paths": [],
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="final_answer",
                detail={"answer": answer, "source": "deterministic"},
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                detail={"reason": termination},
            )
        )
        metrics = _build_metrics(
            candidate_count=candidate_count,
            resolved_count=len(resolved),
            retrieval_hops=retrieval.hops_used,
            entities_retrieved=len(entity_iris),
            relationships_retrieved=0,
            subgraph_count=0,
            context_count=0,
            retrieval_ms=retrieval.execution_ms,
            context_ms=0,
            answer_ms=0,
            total_started=total_started,
            model_turns=0,
            termination=termination,
        )
        return GraphRunResult(
            case_id=case.trace_id,
            mode=mode,
            example_class=case.example_class,
            user_request=case.question,
            resolved_entities=resolved,
            retrieval=retrieval_payload,
            subgraph=subgraph,
            paths=paths,
            context=context,
            answer=answer,
            termination=termination,
            sequence=sequence,
            metrics=metrics,
            provenance=_provenance(mode, model_name),
            model=model_name,
            provider=provider,
            errors=errors,
        )

    sequence.append(
        SequenceEvent(
            kind="subgraph_retrieved",
            detail={
                "tripleCount": len(subgraph),
                "facts": _public_facts(subgraph),
                "paths": _public_paths(paths),
            },
        )
    )

    context, context_assembly_ms = assemble_context(subgraph)
    sequence.append(
        SequenceEvent(
            kind="context_assembled",
            detail={
                "factCount": len(context),
                "facts": context,
                "assemblyMs": context_assembly_ms,
            },
            latency_ms=context_assembly_ms,
        )
    )

    if mode == "graph_grounded":
        answer_started = time.perf_counter()
        answer = generate_deterministic_answer(
            case,
            facts=subgraph,
            paths=paths,
            context=context,
        )
        answer_generation_ms = int((time.perf_counter() - answer_started) * 1000)
        sequence.append(
            SequenceEvent(
                kind="final_answer",
                detail={"answer": answer, "source": "deterministic"},
                latency_ms=answer_generation_ms,
            )
        )
    else:
        if llm_client is None:
            termination = "model_unavailable"
            answer = INSUFFICIENT_EVIDENCE
            errors.append(
                {
                    "code": "model_unavailable",
                    "message": (
                        "OPENAI_API_KEY is not configured for GRAPHRAG_LLM mode."
                    ),
                }
            )
        else:
            model_name = llm_client.model_name
            provider = llm_client.provider
            user_prompt = build_user_prompt(case.question, context)
            sequence.append(
                SequenceEvent(
                    kind="model_request",
                    detail={
                        "provider": provider,
                        "model": model_name,
                        "contextFactCount": len(context),
                        "userPrompt": user_prompt,
                    },
                )
            )
            try:
                answer, model_latency_ms = llm_client.complete(
                    question=case.question,
                    context=context,
                )
                model_turns = 1
                sequence.append(
                    SequenceEvent(
                        kind="model_response",
                        detail={
                            "answer": answer,
                            "model": model_name,
                            "provider": provider,
                        },
                        latency_ms=model_latency_ms,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — surface provider failure
                termination = "model_failed"
                answer = INSUFFICIENT_EVIDENCE
                errors.append({"code": "model_failed", "message": str(exc)})
                sequence.append(
                    SequenceEvent(
                        kind="model_response",
                        detail={
                            "error": str(exc),
                            "model": model_name,
                            "provider": provider,
                        },
                        latency_ms=0,
                    )
                )

            answer_generation_ms = model_latency_ms or 0
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    detail={"answer": answer, "source": "llm"},
                    latency_ms=answer_generation_ms,
                )
            )

    sequence.append(
        SequenceEvent(
            kind="termination",
            detail={"reason": termination},
        )
    )

    metrics = _build_metrics(
        candidate_count=candidate_count,
        resolved_count=len(resolved),
        retrieval_hops=retrieval.hops_used,
        entities_retrieved=len(entity_iris),
        relationships_retrieved=len(subgraph),
        subgraph_count=len(subgraph),
        context_count=len(context),
        retrieval_ms=retrieval.execution_ms,
        context_ms=context_assembly_ms,
        answer_ms=answer_generation_ms,
        total_started=total_started,
        model_turns=model_turns,
        termination=termination,
    )

    return GraphRunResult(
        case_id=case.trace_id,
        mode=mode,
        example_class=case.example_class,
        user_request=case.question,
        resolved_entities=resolved,
        retrieval=retrieval_payload,
        subgraph=subgraph,
        paths=paths,
        context=context,
        answer=answer,
        termination=termination,
        sequence=sequence,
        metrics=metrics,
        provenance=_provenance(mode, model_name),
        model=model_name,
        provider=provider,
        model_latency_ms=model_latency_ms,
        errors=errors,
    )


def _provenance(mode: Mode, model_name: str | None) -> dict[str, str]:
    if mode == "graph_grounded":
        return {"model": "not_used", "tools": "measured", "metrics": "measured"}
    return {
        "model": model_name or "unconfigured",
        "tools": "measured",
        "metrics": "measured",
    }


def _build_metrics(
    *,
    candidate_count: int,
    resolved_count: int,
    retrieval_hops: int,
    entities_retrieved: int,
    relationships_retrieved: int,
    subgraph_count: int,
    context_count: int,
    retrieval_ms: int,
    context_ms: int,
    answer_ms: int,
    total_started: float,
    model_turns: int,
    termination: str,
) -> GraphRunMetrics:
    total_ms = int((time.perf_counter() - total_started) * 1000)
    return GraphRunMetrics(
        entity_candidates=candidate_count,
        resolved_entity_count=resolved_count,
        retrieval_hops=retrieval_hops,
        entities_retrieved=entities_retrieved,
        relationships_retrieved=relationships_retrieved,
        subgraph_triple_count=subgraph_count,
        context_fact_count=context_count,
        retrieval_execution_ms=retrieval_ms,
        context_assembly_ms=context_ms,
        answer_generation_ms=answer_ms,
        total_ms=total_ms,
        model_turns=model_turns,
        termination_reason=termination,  # type: ignore[arg-type]
    )
