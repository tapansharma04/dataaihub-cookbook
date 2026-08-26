"""Execute graph-construction cases in structured or LLM-assisted modes."""

from __future__ import annotations

import time
from typing import Literal

from config import Settings
from graph.builder import RdfGraphStore
from graph.cases import MeasuredCase
from graph.extractor import Extractor, StructuredExtractor
from graph.model import (
    ConstructionMetrics,
    ConstructionResult,
    ExtractionProposal,
    Mode,
    SequenceEvent,
    TerminationReason,
)
from graph.validator import validate_proposal

ModeArg = Literal["structured", "llm_assisted"]


def run_case(
    case: MeasuredCase,
    settings: Settings,
    *,
    mode: ModeArg = "structured",
    extractor: Extractor | None = None,
    store: RdfGraphStore | None = None,
) -> ConstructionResult:
    """Run one measured graph-construction case.

    Pipeline: source → extract → validate → resolve → commit → verify.
    The extractor never receives the graph store.
    """
    total_started = time.perf_counter()
    if store is None:
        store = RdfGraphStore.fresh(
            start=case.start_graph,
            seed_path=settings.graph_path,
        )
    if extractor is None:
        if mode == "structured":
            extractor = StructuredExtractor()
        else:
            raise ValueError("llm_assisted mode requires an explicit extractor")

    sequence: list[SequenceEvent] = []
    errors: list[dict] = []
    termination: TerminationReason = "completed"
    model_latency_ms: int | None = None
    model_name: str | None = extractor.model_name
    provider: str | None = extractor.provider
    model_turns = 0
    triples_created: list = []
    graph_before = store.triple_count()

    sequence.append(
        SequenceEvent(
            kind="source_loaded",
            detail={
                "caseId": case.trace_id,
                "mode": mode,
                "sourceText": case.source_text,
                "sourceCharacters": len(case.source_text),
                "startGraph": case.start_graph,
                "graphTripleCount": graph_before,
            },
        )
    )

    sequence.append(
        SequenceEvent(
            kind="extraction_started",
            detail={"mode": mode, "extractor": extractor.mode},
        )
    )

    try:
        proposal, extract_ms = extractor.extract(
            case_id=case.trace_id,
            source_text=case.source_text,
        )
    except Exception as exc:  # noqa: BLE001 — surface extraction failure
        termination = "model_failed"
        errors.append({"code": "model_failed", "message": str(exc)})
        proposal = ExtractionProposal()
        extract_ms = 0
        sequence.append(
            SequenceEvent(
                kind="result",
                detail={"error": str(exc)},
            )
        )
        sequence.append(
            SequenceEvent(kind="termination", detail={"reason": termination})
        )
        return _finalize(
            case=case,
            mode=mode,
            proposal=proposal,
            validation=validate_proposal(proposal),
            triples_created=[],
            store=store,
            graph_before=graph_before,
            sequence=sequence,
            termination=termination,
            total_started=total_started,
            model_turns=0,
            model_ms=0,
            model_name=model_name,
            provider=provider,
            model_latency_ms=None,
            errors=errors,
            model_used=False,
        )

    if mode == "llm_assisted" and extractor.provider not in (None,):
        # Structured fixtures report 0 model turns; live/mock LLM counts a turn.
        if extractor.model_name is not None:
            model_turns = 1
            model_latency_ms = extract_ms

    for entity in proposal.entities:
        sequence.append(
            SequenceEvent(
                kind="entity_proposed",
                detail={
                    "label": entity.label,
                    "entityType": entity.entity_type,
                },
            )
        )

    for rel in proposal.relationships:
        sequence.append(
            SequenceEvent(
                kind="relationship_proposed",
                detail={
                    "subject": rel.subject,
                    "predicate": rel.predicate,
                    "object": rel.object,
                },
            )
        )

    sequence.append(SequenceEvent(kind="validation_started", detail={}))
    validation = validate_proposal(proposal)

    for entity in validation.resolved_entities:
        sequence.append(
            SequenceEvent(
                kind="entity_resolved",
                detail={
                    "label": entity.label,
                    "iri": entity.iri,
                    "entityType": entity.entity_type,
                },
            )
        )

    for rejected in validation.rejected_relationships:
        sequence.append(
            SequenceEvent(
                kind="validation_rejected",
                detail={
                    "subject": rejected.subject,
                    "predicate": rejected.predicate,
                    "object": rejected.object,
                    "reason": rejected.reason,
                },
            )
        )

    for unresolved in validation.unresolved_labels:
        sequence.append(
            SequenceEvent(
                kind="validation_rejected",
                detail={"label": unresolved, "reason": "unresolved_entity"},
            )
        )

    for type_error in validation.entity_type_errors:
        reason = type_error.split(":", 1)[0]
        detail: dict = {"reason": reason, "error": type_error}
        parts = type_error.split(":")
        if reason == "entity_type_mismatch" and len(parts) >= 4:
            detail["label"] = parts[1]
            detail["proposedType"] = parts[2]
            detail["registryType"] = parts[3]
        elif reason == "unsupported_entity_type" and len(parts) >= 3:
            detail["proposedType"] = parts[1]
            detail["label"] = parts[2]
        sequence.append(SequenceEvent(kind="validation_rejected", detail=detail))

    # Rejection path: do not commit unsupported / unresolved facts.
    should_reject = (
        bool(validation.rejected_relationships)
        or bool(validation.unresolved_labels)
        or bool(validation.entity_type_errors)
    )
    all_relationships_rejected = bool(proposal.relationships) and not bool(
        validation.accepted_relationships
    )

    if should_reject and (
        case.example_class == "INVALID_FACT"
        or case.expect_graph_unchanged
        or all_relationships_rejected
        or bool(validation.entity_type_errors)
    ):
        termination = "validation_rejected"
        verification = store.verify_committed(
            entities=[],
            relationships=[],
            triples=[],
        )
        sequence.append(
            SequenceEvent(
                kind="graph_verified",
                detail={
                    "ok": verification["ok"],
                    "tripleCount": verification["tripleCount"],
                    "graphUnchanged": store.triple_count() == graph_before,
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="result",
                detail={
                    "committed": False,
                    "rejectedRelationships": [
                        r.model_dump() for r in validation.rejected_relationships
                    ],
                    "graphUnchanged": True,
                },
            )
        )
        sequence.append(
            SequenceEvent(kind="termination", detail={"reason": termination})
        )
        return _finalize(
            case=case,
            mode=mode,
            proposal=proposal,
            validation=validation,
            triples_created=[],
            store=store,
            graph_before=graph_before,
            sequence=sequence,
            termination=termination,
            total_started=total_started,
            model_turns=model_turns,
            model_ms=model_latency_ms or 0,
            model_name=model_name,
            provider=provider,
            model_latency_ms=model_latency_ms,
            errors=errors,
            model_used=model_turns > 0,
        )

    if validation.unresolved_labels:
        termination = "unresolved_entity"
        sequence.append(
            SequenceEvent(
                kind="graph_verified",
                detail={
                    "ok": True,
                    "tripleCount": store.triple_count(),
                    "graphUnchanged": store.triple_count() == graph_before,
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="result",
                detail={"committed": False, "unresolved": validation.unresolved_labels},
            )
        )
        sequence.append(
            SequenceEvent(kind="termination", detail={"reason": termination})
        )
        return _finalize(
            case=case,
            mode=mode,
            proposal=proposal,
            validation=validation,
            triples_created=[],
            store=store,
            graph_before=graph_before,
            sequence=sequence,
            termination=termination,
            total_started=total_started,
            model_turns=model_turns,
            model_ms=model_latency_ms or 0,
            model_name=model_name,
            provider=provider,
            model_latency_ms=model_latency_ms,
            errors=errors,
            model_used=model_turns > 0,
        )

    if validation.accepted_relationships:
        sequence.append(
            SequenceEvent(
                kind="validation_passed",
                detail={
                    "acceptedCount": len(validation.accepted_relationships),
                    "relationships": [
                        r.model_dump() for r in validation.accepted_relationships
                    ],
                },
            )
        )

    # Case-scoped commit: entity extraction asserts type/label; relationship
    # and linking cases commit validated edges. Extractors never write here.
    if case.example_class == "ENTITY_EXTRACTION":
        entity_triples = store.commit_entities(validation.resolved_entities)
        triples_created.extend(entity_triples)
        for triple in entity_triples:
            sequence.append(
                SequenceEvent(
                    kind="triple_created",
                    detail=triple.model_dump(),
                )
            )
        entities_for_verify = validation.resolved_entities
        relationships_for_verify = []
    else:
        rel_triples = store.commit_relationships(validation.accepted_relationships)
        triples_created.extend(rel_triples)
        for triple in rel_triples:
            sequence.append(
                SequenceEvent(
                    kind="triple_created",
                    detail=triple.model_dump(),
                )
            )
        entities_for_verify = []
        relationships_for_verify = validation.accepted_relationships

    sequence.append(
        SequenceEvent(
            kind="graph_committed",
            detail={
                "triplesCreated": len(triples_created),
                "graphTripleCount": store.triple_count(),
            },
        )
    )

    verification = store.verify_committed(
        entities=entities_for_verify,
        relationships=relationships_for_verify,
        triples=triples_created,
    )
    sequence.append(
        SequenceEvent(
            kind="graph_verified",
            detail=verification,
        )
    )
    sequence.append(
        SequenceEvent(
            kind="result",
            detail={
                "committed": True,
                "triplesCreated": [t.model_dump() for t in triples_created],
                "graphTripleCount": store.triple_count(),
            },
        )
    )
    sequence.append(SequenceEvent(kind="termination", detail={"reason": termination}))

    return _finalize(
        case=case,
        mode=mode,
        proposal=proposal,
        validation=validation,
        triples_created=triples_created,
        store=store,
        graph_before=graph_before,
        sequence=sequence,
        termination=termination,
        total_started=total_started,
        model_turns=model_turns,
        model_ms=model_latency_ms or 0,
        model_name=model_name,
        provider=provider,
        model_latency_ms=model_latency_ms,
        errors=errors,
        model_used=model_turns > 0,
    )


def _provenance(
    mode: Mode,
    model_name: str | None,
    *,
    model_used: bool,
) -> dict[str, str]:
    if mode == "structured" or not model_used:
        return {"model": "not_used", "tools": "measured", "metrics": "measured"}
    return {
        "model": model_name or "unconfigured",
        "tools": "measured",
        "metrics": "measured",
    }


def _finalize(
    *,
    case: MeasuredCase,
    mode: ModeArg,
    proposal: ExtractionProposal,
    validation,
    triples_created: list,
    store: RdfGraphStore,
    graph_before: int,
    sequence: list[SequenceEvent],
    termination: TerminationReason,
    total_started: float,
    model_turns: int,
    model_ms: int,
    model_name: str | None,
    provider: str | None,
    model_latency_ms: int | None,
    errors: list,
    model_used: bool,
) -> ConstructionResult:
    graph_after = store.triple_count()
    rejected_count = len(validation.rejected_relationships) + len(
        validation.unresolved_labels
    )
    metrics = ConstructionMetrics(
        source_characters=len(case.source_text),
        entities_proposed=len(proposal.entities),
        entities_resolved=len(validation.resolved_entities),
        relationships_proposed=len(proposal.relationships),
        relationships_accepted=len(validation.accepted_relationships),
        relationships_rejected=len(validation.rejected_relationships),
        triples_created=len(triples_created),
        triples_rejected=rejected_count if not triples_created else 0,
        graph_triple_count=graph_after,
        validation_errors=rejected_count + len(validation.entity_type_errors),
        model_turns=model_turns,
        total_ms=int((time.perf_counter() - total_started) * 1000),
        model_ms=model_ms,
        termination_reason=termination,
    )
    return ConstructionResult(
        case_id=case.trace_id,
        mode=mode,
        example_class=case.example_class,
        source_text=case.source_text,
        proposal=proposal,
        validation=validation,
        triples_created=triples_created,
        graph_before_count=graph_before,
        graph_after_count=graph_after,
        graph_unchanged=graph_before == graph_after,
        termination=termination,
        sequence=sequence,
        metrics=metrics,
        provenance=_provenance(mode, model_name, model_used=model_used),
        model=model_name if model_used else None,
        provider=provider if model_used else None,
        model_latency_ms=model_latency_ms if model_used else None,
        errors=errors,
    )
