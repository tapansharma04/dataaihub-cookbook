"""Execute predefined SPARQL UPDATEs against a fresh local RDF graph per case."""

from __future__ import annotations

import time
from typing import Any

from config import Settings
from sparql.cases import MeasuredCase
from sparql.graph import RdfGraphStore
from sparql.model import (
    BindingRow,
    BindingValue,
    SequenceEvent,
    SparqlError,
    TripleState,
    UpdateRunMetrics,
    UpdateRunResult,
)
from sparql.queries import get_update, parse_prefixes, validate_sparql_text


def _row_to_binding(
    row: Any,
    *,
    store: RdfGraphStore,
    select_vars: tuple[str, ...],
) -> BindingRow:
    variables: dict[str, BindingValue] = {}
    for var in select_vars:
        term = row[var]
        if term is None:
            continue
        variables[var] = store.binding_value(term)
    return BindingRow(variables=variables)


def _public_bindings(rows: list[BindingRow]) -> list[dict[str, dict[str, str | None]]]:
    return [row.public() for row in rows]


def _public_triples(
    triples: list[TripleState],
) -> list[dict[str, dict[str, str | None]]]:
    return [triple.public() for triple in triples]


def _triple_key(triple: TripleState) -> tuple[str, str, str]:
    return triple.sort_key()


def _diff_triples(
    before: list[TripleState],
    after: list[TripleState],
) -> tuple[list[TripleState], list[TripleState]]:
    """Derive inserted/deleted from focused before/after state."""
    before_map = {_triple_key(t): t for t in before}
    after_map = {_triple_key(t): t for t in after}
    inserted = [after_map[key] for key in sorted(set(after_map) - set(before_map))]
    deleted = [before_map[key] for key in sorted(set(before_map) - set(after_map))]
    return inserted, deleted


def _focus_state(store: RdfGraphStore, focus: Any) -> list[TripleState]:
    return store.focused_state(
        subjects=focus.subjects,
        predicates=focus.predicates,
        objects=focus.objects,
    )


def run_case(
    case: MeasuredCase,
    settings: Settings,
    *,
    store: RdfGraphStore | None = None,
) -> UpdateRunResult:
    """Run one measured UPDATE case on a fresh graph (unless store is injected)."""
    predefined = get_update(case.update_name)
    update_query = predefined.update_query
    verification_query = predefined.verification_query
    prefixes = parse_prefixes(update_query)

    if store is None:
        store = RdfGraphStore.fresh_from_path(settings.graph_path)

    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    update_ms = 0
    verify_ms = 0
    before_state: list[TripleState] = []
    after_state: list[TripleState] = []
    inserted: list[TripleState] = []
    deleted: list[TripleState] = []
    bindings: list[BindingRow] = []
    termination_reason = "completed"

    sequence.append(
        SequenceEvent(
            kind="user_request",
            detail={
                "case": case.trace_id,
                "question": case.question,
                "updateName": predefined.name,
            },
        )
    )

    try:
        validate_sparql_text(update_query)
        validate_sparql_text(verification_query)
    except SparqlError as exc:
        sequence.append(
            SequenceEvent(
                kind="termination",
                detail={"reason": exc.code},
            )
        )
        metrics = UpdateRunMetrics(
            update_execution_ms=0,
            verification_execution_ms=0,
            inserted_triple_count=0,
            deleted_triple_count=0,
            before_triple_count=0,
            after_triple_count=0,
            verification_rows=0,
            update_type=predefined.name,
            termination_reason=exc.code,
        )
        return UpdateRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            question=case.question,
            update_name=predefined.name,
            update_query=update_query,
            verification_query=verification_query,
            prefixes=prefixes,
            sequence=sequence,
            metrics=metrics,
            output={
                "before": [],
                "after": [],
                "insertedTriples": [],
                "deletedTriples": [],
                "verificationBindings": [],
                "verificationRows": 0,
                "updateQuery": update_query,
                "verificationQuery": verification_query,
                "updateName": predefined.name,
                "terminationReason": exc.code,
            },
            errors=[{"code": exc.code, "message": exc.message}],
        )

    before_state = _focus_state(store, predefined.focus)

    sequence.append(
        SequenceEvent(
            kind="update_started",
            detail={
                "updateName": predefined.name,
                "updateQuery": update_query,
                "prefixes": prefixes,
                "before": _public_triples(before_state),
                "beforeTripleCount": len(before_state),
            },
        )
    )

    try:
        started = time.perf_counter()
        store.rdf.update(update_query)
        update_ms = int((time.perf_counter() - started) * 1000)
        sequence.append(
            SequenceEvent(
                kind="update_executed",
                detail={
                    "engine": "rdflib",
                    "method": "Graph.update",
                    "updateQuery": update_query,
                    "executionMs": update_ms,
                    "success": True,
                },
                latency_ms=update_ms,
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface as measured failure
        update_ms = int((time.perf_counter() - started) * 1000)
        termination_reason = "update_failed"
        errors.append({"code": "update_failed", "message": str(exc)})
        sequence.append(
            SequenceEvent(
                kind="update_executed",
                detail={
                    "engine": "rdflib",
                    "method": "Graph.update",
                    "updateQuery": update_query,
                    "executionMs": update_ms,
                    "success": False,
                    "error": str(exc),
                },
                latency_ms=update_ms,
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                detail={"reason": termination_reason},
            )
        )
        metrics = UpdateRunMetrics(
            update_execution_ms=update_ms,
            verification_execution_ms=0,
            inserted_triple_count=0,
            deleted_triple_count=0,
            before_triple_count=len(before_state),
            after_triple_count=0,
            verification_rows=0,
            update_type=predefined.name,
            termination_reason=termination_reason,
        )
        return UpdateRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            question=case.question,
            update_name=predefined.name,
            update_query=update_query,
            verification_query=verification_query,
            prefixes=prefixes,
            before_state=before_state,
            sequence=sequence,
            metrics=metrics,
            output={
                "before": _public_triples(before_state),
                "after": [],
                "insertedTriples": [],
                "deletedTriples": [],
                "verificationBindings": [],
                "verificationRows": 0,
                "updateQuery": update_query,
                "verificationQuery": verification_query,
                "updateName": predefined.name,
                "terminationReason": termination_reason,
            },
            errors=errors,
        )

    after_state = _focus_state(store, predefined.focus)
    inserted, deleted = _diff_triples(before_state, after_state)

    sequence.append(
        SequenceEvent(
            kind="graph_state",
            detail={
                "before": _public_triples(before_state),
                "after": _public_triples(after_state),
                "insertedTriples": _public_triples(inserted),
                "deletedTriples": _public_triples(deleted),
                "beforeTripleCount": len(before_state),
                "afterTripleCount": len(after_state),
                "insertedTripleCount": len(inserted),
                "deletedTripleCount": len(deleted),
            },
        )
    )

    sequence.append(
        SequenceEvent(
            kind="verification_started",
            detail={
                "verificationQuery": verification_query,
                "variables": list(predefined.verification_variables),
            },
        )
    )

    try:
        started = time.perf_counter()
        result = store.rdf.query(verification_query)
        verify_ms = int((time.perf_counter() - started) * 1000)
        raw_rows = list(result)
        if len(raw_rows) > settings.max_result_rows:
            termination_reason = "row_limit"
            raw_rows = raw_rows[: settings.max_result_rows]
        for row in raw_rows:
            bindings.append(
                _row_to_binding(
                    row,
                    store=store,
                    select_vars=predefined.verification_variables,
                )
            )
        sequence.append(
            SequenceEvent(
                kind="verification_result",
                detail={
                    "engine": "rdflib",
                    "method": "Graph.query",
                    "executionMs": verify_ms,
                    "rowCount": len(bindings),
                    "bindings": _public_bindings(bindings),
                    "success": True,
                },
                latency_ms=verify_ms,
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface as measured failure
        verify_ms = int((time.perf_counter() - started) * 1000)
        termination_reason = "verification_failed"
        errors.append({"code": "verification_failed", "message": str(exc)})
        sequence.append(
            SequenceEvent(
                kind="verification_result",
                detail={
                    "engine": "rdflib",
                    "method": "Graph.query",
                    "executionMs": verify_ms,
                    "rowCount": 0,
                    "bindings": [],
                    "success": False,
                    "error": str(exc),
                },
                latency_ms=verify_ms,
            )
        )

    sequence.append(
        SequenceEvent(
            kind="update_completed",
            detail={
                "insertedTripleCount": len(inserted),
                "deletedTripleCount": len(deleted),
                "verificationRows": len(bindings),
                "terminationReason": termination_reason,
            },
        )
    )
    sequence.append(
        SequenceEvent(
            kind="termination",
            detail={"reason": termination_reason},
        )
    )

    metrics = UpdateRunMetrics(
        update_execution_ms=update_ms,
        verification_execution_ms=verify_ms,
        inserted_triple_count=len(inserted),
        deleted_triple_count=len(deleted),
        before_triple_count=len(before_state),
        after_triple_count=len(after_state),
        verification_rows=len(bindings),
        update_type=predefined.name,
        termination_reason=termination_reason,  # type: ignore[arg-type]
    )

    public_bindings = _public_bindings(bindings)
    output = {
        "before": _public_triples(before_state),
        "after": _public_triples(after_state),
        "insertedTriples": _public_triples(inserted),
        "deletedTriples": _public_triples(deleted),
        "verificationBindings": public_bindings,
        "verificationRows": len(bindings),
        "updateQuery": update_query,
        "verificationQuery": verification_query,
        "updateName": predefined.name,
        "terminationReason": termination_reason,
    }

    return UpdateRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        question=case.question,
        update_name=predefined.name,
        update_query=update_query,
        verification_query=verification_query,
        prefixes=prefixes,
        before_state=before_state,
        after_state=after_state,
        inserted_triples=inserted,
        deleted_triples=deleted,
        verification_bindings=bindings,
        sequence=sequence,
        metrics=metrics,
        output=output,
        errors=errors,
    )
