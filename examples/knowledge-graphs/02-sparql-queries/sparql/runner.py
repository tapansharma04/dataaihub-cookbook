"""Execute predefined SPARQL queries against a local RDF graph."""

from __future__ import annotations

import time
from typing import Any

from config import Settings
from sparql.cases import MeasuredCase
from sparql.graph import RdfGraphStore
from sparql.model import (
    BindingRow,
    BindingValue,
    QueryRunMetrics,
    QueryRunResult,
    SequenceEvent,
    SparqlError,
)
from sparql.queries import get_query, parse_prefixes, validate_query_text


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


def run_case(
    case: MeasuredCase,
    store: RdfGraphStore,
    settings: Settings,
) -> QueryRunResult:
    predefined = get_query(case.query_name)
    query = predefined.query

    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    elapsed_ms = 0

    try:
        validate_query_text(query)
    except SparqlError as exc:
        sequence.append(
            SequenceEvent(
                kind="user_request",
                detail={
                    "case": case.trace_id,
                    "question": case.question,
                    "queryName": predefined.name,
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                detail={"reason": exc.code},
            )
        )
        metrics = QueryRunMetrics(
            query_execution_ms=0,
            result_rows=0,
            triple_patterns=predefined.triple_patterns,
            filter_count=predefined.filter_count,
            variables=list(predefined.variables),
            query_case=case.example_class,
            bindings_returned=0,
            termination_reason=exc.code,
        )
        return QueryRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            question=case.question,
            query_name=predefined.name,
            query=query,
            prefixes=parse_prefixes(query),
            patterns=list(predefined.patterns),
            sequence=sequence,
            metrics=metrics,
            output={
                "bindings": [],
                "matches": [],
                "rowCount": 0,
                "query": query,
                "queryName": predefined.name,
                "terminationReason": exc.code,
            },
            errors=[{"code": exc.code, "message": exc.message}],
        )

    sequence.append(
        SequenceEvent(
            kind="user_request",
            detail={
                "case": case.trace_id,
                "question": case.question,
                "queryName": predefined.name,
            },
        )
    )
    sequence.append(
        SequenceEvent(
            kind="query_started",
            detail={
                "queryName": predefined.name,
                "query": query,
                "prefixes": parse_prefixes(query),
                "patterns": list(predefined.patterns),
                "triplePatterns": predefined.triple_patterns,
                "filterCount": predefined.filter_count,
            },
        )
    )

    started = time.perf_counter()
    termination_reason = "completed"
    bindings: list[BindingRow] = []

    try:
        result = store.rdf.query(query)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        sequence.append(
            SequenceEvent(
                kind="query_executed",
                detail={
                    "engine": "rdflib",
                    "query": query,
                    "executionMs": elapsed_ms,
                    "success": True,
                },
                latency_ms=elapsed_ms,
            )
        )

        select_vars = predefined.variables
        raw_rows = list(result)
        if len(raw_rows) > settings.max_result_rows:
            termination_reason = "row_limit"
            raw_rows = raw_rows[: settings.max_result_rows]

        for row in raw_rows:
            bindings.append(_row_to_binding(row, store=store, select_vars=select_vars))

        if not bindings and termination_reason == "completed":
            termination_reason = "no_match"

        sequence.append(
            SequenceEvent(
                kind="result_bindings",
                detail={
                    "rowCount": len(bindings),
                    "bindings": _public_bindings(bindings),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface as measured failure
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        termination_reason = "query_failed"
        errors.append({"code": "query_failed", "message": str(exc)})
        sequence.append(
            SequenceEvent(
                kind="query_executed",
                detail={
                    "engine": "rdflib",
                    "query": query,
                    "executionMs": elapsed_ms,
                    "success": False,
                    "error": str(exc),
                },
                latency_ms=elapsed_ms,
            )
        )

    sequence.append(
        SequenceEvent(
            kind="query_completed",
            detail={
                "rowCount": len(bindings),
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

    metrics = QueryRunMetrics(
        query_execution_ms=elapsed_ms,
        result_rows=len(bindings),
        triple_patterns=predefined.triple_patterns,
        filter_count=predefined.filter_count,
        variables=list(predefined.variables),
        query_case=case.example_class,
        bindings_returned=len(bindings),
        termination_reason=termination_reason,  # type: ignore[arg-type]
    )

    public_matches = _public_bindings(bindings)
    output = {
        "bindings": public_matches,
        "matches": public_matches,
        "rowCount": len(bindings),
        "query": query,
        "queryName": predefined.name,
        "terminationReason": termination_reason,
    }

    return QueryRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        question=case.question,
        query_name=predefined.name,
        query=query,
        prefixes=parse_prefixes(query),
        patterns=list(predefined.patterns),
        bindings=bindings,
        matches=public_matches,
        sequence=sequence,
        metrics=metrics,
        output=output,
        errors=errors,
    )


def assert_query_not_python_filtered(case: MeasuredCase, store: RdfGraphStore) -> bool:
    """Return True when removing FILTER from the query changes FILTER_QUERY results."""
    if case.query_name != "FILTER_QUERY":
        return True
    predefined = get_query("FILTER_QUERY")
    unfiltered = predefined.query.replace('  FILTER(?team = "platform")\n', "")
    validate_query_text(unfiltered)
    unfiltered_rows = list(store.rdf.query(unfiltered))
    filtered_rows = list(store.rdf.query(predefined.query))
    return len(unfiltered_rows) > len(filtered_rows)
