"""sparql-queries — declarative SPARQL over a local RDF graph.

Example ID: sparql-queries

Teaching chain:
  Question → SPARQL query → Triple patterns → Bindings → Result rows
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from sparql.cases import CASES, get_case
from sparql.graph import RdfGraphStore
from sparql.model import QueryRunResult
from sparql.runner import run_case


def _print_run(result: QueryRunResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Question: {result.question}")
    print(f"Query:    {result.query_name}")
    print(
        "Metrics: "
        f"rows={result.metrics.result_rows} "
        f"patterns={result.metrics.triple_patterns} "
        f"filters={result.metrics.filter_count} "
        f"reason={result.metrics.termination_reason} "
        f"executionMs={result.metrics.query_execution_ms}"
    )
    if result.matches:
        print("Bindings:")
        for index, row in enumerate(result.matches, start=1):
            parts = ", ".join(
                f"{key}={value.get('label') or value.get('literal')}"
                for key, value in row.items()
            )
            print(f"  [{index}] {parts}")
    else:
        print("Bindings: (none)")
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            print(f"  [{event.kind}] {event.detail}")


def main(argv: list[str] | None = None) -> int:
    case_ids = [case.trace_id for case in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured SPARQL query case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable SPARQL operation sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the QueryRunResult as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "basic-select-knowledge-platform-people"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    store = RdfGraphStore.from_path(settings.graph_path)
    result = run_case(case, store, settings)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
