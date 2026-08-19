"""graph-traversal — entities, relationships, and explicit graph walks.

Example ID: graph-traversal

Teaching chain:
  Entity → Relationship → Graph → Traversal → Evidence
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from graph.cases import CASES, get_case
from graph.model import GraphRunResult
from graph.store import GraphStore
from graph.traversal import run_case


def _print_run(result: GraphRunResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Question: {result.question}")
    print(
        "Metrics: "
        f"entities={result.metrics.entities_visited} "
        f"relationships={result.metrics.relationships_visited} "
        f"depth={result.metrics.traversal_depth} "
        f"matched={result.metrics.matched_relationships} "
        f"pathFound={result.metrics.path_found} "
        f"reason={result.metrics.termination_reason} "
        f"executionMs={result.metrics.execution_ms}"
    )
    answers = ", ".join(f"{a.label} ({a.id})" for a in result.answers) or "(none)"
    print(f"Answers:  {answers}")
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
        help="Run a measured graph-traversal case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable graph operation sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the GraphRunResult as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "direct-relationship-employs"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    store = GraphStore.from_path(settings.graph_path)
    result = run_case(case, store, settings)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
