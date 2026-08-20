"""sparql-updates — SPARQL UPDATE mutation over a local RDF graph.

Example ID: sparql-updates

Teaching chain:
  RDF graph → SPARQL UPDATE → changed graph → SELECT verification
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from sparql.cases import CASES, get_case
from sparql.model import UpdateRunResult
from sparql.runner import run_case


def _print_run(result: UpdateRunResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Question: {result.question}")
    print(f"Update:   {result.update_name}")
    print(
        "Metrics: "
        f"inserted={result.metrics.inserted_triple_count} "
        f"deleted={result.metrics.deleted_triple_count} "
        f"verifyRows={result.metrics.verification_rows} "
        f"reason={result.metrics.termination_reason} "
        f"updateMs={result.metrics.update_execution_ms} "
        f"verifyMs={result.metrics.verification_execution_ms}"
    )
    print(f"Before ({len(result.before_state)}):")
    for triple in result.before_state:
        subj = triple.subject.label or triple.subject.iri
        pred = triple.predicate.label or triple.predicate.iri
        obj = triple.object.label or triple.object.literal or triple.object.iri
        print(f"  {subj} --{pred}--> {obj}")
    print(f"After ({len(result.after_state)}):")
    for triple in result.after_state:
        subj = triple.subject.label or triple.subject.iri
        pred = triple.predicate.label or triple.predicate.iri
        obj = triple.object.label or triple.object.literal or triple.object.iri
        print(f"  {subj} --{pred}--> {obj}")
    if result.verification_bindings:
        print("Verification bindings:")
        for index, row in enumerate(result.output["verificationBindings"], start=1):
            parts = ", ".join(
                f"{key}={value.get('label') or value.get('literal')}"
                for key, value in row.items()
            )
            print(f"  [{index}] {parts}")
    else:
        print("Verification bindings: (none)")
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
        help="Run a measured SPARQL UPDATE case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable SPARQL UPDATE operation sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the UpdateRunResult as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "insert-data-billing-portal-redis"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    result = run_case(case, settings)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
