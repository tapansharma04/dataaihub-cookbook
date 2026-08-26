"""graph-construction — Source → proposal → validation → RDF graph.

Example ID: graph-construction

Pipeline:
  Source → Extractor → Structured proposal → Application validation
  → RDF mapping → rdflib.Graph → Measured trace
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from graph.builder import RdfGraphStore
from graph.cases import CASES, get_case
from graph.extractor import build_extractor
from graph.model import ConstructionResult
from graph.runner import run_case


def _print_run(result: ConstructionResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Mode:     {result.mode}")
    print(f"Source:   {result.source_text}")
    print(
        "Metrics: "
        f"entities={result.metrics.entities_resolved} "
        f"relsAccepted={result.metrics.relationships_accepted} "
        f"relsRejected={result.metrics.relationships_rejected} "
        f"triples={result.metrics.triples_created} "
        f"graph={result.metrics.graph_triple_count} "
        f"reason={result.metrics.termination_reason} "
        f"totalMs={result.metrics.total_ms}"
    )
    if result.validation.resolved_entities:
        links = ", ".join(
            f"{e.label}→{e.iri}" for e in result.validation.resolved_entities
        )
        print(f"Resolved: {links}")
    if result.validation.rejected_relationships:
        for rejected in result.validation.rejected_relationships:
            print(
                f"Rejected: {rejected.subject} -[{rejected.predicate}]-> "
                f"{rejected.object} ({rejected.reason})"
            )
    if result.triples_created:
        print("Triples:")
        for triple in result.triples_created:
            print(
                f"  {triple.subject} {triple.predicate} {triple.object} ({triple.kind})"
            )
    print(
        f"Graph:    before={result.graph_before_count} "
        f"after={result.graph_after_count} unchanged={result.graph_unchanged}"
    )
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
        help="Run a measured graph-construction case",
    )
    parser.add_argument(
        "--mode",
        choices=["structured", "llm_assisted"],
        default="structured",
        help="Execution mode (default: structured)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable construction sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the ConstructionResult as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "entity-extraction-alice"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    store = RdfGraphStore.fresh(
        start=case.start_graph,
        seed_path=settings.graph_path,
    )
    extractor = build_extractor(settings, mode=args.mode)
    if extractor is None:
        print(
            "llm_assisted mode requires OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    result = run_case(
        case,
        settings,
        mode=args.mode,
        extractor=extractor,
        store=store,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} mode={args.mode}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
