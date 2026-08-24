"""graphrag — Graph-grounded retrieval with optional LLM answering.

Example ID: graphrag

Teaching chain:
  Question → entity resolution → graph retrieval → subgraph → context
  → [optional LLM] → answer
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from graphrag.cases import CASES, get_case
from graphrag.graph import RdfGraphStore
from graphrag.llm import build_llm_client
from graphrag.model import GraphRunResult
from graphrag.runner import run_case


def _print_run(result: GraphRunResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Mode:     {result.mode}")
    print(f"Question: {result.user_request}")
    print(
        "Metrics: "
        f"entities={result.metrics.resolved_entity_count} "
        f"triples={result.metrics.subgraph_triple_count} "
        f"contextFacts={result.metrics.context_fact_count} "
        f"modelTurns={result.metrics.model_turns} "
        f"reason={result.metrics.termination_reason} "
        f"totalMs={result.metrics.total_ms}"
    )
    if result.resolved_entities:
        labels = ", ".join(entity.label for entity in result.resolved_entities)
        print(f"Resolved: {labels}")
    if result.context:
        print("Context:")
        for line in result.context:
            print(f"  - {line}")
    print(f"Answer:   {result.answer}")
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
        help="Run a measured GraphRAG case",
    )
    parser.add_argument(
        "--mode",
        choices=["graph_grounded", "graphrag_llm"],
        default="graph_grounded",
        help="Execution mode (default: graph_grounded)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable GraphRAG operation sequence",
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

    trace_id = args.case or "entity-retrieval-knowledge-platform"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    store = RdfGraphStore.from_path(settings.graph_path)
    llm_client = None
    if args.mode == "graphrag_llm":
        llm_client = build_llm_client(settings)
        if llm_client is None:
            print(
                "GRAPHRAG_LLM mode requires OPENAI_API_KEY.",
                file=sys.stderr,
            )
            return 1

    result = run_case(
        case,
        settings,
        mode=args.mode,
        store=store,
        llm_client=llm_client,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} mode={args.mode}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
