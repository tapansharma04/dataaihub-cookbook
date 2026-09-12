"""multi-agent-collaboration — coordinator-owned specialist collaboration.

Example ID: multi-agent-collaboration

Runtime:
  Coordinator → delegate → specialized agents → explicit messages/results
  → shared task state → aggregate → terminate
"""

from __future__ import annotations

import argparse
import sys

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_collaboration
from agent.schemas import CollaborationRunResult
from agent.synthesizer import LiveSynthesizer, MockSynthesizer
from config import EXAMPLE_ID, get_settings


def _print_run(result: CollaborationRunResult, *, show_sequence: bool) -> None:
    print(f"Request:  {result.request}")
    print(f"Answer:   {result.answer}")
    print(f"OK:       {result.ok}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(
        "Agents: "
        f"invoked={result.metrics.agents_invoked} "
        f"ok={result.metrics.successful_agent_results} "
        f"fail={result.metrics.failed_agent_results} "
        f"skipped={result.metrics.skipped_delegations}"
    )
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"coordinator={result.metrics.coordinator_ms}ms "
        f"agent={result.metrics.agent_ms}ms "
        f"synthesis={result.metrics.synthesis_ms}ms"
    )
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            extra = ""
            if event.kind == "coordinator_decision":
                extra = f" decision={event.detail.get('decision')}"
            elif event.kind in {"delegation", "agent_selected", "agent_result"}:
                extra = f" agent={event.detail.get('agentId')}"
            elif event.kind == "agent_message":
                extra = f" to={event.detail.get('to')}"
            elif event.kind == "termination":
                extra = f" reason={event.detail.get('reason')}"
            print(f"  [{event.kind}]{extra}")


def run_measured_case(
    trace_id: str,
    *,
    live: bool = False,
) -> CollaborationRunResult:
    settings = get_settings()
    case = get_case(trace_id)
    synthesizer: MockSynthesizer | LiveSynthesizer
    if live:
        synthesizer = LiveSynthesizer(settings)
    else:
        synthesizer = MockSynthesizer()
    return run_collaboration(
        case,
        catalog=Catalog(settings.data_dir),
        synthesizer=synthesizer,
        max_delegations=settings.max_delegations,
    )


def main(argv: list[str] | None = None) -> int:
    case_ids = [item.trace_id for item in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured collaboration case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable collaboration sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the CollaborationRunResult as JSON",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a live model to synthesize the final brief from actual results",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "independent-status-and-docs"
    try:
        get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.live and not get_settings().openai_api_key:
        print("Live synthesis requires OPENAI_API_KEY", file=sys.stderr)
        return 1

    try:
        result = run_measured_case(trace_id, live=args.live)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    case = get_case(trace_id)
    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    print(f"[{EXAMPLE_ID}] model_driver={result.model_driver}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
