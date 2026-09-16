"""agent-handoffs — application-owned ownership transfer.

Example ID: agent-handoffs

Runtime:
  Active owner → complete | fail | request handoff
  → runtime validates → ownership transfers → new owner
  → terminate
"""

from __future__ import annotations

import argparse
import sys

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_handoffs
from agent.schemas import HandoffRunResult
from agent.synthesizer import LiveSynthesizer, MockSynthesizer
from config import EXAMPLE_ID, get_settings


def _print_run(result: HandoffRunResult, *, show_sequence: bool) -> None:
    print(f"Request:  {result.request}")
    print(f"Answer:   {result.answer}")
    print(f"OK:       {result.ok}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(
        "Owner: "
        f"current={result.state.get('currentOwner')} "
        f"previous={result.state.get('previousOwner')}"
    )
    print(
        "Handoffs: "
        f"requested={result.metrics.handoffs_requested} "
        f"accepted={result.metrics.handoffs_accepted} "
        f"rejected={result.metrics.handoffs_rejected} "
        f"transfers={result.metrics.ownership_transfers}"
    )
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"agent={result.metrics.agent_ms}ms "
        f"runtime={result.metrics.runtime_ms}ms"
    )
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            extra = ""
            if event.kind == "agent_activated":
                extra = (
                    f" agent={event.detail.get('agentId')}"
                    f" owner={event.detail.get('currentOwner')}"
                )
            elif event.kind == "agent_result":
                extra = (
                    f" agent={event.detail.get('agentId')}"
                    f" kind={event.detail.get('kind')}"
                )
            elif event.kind in {
                "handoff_requested",
                "handoff_accepted",
                "handoff_rejected",
                "ownership_transferred",
            }:
                extra = f" {event.detail.get('from')} -> {event.detail.get('to')}"
            elif event.kind == "termination":
                extra = f" reason={event.detail.get('reason')}"
            print(f"  [{event.kind}]{extra}")


def run_measured_case(
    trace_id: str,
    *,
    live: bool = False,
) -> HandoffRunResult:
    settings = get_settings()
    case = get_case(trace_id)
    synthesizer: MockSynthesizer | LiveSynthesizer
    if live:
        synthesizer = LiveSynthesizer(settings)
    else:
        synthesizer = MockSynthesizer()
    return run_handoffs(
        case,
        catalog=Catalog(settings.data_dir),
        synthesizer=synthesizer,
        max_handoffs=settings.max_handoffs,
    )


def main(argv: list[str] | None = None) -> int:
    case_ids = [item.trace_id for item in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured handoff case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable handoff sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the HandoffRunResult as JSON",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a live model to format the terminating owner's actual result",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "invoice-duplicate-basic-handoff"
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
