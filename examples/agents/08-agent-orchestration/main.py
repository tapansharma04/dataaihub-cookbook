"""agent-orchestration — application-owned workflow of agent executions.

Example ID: agent-orchestration

Runtime:
  Plan → resolve READY nodes from dependencies/conditions
  → execute READY wave → record results → repeat
  → terminate
"""

from __future__ import annotations

import argparse
import sys

from agent.cases import CASES, get_case
from agent.catalog import Catalog
from agent.runtime import run_orchestration
from agent.schemas import OrchestrationRunResult
from agent.synthesizer import LiveSynthesizer, MockSynthesizer
from config import EXAMPLE_ID, get_settings


def _print_run(result: OrchestrationRunResult, *, show_sequence: bool) -> None:
    print(f"Request:  {result.request}")
    print(f"Answer:   {result.answer}")
    print(f"OK:       {result.ok}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(f"Node states: {result.state.get('nodeStates')}")
    print(f"Execution: {result.state.get('executionOrder')}")
    print(
        "Nodes: "
        f"ready={result.metrics.nodes_ready} "
        f"executed={result.metrics.nodes_executed} "
        f"completed={result.metrics.nodes_completed} "
        f"failed={result.metrics.nodes_failed} "
        f"skipped={result.metrics.nodes_skipped}"
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
            if event.kind in {
                "node_ready",
                "node_started",
                "node_result",
                "node_completed",
                "node_failed",
                "node_skipped",
            }:
                extra = (
                    f" node={event.detail.get('nodeId')}"
                    f" agent={event.detail.get('agentId')}"
                )
                if event.kind == "node_skipped":
                    extra += f" reason={event.detail.get('reason')}"
            elif event.kind == "condition_evaluated":
                extra = (
                    f" node={event.detail.get('nodeId')}"
                    f" selected={event.detail.get('selected')}"
                )
            elif event.kind == "plan_created":
                extra = f" nodes={event.detail.get('nodeIds')}"
            elif event.kind == "termination":
                extra = f" reason={event.detail.get('reason')}"
            print(f"  [{event.kind}]{extra}")


def run_measured_case(
    trace_id: str,
    *,
    live: bool = False,
) -> OrchestrationRunResult:
    settings = get_settings()
    case = get_case(trace_id)
    synthesizer: MockSynthesizer | LiveSynthesizer
    if live:
        synthesizer = LiveSynthesizer(settings)
    else:
        synthesizer = MockSynthesizer()
    return run_orchestration(
        case,
        catalog=Catalog(settings.data_dir),
        synthesizer=synthesizer,
        max_nodes=settings.max_nodes,
    )


def main(argv: list[str] | None = None) -> int:
    case_ids = [item.trace_id for item in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured orchestration case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable orchestration sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the OrchestrationRunResult as JSON",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a live model to format the actual workflow outcome",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "payments-incident-basic-orchestration"
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
