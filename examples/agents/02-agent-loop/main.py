"""agent-loop — application-controlled agent runtime.

Example ID: agent-loop

Loop:
  User request → Model decision → (Tool call → Observation → Model decision)*
  → Final answer / termination
"""

from __future__ import annotations

import argparse
import sys

from agent.cases import CASES, get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_agent_loop
from agent.model import OpenAIModelClient, get_openai_client
from agent.schemas import AgentRunResult
from agent.tools import build_registry
from config import EXAMPLE_ID, get_settings


def _print_run(result: AgentRunResult, *, show_sequence: bool) -> None:
    print(f"\nRequest: {result.request}")
    print(f"Answer:  {result.answer}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"model={result.metrics.model_ms}ms "
        f"tool={result.metrics.tool_ms}ms "
        f"turns={result.metrics.model_turns}/{result.metrics.max_turns} "
        f"tool_calls={result.metrics.tool_calls} "
        f"(ok={result.metrics.successful_tool_calls} "
        f"fail={result.metrics.failed_tool_calls})"
    )
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            if event.kind == "model_decision":
                decision = event.detail.get("decision")
                print(f"  [{event.kind}] turn={event.turn} decision={decision}")
            elif event.kind == "tool_call":
                print(
                    f"  [{event.kind}] {event.detail.get('name')} "
                    f"args={event.detail.get('arguments')}"
                )
            elif event.kind == "observation":
                ok = event.detail.get("ok")
                print(f"  [{event.kind}] ok={ok} name={event.detail.get('name')}")
            elif event.kind == "termination":
                print(f"  [{event.kind}] reason={event.detail.get('reason')}")
            elif event.kind == "final_answer":
                print(f"  [{event.kind}]")
            else:
                print(f"  [{event.kind}]")


def _max_turns_for_case(trace_id: str) -> int:
    settings = get_settings()
    case = get_case(trace_id)
    return case.max_turns if case.max_turns is not None else settings.max_turns


def run_measured_case(trace_id: str) -> AgentRunResult:
    settings = get_settings()
    case = get_case(trace_id)
    registry = build_registry(settings.data_dir)
    executor = ToolExecutor(registry, timeout_ms=settings.tool_timeout_ms)
    return run_agent_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
        max_turns=_max_turns_for_case(trace_id),
        max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
    )


def run_live(request: str) -> AgentRunResult:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    registry = build_registry(settings.data_dir)
    executor = ToolExecutor(registry, timeout_ms=settings.tool_timeout_ms)
    model = OpenAIModelClient(get_openai_client(settings), settings.chat_model)
    return run_agent_loop(
        request=request,
        model=model,
        registry=registry,
        executor=executor,
        max_turns=settings.max_turns,
        max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
    )


def main(argv: list[str] | None = None) -> int:
    case_ids = [c.trace_id for c in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="Live user request (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured case harness (no paid API required)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable model/tool/loop sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the AgentRunResult as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}\t{case.request[:72]}")
        return 0

    if args.case:
        case = get_case(args.case)
        result = run_measured_case(args.case)
        if args.json:
            print(result.model_dump_json(indent=2))
            return 0
        print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
        print(f"[{EXAMPLE_ID}] model_driver={result.model_driver}")
        _print_run(result, show_sequence=args.show_sequence)
        return 0

    request = args.request or (
        "Check the payments service. If it is degraded, inspect the relevant "
        "documentation and summarize what the user should know."
    )
    try:
        result = run_live(request)
    except RuntimeError as exc:
        print(
            f"{exc}. Copy .env.example to .env and set your key,\n"
            "or run a measured case with: "
            "uv run python main.py --case simple-loop-payments-docs",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] live model={result.model}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
