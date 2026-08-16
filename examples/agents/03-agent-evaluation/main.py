"""agent-evaluation — measured agent run plus outcome/trajectory evaluation.

Example ID: agent-evaluation

Lifecycle:
  Task → Agent Run → Trace → Outcome Evaluation + Trajectory Evaluation
  → Evaluation Result → Improve / Regression
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.cases import CASES, get_case
from agent.executor import ToolExecutor
from agent.loop import run_agent_loop
from agent.model import OpenAIModelClient, get_openai_client
from agent.run import run_measured_case
from agent.schemas import AgentRunResult
from agent.tools import build_registry
from config import EXAMPLE_ID, get_settings
from evaluation.evaluator import evaluate_run
from evaluation.schemas import EvaluationResult


def _print_run(
    result: AgentRunResult,
    *,
    show_sequence: bool,
    evaluation: EvaluationResult | None = None,
) -> None:
    print(f"\nRequest: {result.request}")
    print(f"Answer:  {result.answer}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(
        "Operational metrics: "
        f"total={result.metrics.total_ms}ms "
        f"model={result.metrics.model_ms}ms "
        f"tool={result.metrics.tool_ms}ms "
        f"turns={result.metrics.model_turns}/{result.metrics.max_turns} "
        f"tool_calls={result.metrics.tool_calls} "
        f"(ok={result.metrics.successful_tool_calls} "
        f"fail={result.metrics.failed_tool_calls})"
    )
    if evaluation is not None:
        print(
            "Evaluation: "
            f"task_success={evaluation.task_success} "
            f"final_answer_correct={evaluation.final_answer_correct} "
            f"trajectory_success={evaluation.trajectory_success} "
            f"efficiency={evaluation.step_efficiency.status} "
            f"recovery={evaluation.recovery.status}"
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
        help="Print the observable model/tool sequence",
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
        result = run_measured_case(case)
        evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
        if args.json:
            payload = result.model_dump()
            payload["evaluation"] = evaluation.to_public_dict()
            print(json.dumps(payload, indent=2))
            return 0
        print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
        print(f"[{EXAMPLE_ID}] model_driver={result.model_driver}")
        _print_run(result, show_sequence=args.show_sequence, evaluation=evaluation)
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
            "uv run python main.py --case task-success-payments-docs",
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
