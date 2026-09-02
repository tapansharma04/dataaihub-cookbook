"""mcp-composition — composed MCP workflow with client-owned Sampling.

Example ID: mcp-composition

Protocol lifecycle:
  Client → initialize → MCP primitives → tools/call → sampling/createMessage
  → client sampling callback → LLM or mock → server → final result
"""

from __future__ import annotations

import argparse
import sys

from client.cases import CASES, get_case
from client.runner import run_case
from client.schemas import McpRunResult
from config import EXAMPLE_ID, get_settings


def _print_run(result: McpRunResult, *, show_sequence: bool) -> None:
    print(f"Case:     {result.case_id} ({result.example_class})")
    print(f"Transport: {result.transport}")
    print(f"Protocol:  {result.protocol_version}")
    print(f"Sampling:  {result.sampling_mode}")
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"init={result.metrics.initialize_ms}ms "
        f"tools={result.metrics.tool_calls} "
        f"resources={result.metrics.resources_read} "
        f"prompts={result.metrics.prompts_requested} "
        f"sampling={result.metrics.sampling_requests} "
        f"(ok={result.metrics.successful_samplings} "
        f"fail={result.metrics.failed_samplings})"
    )
    for invocation in result.output.get("invocations", []):
        status = "REJECTED" if invocation.get("isError") else "ok"
        print(f"Tool:     {invocation['tool']} {status}")
    for sample in result.output.get("sampling", []):
        if sample.get("isError"):
            print("Sample:   REJECTED")
        else:
            model = (sample.get("result") or {}).get("model")
            print(f"Sample:   model={model}")
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            print(f"  [{event.kind}] {event.detail}")


def main(argv: list[str] | None = None) -> int:
    case_ids = [c.trace_id for c in CASES]
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--case",
        choices=case_ids,
        help="Run a measured MCP composition case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print measured case ids and exit",
    )
    parser.add_argument(
        "--show-sequence",
        action="store_true",
        help="Print the observable MCP protocol sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the McpRunResult as JSON",
    )
    parser.add_argument(
        "--sampling",
        choices=["mock", "reject", "live"],
        help="Override the case sampling callback (default: case-owned)",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}\t{case.sampling_mode}")
        return 0

    trace_id = args.case or "resource-to-sampling"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    if args.sampling == "live" and not settings.openai_api_key:
        print("Live sampling requires OPENAI_API_KEY", file=sys.stderr)
        return 1

    result = run_case(case, settings=settings, sampling_mode=args.sampling)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
