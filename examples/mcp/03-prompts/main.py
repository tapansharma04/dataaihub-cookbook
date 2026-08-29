"""mcp-prompts — MCP initialize → prompts/list → prompts/get lifecycle.

Example ID: mcp-prompts

Protocol lifecycle:
  Client → initialize → MCP Server → prompts/list → prompts/get → messages
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
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"init={result.metrics.initialize_ms}ms "
        f"discover={result.metrics.discovery_ms}ms "
        f"get={result.metrics.prompt_get_ms}ms "
        f"prompts={result.metrics.prompts_discovered} "
        f"gets={result.metrics.prompts_requested} "
        f"(ok={result.metrics.successful_gets} "
        f"fail={result.metrics.failed_gets}) "
        f"messages={result.metrics.message_count} "
        f"bytes={result.metrics.message_bytes}"
    )
    if result.discovered_prompts:
        names = ", ".join(prompt.name for prompt in result.discovered_prompts)
        print(f"Prompts:  {names}")
    for get in result.output.get("gets", []):
        if get.get("isError"):
            print(f"Get:      {get['requestedPrompt']} REJECTED")
        else:
            print(
                f"Get:      prompt={get['requestedPrompt']} "
                f"messages={len(get.get('messages', []))}"
            )
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
        help="Run a measured MCP prompt protocol case",
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
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}")
        return 0

    trace_id = args.case or "prompt-discovery"
    try:
        case = get_case(trace_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    settings = get_settings()
    result = run_case(case, settings=settings)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"[{EXAMPLE_ID}] case={case.trace_id} class={case.example_class}")
    _print_run(result, show_sequence=args.show_sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
