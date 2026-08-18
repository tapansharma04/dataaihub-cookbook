"""agent-memory — application-managed store / retrieve across interactions.

Example ID: agent-memory

Loop:
  Interaction → Agent → Memory Store (STORE / RETRIEVE)
  → Later Interaction → Agent uses memory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.cases import CASES, get_case, scripted_client_for
from agent.loop import run_memory_loop
from agent.memory import FixedClock, MemoryStore
from agent.model import OpenAIModelClient, get_openai_client
from agent.schemas import AgentRunResult
from agent.source import AuthoritativeStore
from config import EXAMPLE_ID, get_settings


def _known_scopes(data_dir: Path) -> set[str]:
    users = json.loads((data_dir / "users.json").read_text(encoding="utf-8"))
    return set(users)


def _print_run(result: AgentRunResult, *, show_sequence: bool) -> None:
    print(f"\nRequest: {result.request}")
    print(f"Answer:  {result.answer}")
    print(f"Termination: {result.metrics.termination_reason}")
    print(
        "Memory: "
        f"scope={result.metrics.memory_scope} "
        f"writes={result.metrics.memory_writes} "
        f"reads={result.metrics.memory_reads} "
        f"(hits={result.metrics.memory_hits} "
        f"misses={result.metrics.memory_misses}) "
        f"version={result.metrics.memory_version} "
        f"stale={result.metrics.stale_memory_detected}"
    )
    print(
        "Metrics: "
        f"total={result.metrics.total_ms}ms "
        f"model={result.metrics.model_ms}ms "
        f"tool={result.metrics.tool_ms}ms "
        f"turns={result.metrics.model_turns}/{result.metrics.max_turns}"
    )
    if show_sequence:
        print("\nSequence:")
        for event in result.sequence:
            iid = event.interaction_id
            prefix = f"  [{event.kind}] interaction={iid}"
            if event.kind == "model_decision":
                print(
                    f"{prefix} turn={event.turn} "
                    f"decision={event.detail.get('decision')}"
                )
            elif event.kind == "memory_stored":
                print(
                    f"{prefix} key={event.detail.get('key')} "
                    f"v{event.detail.get('version')} "
                    f"source={event.detail.get('source')}"
                )
            elif event.kind == "memory_retrieved":
                record = event.detail.get("record") or {}
                print(
                    f"{prefix} key={event.detail.get('key')} v{record.get('version')}"
                )
            elif event.kind == "memory_not_found":
                print(f"{prefix} key={event.detail.get('key')}")
            elif event.kind == "observation":
                print(
                    f"{prefix} stale={event.detail.get('staleMemoryDetected')} "
                    f"resolution={event.detail.get('resolution')}"
                )
            elif event.kind == "termination":
                print(f"{prefix} reason={event.detail.get('reason')}")
            elif event.kind == "user_request":
                print(f"{prefix}")
            elif event.kind == "final_answer":
                print(f"{prefix}")
            else:
                print(prefix)


def _max_turns_for_case(trace_id: str) -> int:
    settings = get_settings()
    case = get_case(trace_id)
    return case.max_turns if case.max_turns is not None else settings.max_turns


def run_measured_case(trace_id: str) -> AgentRunResult:
    settings = get_settings()
    case = get_case(trace_id)
    data_dir = settings.data_dir
    store = MemoryStore(
        known_scopes=_known_scopes(data_dir),
        clock=FixedClock(),
    )
    return run_memory_loop(
        interactions=[item.request for item in case.interactions],
        scope=case.scope,
        model=scripted_client_for(case),
        memory_store=store,
        authoritative=AuthoritativeStore.from_data_dir(data_dir),
        max_turns=_max_turns_for_case(trace_id),
    )


def run_live(request: str, *, scope: str = "u-1001") -> AgentRunResult:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    data_dir = settings.data_dir
    store = MemoryStore(
        known_scopes=_known_scopes(data_dir),
        clock=FixedClock(),
    )
    model = OpenAIModelClient(get_openai_client(settings), settings.chat_model)
    return run_memory_loop(
        interactions=[request],
        scope=scope,
        model=model,
        memory_store=store,
        authoritative=AuthoritativeStore.from_data_dir(data_dir),
        max_turns=settings.max_turns,
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
        help="Print the observable memory sequence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the AgentRunResult as JSON",
    )
    parser.add_argument(
        "--scope",
        default="u-1001",
        help="Session scope / user id for live mode (default u-1001)",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASES:
            print(f"{case.trace_id}\t{case.example_class}\t{case.scope}")
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

    request = args.request or ("I prefer email notifications for service incidents.")
    try:
        result = run_live(request, scope=args.scope)
    except RuntimeError as exc:
        print(
            f"{exc}. Copy .env.example to .env and set your key,\n"
            "or run a measured case with: "
            "uv run python main.py --case store-email-notification-preference",
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
