"""Export MCP composition traces for a future Interactive Lab.

Mock traces are written to lab_traces.json (deterministic, no API key).
Live traces are written to lab_traces_llm.json and require OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from client.cases import CASES
from client.runner import run_case
from client.trace import build_trace
from config import get_settings

ROOT = Path(__file__).resolve().parent
LAB_TRACES_PATH = ROOT / "lab_traces.json"
LAB_TRACES_LLM_PATH = ROOT / "lab_traces_llm.json"


def export_mode(mode: str, *, force: bool = False) -> None:
    out_path = LAB_TRACES_LLM_PATH if mode == "live" else LAB_TRACES_PATH
    if out_path.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing traces in {out_path}. "
            "Pass --force to replace them."
        )

    settings = get_settings()
    if mode == "live" and not settings.openai_api_key:
        raise SystemExit(
            "Live sampling export requires OPENAI_API_KEY and a configured model."
        )

    traces = []
    for case in CASES:
        sampling_mode = None
        if mode == "live" and case.sampling_mode == "mock":
            sampling_mode = "live"
        result = run_case(
            case,
            settings=settings,
            sampling_mode=sampling_mode,
        )
        traces.append(build_trace(case=case, result=result, settings=settings))

    out_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MCP composition lab traces")
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock writes lab_traces.json; live writes lab_traces_llm.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the selected traces file",
    )
    args = parser.parse_args()
    export_mode(args.mode, force=args.force)


if __name__ == "__main__":
    main()
