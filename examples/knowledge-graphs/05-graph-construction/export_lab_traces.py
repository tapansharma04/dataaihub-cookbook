"""Export graph-construction traces for a future Interactive Lab.

structured traces are written to lab_traces.json (deterministic).
llm_assisted traces are written to lab_traces_llm.json (separate file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import FIXED_RECORDED_AT, get_settings
from graph.builder import RdfGraphStore
from graph.cases import CASES
from graph.extractor import build_extractor
from graph.runner import run_case
from graph.trace import build_trace

ROOT = Path(__file__).resolve().parent
LAB_TRACES_PATH = ROOT / "lab_traces.json"
LAB_TRACES_LLM_PATH = ROOT / "lab_traces_llm.json"

OUTPUT_BY_MODE = {
    "structured": LAB_TRACES_PATH,
    "llm_assisted": LAB_TRACES_LLM_PATH,
}


def export_mode(mode: str, *, force: bool = False) -> None:
    out_path = OUTPUT_BY_MODE[mode]
    if out_path.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing traces in {out_path}. "
            "Pass --force to replace them."
        )

    settings = get_settings()
    extractor = build_extractor(settings, mode=mode)
    if extractor is None:
        raise SystemExit(
            "llm_assisted export requires OPENAI_API_KEY and a configured model."
        )

    traces = []
    for case in CASES:
        store = RdfGraphStore.fresh(
            start=case.start_graph,
            seed_path=settings.graph_path,
        )
        result = run_case(
            case,
            settings,
            mode=mode,  # type: ignore[arg-type]
            extractor=extractor,
            store=store,
        )
        recorded_at = FIXED_RECORDED_AT if mode == "structured" else None
        traces.append(
            build_trace(
                case=case,
                result=result,
                settings=settings,
                store=store,
                recorded_at=recorded_at,
            )
        )

    traces.sort(key=lambda trace: trace.get("traceId", ""))
    out_path.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(traces)} {mode} trace(s) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export graph-construction lab traces")
    parser.add_argument(
        "--mode",
        choices=["structured", "llm_assisted"],
        default="structured",
        help="Which execution mode to export (default: structured)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing traces file for the selected mode",
    )
    args = parser.parse_args()
    export_mode(args.mode, force=args.force)


if __name__ == "__main__":
    main()
