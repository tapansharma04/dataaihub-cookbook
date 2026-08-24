"""Export GraphRAG traces for a future Interactive Lab.

GRAPH_GROUNDED traces are written to lab_traces.json (deterministic).
GRAPHRAG_LLM traces are written to lab_traces_llm.json (separate file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from graphrag.cases import CASES
from graphrag.graph import RdfGraphStore
from graphrag.llm import build_llm_client
from graphrag.runner import run_case
from graphrag.trace import build_trace

ROOT = Path(__file__).resolve().parent
LAB_TRACES_PATH = ROOT / "lab_traces.json"
LAB_TRACES_LLM_PATH = ROOT / "lab_traces_llm.json"

OUTPUT_BY_MODE = {
    "graph_grounded": LAB_TRACES_PATH,
    "graphrag_llm": LAB_TRACES_LLM_PATH,
}


def export_mode(mode: str, *, force: bool = False) -> None:
    out_path = OUTPUT_BY_MODE[mode]
    if out_path.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing traces in {out_path}. "
            "Pass --force to replace them."
        )

    settings = get_settings()
    store = RdfGraphStore.from_path(settings.graph_path)
    llm_client = None
    if mode == "graphrag_llm":
        llm_client = build_llm_client(settings)
        if llm_client is None:
            raise SystemExit(
                "GRAPHRAG_LLM export requires OPENAI_API_KEY and a configured model."
            )

    traces = []
    for case in CASES:
        result = run_case(
            case,
            settings,
            mode=mode,  # type: ignore[arg-type]
            store=store,
            llm_client=llm_client,
        )
        traces.append(
            build_trace(case=case, result=result, settings=settings, store=store)
        )

    traces.sort(key=lambda trace: trace.get("traceId", ""))
    out_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} {mode} trace(s) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GraphRAG lab traces")
    parser.add_argument(
        "--mode",
        choices=["graph_grounded", "graphrag_llm"],
        required=True,
        help="Which execution mode to export",
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
