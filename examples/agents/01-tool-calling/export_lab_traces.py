"""Export tool-calling traces for a future Interactive Lab.

Model turns: reproducible case harness (provenance.model=case-harness).
Tool execution: real ToolExecutor against local fixtures (provenance.tools=measured).
Metrics: recorded from the run (provenance.metrics=measured).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import CASES, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_tool_calling_loop
from agent.tools import build_registry
from agent.trace import build_trace
from config import get_settings


def main() -> None:
    settings = get_settings()
    registry = build_registry(settings.data_dir)
    executor = ToolExecutor(registry, timeout_ms=settings.tool_timeout_ms)

    traces = []
    for case in CASES:
        result = run_tool_calling_loop(
            request=case.request,
            model=scripted_client_for(case),
            registry=registry,
            executor=executor,
            max_model_turns=settings.max_model_turns,
            max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
        )
        traces.append(build_trace(case=case, result=result, settings=settings))

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
