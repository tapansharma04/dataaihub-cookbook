"""Export MCP resource traces for a future Interactive Lab.

No LLM is used. Protocol interactions are measured through the official MCP
Python SDK client/server boundary (in-process InMemoryTransport).
"""

from __future__ import annotations

import json
from pathlib import Path

from client.cases import CASES
from client.runner import run_case
from client.trace import build_trace
from config import get_settings


def main() -> None:
    settings = get_settings()
    traces = []
    for case in CASES:
        result = run_case(case, settings=settings)
        traces.append(build_trace(case=case, result=result, settings=settings))

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
