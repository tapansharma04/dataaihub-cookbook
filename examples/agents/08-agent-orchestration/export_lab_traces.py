"""Export orchestration traces as a recorded execution artifact.

Synthesizer: deterministic mock (provenance.model=mock).
Agent execution: real specialists against local catalogs / upstream results
(provenance.tools=measured).
Metrics: recorded from the run (provenance.metrics=measured).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import CASES
from agent.catalog import Catalog
from agent.runtime import run_orchestration
from agent.synthesizer import MockSynthesizer
from agent.trace import build_trace
from config import get_settings


def main() -> None:
    settings = get_settings()
    catalog = Catalog(settings.data_dir)
    traces = []
    for case in CASES:
        result = run_orchestration(
            case,
            catalog=catalog,
            synthesizer=MockSynthesizer(),
            max_nodes=settings.max_nodes,
        )
        traces.append(build_trace(case=case, result=result, settings=settings))

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
