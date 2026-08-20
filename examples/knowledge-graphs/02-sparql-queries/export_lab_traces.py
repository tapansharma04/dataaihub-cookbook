"""Export SPARQL query traces for a future Interactive Lab.

No LLM is used. SPARQL queries are measured against a local in-memory RDF
graph loaded from fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from sparql.cases import CASES
from sparql.graph import RdfGraphStore
from sparql.runner import run_case
from sparql.trace import build_trace


def main() -> None:
    settings = get_settings()
    store = RdfGraphStore.from_path(settings.graph_path)
    traces = []
    for case in CASES:
        result = run_case(case, store, settings)
        traces.append(
            build_trace(case=case, result=result, settings=settings, store=store)
        )

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
