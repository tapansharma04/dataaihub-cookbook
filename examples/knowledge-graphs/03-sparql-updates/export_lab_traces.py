"""Export SPARQL UPDATE traces for a future Interactive Lab.

No LLM is used. SPARQL UPDATEs are measured against a local in-memory RDF
graph loaded fresh from fixtures for each case.
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
    traces = []
    for case in CASES:
        # Fresh graph per case — isolation is part of the measured contract.
        store = RdfGraphStore.fresh_from_path(settings.graph_path)
        result = run_case(case, settings, store=store)
        traces.append(
            build_trace(case=case, result=result, settings=settings, store=store)
        )

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
