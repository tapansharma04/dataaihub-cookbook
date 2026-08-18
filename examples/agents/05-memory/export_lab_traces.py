"""Export agent-memory traces for a future Interactive Lab.

Model turns: reproducible case harness (provenance.model=case-harness).
Memory operations: real MemoryStore against local fixtures
(provenance.tools=measured).
Metrics: recorded from the run (provenance.metrics=measured).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import CASES, scripted_client_for
from agent.loop import run_memory_loop
from agent.memory import FixedClock, MemoryStore
from agent.source import AuthoritativeStore
from agent.trace import build_trace
from config import get_settings


def _known_scopes(data_dir: Path) -> set[str]:
    users = json.loads((data_dir / "users.json").read_text(encoding="utf-8"))
    return set(users)


def main() -> None:
    settings = get_settings()
    data_dir = settings.data_dir
    known_scopes = _known_scopes(data_dir)
    authoritative = AuthoritativeStore.from_data_dir(data_dir)

    traces = []
    for case in CASES:
        max_turns = case.max_turns if case.max_turns is not None else settings.max_turns
        store = MemoryStore(known_scopes=known_scopes, clock=FixedClock())
        result = run_memory_loop(
            interactions=[item.request for item in case.interactions],
            scope=case.scope,
            model=scripted_client_for(case),
            memory_store=store,
            authoritative=authoritative,
            max_turns=max_turns,
        )
        traces.append(
            build_trace(
                case=case,
                result=result,
                settings=settings,
                max_turns=max_turns,
            )
        )

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
