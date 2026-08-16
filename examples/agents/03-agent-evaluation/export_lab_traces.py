"""Export agent-evaluation traces for a future Interactive Lab.

Model turns: reproducible case harness (provenance.model=case-harness).
Tool execution: real ToolExecutor against local fixtures (provenance.tools=measured).
Metrics: recorded from the run (provenance.metrics=measured).
Evaluation: computed from the measured trace + explicit case criteria
(evaluationProvenance=computed). Evaluation does not rewrite the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import CASES
from agent.run import run_measured_case
from agent.trace import build_trace
from config import get_settings
from evaluation.evaluator import evaluate_run


def main() -> None:
    settings = get_settings()
    traces = []
    for case in CASES:
        max_turns = case.max_turns if case.max_turns is not None else settings.max_turns
        result = run_measured_case(case, settings=settings)
        evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
        traces.append(
            build_trace(
                case=case,
                result=result,
                settings=settings,
                max_turns=max_turns,
                evaluation=evaluation,
            )
        )

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
