"""Trace schema, provenance, CoT, and export tests — no network / no paid APIs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent.cases import CASES, get_case
from agent.run import run_measured_case
from agent.trace import build_signature_view, build_trace
from config import EXAMPLE_ID, Settings
from evaluation.evaluator import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAB_TRACES_PATH = ROOT / "lab_traces.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {"TASK_SUCCESS", "PARTIAL_SUCCESS", "TOOL_ERROR_RECOVERY", "GOAL_MISS"}
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "task-success-payments-docs",
        "partial-success-extra-profile",
        "tool-error-recovery-payments",
        "goal-miss-wrong-answer",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {"final_answer", "max_turns", "invalid_action", "error"}
)
COT_FIELD_NAMES = frozenset(
    {
        "chainOfThought",
        "chain_of_thought",
        "reasoning",
        "hiddenReasoning",
        "hidden_reasoning",
        "internalReasoning",
        "internal_reasoning",
        "thoughtProcess",
        "thought_process",
        "thought",
        "thoughts",
        "privateReasoning",
        "private_reasoning",
    }
)
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "totalMs",
        "modelMs",
        "toolMs",
        "total_ms",
        "model_ms",
        "tool_ms",
    }
)


def _collect_cot_violations(obj: Any, path: str = "") -> list[str]:
    """Return dotted paths where hidden-reasoning field names appear."""
    violations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in COT_FIELD_NAMES:
                violations.append(child_path)
            violations.extend(_collect_cot_violations(value, child_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(_collect_cot_violations(item, f"{path}[{i}]"))
    return violations


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_volatile(value)
            for key, value in obj.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(item) for item in obj]
    return obj


def _build(trace_id: str) -> dict[str, Any]:
    settings = Settings(openai_api_key="", data_dir=DATA)
    case = get_case(trace_id)
    result = run_measured_case(case, settings=settings, data_dir=DATA)
    evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
    max_turns = case.max_turns if case.max_turns is not None else settings.max_turns
    return build_trace(
        case=case,
        result=result,
        settings=settings,
        max_turns=max_turns,
        evaluation=evaluation,
    )


def _assert_evaluation_schema(evaluation: dict[str, Any]) -> None:
    assert set(evaluation) >= {
        "caseId",
        "provenance",
        "taskSuccess",
        "finalAnswerCorrect",
        "trajectorySuccess",
        "toolSelection",
        "toolArguments",
        "toolExecution",
        "resultInterpretation",
        "stepEfficiency",
        "recovery",
        "constraintsSatisfied",
        "constraintsViolated",
    }
    assert evaluation["provenance"] == "computed"
    assert isinstance(evaluation["taskSuccess"], bool)
    assert isinstance(evaluation["finalAnswerCorrect"], bool)
    assert isinstance(evaluation["trajectorySuccess"], bool)
    assert isinstance(evaluation["toolSelection"]["passed"], bool)
    assert isinstance(evaluation["toolArguments"]["passed"], bool)
    assert evaluation["stepEfficiency"]["status"] in {
        "pass",
        "fail",
        "not_applicable",
    }
    assert evaluation["recovery"]["status"] in {
        "not_applicable",
        "recovered",
        "not_recovered",
    }
    assert "attempted" in evaluation["recovery"]
    assert "succeeded" in evaluation["recovery"]
    assert "agentScore" not in evaluation


def _assert_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert isinstance(trace["traceId"], str) and trace["traceId"]
    assert trace["metricsProvenance"] == "measured"
    assert trace["evaluationProvenance"] == "computed"
    assert trace["provenance"] == {
        "model": "case-harness",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert "tools" in trace and trace["tools"]
    assert "sequence" in trace and len(trace["sequence"]) >= 2
    assert "steps" in trace and len(trace["steps"]) >= 2
    assert "state" in trace
    assert "metrics" in trace
    assert "evaluation" in trace
    assert "criteria" in trace["input"]
    _assert_evaluation_schema(trace["evaluation"])
    assert set(trace["metrics"]) >= {
        "totalMs",
        "modelMs",
        "toolMs",
        "modelTurns",
        "toolCalls",
        "successfulToolCalls",
        "failedToolCalls",
        "terminationReason",
        "maxTurns",
        "provenance",
    }
    assert trace["metrics"]["provenance"] == "measured"
    assert "taskSuccess" not in trace["metrics"]

    termination_reason = trace["metrics"]["terminationReason"]
    assert termination_reason in SUPPORTED_TERMINATION_REASONS
    assert trace["output"]["terminationReason"] == termination_reason
    assert trace["state"]["terminationReason"] == termination_reason

    termination_steps = [s for s in trace["steps"] if s["type"] == "termination"]
    assert len(termination_steps) >= 1
    assert termination_steps[-1]["detail"]["reason"] == termination_reason
    termination_events = [e for e in trace["sequence"] if e["kind"] == "termination"]
    assert len(termination_events) >= 1
    assert termination_events[-1]["detail"]["reason"] == termination_reason

    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    assert trace["architecture"]["layout"] == "agent-evaluation"
    signature_phases = [v["phase"] for v in trace["presentation"]["signatureView"]]
    assert signature_phases[-1] == "EVALUATION"

    for event in trace["sequence"]:
        assert "explanation" not in event
        assert "teachingNote" not in event.get("detail", {})
        assert event["kind"] != "evaluation"

    for step in trace["steps"]:
        if step["type"] == "llm":
            assert step["metrics"]["provenance"] == "case-harness"
        elif step["type"] in {"tool_call", "observation"}:
            assert step["metrics"]["provenance"] == "measured"

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []


def test_all_four_cases_have_corresponding_traces():
    traces = [_build(case.trace_id) for case in CASES]
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    assert {t["exampleClass"] for t in traces} == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_trace_contract(trace)


def test_build_trace_separates_teaching_metadata():
    trace = _build("task-success-payments-docs")
    _assert_trace_contract(trace)
    assert "stageNotes" in trace["presentation"]
    assert "evaluationNote" in trace["presentation"]
    assert trace["evaluation"]["taskSuccess"] is True


def test_signature_view_appends_evaluation_without_rewriting_sequence():
    case = get_case("task-success-payments-docs")
    result = run_measured_case(case, data_dir=DATA)
    evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
    kinds = [e.kind for e in result.sequence]
    assert "evaluation" not in kinds
    view = build_signature_view(result.sequence, evaluation)
    phases = [v["phase"] for v in view]
    assert phases[-1] == "EVALUATION"
    assert kinds == [e.kind for e in result.sequence]


def test_evaluation_does_not_modify_sequence_in_exported_trace():
    case = get_case("tool-error-recovery-payments")
    result = run_measured_case(case, data_dir=DATA)
    before = copy.deepcopy([e.model_dump() for e in result.sequence])
    settings = Settings(openai_api_key="", data_dir=DATA)
    trace = build_trace(
        case=case,
        result=result,
        settings=settings,
        max_turns=settings.max_turns,
    )
    assert [e.model_dump() for e in result.sequence] == before
    exported_kinds = [e["kind"] for e in trace["sequence"]]
    assert exported_kinds == [e.kind for e in result.sequence]
    assert trace["evaluation"]["recovery"]["status"] == "recovered"


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES) == 4
    assert {t["traceId"] for t in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_trace_contract(trace)


def test_committed_traces_match_regenerated_semantically():
    if not LAB_TRACES_PATH.exists():
        return
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    regenerated = [_build(case.trace_id) for case in CASES]
    committed_by_id = {t["traceId"]: t for t in committed}
    for trace in regenerated:
        left = _strip_volatile(committed_by_id[trace["traceId"]])
        right = _strip_volatile(trace)
        assert left == right


def test_no_chain_of_thought_in_lab_traces():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        violations = _collect_cot_violations(trace)
        assert violations == [], (
            f"{trace['traceId']} contains hidden-reasoning fields: {violations}"
        )


def test_no_hidden_reasoning_in_fresh_traces():
    for case in CASES:
        violations = _collect_cot_violations(_build(case.trace_id))
        assert violations == []


def test_trace_provenance_model():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    for trace in traces:
        assert trace["provenance"]["model"] == "case-harness"
        assert trace["provenance"]["tools"] == "measured"
        assert trace["provenance"]["metrics"] == "measured"
        assert trace["evaluationProvenance"] == "computed"
        assert trace["input"]["config"]["modelDriver"] == "case-harness"


def test_recovery_trace_preserves_failed_observation():
    trace = _build("tool-error-recovery-payments")
    observations = [e for e in trace["sequence"] if e["kind"] == "observation"]
    assert observations[0]["detail"]["ok"] is False
    assert observations[1]["detail"]["ok"] is True
    assert trace["evaluation"]["recovery"]["status"] == "recovered"
    assert trace["metrics"]["failedToolCalls"] == 1
