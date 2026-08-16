"""Evaluator tests — deterministic rubric application, no paid APIs."""

from __future__ import annotations

import copy
from pathlib import Path

from agent.cases import CASES, get_case
from agent.executor import ToolExecutor
from agent.loop import run_agent_loop
from agent.model import ScriptedModelClient, ScriptedTurn
from agent.run import run_measured_case
from agent.tools import build_registry
from evaluation.criteria import (
    PAYMENTS_INVESTIGATION_CRITERIA,
    ArgumentExpectation,
    CaseCriteria,
)
from evaluation.evaluator import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SUCCESS_FACTS_ANSWER = (
    "Payments is degraded in us-east-1 (PAY-2041: elevated card-auth latency). "
    "Per the payments degradation runbook, enterprise operators may see slower "
    "checkout confirmations."
)


def _eval(trace_id: str):
    case = get_case(trace_id)
    result = run_measured_case(case, data_dir=DATA)
    evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
    return case, result, evaluation


def test_all_four_cases_execute():
    assert len(CASES) == 4
    for case in CASES:
        result = run_measured_case(case, data_dir=DATA)
        assert result.metrics.termination_reason == "final_answer"
        assert result.answer
        assert result.sequence[0].kind == "user_request"


def test_task_success_classified_true():
    _, result, evaluation = _eval("task-success-payments-docs")
    assert evaluation.task_success is True
    assert evaluation.final_answer_correct is True
    assert evaluation.trajectory_success is True
    assert evaluation.tool_selection.passed is True
    assert evaluation.tool_arguments.passed is True
    assert evaluation.tool_execution.passed is True
    assert evaluation.result_interpretation.passed is True
    assert evaluation.step_efficiency.status == "pass"
    assert evaluation.recovery.status == "not_applicable"
    assert result.metrics.tool_calls == 2
    assert result.metrics.failed_tool_calls == 0


def test_partial_success_detects_inefficient_trajectory():
    _, result, evaluation = _eval("partial-success-extra-profile")
    assert evaluation.task_success is True
    assert evaluation.final_answer_correct is True
    assert evaluation.trajectory_success is False
    assert evaluation.step_efficiency.status == "fail"
    assert evaluation.step_efficiency.observed_tool_calls == 3
    assert evaluation.step_efficiency.max_useful_tool_calls == 2
    assert "exceeds" in evaluation.step_efficiency.note.lower()
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names[-1] == "get_user_profile"
    assert "get_user_profile" in evaluation.tool_selection.detail["observedTools"]


def test_recovery_preserves_failure_and_classifies_recovered():
    _, result, evaluation = _eval("tool-error-recovery-payments")
    observations = [e for e in result.sequence if e.kind == "observation"]
    assert observations[0].detail["ok"] is False
    assert observations[0].detail["error"]["code"] == "unknown_service"
    assert observations[1].detail["ok"] is True
    assert result.metrics.failed_tool_calls == 1
    assert result.metrics.successful_tool_calls == 2

    assert evaluation.task_success is True
    assert evaluation.final_answer_correct is True
    assert evaluation.trajectory_success is True
    assert evaluation.recovery.status == "recovered"
    assert evaluation.recovery.attempted is True
    assert evaluation.recovery.succeeded is True
    assert evaluation.recovery.failures == 1
    assert evaluation.recovery.recovered_failures == 1
    assert evaluation.recovery.error_recovery_rate == 1.0
    assert "single-case" in evaluation.recovery.note.lower()


def test_goal_miss_classified_as_failure():
    _, result, evaluation = _eval("goal-miss-wrong-answer")
    assert result.metrics.termination_reason == "final_answer"
    assert result.answer
    assert evaluation.task_success is False
    assert evaluation.final_answer_correct is False
    assert evaluation.result_interpretation.passed is False
    assert evaluation.trajectory_success is True
    assert evaluation.tool_selection.passed is True
    assert evaluation.tool_arguments.passed is True
    assert evaluation.step_efficiency.status == "pass"


def test_goal_miss_separates_outcome_from_trajectory_constraints():
    """GOAL_MISS: interpretation/answer can fail without failing trajectory.

    Result interpretation is a reported dimension, not a hard trajectory
    constraint. Do not fold it into trajectory_success.
    """
    _, result, evaluation = _eval("goal-miss-wrong-answer")
    assert result.metrics.termination_reason == "final_answer"
    assert evaluation.final_answer_correct is False
    assert evaluation.result_interpretation.passed is False
    assert evaluation.trajectory_success is True
    assert evaluation.task_success is False
    assert evaluation.tool_selection.passed is True
    assert evaluation.tool_arguments.passed is True
    assert evaluation.tool_execution.passed is True
    assert evaluation.step_efficiency.status == "pass"
    assert evaluation.recovery.status == "not_applicable"


def test_tool_selection_evaluation_fails_when_required_tool_missing():
    registry = build_registry(DATA)
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "get_user_profile",
                        "arguments": {"user_id": "u-1001"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(content=SUCCESS_FACTS_ANSWER, finish_reason="stop"),
        ]
    )
    result = run_agent_loop(
        request="Check payments",
        model=model,
        registry=registry,
        executor=ToolExecutor(registry),
    )
    evaluation = evaluate_run(
        result, PAYMENTS_INVESTIGATION_CRITERIA, case_id="synthetic-selection"
    )
    assert evaluation.tool_selection.passed is False
    assert "get_service_status" in evaluation.tool_selection.detail["missingTools"]
    assert evaluation.task_success is False


def test_argument_evaluation_fails_on_wrong_service():
    registry = build_registry(DATA)
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "get_service_status",
                        "arguments": {"service": "billing"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "search_documentation",
                        "arguments": {"query": "payments degradation"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(content=SUCCESS_FACTS_ANSWER, finish_reason="stop"),
        ]
    )
    result = run_agent_loop(
        request="Check payments",
        model=model,
        registry=registry,
        executor=ToolExecutor(registry),
    )
    evaluation = evaluate_run(
        result, PAYMENTS_INVESTIGATION_CRITERIA, case_id="synthetic-args"
    )
    assert evaluation.tool_selection.passed is True
    assert evaluation.tool_arguments.passed is False
    assert "get_service_status" in evaluation.tool_arguments.detail["failedTools"]
    assert evaluation.trajectory_success is False
    assert evaluation.task_success is False


def test_result_interpretation_is_independent_of_answer_facts():
    """Answer facts can pass while interpretation fails on observed status.

    Uses a synthetic run and a custom fact list so the four measured cases
    stay unchanged. Interpretation remains a reported dimension, not a hard
    trajectory constraint.
    """
    registry = build_registry(DATA)
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "get_service_status",
                        "arguments": {"service": "payments"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "search_documentation",
                        "arguments": {"query": "payments degradation"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=(
                    "Incident PAY-2041 is on file and checkout may be slower, "
                    "but the payments service is operational."
                ),
                finish_reason="stop",
            ),
        ]
    )
    result = run_agent_loop(
        request="Check payments",
        model=model,
        registry=registry,
        executor=ToolExecutor(registry),
    )
    criteria = CaseCriteria(
        required_tools=("get_service_status", "search_documentation"),
        required_arguments=(
            ArgumentExpectation(
                tool="get_service_status",
                arguments={"service": "payments"},
            ),
            ArgumentExpectation(
                tool="search_documentation",
                argument_contains={"query": "payment"},
            ),
        ),
        required_answer_facts=("PAY-2041", "checkout"),
        max_useful_tool_calls=2,
        recovery_expected=False,
    )
    evaluation = evaluate_run(result, criteria, case_id="synthetic-interpretation")

    payments_obs = next(
        e
        for e in result.sequence
        if e.kind == "observation"
        and e.detail.get("name") == "get_service_status"
        and e.detail.get("ok") is True
    )
    assert payments_obs.detail["result"]["service"]["status"] == "degraded"
    assert evaluation.tool_selection.passed is True
    assert evaluation.tool_arguments.passed is True
    assert evaluation.tool_execution.passed is True
    assert evaluation.final_answer_correct is True
    assert evaluation.result_interpretation.passed is False
    assert evaluation.result_interpretation.detail["paymentsStatusObserved"] == (
        "degraded"
    )
    assert evaluation.trajectory_success is True
    assert evaluation.task_success is True
    assert evaluation.step_efficiency.status == "pass"


def test_final_answer_correct_is_not_task_success_without_evidence():
    """Hallucinated facts are not task success if required tools were skipped."""
    registry = build_registry(DATA)
    model = ScriptedModelClient(
        [ScriptedTurn(content=SUCCESS_FACTS_ANSWER, finish_reason="stop")]
    )
    result = run_agent_loop(
        request="Check payments",
        model=model,
        registry=registry,
        executor=ToolExecutor(registry),
    )
    evaluation = evaluate_run(
        result, PAYMENTS_INVESTIGATION_CRITERIA, case_id="synthetic-hallucination"
    )
    assert evaluation.final_answer_correct is True
    assert evaluation.tool_selection.passed is False
    assert evaluation.task_success is False
    assert evaluation.trajectory_success is False


def test_evaluation_does_not_modify_measured_trace():
    case = get_case("tool-error-recovery-payments")
    result = run_measured_case(case, data_dir=DATA)
    before_sequence = copy.deepcopy([e.model_dump() for e in result.sequence])
    before_metrics = result.metrics.model_dump()
    before_answer = result.answer
    before_errors = copy.deepcopy(result.errors)

    evaluate_run(result, case.criteria, case_id=case.trace_id)

    assert [e.model_dump() for e in result.sequence] == before_sequence
    assert result.metrics.model_dump() == before_metrics
    assert result.answer == before_answer
    assert result.errors == before_errors


def test_operational_metrics_are_present_and_not_quality_scores():
    _, result, evaluation = _eval("task-success-payments-docs")
    metrics = result.metrics
    assert metrics.provenance == "measured"
    assert metrics.total_ms >= 0
    assert metrics.model_ms >= 0
    assert metrics.tool_ms >= 0
    assert metrics.model_turns == 3
    assert metrics.tool_calls == 2
    dumped = evaluation.to_public_dict()
    assert "totalMs" not in dumped
    assert "modelMs" not in dumped
    assert "agentScore" not in dumped
    assert dumped["provenance"] == "computed"


def test_deterministic_evaluation_is_stable():
    first = _eval("partial-success-extra-profile")[2].to_public_dict()
    second = _eval("partial-success-extra-profile")[2].to_public_dict()
    assert first == second


def test_no_api_key_required_for_measured_cases(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    for case in CASES:
        result = run_measured_case(case, data_dir=DATA)
        evaluation = evaluate_run(result, case.criteria, case_id=case.trace_id)
        assert evaluation.case_id == case.trace_id


def test_custom_efficiency_constraint_is_case_specific():
    criteria = CaseCriteria(
        required_tools=("get_service_status",),
        required_arguments=(
            ArgumentExpectation(
                tool="get_service_status",
                arguments={"service": "payments"},
            ),
        ),
        required_answer_facts=("degraded",),
        max_useful_tool_calls=1,
    )
    case = get_case("task-success-payments-docs")
    result = run_measured_case(case, data_dir=DATA)
    evaluation = evaluate_run(result, criteria, case_id="tight-efficiency")
    assert evaluation.step_efficiency.status == "fail"
    assert evaluation.step_efficiency.observed_tool_calls == 2
    assert evaluation.step_efficiency.max_useful_tool_calls == 1
