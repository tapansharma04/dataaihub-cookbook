"""Deterministic evaluator: apply explicit case criteria to a measured run.

The evaluator reads the run. It does not execute tools, call a model, or
mutate the measured trace. LLM-as-a-judge is not used here; it would be a
separate evaluator, not ground truth.
"""

from __future__ import annotations

from typing import Any

from agent.schemas import AgentRunResult, SequenceEvent
from evaluation.criteria import ArgumentExpectation, CaseCriteria
from evaluation.schemas import (
    DimensionCheck,
    EvaluationResult,
    RecoveryCheck,
    StepEfficiencyCheck,
)


def evaluate_run(
    result: AgentRunResult,
    criteria: CaseCriteria,
    *,
    case_id: str,
) -> EvaluationResult:
    """Score one measured run against case-level observable constraints."""
    calls = _tool_calls(result.sequence)
    observations = _observations(result.sequence)

    selection = _evaluate_tool_selection(calls, criteria)
    arguments = _evaluate_tool_arguments(calls, criteria)
    execution = _evaluate_tool_execution(observations, criteria)
    interpretation = _evaluate_result_interpretation(result, observations, criteria)
    efficiency = _evaluate_step_efficiency(result.metrics.tool_calls, criteria)
    recovery = _evaluate_recovery(observations, criteria)
    answer = _evaluate_final_answer(result.answer, criteria)

    evidence_gathered = selection.passed and arguments.passed and execution.passed
    recovery_ok = _recovery_satisfies_criteria(recovery, criteria)

    # FINAL ANSWER CORRECTNESS: required observable answer facts are present.
    # Distinct from result interpretation (observed evidence vs answer).
    final_answer_correct = answer.passed

    # TASK SUCCESS: assigned task succeeded under the case-level outcome
    # contract (required evidence gathered + correct answer facts + recovery
    # when the case expects it). Step efficiency is excluded on purpose.
    task_success = final_answer_correct and evidence_gathered and recovery_ok

    # TRAJECTORY SUCCESS: hard trajectory constraints only — not a unique
    # gold path, not final-answer correctness, and not result interpretation.
    # Hard constraints: required tool selection, required arguments, required
    # successful execution, step-efficiency constraint, recovery requirement.
    # result_interpretation is evaluated and reported independently; a
    # reported dimension is not automatically a boolean requirement here.
    trajectory_success = (
        selection.passed
        and arguments.passed
        and execution.passed
        and efficiency.passed()
        and recovery_ok
    )

    satisfied, violated = _constraint_lists(
        selection=selection,
        arguments=arguments,
        execution=execution,
        interpretation=interpretation,
        efficiency=efficiency,
        recovery=recovery,
        answer=answer,
        task_success=task_success,
        trajectory_success=trajectory_success,
        criteria=criteria,
    )

    return EvaluationResult(
        case_id=case_id,
        task_success=task_success,
        final_answer_correct=final_answer_correct,
        trajectory_success=trajectory_success,
        tool_selection=selection,
        tool_arguments=arguments,
        tool_execution=execution,
        result_interpretation=interpretation,
        step_efficiency=efficiency,
        recovery=recovery,
        constraints_satisfied=satisfied,
        constraints_violated=violated,
    )


def _tool_calls(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    return [
        {
            "name": event.detail.get("name"),
            "arguments": event.detail.get("arguments") or {},
        }
        for event in sequence
        if event.kind == "tool_call"
    ]


def _observations(sequence: list[SequenceEvent]) -> list[dict[str, Any]]:
    return [
        {
            "name": event.detail.get("name"),
            "ok": event.detail.get("ok"),
            "result": event.detail.get("result"),
            "error": event.detail.get("error"),
        }
        for event in sequence
        if event.kind == "observation"
    ]


def _evaluate_tool_selection(
    calls: list[dict[str, Any]],
    criteria: CaseCriteria,
) -> DimensionCheck:
    observed = [str(call["name"]) for call in calls if call.get("name")]
    missing = [name for name in criteria.required_tools if name not in observed]
    passed = not missing
    note = (
        "Required tools were selected."
        if passed
        else f"Missing required tools: {missing}."
    )
    return DimensionCheck(
        passed=passed,
        note=note,
        detail={
            "requiredTools": list(criteria.required_tools),
            "observedTools": observed,
            "missingTools": missing,
        },
    )


def _call_matches(call: dict[str, Any], expectation: ArgumentExpectation) -> bool:
    if call.get("name") != expectation.tool:
        return False
    args = call.get("arguments") or {}
    for key, expected in expectation.arguments.items():
        if args.get(key) != expected:
            return False
    for key, needle in expectation.argument_contains.items():
        value = args.get(key)
        if not isinstance(value, str) or needle.lower() not in value.lower():
            return False
    return True


def _evaluate_tool_arguments(
    calls: list[dict[str, Any]],
    criteria: CaseCriteria,
) -> DimensionCheck:
    checks: list[dict[str, Any]] = []
    failed: list[str] = []
    for expectation in criteria.required_arguments:
        matched = any(_call_matches(call, expectation) for call in calls)
        checks.append(
            {
                "tool": expectation.tool,
                "arguments": expectation.arguments,
                "argumentContains": expectation.argument_contains,
                "matched": matched,
            }
        )
        if not matched:
            failed.append(expectation.tool)
    passed = not failed
    note = (
        "Required tool arguments were satisfied by at least one call each."
        if passed
        else f"Argument constraints failed for: {failed}."
    )
    return DimensionCheck(
        passed=passed,
        note=note,
        detail={"checks": checks, "failedTools": failed},
    )


def _evaluate_tool_execution(
    observations: list[dict[str, Any]],
    criteria: CaseCriteria,
) -> DimensionCheck:
    successful = {str(obs["name"]) for obs in observations if obs.get("ok") is True}
    failed_names = [str(obs["name"]) for obs in observations if obs.get("ok") is False]
    missing = [name for name in criteria.required_tools if name not in successful]
    passed = not missing
    note = (
        "Required tools produced at least one successful observation each."
        if passed
        else f"No successful execution for required tools: {missing}."
    )
    return DimensionCheck(
        passed=passed,
        note=note,
        detail={
            "successfulTools": sorted(successful),
            "failedToolNames": failed_names,
            "missingSuccessfulTools": missing,
        },
    )


def _payments_status(observations: list[dict[str, Any]]) -> str | None:
    for obs in observations:
        if obs.get("name") != "get_service_status" or obs.get("ok") is not True:
            continue
        result = obs.get("result") or {}
        service = result.get("service") or {}
        if service.get("service") == "payments" and isinstance(
            service.get("status"), str
        ):
            return service["status"]
    return None


def _evaluate_result_interpretation(
    result: AgentRunResult,
    observations: list[dict[str, Any]],
    criteria: CaseCriteria,
) -> DimensionCheck:
    """Whether the answer reflects the observed payments status/evidence.

    Reported independently. Not a hard `trajectory_success` constraint.
    """
    answer = (result.answer or "").lower()
    status = _payments_status(observations)
    if status is None:
        return DimensionCheck(
            passed=False,
            note=(
                "No successful payments status observation was available to interpret."
            ),
            detail={"paymentsStatusObserved": None},
        )
    reflected = status.lower() in answer
    facts_from_obs = all(
        fact.lower() in answer
        for fact in criteria.required_answer_facts
        if fact.lower() in {"degraded", "pay-2041"}
    )
    passed = reflected and facts_from_obs
    note = (
        "Final answer reflects the payments status observation."
        if passed
        else (f"Final answer does not reflect observed payments status '{status}'.")
    )
    return DimensionCheck(
        passed=passed,
        note=note,
        detail={
            "paymentsStatusObserved": status,
            "statusReflectedInAnswer": reflected,
        },
    )


def _evaluate_step_efficiency(
    observed_tool_calls: int,
    criteria: CaseCriteria,
) -> StepEfficiencyCheck:
    limit = criteria.max_useful_tool_calls
    if limit is None:
        return StepEfficiencyCheck(
            status="not_applicable",
            observed_tool_calls=observed_tool_calls,
            max_useful_tool_calls=None,
            note="This case does not define a maximum useful tool-call count.",
        )
    if observed_tool_calls <= limit:
        return StepEfficiencyCheck(
            status="pass",
            observed_tool_calls=observed_tool_calls,
            max_useful_tool_calls=limit,
            note=(
                f"Observed {observed_tool_calls} tool call(s); case constraint "
                f"allows at most {limit} useful tool call(s)."
            ),
        )
    return StepEfficiencyCheck(
        status="fail",
        observed_tool_calls=observed_tool_calls,
        max_useful_tool_calls=limit,
        note=(
            f"Observed {observed_tool_calls} tool call(s), which exceeds the "
            f"case constraint of at most {limit} useful tool call(s)."
        ),
    )


def _evaluate_recovery(
    observations: list[dict[str, Any]],
    criteria: CaseCriteria,
) -> RecoveryCheck:
    failures = [i for i, obs in enumerate(observations) if obs.get("ok") is False]
    recovered = 0
    for index in failures:
        failed_name = observations[index].get("name")
        later_success = any(
            later.get("ok") is True and later.get("name") == failed_name
            for later in observations[index + 1 :]
        )
        if later_success:
            recovered += 1

    failure_count = len(failures)
    rate = (recovered / failure_count) if failure_count else None
    attempted = failure_count > 0 and (
        recovered > 0
        or any(
            observations[index + 1 :]
            for index in failures
            if index + 1 < len(observations)
        )
    )
    succeeded = failure_count > 0 and recovered == failure_count

    if failure_count == 0:
        status: str = "not_applicable"
        note = "No tool-call failures occurred; recovery is not applicable."
        attempted = False
        succeeded = False
    elif succeeded:
        status = "recovered"
        note = (
            f"Recovered failures / failures = {recovered}/{failure_count}. "
            "This is a single-case ratio, not a production estimate."
        )
    else:
        status = "not_recovered"
        note = (
            f"Recovered failures / failures = {recovered}/{failure_count}. "
            "At least one tool failure was not followed by a successful "
            "retry of the same tool."
        )

    if criteria.recovery_expected and status == "not_applicable":
        note = "Recovery was expected but no tool-call failure occurred."

    return RecoveryCheck(
        status=status,  # type: ignore[arg-type]
        attempted=attempted,
        succeeded=succeeded,
        failures=failure_count,
        recovered_failures=recovered,
        error_recovery_rate=rate,
        note=note,
    )


def _recovery_satisfies_criteria(
    recovery: RecoveryCheck,
    criteria: CaseCriteria,
) -> bool:
    if criteria.recovery_expected:
        return recovery.status == "recovered" and recovery.succeeded
    return recovery.status in {"not_applicable", "recovered"}


def _evaluate_final_answer(answer: str, criteria: CaseCriteria) -> DimensionCheck:
    """Whether required observable answer facts are present in the text."""
    text = answer or ""
    lowered = text.lower()
    missing = [
        fact for fact in criteria.required_answer_facts if fact.lower() not in lowered
    ]
    forbidden_hits = [
        fact for fact in criteria.forbidden_answer_facts if fact.lower() in lowered
    ]
    passed = not missing and not forbidden_hits
    if passed:
        note = "Final answer contains the required observable facts."
    elif missing:
        note = f"Final answer is missing required facts: {missing}."
    else:
        note = f"Final answer contains forbidden claims: {forbidden_hits}."
    return DimensionCheck(
        passed=passed,
        note=note,
        detail={"missingFacts": missing, "forbiddenHits": forbidden_hits},
    )


def _constraint_lists(
    *,
    selection: DimensionCheck,
    arguments: DimensionCheck,
    execution: DimensionCheck,
    interpretation: DimensionCheck,
    efficiency: StepEfficiencyCheck,
    recovery: RecoveryCheck,
    answer: DimensionCheck,
    task_success: bool,
    trajectory_success: bool,
    criteria: CaseCriteria,
) -> tuple[list[str], list[str]]:
    satisfied: list[str] = []
    violated: list[str] = []

    def record(label: str, ok: bool) -> None:
        (satisfied if ok else violated).append(label)

    record("tool_selection", selection.passed)
    record("tool_arguments", arguments.passed)
    record("tool_execution", execution.passed)
    # Reported dimension; not an input to trajectory_success.
    record("result_interpretation", interpretation.passed)
    record("step_efficiency", efficiency.passed())
    record(
        "error_recovery",
        _recovery_satisfies_criteria(recovery, criteria),
    )
    record("final_answer_correct", answer.passed)
    record("task_success", task_success)
    record("trajectory_success", trajectory_success)
    return satisfied, violated
