"""Structured evaluation result schemas.

Evaluation is computed from a measured trace plus an explicit case rubric.
It is not a live-model score and not an industry-standard agent quality index.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EfficiencyStatus = Literal["pass", "fail", "not_applicable"]
RecoveryStatus = Literal["not_applicable", "recovered", "not_recovered"]
CheckProvenance = Literal["computed"]


class DimensionCheck(BaseModel):
    passed: bool
    note: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class StepEfficiencyCheck(BaseModel):
    status: EfficiencyStatus
    observed_tool_calls: int
    max_useful_tool_calls: int | None = None
    note: str = ""

    def passed(self) -> bool:
        return self.status in {"pass", "not_applicable"}


class RecoveryCheck(BaseModel):
    status: RecoveryStatus
    attempted: bool
    succeeded: bool
    failures: int
    recovered_failures: int
    error_recovery_rate: float | None = None
    note: str = ""


class EvaluationResult(BaseModel):
    """Outcome + trajectory evaluation for one agent run.

    Operational metrics live on the measured run, not here. This object is a
    demonstration-specific rubric application, not a benchmark score.

    `final_answer_correct` — required observable answer facts are present.
    `result_interpretation` — answer reflects observed tool evidence; reported
    independently, not a hard trajectory constraint.
    `task_success` — assigned task succeeded under the case outcome contract.
    `trajectory_success` — designated hard trajectory constraints were
    satisfied (selection, arguments, execution, efficiency, recovery).
    Not every reported dimension is a boolean requirement for trajectory
    success.
    """

    case_id: str
    provenance: CheckProvenance = "computed"
    task_success: bool
    final_answer_correct: bool
    trajectory_success: bool
    tool_selection: DimensionCheck
    tool_arguments: DimensionCheck
    tool_execution: DimensionCheck
    result_interpretation: DimensionCheck
    step_efficiency: StepEfficiencyCheck
    recovery: RecoveryCheck
    constraints_satisfied: list[str] = Field(default_factory=list)
    constraints_violated: list[str] = Field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "provenance": self.provenance,
            "taskSuccess": self.task_success,
            "finalAnswerCorrect": self.final_answer_correct,
            "trajectorySuccess": self.trajectory_success,
            "toolSelection": {
                "passed": self.tool_selection.passed,
                "note": self.tool_selection.note,
                **self.tool_selection.detail,
            },
            "toolArguments": {
                "passed": self.tool_arguments.passed,
                "note": self.tool_arguments.note,
                **self.tool_arguments.detail,
            },
            "toolExecution": {
                "passed": self.tool_execution.passed,
                "note": self.tool_execution.note,
                **self.tool_execution.detail,
            },
            "resultInterpretation": {
                "passed": self.result_interpretation.passed,
                "note": self.result_interpretation.note,
                **self.result_interpretation.detail,
            },
            "stepEfficiency": {
                "status": self.step_efficiency.status,
                "observedToolCalls": self.step_efficiency.observed_tool_calls,
                "maxUsefulToolCalls": self.step_efficiency.max_useful_tool_calls,
                "note": self.step_efficiency.note,
            },
            "recovery": {
                "status": self.recovery.status,
                "attempted": self.recovery.attempted,
                "succeeded": self.recovery.succeeded,
                "failures": self.recovery.failures,
                "recoveredFailures": self.recovery.recovered_failures,
                "errorRecoveryRate": self.recovery.error_recovery_rate,
                "note": self.recovery.note,
            },
            "constraintsSatisfied": list(self.constraints_satisfied),
            "constraintsViolated": list(self.constraints_violated),
        }
