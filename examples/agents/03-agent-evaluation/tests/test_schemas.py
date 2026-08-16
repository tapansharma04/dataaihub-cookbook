"""Schema serialization smoke tests for runs and evaluation results."""

from __future__ import annotations

from agent.schemas import AgentRunMetrics, ToolCallRequest
from evaluation.schemas import (
    DimensionCheck,
    EvaluationResult,
    RecoveryCheck,
    StepEfficiencyCheck,
)


def test_metrics_are_operational_only():
    metrics = AgentRunMetrics(
        total_ms=1,
        model_ms=1,
        tool_ms=0,
        model_turns=1,
        tool_calls=0,
        successful_tool_calls=0,
        failed_tool_calls=0,
        termination_reason="final_answer",
        max_turns=6,
    )
    payload = metrics.model_dump()
    assert payload["provenance"] == "measured"
    assert "task_success" not in payload
    assert "score" not in payload


def test_tool_call_request_roundtrip():
    call = ToolCallRequest(
        id="1",
        name="get_service_status",
        arguments={"service": "payments"},
    )
    assert call.arguments["service"] == "payments"


def test_evaluation_result_public_dict_uses_camel_case():
    result = EvaluationResult(
        case_id="demo",
        task_success=True,
        final_answer_correct=True,
        trajectory_success=True,
        tool_selection=DimensionCheck(passed=True, note="ok"),
        tool_arguments=DimensionCheck(passed=True, note="ok"),
        tool_execution=DimensionCheck(passed=True, note="ok"),
        result_interpretation=DimensionCheck(passed=True, note="ok"),
        step_efficiency=StepEfficiencyCheck(
            status="pass",
            observed_tool_calls=2,
            max_useful_tool_calls=2,
            note="within case limit",
        ),
        recovery=RecoveryCheck(
            status="not_applicable",
            attempted=False,
            succeeded=False,
            failures=0,
            recovered_failures=0,
            error_recovery_rate=None,
            note="n/a",
        ),
    )
    payload = result.to_public_dict()
    assert payload["caseId"] == "demo"
    assert payload["taskSuccess"] is True
    assert payload["toolSelection"]["passed"] is True
    assert payload["stepEfficiency"]["status"] == "pass"
    assert payload["recovery"]["status"] == "not_applicable"
    assert "reasoning" not in payload
    assert "thought" not in payload
    assert "agentScore" not in payload
