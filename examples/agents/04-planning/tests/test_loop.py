"""Planning loop tests — scripted model, real tools, no paid APIs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent.cases import get_case, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import classify_decision, run_planning_loop
from agent.model import ScriptedModelClient, ScriptedTurn
from agent.schemas import ModelTurn
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _run(case_id: str):
    case = get_case(case_id)
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    max_turns = case.max_turns if case.max_turns is not None else 6
    return run_planning_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
        max_turns=max_turns,
    )


def test_simple_plan_completes_without_revision():
    result = _run("simple-plan-billing-docs")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.plan_status == "completed"
    assert result.metrics.plan_revisions == 0
    assert result.metrics.plan_version == 1
    assert result.metrics.completed_steps == 3
    assert result.metrics.failed_steps == 0
    assert result.metrics.model_turns == 2
    assert result.metrics.tool_calls == 2
    assert result.metrics.successful_tool_calls == 2
    kinds = [e.kind for e in result.sequence]
    assert kinds[0] == "user_request"
    assert "plan_created" in kinds
    assert kinds.count("plan_step_started") == 3
    assert kinds.count("plan_step_completed") == 3
    assert "plan_revised" not in kinds
    assert kinds[-2:] == ["final_answer", "termination"]
    names = [e.detail["name"] for e in result.sequence if e.kind == "tool_call"]
    assert names == ["get_service_status", "search_documentation"]
    assert "operational" in result.answer.lower() or "billing" in result.answer.lower()


def test_plan_execution_tracks_step_progress():
    result = _run("plan-execution-payments-progress")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.plan_status == "completed"
    assert result.metrics.plan_revisions == 0
    assert result.metrics.tool_calls == 3
    assert result.metrics.model_turns == 2
    started = [e for e in result.sequence if e.kind == "plan_step_started"]
    completed = [e for e in result.sequence if e.kind == "plan_step_completed"]
    assert len(started) == 4
    assert len(completed) == 4
    first = started[0]
    assert first.detail["status"] == "in_progress"
    assert "step-1" in first.detail["progress"]["inProgress"]
    remaining_after_first = first.detail["progress"]["remaining"]
    assert len(remaining_after_first) >= 3
    after_two_tools = completed[1]
    assert after_two_tools.detail["progress"]["completed"] == ["step-1", "step-2"]
    assert "step-3" in after_two_tools.detail["progress"]["pending"]
    last = completed[-1]
    assert last.detail["progress"]["completed"] == [
        "step-1",
        "step-2",
        "step-3",
        "step-4",
    ]
    assert last.detail["progress"]["remaining"] == []
    assert result.metrics.model_turns < result.metrics.tool_calls


def test_plan_revision_preserves_v1_and_continues():
    result = _run("plan-revision-payments-healthy")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.plan_status == "completed"
    assert result.metrics.plan_revisions == 1
    assert result.metrics.plan_version == 2
    kinds = [e.kind for e in result.sequence]
    created = kinds.index("plan_created")
    revised = kinds.index("plan_revised")
    assert created < kinds.index("observation") < revised
    assert "plan_step_completed" in kinds[created:revised]
    assert "plan_step_started" in kinds[revised:]
    event = next(e for e in result.sequence if e.kind == "plan_revised")
    assert event.detail["fromVersion"] == 1
    assert event.detail["toVersion"] == 2
    assert event.detail["supersedes"] is True
    original_ids = [s["id"] for s in event.detail["originalPlan"]["steps"]]
    revised_ids = [s["id"] for s in event.detail["revisedPlan"]["steps"]]
    assert original_ids == ["step-1", "step-2", "step-3"]
    assert "step-2" in original_ids
    assert "step-2-revised" in revised_ids
    assert "operational" in event.detail["reason"]
    assert event.detail["originalPlan"]["status"] == "superseded"
    history = result.state["planHistory"]
    assert history[0]["version"] == 1
    assert history[0]["status"] == "superseded"
    assert result.state["plan"]["version"] == 2
    assert (
        "remediation" in result.answer.lower() or "operational" in result.answer.lower()
    )


def test_plan_failure_does_not_claim_success():
    result = _run("plan-failure-auth-runbook-missing")
    assert result.metrics.termination_reason == "plan_failed"
    assert result.metrics.plan_status == "failed"
    assert result.metrics.failed_steps == 1
    assert result.metrics.skipped_steps == 1
    assert result.metrics.completed_steps == 1
    assert result.metrics.plan_revisions == 0
    term = next(e for e in result.sequence if e.kind == "termination")
    assert term.detail["reason"] == "plan_failed"
    assert term.detail["planStatus"] == "failed"
    failed_step = next(
        e
        for e in result.sequence
        if e.kind == "plan_step_completed" and e.detail.get("status") == "failed"
    )
    assert failed_step.detail["stepId"] == "step-2"
    assert "not found" in result.answer.lower() or "failed" in result.answer.lower()
    assert "runbook" in result.answer.lower()


def test_direct_tool_call_is_rejected():
    registry = build_registry(DATA)
    executor = MagicMock(spec=ToolExecutor)
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                content=None,
                decision="create_plan",
                tool_calls=[
                    {
                        "id": "call_direct",
                        "name": "get_service_status",
                        "arguments": {"service": "billing"},
                    }
                ],
            )
        ]
    )
    result = run_planning_loop(
        request="Check billing",
        model=model,
        registry=registry,
        executor=executor,
        max_turns=6,
    )
    executor.execute.assert_not_called()
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.tool_calls == 0


def test_completed_plan_steps_are_not_reexecuted():
    case = get_case("simple-plan-billing-docs")
    registry = build_registry(DATA)
    real = ToolExecutor(registry)
    from unittest.mock import patch

    with patch.object(real, "execute", wraps=real.execute) as spy:
        run_planning_loop(
            request=case.request,
            model=scripted_client_for(case),
            registry=registry,
            executor=real,
            max_turns=6,
        )
        names = [call.args[0].name for call in spy.call_args_list]
        assert names == ["get_service_status", "search_documentation"]
        assert spy.call_count == 2


def test_metrics_are_measured_non_negative():
    result = _run("simple-plan-billing-docs")
    m = result.metrics
    assert m.provenance == "measured"
    assert m.total_ms >= 0
    assert m.model_ms >= 0
    assert m.tool_ms >= 0


def test_classify_decision_rejects_unknown():
    turn = ModelTurn(content="keep going", decision="invalid_action")
    assert classify_decision(turn) == "invalid_action"


def test_scripted_unrecognized_decision_is_invalid_action():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    model = ScriptedModelClient(
        [ScriptedTurn(content="keep going", decision="continue")]
    )
    result = run_planning_loop(
        request="Check billing",
        model=model,
        registry=registry,
        executor=executor,
    )
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.tool_calls == 0


def test_repeated_execution_is_semantically_stable():
    first = _run("plan-revision-payments-healthy")
    second = _run("plan-revision-payments-healthy")
    left = [
        (
            e.kind,
            e.detail.get("stepId"),
            e.detail.get("fromVersion"),
            e.detail.get("reason"),
        )
        for e in first.sequence
    ]
    right = [
        (
            e.kind,
            e.detail.get("stepId"),
            e.detail.get("fromVersion"),
            e.detail.get("reason"),
        )
        for e in second.sequence
    ]
    assert left == right
    assert first.metrics.termination_reason == second.metrics.termination_reason
    assert first.answer == second.answer
    assert first.state["plans"][0]["steps"] == second.state["plans"][0]["steps"]
