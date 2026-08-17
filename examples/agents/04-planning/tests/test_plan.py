"""Plan state, transitions, and application ownership tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.plan import (
    PlanValidationError,
    complete_step,
    create_plan_from_proposal,
    fail_step,
    interpret_observation,
    mark_plan_completed,
    mark_plan_failed,
    progress_snapshot,
    skip_pending_steps,
    start_next_step,
)
from agent.schemas import ToolCallResult
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BILLING_STEPS = [
    {
        "id": "step-1",
        "description": "Check billing",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "billing"},
    },
    {
        "id": "step-2",
        "description": "Inspect docs",
        "action_kind": "tool_call",
        "intent": "docs_lookup",
        "tool": "search_documentation",
        "arguments": {"query": "billing operations"},
    },
    {
        "id": "step-3",
        "description": "Summarize",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]


def _registry():
    return build_registry(DATA)


def test_plan_can_be_created():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    assert plan.id == "plan-1"
    assert plan.version == 1
    assert plan.status == "pending"
    assert len(plan.steps) == 3
    assert all(step.status == "pending" for step in plan.steps)


def test_proposed_status_is_ignored():
    raw = [
        {**BILLING_STEPS[0], "status": "completed"},
        BILLING_STEPS[1],
        BILLING_STEPS[2],
    ]
    plan = create_plan_from_proposal(raw, registry=_registry())
    assert plan.steps[0].status == "pending"


def test_unknown_tool_is_rejected():
    raw = [
        {
            "id": "step-1",
            "description": "Drop the database",
            "action_kind": "tool_call",
            "intent": "status_check",
            "tool": "drop_database",
            "arguments": {},
        }
    ]
    with pytest.raises(PlanValidationError, match="unknown tool"):
        create_plan_from_proposal(raw, registry=_registry())


def test_invalid_arguments_rejected_at_plan_create():
    raw = [
        {
            "id": "step-1",
            "description": "Check service",
            "action_kind": "tool_call",
            "intent": "status_check",
            "tool": "get_service_status",
            "arguments": {},
        }
    ]
    with pytest.raises(PlanValidationError, match="invalid arguments"):
        create_plan_from_proposal(raw, registry=_registry())


def test_pending_in_progress_completed():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    step = start_next_step(plan)
    assert step.id == "step-1"
    assert step.status == "in_progress"
    assert plan.status == "in_progress"
    complete_step(plan, "step-1")
    assert plan.step_by_id("step-1").status == "completed"
    nxt = start_next_step(plan)
    assert nxt.id == "step-2"


def test_completed_steps_are_not_started_again():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    start_next_step(plan)
    complete_step(plan, "step-1")
    nxt = start_next_step(plan)
    assert nxt.id == "step-2"
    with pytest.raises(PlanValidationError, match="Invalid step transition"):
        complete_step(plan, "step-1")


def test_invalid_step_transition_rejected():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    with pytest.raises(PlanValidationError, match="Invalid step transition"):
        complete_step(plan, "step-1")


def test_progress_snapshot_lists_remaining():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    start_next_step(plan)
    complete_step(plan, "step-1")
    snapshot = progress_snapshot(plan)
    assert snapshot["completed"] == ["step-1"]
    assert snapshot["pending"] == ["step-2", "step-3"]
    assert [item["id"] for item in snapshot["remaining"]] == ["step-2", "step-3"]


def test_failed_plan_cannot_become_completed():
    plan = create_plan_from_proposal(BILLING_STEPS, registry=_registry())
    start_next_step(plan)
    fail_step(plan, "step-1")
    skip_pending_steps(plan)
    mark_plan_failed(plan)
    assert plan.status == "failed"
    with pytest.raises(PlanValidationError, match="failed plan cannot"):
        mark_plan_completed(plan)


def test_healthy_status_invalidates_remediation_remaining():
    remaining_raw = [
        {
            "id": "step-2",
            "description": "Remediate",
            "action_kind": "tool_call",
            "intent": "remediation",
            "tool": "search_documentation",
            "arguments": {"query": "payments remediation"},
        },
        {
            "id": "step-3",
            "description": "Recommend remediation",
            "action_kind": "finalize",
            "intent": "remediation",
        },
    ]
    remaining = create_plan_from_proposal(
        [
            BILLING_STEPS[0],
            *remaining_raw,
        ],
        registry=_registry(),
    ).steps[1:]
    step = create_plan_from_proposal(
        [BILLING_STEPS[0], remaining_raw[0], remaining_raw[1]],
        registry=_registry(),
    ).steps[0]
    result = ToolCallResult(
        call_id="c1",
        name="get_service_status",
        ok=True,
        result={
            "ok": True,
            "service": {"service": "payments", "status": "operational"},
        },
        latency_ms=0,
    )
    effect = interpret_observation(step=step, result=result, remaining=remaining)
    assert effect.kind == "revise"
    assert effect.reason is not None
    assert "operational" in effect.reason


def test_missing_required_doc_blocks_plan():
    raw = [
        {
            "id": "step-1",
            "description": "Find runbook",
            "action_kind": "tool_call",
            "intent": "required_docs",
            "tool": "search_documentation",
            "arguments": {"query": "AUTH-881 incident runbook"},
            "requires_doc_id": "doc-auth-881-runbook",
        },
        {
            "id": "step-2",
            "description": "Recommend",
            "action_kind": "finalize",
            "intent": "remediation",
        },
    ]
    plan = create_plan_from_proposal(raw, registry=_registry())
    step = plan.steps[0]
    result = ToolCallResult(
        call_id="c1",
        name="search_documentation",
        ok=True,
        result={"ok": True, "hitCount": 1, "hits": [{"id": "doc-other"}]},
        latency_ms=0,
    )
    effect = interpret_observation(
        step=step,
        result=result,
        remaining=[plan.steps[1]],
    )
    assert effect.kind == "block"
    assert "unavailable" in (effect.reason or "")
