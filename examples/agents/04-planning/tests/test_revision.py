"""Plan revision preserves prior versions and continues execution."""

from __future__ import annotations

from pathlib import Path

from agent.plan import (
    complete_step,
    create_plan_from_proposal,
    revise_plan,
    start_next_step,
)
from agent.state import initial_state
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INITIAL = [
    {
        "id": "step-1",
        "description": "Check payments",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "payments"},
    },
    {
        "id": "step-2",
        "description": "Inspect remediation docs",
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

REVISED_REMAINING = [
    {
        "id": "step-2-revised",
        "description": "Check recent deployment",
        "action_kind": "tool_call",
        "intent": "docs_lookup",
        "tool": "search_documentation",
        "arguments": {"query": "payments recent deployment"},
    },
    {
        "id": "step-3-revised",
        "description": "Summarize current status",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]


def test_revision_preserves_previous_plan_and_creates_new_version():
    registry = build_registry(DATA)
    current = create_plan_from_proposal(INITIAL, registry=registry)
    start_next_step(current)
    complete_step(current, "step-1")
    original_version = current.version
    original_step_ids = [step.id for step in current.steps]

    superseded, revised, record = revise_plan(
        current,
        proposed_remaining=REVISED_REMAINING,
        registry=registry,
        reason="payments is operational",
        observation_call_id="call_v1_step-1",
    )

    assert superseded.version == original_version == 1
    assert superseded.status == "superseded"
    assert [step.id for step in superseded.steps] == original_step_ids
    assert superseded.step_by_id("step-1").status == "completed"
    assert superseded.step_by_id("step-2").status == "skipped"
    assert superseded.step_by_id("step-3").status == "skipped"

    assert revised.version == 2
    assert revised.supersedes_version == 1
    assert record.from_version == 1
    assert record.to_version == 2
    assert record.supersedes is True if hasattr(record, "supersedes") else True
    assert record.to_public_dict()["supersedes"] is True

    assert revised.step_by_id("step-1").status == "completed"
    assert [step.id for step in revised.steps] == [
        "step-1",
        "step-2-revised",
        "step-3-revised",
    ]
    assert revised.step_by_id("step-2-revised").status == "pending"
    nxt = start_next_step(revised)
    assert nxt.id == "step-2-revised"


def test_state_revision_does_not_overwrite_history():
    registry = build_registry(DATA)
    state = initial_state(request="r", max_turns=6, system_prompt="s")
    state.install_plan(INITIAL, registry=registry)
    state.start_current_step()
    state.complete_current_step("step-1")
    v1_before = state.plan.to_public_dict() if state.plan else None
    state.revise_current_plan(
        REVISED_REMAINING,
        registry=registry,
        reason="payments is operational",
        observation_call_id="c1",
    )
    assert len(state.plan_history) == 1
    assert state.plan_history[0].version == 1
    assert state.plan_history[0].status == "superseded"
    assert state.plan is not None
    assert state.plan.version == 2
    history_v1 = state.plan_history[0].to_public_dict()
    assert history_v1["steps"][0]["id"] == "step-1"
    assert history_v1["steps"][1]["status"] == "skipped"
    assert v1_before is not None
    plans = state.exported_plans()
    assert [p["version"] for p in plans] == [1, 2]
    assert plans[0]["status"] == "superseded"
    assert plans[1]["version"] == 2
