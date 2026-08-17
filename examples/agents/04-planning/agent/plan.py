"""Application-owned plan state, validation, and revision.

The model/case harness may propose plan steps. This module is the only place
that creates plans, mutates step status, or records a new plan version.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import (
    ActionKind,
    ObservationEffectKind,
    PlanStatus,
    StepIntent,
    StepStatus,
    ToolCallResult,
)
from agent.tools import ToolRegistry

VALID_STEP_TRANSITIONS: dict[StepStatus, set[StepStatus]] = {
    "pending": {"in_progress", "skipped"},
    "in_progress": {"completed", "failed"},
    "completed": set(),
    "skipped": set(),
    "failed": set(),
}

ALLOWED_INTENTS: frozenset[str] = frozenset(
    {"status_check", "docs_lookup", "remediation", "required_docs", "summarize"}
)
ALLOWED_ACTION_KINDS: frozenset[str] = frozenset({"tool_call", "finalize"})


class PlanValidationError(ValueError):
    """Proposed plan/revision failed application validation."""


class PlanStep(BaseModel):
    """One runtime-managed plan step. Status is application-owned."""

    id: str
    description: str
    action_kind: ActionKind
    intent: StepIntent
    status: StepStatus = "pending"
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_doc_id: str | None = None


class Plan(BaseModel):
    """Explicit multi-step plan. A version is never overwritten in place."""

    id: str
    version: int
    status: PlanStatus = "pending"
    steps: list[PlanStep]
    supersedes_version: int | None = None
    revision_reason: str | None = None

    def step_by_id(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def next_pending_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "in_progress":
                return step
            if step.status == "pending":
                return step
        return None

    def counts(self) -> dict[str, int]:
        return {
            "plan_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == "completed"),
            "skipped_steps": sum(1 for s in self.steps if s.status == "skipped"),
            "failed_steps": sum(1 for s in self.steps if s.status == "failed"),
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "status": self.status,
            "supersedesVersion": self.supersedes_version,
            "revisionReason": self.revision_reason,
            "steps": [step_to_public(s) for s in self.steps],
        }


class PlanRevision(BaseModel):
    """Observable record that version N+1 supersedes version N."""

    from_version: int
    to_version: int
    reason: str
    observation_call_id: str | None = None
    completed_step_ids: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    added_step_ids: list[str] = Field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "fromVersion": self.from_version,
            "toVersion": self.to_version,
            "reason": self.reason,
            "observationCallId": self.observation_call_id,
            "completedStepIds": list(self.completed_step_ids),
            "skippedStepIds": list(self.skipped_step_ids),
            "addedStepIds": list(self.added_step_ids),
            "supersedes": True,
        }


class ObservationEffect(BaseModel):
    """Application interpretation of an observation against remaining plan steps."""

    kind: ObservationEffectKind
    reason: str | None = None


def step_to_public(step: PlanStep) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": step.id,
        "description": step.description,
        "actionKind": step.action_kind,
        "intent": step.intent,
        "status": step.status,
        "tool": step.tool,
        "arguments": dict(step.arguments),
    }
    if step.requires_doc_id is not None:
        payload["requiresDocId"] = step.requires_doc_id
    return payload


def progress_snapshot(plan: Plan) -> dict[str, Any]:
    """Observable progress against the current plan version."""
    remaining = [
        {
            "id": step.id,
            "description": step.description,
            "status": step.status,
        }
        for step in plan.steps
        if step.status in {"pending", "in_progress"}
    ]
    return {
        "version": plan.version,
        "status": plan.status,
        "completed": [s.id for s in plan.steps if s.status == "completed"],
        "inProgress": [s.id for s in plan.steps if s.status == "in_progress"],
        "pending": [s.id for s in plan.steps if s.status == "pending"],
        "skipped": [s.id for s in plan.steps if s.status == "skipped"],
        "failed": [s.id for s in plan.steps if s.status == "failed"],
        "remaining": remaining,
    }


def _require_unique_ids(steps: list[PlanStep]) -> None:
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("Plan steps must have unique ids")
    if any(not step_id.strip() for step_id in ids):
        raise PlanValidationError("Plan step ids must be non-empty")


def _strip_harness_status(raw: dict[str, Any]) -> dict[str, Any]:
    """Ignore any status the harness tries to set. Application owns status."""
    cleaned = dict(raw)
    cleaned.pop("status", None)
    return cleaned


def proposed_step_from_raw(
    raw: dict[str, Any],
    *,
    registry: ToolRegistry,
) -> PlanStep:
    data = _strip_harness_status(raw)
    step_id = str(data.get("id") or "").strip()
    description = str(data.get("description") or "").strip()
    action_kind = data.get("action_kind") or data.get("actionKind")
    intent = data.get("intent")
    if not step_id:
        raise PlanValidationError("Each plan step requires an id")
    if not description:
        raise PlanValidationError(f"Step '{step_id}' requires a description")
    if action_kind not in ALLOWED_ACTION_KINDS:
        raise PlanValidationError(
            f"Step '{step_id}' has invalid action_kind '{action_kind}'"
        )
    if intent not in ALLOWED_INTENTS:
        raise PlanValidationError(f"Step '{step_id}' has invalid intent '{intent}'")

    tool = data.get("tool")
    arguments = data.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise PlanValidationError(f"Step '{step_id}' arguments must be an object")
    requires_doc_id = data.get("requires_doc_id") or data.get("requiresDocId")

    if action_kind == "tool_call":
        if not isinstance(tool, str) or not tool.strip():
            raise PlanValidationError(f"Step '{step_id}' tool_call requires a tool")
        if tool not in registry.names():
            raise PlanValidationError(
                f"Step '{step_id}' references unknown tool '{tool}'"
            )
        try:
            validated = registry.parse_arguments(tool, arguments)
        except (KeyError, ValueError) as exc:
            raise PlanValidationError(
                f"Step '{step_id}' has invalid arguments for '{tool}': {exc}"
            ) from exc
        arguments = validated.model_dump()
        tool = tool.strip()
    else:
        tool = None
        arguments = {}

    return PlanStep(
        id=step_id,
        description=description,
        action_kind=action_kind,
        intent=intent,
        status="pending",
        tool=tool,
        arguments=arguments,
        requires_doc_id=str(requires_doc_id) if requires_doc_id else None,
    )


def validate_proposed_steps(
    raw_steps: list[dict[str, Any]],
    *,
    registry: ToolRegistry,
) -> list[PlanStep]:
    if not raw_steps:
        raise PlanValidationError("A plan must contain at least one step")
    steps = [proposed_step_from_raw(item, registry=registry) for item in raw_steps]
    _require_unique_ids(steps)
    finalize_indexes = [
        i for i, step in enumerate(steps) if step.action_kind == "finalize"
    ]
    if len(finalize_indexes) > 1:
        raise PlanValidationError("A plan may contain at most one finalize step")
    if finalize_indexes and finalize_indexes[0] != len(steps) - 1:
        raise PlanValidationError("A finalize step must be last")
    if not any(step.action_kind == "tool_call" for step in steps):
        raise PlanValidationError("A plan must include at least one tool step")
    return steps


def create_plan_from_proposal(
    raw_steps: list[dict[str, Any]],
    *,
    registry: ToolRegistry,
    plan_id: str = "plan-1",
) -> Plan:
    steps = validate_proposed_steps(raw_steps, registry=registry)
    return Plan(id=plan_id, version=1, status="pending", steps=steps)


def set_step_status(plan: Plan, step_id: str, new_status: StepStatus) -> PlanStep:
    step = plan.step_by_id(step_id)
    allowed = VALID_STEP_TRANSITIONS[step.status]
    if new_status not in allowed:
        raise PlanValidationError(
            f"Invalid step transition {step.status} → {new_status} for '{step_id}'"
        )
    step.status = new_status
    return step


def start_next_step(plan: Plan) -> PlanStep:
    if plan.status == "pending":
        plan.status = "in_progress"
    step = plan.next_pending_step()
    if step is None:
        raise PlanValidationError("No pending plan step to start")
    if step.status == "pending":
        set_step_status(plan, step.id, "in_progress")
    return step


def complete_step(plan: Plan, step_id: str) -> PlanStep:
    return set_step_status(plan, step_id, "completed")


def fail_step(plan: Plan, step_id: str) -> PlanStep:
    return set_step_status(plan, step_id, "failed")


def skip_pending_steps(plan: Plan) -> list[str]:
    skipped: list[str] = []
    for step in plan.steps:
        if step.status == "pending":
            set_step_status(plan, step.id, "skipped")
            skipped.append(step.id)
    return skipped


def mark_plan_completed(plan: Plan) -> None:
    if plan.status == "failed":
        raise PlanValidationError("A failed plan cannot be marked completed")
    if any(step.status not in {"completed", "skipped"} for step in plan.steps):
        raise PlanValidationError("Cannot complete a plan with unfinished steps")
    if not any(step.status == "completed" for step in plan.steps):
        raise PlanValidationError("Cannot complete a plan with no completed steps")
    plan.status = "completed"


def mark_plan_failed(plan: Plan) -> None:
    if plan.status == "completed":
        raise PlanValidationError("A completed plan cannot be marked failed")
    plan.status = "failed"


def freeze_plan(plan: Plan) -> Plan:
    return plan.model_copy(deep=True)


def revise_plan(
    current: Plan,
    *,
    proposed_remaining: list[dict[str, Any]],
    registry: ToolRegistry,
    reason: str,
    observation_call_id: str | None = None,
) -> tuple[Plan, Plan, PlanRevision]:
    """Return (frozen vN, live vN+1, revision record).

    Completed steps stay completed. Remaining pending steps on vN are skipped
    on the frozen snapshot. vN is not mutated after the snapshot is taken
    beyond that skip, then replaced by vN+1 as the live plan.
    """
    if current.status in {"completed", "failed", "superseded"}:
        raise PlanValidationError(f"Cannot revise a plan in status '{current.status}'")
    completed = [
        step.model_copy(deep=True)
        for step in current.steps
        if step.status == "completed"
    ]
    if not completed:
        raise PlanValidationError("Revision requires at least one completed step")

    new_remaining = validate_proposed_steps(proposed_remaining, registry=registry)
    completed_ids = {step.id for step in completed}
    for step in new_remaining:
        if step.id in completed_ids:
            raise PlanValidationError(
                f"Revised step '{step.id}' collides with a completed step id"
            )

    superseded = freeze_plan(current)
    skipped_ids = skip_pending_steps(superseded)
    if superseded.status == "in_progress" or superseded.status == "pending":
        superseded.status = "superseded"

    next_version = current.version + 1
    revised = Plan(
        id=current.id,
        version=next_version,
        status="in_progress",
        steps=[*completed, *new_remaining],
        supersedes_version=current.version,
        revision_reason=reason,
    )
    revision = PlanRevision(
        from_version=current.version,
        to_version=next_version,
        reason=reason,
        observation_call_id=observation_call_id,
        completed_step_ids=[step.id for step in completed],
        skipped_step_ids=skipped_ids,
        added_step_ids=[step.id for step in new_remaining],
    )
    return superseded, revised, revision


def interpret_observation(
    *,
    step: PlanStep,
    result: ToolCallResult,
    remaining: list[PlanStep],
) -> ObservationEffect:
    """Decide continue / revise / block from an observable tool result.

    Revision and failure are application policy, not model discretion.
    """
    payload = result.result or {}

    if step.intent == "required_docs":
        hits = payload.get("hits") if result.ok else None
        hit_ids = {hit.get("id") for hit in hits or [] if isinstance(hit, dict)}
        required = step.requires_doc_id
        if required and required not in hit_ids:
            return ObservationEffect(
                kind="block",
                reason=(
                    f"required documentation '{required}' is unavailable; "
                    "remaining plan cannot be completed"
                ),
            )
        if result.ok and int(payload.get("hitCount") or 0) == 0:
            return ObservationEffect(
                kind="block",
                reason=(
                    "required documentation returned no hits; "
                    "remaining plan cannot be completed"
                ),
            )
        if not result.ok:
            return ObservationEffect(
                kind="block",
                reason=(
                    "required documentation lookup failed; "
                    "remaining plan cannot be completed"
                ),
            )

    if step.intent == "status_check" and not result.ok:
        return ObservationEffect(
            kind="block",
            reason=(
                "required service information is unavailable; "
                "remaining plan cannot be completed"
            ),
        )

    if step.intent == "status_check" and result.ok:
        service = payload.get("service") or {}
        status = service.get("status")
        service_name = service.get("service") or "service"
        if status == "operational" and any(
            item.intent == "remediation" for item in remaining
        ):
            return ObservationEffect(
                kind="revise",
                reason=(
                    "observation invalidated remaining remediation step: "
                    f"{service_name} is operational"
                ),
            )

    return ObservationEffect(kind="continue")
