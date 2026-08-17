"""Explicit serializable planning runtime state.

Stores observable plan versions, decisions, actions, and results — never
hidden chain-of-thought.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.plan import (
    Plan,
    PlanRevision,
    PlanStep,
    complete_step,
    create_plan_from_proposal,
    fail_step,
    freeze_plan,
    mark_plan_completed,
    mark_plan_failed,
    progress_snapshot,
    revise_plan,
    skip_pending_steps,
    start_next_step,
)
from agent.schemas import TerminationReason
from agent.tools import ToolRegistry


class RecordedDecision(BaseModel):
    """One observable model decision recorded in state."""

    turn: int
    decision: str
    content: str | None = None
    proposed_steps: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    latency_ms: int = 0


class RecordedToolCall(BaseModel):
    turn: int
    call_id: str
    step_id: str
    plan_version: int
    name: str
    arguments: dict[str, Any]
    validated_arguments: dict[str, Any] | None = None
    latency_ms: int = 0


class RecordedObservation(BaseModel):
    turn: int
    call_id: str
    step_id: str
    plan_version: int
    name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_ms: int = 0


class AgentState(BaseModel):
    """Application-owned planning state across plan versions."""

    request: str
    current_turn: int = 0
    max_turns: int
    messages: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[RecordedDecision] = Field(default_factory=list)
    tool_calls: list[RecordedToolCall] = Field(default_factory=list)
    observations: list[RecordedObservation] = Field(default_factory=list)
    plan: Plan | None = None
    plan_history: list[Plan] = Field(default_factory=list)
    revisions: list[PlanRevision] = Field(default_factory=list)
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not None

    def begin_turn(self) -> int:
        self.current_turn += 1
        return self.current_turn

    def record_decision(self, decision: RecordedDecision) -> None:
        self.decisions.append(decision)

    def record_tool_call(self, call: RecordedToolCall) -> None:
        self.tool_calls.append(call)

    def record_observation(self, observation: RecordedObservation) -> None:
        self.observations.append(observation)

    def install_plan(
        self,
        raw_steps: list[dict[str, Any]],
        *,
        registry: ToolRegistry,
    ) -> Plan:
        if self.plan is not None:
            raise ValueError("A plan is already installed; use revise_current_plan")
        plan = create_plan_from_proposal(raw_steps, registry=registry)
        self.plan = plan
        return plan

    def start_current_step(self) -> PlanStep:
        if self.plan is None:
            raise ValueError("No plan is installed")
        return start_next_step(self.plan)

    def complete_current_step(self, step_id: str) -> PlanStep:
        if self.plan is None:
            raise ValueError("No plan is installed")
        return complete_step(self.plan, step_id)

    def fail_current_step(self, step_id: str) -> PlanStep:
        if self.plan is None:
            raise ValueError("No plan is installed")
        return fail_step(self.plan, step_id)

    def skip_remaining_steps(self) -> list[str]:
        if self.plan is None:
            raise ValueError("No plan is installed")
        return skip_pending_steps(self.plan)

    def complete_plan(self) -> None:
        if self.plan is None:
            raise ValueError("No plan is installed")
        mark_plan_completed(self.plan)

    def fail_plan(self) -> None:
        if self.plan is None:
            raise ValueError("No plan is installed")
        mark_plan_failed(self.plan)

    def revise_current_plan(
        self,
        proposed_remaining: list[dict[str, Any]],
        *,
        registry: ToolRegistry,
        reason: str,
        observation_call_id: str | None = None,
    ) -> PlanRevision:
        if self.plan is None:
            raise ValueError("No plan is installed")
        superseded, revised, revision = revise_plan(
            self.plan,
            proposed_remaining=proposed_remaining,
            registry=registry,
            reason=reason,
            observation_call_id=observation_call_id,
        )
        self.plan_history.append(superseded)
        self.plan = revised
        self.revisions.append(revision)
        return revision

    def progress(self) -> dict[str, Any]:
        if self.plan is None:
            return {}
        return progress_snapshot(self.plan)

    def exported_plans(self) -> list[dict[str, Any]]:
        """All plan versions: superseded history first, then current."""
        plans = [freeze_plan(item).to_public_dict() for item in self.plan_history]
        if self.plan is not None:
            plans.append(freeze_plan(self.plan).to_public_dict())
        return plans

    def terminate(
        self,
        reason: TerminationReason,
        *,
        answer: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if self.termination_reason is not None:
            return
        self.termination_reason = reason
        if answer is not None:
            self.final_answer = answer
        if error is not None:
            self.errors.append(error)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize runtime state for traces / JSON (no CoT fields)."""
        counts = self.plan.counts() if self.plan is not None else {}
        return {
            "request": self.request,
            "currentTurn": self.current_turn,
            "maxTurns": self.max_turns,
            "decisions": [
                {
                    "turn": d.turn,
                    "decision": d.decision,
                    "content": d.content,
                    "proposedSteps": d.proposed_steps,
                    "finishReason": d.finish_reason,
                    "latencyMs": d.latency_ms,
                }
                for d in self.decisions
            ],
            "toolCalls": [c.model_dump() for c in self.tool_calls],
            "observations": [o.model_dump() for o in self.observations],
            "plan": self.plan.to_public_dict() if self.plan is not None else None,
            "planHistory": [p.to_public_dict() for p in self.plan_history],
            "plans": self.exported_plans(),
            "revisions": [r.to_public_dict() for r in self.revisions],
            "progress": self.progress(),
            "planVersion": self.plan.version if self.plan is not None else None,
            "planStatus": self.plan.status if self.plan is not None else None,
            "planRevisions": len(self.revisions),
            "completedSteps": counts.get("completed_steps", 0),
            "skippedSteps": counts.get("skipped_steps", 0),
            "failedSteps": counts.get("failed_steps", 0),
            "finalAnswer": self.final_answer,
            "terminationReason": self.termination_reason,
            "errors": list(self.errors),
            "messageCount": len(self.messages),
        }


def initial_state(
    *,
    request: str,
    max_turns: int,
    system_prompt: str,
) -> AgentState:
    return AgentState(
        request=request,
        max_turns=max_turns,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ],
    )
