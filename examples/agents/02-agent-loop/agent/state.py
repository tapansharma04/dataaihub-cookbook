"""Explicit serializable agent-loop runtime state.

Stores only observable decisions, actions, and results — never hidden
chain-of-thought.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import TerminationReason


class RecordedDecision(BaseModel):
    """One observable model decision recorded in state."""

    turn: int
    decision: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    latency_ms: int = 0


class RecordedToolCall(BaseModel):
    turn: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    validated_arguments: dict[str, Any] | None = None
    latency_ms: int = 0


class RecordedObservation(BaseModel):
    turn: int
    call_id: str
    name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_ms: int = 0


class AgentState(BaseModel):
    """Application-owned loop state across iterations."""

    request: str
    current_turn: int = 0
    max_turns: int
    messages: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[RecordedDecision] = Field(default_factory=list)
    tool_calls: list[RecordedToolCall] = Field(default_factory=list)
    observations: list[RecordedObservation] = Field(default_factory=list)
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
        return {
            "request": self.request,
            "currentTurn": self.current_turn,
            "maxTurns": self.max_turns,
            "decisions": [d.model_dump() for d in self.decisions],
            "toolCalls": [c.model_dump() for c in self.tool_calls],
            "observations": [o.model_dump() for o in self.observations],
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
