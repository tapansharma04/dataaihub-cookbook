"""Explicit serializable memory runtime state.

Stores observable memory operations, decisions, and answers — never
hidden chain-of-thought.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.memory import MemoryRecord
from agent.schemas import TerminationReason


class RecordedDecision(BaseModel):
    """One observable model decision recorded in state."""

    turn: int
    interaction_id: str
    decision: str
    content: str | None = None
    memory_write: dict[str, Any] | None = None
    memory_read: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: int = 0


class RecordedInteraction(BaseModel):
    id: str
    request: str
    final_answer: str | None = None


class AgentState(BaseModel):
    """Application-owned memory session state across interactions."""

    scope: str
    current_turn: int = 0
    max_turns: int
    current_interaction_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    interactions: list[RecordedInteraction] = Field(default_factory=list)
    decisions: list[RecordedDecision] = Field(default_factory=list)
    memory_writes: list[dict[str, Any]] = Field(default_factory=list)
    memory_reads: list[dict[str, Any]] = Field(default_factory=list)
    freshness: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not None

    def begin_turn(self) -> int:
        self.current_turn += 1
        return self.current_turn

    def begin_interaction(
        self, interaction_id: str, request: str, system_prompt: str
    ) -> None:
        self.current_interaction_id = interaction_id
        self.interactions.append(
            RecordedInteraction(id=interaction_id, request=request)
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

    def record_decision(self, decision: RecordedDecision) -> None:
        self.decisions.append(decision)

    def record_write(self, record: MemoryRecord) -> None:
        self.memory_writes.append(record.to_public_dict())

    def record_read(self, payload: dict[str, Any]) -> None:
        self.memory_reads.append(payload)

    def record_freshness(self, payload: dict[str, Any]) -> None:
        self.freshness.append(payload)

    def set_interaction_answer(self, answer: str) -> None:
        if not self.interactions:
            return
        self.interactions[-1].final_answer = answer
        self.final_answer = answer

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
            self.set_interaction_answer(answer)
        if error is not None:
            self.errors.append(error)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize runtime state for traces / JSON (no CoT fields)."""
        return {
            "scope": self.scope,
            "currentTurn": self.current_turn,
            "maxTurns": self.max_turns,
            "currentInteractionId": self.current_interaction_id,
            "interactions": [
                {
                    "id": item.id,
                    "request": item.request,
                    "finalAnswer": item.final_answer,
                }
                for item in self.interactions
            ],
            "decisions": [
                {
                    "turn": d.turn,
                    "interactionId": d.interaction_id,
                    "decision": d.decision,
                    "content": d.content,
                    "memoryWrite": d.memory_write,
                    "memoryRead": d.memory_read,
                    "finishReason": d.finish_reason,
                    "latencyMs": d.latency_ms,
                }
                for d in self.decisions
            ],
            "memoryWrites": list(self.memory_writes),
            "memoryReads": list(self.memory_reads),
            "freshness": list(self.freshness),
            "finalAnswer": self.final_answer,
            "terminationReason": self.termination_reason,
            "errors": list(self.errors),
            "messageCount": len(self.messages),
        }


def initial_state(*, scope: str, max_turns: int) -> AgentState:
    return AgentState(scope=scope, max_turns=max_turns)
