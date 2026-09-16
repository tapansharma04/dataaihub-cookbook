"""Application-owned handoff state.

The runtime is the only writer. There is exactly one current owner at a
time. A successful handoff replaces that owner; the previous agent is no
longer active.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import (
    AgentResult,
    EnteredVia,
    HandoffRejection,
    HandoffRequest,
    OwnershipRecord,
    TerminationReason,
)


class HandoffState(BaseModel):
    task_id: str
    request: str
    ticket_id: str | None = None
    current_owner: str | None = None
    previous_owner: str | None = None
    ownership_history: list[OwnershipRecord] = Field(default_factory=list)
    accepted_handoffs: list[HandoffRequest] = Field(default_factory=list)
    rejections: list[HandoffRejection] = Field(default_factory=list)
    results: list[AgentResult] = Field(default_factory=list)
    active_handoff: HandoffRequest | None = None
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not None

    def activate(
        self,
        agent_id: str,
        *,
        entered_via: EnteredVia,
        handoff_id: str | None = None,
        parent_result_id: str | None = None,
    ) -> None:
        self.current_owner = agent_id
        self.ownership_history.append(
            OwnershipRecord(
                agent_id=agent_id,
                entered_via=entered_via,
                handoff_id=handoff_id,
                parent_result_id=parent_result_id,
            )
        )

    def transfer_ownership(self, handoff: HandoffRequest) -> None:
        """Replace the current owner. The previous agent is no longer active."""
        self.previous_owner = self.current_owner
        self.current_owner = handoff.to_agent
        self.active_handoff = handoff.model_copy(deep=True)
        self.accepted_handoffs.append(handoff.model_copy(deep=True))
        self.ownership_history.append(
            OwnershipRecord(
                agent_id=handoff.to_agent,
                entered_via="handoff",
                handoff_id=handoff.handoff_id,
                parent_result_id=handoff.parent_result_id,
            )
        )

    def record_result(self, result: AgentResult) -> None:
        self.results.append(result)

    def record_rejection(self, rejection: HandoffRejection) -> None:
        self.rejections.append(rejection)

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
        return {
            "taskId": self.task_id,
            "request": self.request,
            "ticketId": self.ticket_id,
            "currentOwner": self.current_owner,
            "previousOwner": self.previous_owner,
            "ownershipHistory": [
                item.model_dump(mode="json") for item in self.ownership_history
            ],
            "acceptedHandoffs": [
                item.model_dump(mode="json") for item in self.accepted_handoffs
            ],
            "rejections": [item.model_dump(mode="json") for item in self.rejections],
            "results": [item.model_dump(mode="json") for item in self.results],
            "activeHandoff": (
                self.active_handoff.model_dump(mode="json")
                if self.active_handoff is not None
                else None
            ),
            "invokedAgentIds": [item.agent_id for item in self.results],
            "finalAnswer": self.final_answer,
            "terminationReason": self.termination_reason,
            "errors": list(self.errors),
        }
