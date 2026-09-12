"""Application-owned shared task state.

Specialist agents never write this object. The coordinator records
delegations, messages, and results after each agent returns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import AgentMessage, AgentResult, Delegation, TerminationReason


class TaskState(BaseModel):
    task_id: str
    request: str
    service: str | None = None
    delegations: list[Delegation] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    results: list[AgentResult] = Field(default_factory=list)
    skipped: list[Delegation] = Field(default_factory=list)
    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not None

    def result_for(self, agent_id: str) -> AgentResult | None:
        for item in reversed(self.results):
            if item.agent_id == agent_id:
                return item
        return None

    def record_delegation(self, delegation: Delegation) -> None:
        self.delegations.append(delegation)

    def record_skip(self, delegation: Delegation) -> None:
        self.skipped.append(delegation)

    def record_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def record_result(self, result: AgentResult) -> None:
        self.results.append(result)

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
            "service": self.service,
            "delegations": [item.model_dump(mode="json") for item in self.delegations],
            "messages": [item.model_dump(mode="json") for item in self.messages],
            "results": [item.model_dump(mode="json") for item in self.results],
            "skipped": [item.model_dump(mode="json") for item in self.skipped],
            "invokedAgentIds": [item.agent_id for item in self.results],
            "skippedAgentIds": [item.agent_id for item in self.skipped],
            "finalAnswer": self.final_answer,
            "terminationReason": self.termination_reason,
            "errors": list(self.errors),
        }
