"""Application-owned handoff runtime.

The runtime keeps exactly one active owner. When that owner requests a
handoff, the runtime validates the contract and, if valid, transfers
ownership. Control does not return to a coordinator or to the previous
agent.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from agent.agents import Agent, default_agents
from agent.cases import MeasuredCase
from agent.catalog import Catalog
from agent.schemas import (
    AgentContext,
    AgentResult,
    HandoffMetrics,
    HandoffRejection,
    HandoffRequest,
    HandoffRunResult,
    SequenceEvent,
    TerminationReason,
)
from agent.state import HandoffState
from agent.synthesizer import MockSynthesizer, Synthesizer


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


class HandoffRuntime:
    """Owns current_owner, validation, transfer, and termination."""

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        synthesizer: Synthesizer,
        max_handoffs: int,
    ) -> None:
        self._agents = agents
        self._synthesizer = synthesizer
        self._max_handoffs = max_handoffs
        self._result_n = 0
        self._handoff_n = 0

    def _stamp_result(self, result: AgentResult) -> AgentResult:
        self._result_n += 1
        return result.model_copy(update={"result_id": f"result-{self._result_n}"})

    def _stamp_handoff(
        self,
        request: HandoffRequest,
        *,
        parent_result_id: str,
    ) -> HandoffRequest:
        self._handoff_n += 1
        claimed = (request.from_agent or "").strip()
        target = (request.to_agent or "").strip()
        return request.model_copy(
            update={
                "handoff_id": f"handoff-{self._handoff_n}",
                "parent_result_id": parent_result_id,
                "from_agent": claimed,
                "to_agent": target,
                "payload": copy.deepcopy(request.payload),
                "context": copy.deepcopy(request.context),
            }
        )

    def _validate_handoff(
        self,
        request: HandoffRequest,
        *,
        current_owner: str,
    ) -> HandoffRejection | None:
        target = (request.to_agent or "").strip()
        if not target:
            return HandoffRejection(
                handoff_id=request.handoff_id,
                from_agent=request.from_agent,
                to_agent=request.to_agent,
                code="empty_target",
                message="Handoff target is empty.",
                parent_result_id=request.parent_result_id,
            )
        if request.from_agent != current_owner:
            return HandoffRejection(
                handoff_id=request.handoff_id,
                from_agent=request.from_agent,
                to_agent=target,
                code="not_current_owner",
                message=(
                    f"Handoff from_agent '{request.from_agent}' is not the "
                    f"current owner '{current_owner}'."
                ),
                parent_result_id=request.parent_result_id,
            )
        if target == current_owner:
            return HandoffRejection(
                handoff_id=request.handoff_id,
                from_agent=request.from_agent,
                to_agent=target,
                code="self_handoff",
                message="An owner cannot hand off to itself.",
                parent_result_id=request.parent_result_id,
            )
        if target not in self._agents:
            return HandoffRejection(
                handoff_id=request.handoff_id,
                from_agent=request.from_agent,
                to_agent=target,
                code="unknown_target",
                message=f"Unknown handoff target '{target}'.",
                parent_result_id=request.parent_result_id,
            )
        return None

    def run(self, case: MeasuredCase) -> HandoffRunResult:
        started = time.perf_counter()
        agent_ms = 0
        sequence: list[SequenceEvent] = []
        requested = 0
        accepted = 0
        rejected = 0
        state = HandoffState(
            task_id=f"task-{case.trace_id}",
            request=case.request,
            ticket_id=case.ticket_id,
        )

        def emit(
            kind: str,
            detail: dict[str, Any],
            latency_ms: int | None = None,
        ) -> None:
            sequence.append(
                SequenceEvent.model_validate(
                    {"kind": kind, "detail": detail, "latency_ms": latency_ms}
                )
            )

        emit(
            "task_created",
            {
                "taskId": state.task_id,
                "request": state.request,
                "ticketId": state.ticket_id,
                "exampleClass": case.example_class,
            },
        )

        if not case.request.strip():
            state.terminate(
                "invalid_task",
                answer="Runtime rejected an empty request.",
                error={"code": "invalid_task", "message": "empty request"},
            )
            emit("error", {"code": "invalid_task", "message": "empty request"})
            emit("termination", {"reason": state.termination_reason})
            return self._finish(
                case,
                state,
                sequence,
                started,
                agent_ms,
                requested=0,
                accepted=0,
                rejected=0,
            )

        initial = case.initial_agent
        if initial not in self._agents:
            state.terminate(
                "error",
                answer=f"Unknown initial agent '{initial}'.",
                error={"code": "unknown_initial_agent", "agentId": initial},
            )
            emit("error", {"code": "unknown_initial_agent", "agentId": initial})
            emit("termination", {"reason": state.termination_reason})
            return self._finish(
                case,
                state,
                sequence,
                started,
                agent_ms,
                requested=0,
                accepted=0,
                rejected=0,
            )

        state.activate(initial, entered_via="initial")
        emit(
            "agent_activated",
            {
                "agentId": initial,
                "enteredVia": "initial",
                "currentOwner": state.current_owner,
                "previousOwner": state.previous_owner,
            },
        )

        while not state.terminated:
            owner = state.current_owner
            assert owner is not None
            agent = self._agents[owner]
            initial_owner = state.active_handoff is None
            context = AgentContext(
                task_id=state.task_id,
                request=state.request if initial_owner else None,
                ticket_id=state.ticket_id if initial_owner else None,
                current_owner=owner,
                inbound_handoff=(
                    state.active_handoff.model_copy(deep=True)
                    if state.active_handoff is not None
                    else None
                ),
            )
            agent_start = time.perf_counter()
            raw = agent.handle(context.model_copy(deep=True))
            latency = _elapsed_ms(agent_start)
            agent_ms += latency
            result = self._stamp_result(raw.model_copy(deep=True))
            if result.kind == "handoff" and result.handoff is not None:
                result = result.model_copy(
                    update={
                        "handoff": self._stamp_handoff(
                            result.handoff,
                            parent_result_id=result.result_id,
                        )
                    }
                )
            state.record_result(result.model_copy(deep=True))
            emit(
                "agent_result",
                {
                    "resultId": result.result_id,
                    "agentId": result.agent_id,
                    "role": result.role,
                    "kind": result.kind,
                    "ok": result.ok,
                    "payload": copy.deepcopy(result.payload),
                    "handoff": (
                        result.handoff.model_dump(mode="json")
                        if result.handoff is not None
                        else None
                    ),
                    "error": result.error,
                    "currentOwner": state.current_owner,
                },
                latency_ms=latency,
            )

            if result.agent_id != owner:
                state.terminate(
                    "error",
                    error={
                        "code": "owner_mismatch",
                        "agentId": result.agent_id,
                        "currentOwner": owner,
                        "message": (
                            f"Result agent_id '{result.agent_id}' is not the "
                            f"current owner '{owner}'."
                        ),
                    },
                )
                emit(
                    "error",
                    {
                        "code": "owner_mismatch",
                        "agentId": result.agent_id,
                        "currentOwner": owner,
                    },
                )
                answer, _ = self._synthesizer.synthesize(state)
                state.final_answer = answer
                break

            if result.kind == "complete":
                state.terminate("completed")
                answer, format_ms = self._synthesizer.synthesize(state)
                state.final_answer = answer
                emit(
                    "completed",
                    {
                        "agentId": result.agent_id,
                        "resultId": result.result_id,
                        "currentOwner": state.current_owner,
                        "answer": answer,
                    },
                    latency_ms=format_ms,
                )
                break

            if result.kind == "failure":
                emit(
                    "failure",
                    {
                        "agentId": result.agent_id,
                        "resultId": result.result_id,
                        "error": result.error,
                        "currentOwner": state.current_owner,
                    },
                )
                state.terminate("agent_failed", error=result.error)
                answer, _ = self._synthesizer.synthesize(state)
                state.final_answer = answer
                break

            if result.kind != "handoff" or result.handoff is None:
                state.terminate(
                    "error",
                    answer="Active owner returned an invalid outcome.",
                    error={
                        "code": "missing_handoff",
                        "agentId": result.agent_id,
                    },
                )
                emit(
                    "error",
                    {"code": "missing_handoff", "agentId": result.agent_id},
                )
                break

            requested += 1
            handoff = result.handoff.model_copy(deep=True)
            emit(
                "handoff_requested",
                {
                    "handoffId": handoff.handoff_id,
                    "from": handoff.from_agent,
                    "to": handoff.to_agent,
                    "reason": handoff.reason,
                    "payload": copy.deepcopy(handoff.payload),
                    "parentResultId": handoff.parent_result_id,
                    "currentOwner": state.current_owner,
                },
            )

            if requested > self._max_handoffs:
                rejected += 1
                rejection = HandoffRejection(
                    handoff_id=handoff.handoff_id,
                    from_agent=handoff.from_agent,
                    to_agent=handoff.to_agent,
                    code="max_handoffs",
                    message="Handoff budget exhausted.",
                    parent_result_id=handoff.parent_result_id,
                )
                state.record_rejection(rejection)
                emit(
                    "handoff_rejected",
                    {
                        "handoffId": rejection.handoff_id,
                        "from": rejection.from_agent,
                        "to": rejection.to_agent,
                        "code": rejection.code,
                        "message": rejection.message,
                        "parentResultId": rejection.parent_result_id,
                        "currentOwner": state.current_owner,
                    },
                )
                state.terminate(
                    "max_handoffs",
                    error={
                        "code": rejection.code,
                        "from": rejection.from_agent,
                        "to": rejection.to_agent,
                        "message": rejection.message,
                    },
                )
                answer, _ = self._synthesizer.synthesize(state)
                state.final_answer = answer
                break

            rejection = self._validate_handoff(
                handoff,
                current_owner=owner,
            )
            if rejection is not None:
                rejected += 1
                state.record_rejection(rejection)
                emit(
                    "handoff_rejected",
                    {
                        "handoffId": rejection.handoff_id,
                        "from": rejection.from_agent,
                        "to": rejection.to_agent,
                        "code": rejection.code,
                        "message": rejection.message,
                        "parentResultId": rejection.parent_result_id,
                        "currentOwner": state.current_owner,
                    },
                )
                state.terminate(
                    "invalid_handoff",
                    error={
                        "code": rejection.code,
                        "from": rejection.from_agent,
                        "to": rejection.to_agent,
                        "message": rejection.message,
                    },
                )
                answer, _ = self._synthesizer.synthesize(state)
                state.final_answer = answer
                break

            accepted += 1
            emit(
                "handoff_accepted",
                {
                    "handoffId": handoff.handoff_id,
                    "from": handoff.from_agent,
                    "to": handoff.to_agent,
                    "parentResultId": handoff.parent_result_id,
                },
            )
            previous = state.current_owner
            state.transfer_ownership(handoff)
            emit(
                "ownership_transferred",
                {
                    "handoffId": handoff.handoff_id,
                    "from": previous,
                    "to": state.current_owner,
                    "previousOwner": state.previous_owner,
                    "currentOwner": state.current_owner,
                    "parentResultId": handoff.parent_result_id,
                },
            )
            emit(
                "agent_activated",
                {
                    "agentId": state.current_owner,
                    "enteredVia": "handoff",
                    "currentOwner": state.current_owner,
                    "previousOwner": state.previous_owner,
                    "handoffId": handoff.handoff_id,
                },
            )

        if not state.terminated:
            state.terminate("error", answer="Runtime ended without termination.")

        if state.final_answer is None:
            answer, _ = self._synthesizer.synthesize(state)
            state.final_answer = answer

        if sequence[-1].kind != "termination":
            emit(
                "termination",
                {
                    "reason": state.termination_reason,
                    "currentOwner": state.current_owner,
                    "previousOwner": state.previous_owner,
                },
            )

        return self._finish(
            case,
            state,
            sequence,
            started,
            agent_ms,
            requested=requested,
            accepted=accepted,
            rejected=rejected,
        )

    def _finish(
        self,
        case: MeasuredCase,
        state: HandoffState,
        sequence: list[SequenceEvent],
        started: float,
        agent_ms: int,
        *,
        requested: int,
        accepted: int,
        rejected: int,
    ) -> HandoffRunResult:
        reason: TerminationReason = state.termination_reason or "error"
        successful = sum(1 for item in state.results if item.ok)
        failed = sum(1 for item in state.results if not item.ok)
        total = _elapsed_ms(started)
        runtime_ms = max(0, total - agent_ms)
        activated = [event for event in sequence if event.kind == "agent_activated"]
        return HandoffRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            request=case.request,
            answer=state.final_answer or "",
            ok=reason == "completed" and failed == 0,
            model=self._synthesizer.model_name,
            model_driver=self._synthesizer.driver,
            sequence=sequence,
            metrics=HandoffMetrics(
                total_ms=total,
                agent_ms=agent_ms,
                runtime_ms=runtime_ms,
                agents_activated=len(activated),
                handoffs_requested=requested,
                handoffs_accepted=accepted,
                handoffs_rejected=rejected,
                ownership_transfers=len(state.accepted_handoffs),
                successful_agent_results=successful,
                failed_agent_results=failed,
                termination_reason=reason,
                max_handoffs=self._max_handoffs,
            ),
            state=state.to_public_dict(),
            errors=list(state.errors),
        )


def run_handoffs(
    case: MeasuredCase,
    *,
    catalog: Catalog,
    agents: dict[str, Agent] | None = None,
    synthesizer: Synthesizer | None = None,
    max_handoffs: int = 6,
) -> HandoffRunResult:
    runtime = HandoffRuntime(
        agents=agents or default_agents(catalog),
        synthesizer=synthesizer or MockSynthesizer(),
        max_handoffs=max_handoffs,
    )
    return runtime.run(case)
