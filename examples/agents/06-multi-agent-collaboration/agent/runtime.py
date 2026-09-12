"""Coordinator-owned multi-agent runtime.

The coordinator assigns work, records explicit messages, stores specialist
results on TaskState, and aggregates those actual results. Agents never
write shared state.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from agent.agents import Agent, default_agents
from agent.cases import CoordinatorStep, MeasuredCase
from agent.catalog import Catalog
from agent.schemas import (
    AgentMessage,
    AgentResult,
    CollaborationMetrics,
    CollaborationRunResult,
    Delegation,
    SequenceEvent,
    TerminationReason,
)
from agent.state import TaskState
from agent.synthesizer import MockSynthesizer, Synthesizer


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _incident_present(result: AgentResult | None) -> bool:
    if result is None or not result.ok:
        return False
    service = result.payload.get("service")
    if not isinstance(service, dict):
        return False
    incident = service.get("incident")
    return bool(incident)


class Coordinator:
    """Application runtime that owns routing, state, and termination."""

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        synthesizer: Synthesizer,
        max_delegations: int,
        stop_on_failure: bool,
    ) -> None:
        self._agents = agents
        self._synthesizer = synthesizer
        self._max_delegations = max_delegations
        self._stop_on_failure = stop_on_failure
        self._message_n = 0
        self._result_n = 0
        self._delegation_n = 0

    def _ids(self) -> tuple[str, str]:
        self._delegation_n += 1
        self._message_n += 1
        return f"delegation-{self._delegation_n}", f"msg-{self._message_n}"

    def _stamp_result(self, result: AgentResult) -> AgentResult:
        self._result_n += 1
        return result.model_copy(update={"result_id": f"result-{self._result_n}"})

    def _build_assignment(
        self,
        step: CoordinatorStep,
        state: TaskState,
    ) -> tuple[dict[str, Any], str | None]:
        assignment = dict(step.assignment)
        parent_result_id = None
        if step.input_from_agent:
            prior = state.result_for(step.input_from_agent)
            if prior is None:
                raise ValueError(
                    f"Coordinator has no result from '{step.input_from_agent}'"
                )
            assignment["prior_result"] = copy.deepcopy(prior.payload)
            parent_result_id = prior.result_id
        return assignment, parent_result_id

    def run(self, case: MeasuredCase) -> CollaborationRunResult:
        started = time.perf_counter()
        coordinator_ms = 0
        agent_ms = 0
        synthesis_ms = 0
        sequence: list[SequenceEvent] = []
        state = TaskState(
            task_id=f"task-{case.trace_id}",
            request=case.request,
            service=case.service,
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
                "service": state.service,
                "exampleClass": case.example_class,
            },
        )

        if not case.request.strip():
            state.terminate(
                "invalid_task",
                answer="Coordinator rejected an empty request.",
                error={"code": "invalid_task", "message": "empty request"},
            )
            emit("error", {"code": "invalid_task", "message": "empty request"})
            emit("termination", {"reason": state.termination_reason})
            return self._finish(
                case, state, sequence, started, coordinator_ms, agent_ms, 0
            )

        for step in case.steps:
            if state.terminated:
                break
            if step.kind == "aggregate":
                decision_start = time.perf_counter()
                emit(
                    "coordinator_decision",
                    {"decision": "aggregate", "resultCount": len(state.results)},
                )
                answer, synthesis_ms = self._synthesizer.synthesize(state)
                coordinator_ms += _elapsed_ms(decision_start)
                reason: TerminationReason = (
                    "no_further_work" if state.skipped else "aggregated"
                )
                emit(
                    "aggregation",
                    {
                        "resultIds": [item.result_id for item in state.results],
                        "agentIds": [item.agent_id for item in state.results],
                        "answer": answer,
                        "skippedAgentIds": [item.agent_id for item in state.skipped],
                    },
                    latency_ms=synthesis_ms,
                )
                state.terminate(reason, answer=answer)
                break

            if step.kind in {"delegate", "delegate_if_incident"}:
                if len(state.delegations) >= self._max_delegations:
                    state.terminate(
                        "max_delegations",
                        answer="Coordinator stopped at the delegation limit.",
                        error={"code": "max_delegations"},
                    )
                    emit(
                        "termination",
                        {"reason": "max_delegations"},
                    )
                    break

                agent_id = step.agent_id
                assert agent_id is not None
                if step.kind == "delegate_if_incident":
                    prior = (
                        state.result_for(step.input_from_agent)
                        if step.input_from_agent
                        else None
                    )
                    if not _incident_present(prior):
                        skip = Delegation(
                            delegation_id=f"skipped-{agent_id}",
                            agent_id=agent_id,
                            assignment=dict(step.assignment),
                            skipped=True,
                            skip_reason="no_incident",
                        )
                        state.record_skip(skip)
                        emit(
                            "coordinator_decision",
                            {
                                "decision": "skip",
                                "agentId": agent_id,
                                "reason": "no_incident",
                            },
                        )
                        continue

                decision_start = time.perf_counter()
                emit(
                    "coordinator_decision",
                    {"decision": "delegate", "agentId": agent_id},
                )
                agent = self._agents.get(agent_id)
                if agent is None:
                    state.terminate(
                        "error",
                        answer=f"Unknown agent '{agent_id}'.",
                        error={"code": "unknown_agent", "agentId": agent_id},
                    )
                    emit(
                        "error",
                        {"code": "unknown_agent", "agentId": agent_id},
                    )
                    coordinator_ms += _elapsed_ms(decision_start)
                    break

                try:
                    assignment, parent_result_id = self._build_assignment(step, state)
                except ValueError as exc:
                    state.terminate(
                        "error",
                        answer=str(exc),
                        error={"code": "missing_prior_result", "message": str(exc)},
                    )
                    emit("error", {"code": "missing_prior_result", "message": str(exc)})
                    coordinator_ms += _elapsed_ms(decision_start)
                    break

                delegation_id, message_id = self._ids()
                payload = copy.deepcopy(assignment)
                delegation = Delegation(
                    delegation_id=delegation_id,
                    agent_id=agent_id,
                    role=agent.role,
                    assignment=copy.deepcopy(assignment),
                    parent_result_id=parent_result_id,
                )
                message = AgentMessage(
                    message_id=message_id,
                    to_agent=agent_id,
                    task_id=state.task_id,
                    payload=payload,
                    parent_result_id=parent_result_id,
                )
                state.record_delegation(delegation.model_copy(deep=True))
                state.record_message(message.model_copy(deep=True))
                emit(
                    "delegation",
                    {
                        "delegationId": delegation_id,
                        "agentId": agent_id,
                        "role": agent.role,
                        "assignment": copy.deepcopy(assignment),
                        "parentResultId": parent_result_id,
                    },
                )
                emit(
                    "agent_selected",
                    {"agentId": agent_id, "role": agent.role},
                )
                emit(
                    "agent_message",
                    {
                        "messageId": message_id,
                        "from": "coordinator",
                        "to": agent_id,
                        "payload": copy.deepcopy(payload),
                        "parentResultId": parent_result_id,
                    },
                )
                coordinator_ms += _elapsed_ms(decision_start)

                agent_start = time.perf_counter()
                raw = agent.handle(message.model_copy(deep=True))
                latency = _elapsed_ms(agent_start)
                agent_ms += latency
                result = self._stamp_result(raw.model_copy(deep=True))
                state.record_result(result)
                emit(
                    "agent_result",
                    {
                        "resultId": result.result_id,
                        "agentId": result.agent_id,
                        "role": result.role,
                        "ok": result.ok,
                        "inReplyTo": result.in_reply_to,
                        "payload": result.payload,
                        "error": result.error,
                    },
                    latency_ms=latency,
                )
                emit(
                    "state_updated",
                    {
                        "taskId": state.task_id,
                        "resultId": result.result_id,
                        "agentId": result.agent_id,
                        "resultCount": len(state.results),
                    },
                )
                if not result.ok:
                    emit(
                        "failure",
                        {
                            "agentId": result.agent_id,
                            "resultId": result.result_id,
                            "error": result.error,
                        },
                    )
                    if self._stop_on_failure:
                        answer, synthesis_ms = self._synthesizer.synthesize(state)
                        state.terminate(
                            "agent_failed",
                            answer=answer,
                            error=result.error,
                        )
                        emit(
                            "aggregation",
                            {
                                "resultIds": [item.result_id for item in state.results],
                                "agentIds": [item.agent_id for item in state.results],
                                "answer": answer,
                                "failed": True,
                            },
                            latency_ms=synthesis_ms,
                        )
                        break

        if not state.terminated:
            answer, synthesis_ms = self._synthesizer.synthesize(state)
            fallback_reason: TerminationReason = (
                "no_further_work" if state.skipped else "aggregated"
            )
            state.terminate(fallback_reason, answer=answer)
            emit(
                "aggregation",
                {
                    "resultIds": [item.result_id for item in state.results],
                    "agentIds": [item.agent_id for item in state.results],
                    "answer": answer,
                },
                latency_ms=synthesis_ms,
            )

        if sequence[-1].kind != "termination":
            emit("termination", {"reason": state.termination_reason})

        return self._finish(
            case, state, sequence, started, coordinator_ms, agent_ms, synthesis_ms
        )

    def _finish(
        self,
        case: MeasuredCase,
        state: TaskState,
        sequence: list[SequenceEvent],
        started: float,
        coordinator_ms: int,
        agent_ms: int,
        synthesis_ms: int,
    ) -> CollaborationRunResult:
        reason = state.termination_reason or "error"
        successful = sum(1 for item in state.results if item.ok)
        failed = sum(1 for item in state.results if not item.ok)
        return CollaborationRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            request=case.request,
            answer=state.final_answer or "",
            ok=reason in {"aggregated", "no_further_work"} and failed == 0,
            model=self._synthesizer.model_name,
            model_driver=self._synthesizer.driver,
            sequence=sequence,
            metrics=CollaborationMetrics(
                total_ms=_elapsed_ms(started),
                coordinator_ms=coordinator_ms,
                agent_ms=agent_ms,
                synthesis_ms=synthesis_ms,
                agents_invoked=len({item.agent_id for item in state.results}),
                successful_agent_results=successful,
                failed_agent_results=failed,
                delegations=len(state.delegations),
                skipped_delegations=len(state.skipped),
                messages=len(state.messages),
                termination_reason=reason,
                max_delegations=self._max_delegations,
            ),
            state=state.to_public_dict(),
            errors=list(state.errors),
        )


def run_collaboration(
    case: MeasuredCase,
    *,
    catalog: Catalog,
    agents: dict[str, Agent] | None = None,
    synthesizer: Synthesizer | None = None,
    max_delegations: int = 6,
) -> CollaborationRunResult:
    runtime = Coordinator(
        agents=agents or default_agents(catalog),
        synthesizer=synthesizer or MockSynthesizer(),
        max_delegations=max_delegations,
        stop_on_failure=case.stop_on_failure,
    )
    return runtime.run(case)
