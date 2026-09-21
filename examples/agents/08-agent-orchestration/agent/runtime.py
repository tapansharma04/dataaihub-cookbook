"""Application-owned orchestration runtime.

The runtime owns the workflow plan and decides WHEN each node may
execute from dependencies, readiness, conditions, and node state.
Agents do not call one another. Control is not a handoff and not a
coordinator script of delegate steps.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from agent.agents import Agent, default_agents
from agent.cases import MeasuredCase
from agent.catalog import Catalog
from agent.plan import PlanValidationError, build_plan
from agent.schemas import (
    AgentResult,
    NodeCondition,
    NodeContext,
    OrchestrationMetrics,
    OrchestrationNode,
    OrchestrationPlan,
    OrchestrationRunResult,
    SequenceEvent,
    SkipReason,
    TerminationReason,
)
from agent.state import NodeRecord, OrchestrationState
from agent.synthesizer import MockSynthesizer, Synthesizer


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _payload_value(payload: dict[str, Any], field: str) -> Any:
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def evaluate_condition(condition: NodeCondition, result: AgentResult) -> bool:
    return _payload_value(result.payload, condition.field) == condition.equals


class OrchestrationRuntime:
    """Owns plan execution, readiness, branching, and termination."""

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        synthesizer: Synthesizer,
        max_nodes: int,
    ) -> None:
        self._agents = agents
        self._synthesizer = synthesizer
        self._max_nodes = max_nodes
        self._result_n = 0

    def _stamp_result(
        self,
        result: AgentResult,
        *,
        upstream_result_ids: list[str],
    ) -> AgentResult:
        self._result_n += 1
        return result.model_copy(
            update={
                "result_id": f"result-{self._result_n}",
                "upstream_result_ids": list(upstream_result_ids),
            }
        )

    def run(self, case: MeasuredCase) -> OrchestrationRunResult:
        started = time.perf_counter()
        agent_ms = 0
        sequence: list[SequenceEvent] = []
        state = OrchestrationState(
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
                answer="Runtime rejected an empty request.",
                error={"code": "invalid_task", "message": "empty request"},
            )
            emit("error", {"code": "invalid_task", "message": "empty request"})
            emit("termination", {"reason": state.termination_reason})
            return self._finish(case, state, sequence, started, agent_ms)

        try:
            plan = build_plan(case)
        except PlanValidationError as exc:
            state.terminate(
                "invalid_plan",
                error={"code": "invalid_plan", "message": str(exc)},
            )
            emit("error", {"code": "invalid_plan", "message": str(exc)})
            answer, _ = self._synthesizer.synthesize(state)
            state.final_answer = answer
            emit("termination", {"reason": state.termination_reason})
            return self._finish(case, state, sequence, started, agent_ms)

        state.initialize_plan(plan)
        emit(
            "plan_created",
            {
                "planId": plan.plan_id,
                "nodeIds": plan.node_ids(),
                "nodes": [
                    {
                        "nodeId": node.node_id,
                        "agentId": node.agent_id,
                        "dependencies": list(node.dependencies),
                        "condition": (
                            node.condition.model_dump(mode="json")
                            if node.condition is not None
                            else None
                        ),
                    }
                    for node in plan.nodes
                ],
            },
        )

        waves = 0
        max_waves = max(len(plan.nodes) * 2, 1)
        while not state.terminated and waves < max_waves:
            waves += 1
            self._resolve_pending(plan, state, emit)
            if state.terminated:
                break
            ready_ids = state.ready_node_ids()
            if not ready_ids:
                break
            if len(state.execution_order) + len(ready_ids) > self._max_nodes:
                state.terminate(
                    "error",
                    error={
                        "code": "max_nodes",
                        "message": "Node execution budget exhausted.",
                        "maxNodes": self._max_nodes,
                        "readyNodeIds": ready_ids,
                    },
                )
                emit(
                    "error",
                    {
                        "code": "max_nodes",
                        "maxNodes": self._max_nodes,
                        "readyNodeIds": ready_ids,
                    },
                )
                break
            for node_id in ready_ids:
                node = plan.node(node_id)
                latency = self._execute_node(node, state, emit)
                agent_ms += latency

        if not state.terminated:
            pending = state.node_ids_in("PENDING")
            ready = state.node_ids_in("READY")
            running = state.node_ids_in("RUNNING")
            failed = state.node_ids_in("FAILED")
            if pending or ready or running:
                state.terminate(
                    "error",
                    error={
                        "code": "unresolved_nodes",
                        "pendingNodeIds": pending,
                        "readyNodeIds": ready,
                        "runningNodeIds": running,
                    },
                )
                emit(
                    "error",
                    {
                        "code": "unresolved_nodes",
                        "pendingNodeIds": pending,
                        "readyNodeIds": ready,
                        "runningNodeIds": running,
                    },
                )
            elif failed:
                state.terminate("workflow_failed")
            else:
                state.terminate("completed")

        if state.final_answer is None:
            answer, format_ms = self._synthesizer.synthesize(state)
            state.final_answer = answer
            if state.termination_reason == "completed":
                terminal = [
                    node_id
                    for node_id, record in state.nodes.items()
                    if record.state == "COMPLETED"
                    and node_id
                    not in {
                        dep
                        for node in plan.nodes
                        if state.nodes[node.node_id].state == "COMPLETED"
                        for dep in node.dependencies
                    }
                ]
                emit(
                    "completed",
                    {
                        "answer": answer,
                        "terminalNodeIds": terminal,
                        "resultIds": [item.result_id for item in state.results],
                    },
                    latency_ms=format_ms,
                )

        if sequence[-1].kind != "termination":
            emit(
                "termination",
                {
                    "reason": state.termination_reason,
                    "ok": state.termination_reason == "completed"
                    and not any(
                        record.state == "FAILED" for record in state.nodes.values()
                    ),
                    "nodeStates": {
                        node_id: record.state for node_id, record in state.nodes.items()
                    },
                },
            )

        return self._finish(case, state, sequence, started, agent_ms)

    def _resolve_pending(
        self,
        plan: OrchestrationPlan,
        state: OrchestrationState,
        emit,
    ) -> None:
        changed = True
        while changed:
            changed = False
            for node in plan.nodes:
                record = state.record(node.node_id)
                if record.state != "PENDING":
                    continue
                dep_records = [state.record(dep) for dep in node.dependencies]
                if any(item.state == "FAILED" for item in dep_records):
                    failed = [
                        item.node_id for item in dep_records if item.state == "FAILED"
                    ]
                    self._skip(
                        record,
                        reason="dependency_failed",
                        emit=emit,
                        detail={
                            "failedDependencyIds": failed,
                            "dependencyIds": list(node.dependencies),
                        },
                    )
                    changed = True
                    continue
                if any(item.state == "SKIPPED" for item in dep_records):
                    skipped = [
                        item.node_id for item in dep_records if item.state == "SKIPPED"
                    ]
                    self._skip(
                        record,
                        reason="dependency_skipped",
                        emit=emit,
                        detail={
                            "skippedDependencyIds": skipped,
                            "dependencyIds": list(node.dependencies),
                        },
                    )
                    changed = True
                    continue
                if any(item.state != "COMPLETED" for item in dep_records):
                    continue
                upstream_ids = [
                    item.result.result_id
                    for item in dep_records
                    if item.result is not None
                ]
                record.upstream_result_ids = list(upstream_ids)
                if node.condition is not None:
                    source = state.result_for_node(node.condition.source_node_id)
                    if source is None:
                        state.terminate(
                            "error",
                            error={
                                "code": "missing_condition_source",
                                "nodeId": node.node_id,
                                "sourceNodeId": node.condition.source_node_id,
                            },
                        )
                        emit(
                            "error",
                            {
                                "code": "missing_condition_source",
                                "nodeId": node.node_id,
                                "sourceNodeId": node.condition.source_node_id,
                            },
                        )
                        return
                    actual = _payload_value(source.payload, node.condition.field)
                    selected = evaluate_condition(node.condition, source)
                    emit(
                        "condition_evaluated",
                        {
                            "nodeId": node.node_id,
                            "agentId": node.agent_id,
                            "sourceNodeId": node.condition.source_node_id,
                            "field": node.condition.field,
                            "expected": node.condition.equals,
                            "actual": actual,
                            "selected": selected,
                            "sourceResultId": source.result_id,
                        },
                    )
                    record.condition_matched = selected
                    if not selected:
                        self._skip(
                            record,
                            reason="condition_not_selected",
                            emit=emit,
                            detail={
                                "sourceNodeId": node.condition.source_node_id,
                                "field": node.condition.field,
                                "expected": node.condition.equals,
                                "actual": actual,
                                "sourceResultId": source.result_id,
                                "dependencyIds": list(node.dependencies),
                                "upstreamResultIds": list(upstream_ids),
                            },
                        )
                        changed = True
                        continue
                ready_reason = (
                    "no_dependencies"
                    if not node.dependencies
                    else "dependencies_completed"
                )
                record.ready_reason = ready_reason
                record.transition("READY", reason=ready_reason)
                emit(
                    "node_ready",
                    {
                        "nodeId": node.node_id,
                        "agentId": node.agent_id,
                        "reason": ready_reason,
                        "dependencyIds": list(node.dependencies),
                        "upstreamResultIds": list(upstream_ids),
                        "conditionMatched": record.condition_matched,
                    },
                )
                changed = True

    def _skip(
        self,
        record: NodeRecord,
        *,
        reason: SkipReason,
        emit,
        detail: dict[str, Any],
    ) -> None:
        record.skip_reason = reason
        record.transition("SKIPPED", reason=reason)
        emit(
            "node_skipped",
            {
                "nodeId": record.node_id,
                "agentId": record.agent_id,
                "reason": reason,
                **detail,
            },
        )

    def _identity_error(
        self,
        node: OrchestrationNode,
        agent: Agent,
        result: AgentResult,
    ) -> dict[str, Any] | None:
        if result.agent_id != node.agent_id:
            return {
                "code": "agent_mismatch",
                "message": (
                    f"Result agent_id '{result.agent_id}' is not "
                    f"node agent '{node.agent_id}'."
                ),
                "agentId": result.agent_id,
                "expectedAgentId": node.agent_id,
                "nodeId": result.node_id,
                "expectedNodeId": node.node_id,
            }
        if result.node_id != node.node_id:
            return {
                "code": "node_mismatch",
                "message": (
                    f"Result node_id '{result.node_id}' is not "
                    f"planned node '{node.node_id}'."
                ),
                "nodeId": result.node_id,
                "expectedNodeId": node.node_id,
                "agentId": result.agent_id,
            }
        if result.role != agent.role:
            return {
                "code": "role_mismatch",
                "message": (
                    f"Result role '{result.role}' is not agent role '{agent.role}'."
                ),
                "role": result.role,
                "expectedRole": agent.role,
                "agentId": result.agent_id,
                "nodeId": result.node_id,
            }
        return None

    def _execute_node(
        self,
        node: OrchestrationNode,
        state: OrchestrationState,
        emit,
    ) -> int:
        record = state.record(node.node_id)
        if record.state != "READY":
            raise RuntimeError(f"Node '{node.node_id}' is {record.state}, not READY")
        record.transition("RUNNING", reason="scheduled")
        emit(
            "node_started",
            {
                "nodeId": node.node_id,
                "agentId": node.agent_id,
                "dependencyIds": list(node.dependencies),
                "upstreamResultIds": list(record.upstream_result_ids),
            },
        )
        agent = self._agents.get(node.agent_id)
        if agent is None:
            error = {
                "code": "unknown_agent",
                "message": f"Unknown agent '{node.agent_id}'",
                "agentId": node.agent_id,
                "nodeId": node.node_id,
            }
            record.transition("FAILED", reason="unknown_agent")
            state.execution_order.append(node.node_id)
            state.errors.append(error)
            emit(
                "node_failed",
                {
                    "nodeId": node.node_id,
                    "agentId": node.agent_id,
                    "error": error,
                },
            )
            return 0

        upstream: dict[str, AgentResult] = {}
        for dep in node.dependencies:
            dep_result = state.result_for_node(dep)
            if dep_result is None:
                raise RuntimeError(
                    f"Node '{node.node_id}' missing upstream result for '{dep}'"
                )
            upstream[dep] = dep_result.model_copy(deep=True)
        context = NodeContext(
            task_id=state.task_id,
            node_id=node.node_id,
            agent_id=node.agent_id,
            assignment=copy.deepcopy(node.assignment),
            upstream_results=upstream,
            upstream_result_ids=list(record.upstream_result_ids),
        )
        agent_start = time.perf_counter()
        raw = agent.handle(context.model_copy(deep=True))
        latency = _elapsed_ms(agent_start)
        stamped = self._stamp_result(
            raw.model_copy(deep=True),
            upstream_result_ids=record.upstream_result_ids,
        )
        identity_error = self._identity_error(node, agent, stamped)
        if identity_error is not None:
            stamped = stamped.model_copy(update={"ok": False, "error": identity_error})
            state.record_result(stamped, bound_node_id=node.node_id)
            record.transition("FAILED", reason=identity_error["code"])
            emit(
                "node_result",
                {
                    "resultId": stamped.result_id,
                    "nodeId": stamped.node_id,
                    "agentId": stamped.agent_id,
                    "role": stamped.role,
                    "ok": False,
                    "payload": copy.deepcopy(stamped.payload),
                    "error": stamped.error,
                    "upstreamResultIds": list(stamped.upstream_result_ids),
                    "dependencyIds": list(node.dependencies),
                    "expectedNodeId": node.node_id,
                    "expectedAgentId": node.agent_id,
                },
                latency_ms=latency,
            )
            emit(
                "node_failed",
                {
                    "nodeId": node.node_id,
                    "agentId": node.agent_id,
                    "resultId": stamped.result_id,
                    "error": identity_error,
                    "actualAgentId": stamped.agent_id,
                    "actualNodeId": stamped.node_id,
                    "actualRole": stamped.role,
                },
            )
            return latency
        state.record_result(stamped, bound_node_id=node.node_id)
        emit(
            "node_result",
            {
                "resultId": stamped.result_id,
                "nodeId": stamped.node_id,
                "agentId": stamped.agent_id,
                "role": stamped.role,
                "ok": stamped.ok,
                "payload": copy.deepcopy(stamped.payload),
                "error": stamped.error,
                "upstreamResultIds": list(stamped.upstream_result_ids),
                "dependencyIds": list(node.dependencies),
            },
            latency_ms=latency,
        )
        if stamped.ok:
            record.transition("COMPLETED", reason="agent_ok")
            emit(
                "node_completed",
                {
                    "nodeId": node.node_id,
                    "agentId": node.agent_id,
                    "resultId": stamped.result_id,
                    "upstreamResultIds": list(stamped.upstream_result_ids),
                },
            )
        else:
            record.transition("FAILED", reason="agent_failed")
            emit(
                "node_failed",
                {
                    "nodeId": node.node_id,
                    "agentId": node.agent_id,
                    "resultId": stamped.result_id,
                    "error": stamped.error,
                    "upstreamResultIds": list(stamped.upstream_result_ids),
                },
            )
        return latency

    def _finish(
        self,
        case: MeasuredCase,
        state: OrchestrationState,
        sequence: list[SequenceEvent],
        started: float,
        agent_ms: int,
    ) -> OrchestrationRunResult:
        reason: TerminationReason = state.termination_reason or "error"
        completed = sum(1 for item in state.nodes.values() if item.state == "COMPLETED")
        failed = sum(1 for item in state.nodes.values() if item.state == "FAILED")
        skipped = sum(1 for item in state.nodes.values() if item.state == "SKIPPED")
        ready_events = [event for event in sequence if event.kind == "node_ready"]
        total = _elapsed_ms(started)
        runtime_ms = max(0, total - agent_ms)
        return OrchestrationRunResult(
            case_id=case.trace_id,
            example_class=case.example_class,
            request=case.request,
            answer=state.final_answer or "",
            ok=reason == "completed" and failed == 0,
            model=self._synthesizer.model_name,
            model_driver=self._synthesizer.driver,
            sequence=sequence,
            metrics=OrchestrationMetrics(
                total_ms=total,
                runtime_ms=runtime_ms,
                agent_ms=agent_ms,
                nodes_ready=len(ready_events),
                nodes_executed=len(state.execution_order),
                nodes_completed=completed,
                nodes_failed=failed,
                nodes_skipped=skipped,
                termination_reason=reason,
                max_nodes=self._max_nodes,
            ),
            state=state.to_public_dict(),
            errors=list(state.errors),
        )


def run_orchestration(
    case: MeasuredCase,
    *,
    catalog: Catalog,
    agents: dict[str, Agent] | None = None,
    synthesizer: Synthesizer | None = None,
    max_nodes: int = 12,
) -> OrchestrationRunResult:
    runtime = OrchestrationRuntime(
        agents=agents or default_agents(catalog),
        synthesizer=synthesizer or MockSynthesizer(),
        max_nodes=max_nodes,
    )
    return runtime.run(case)
