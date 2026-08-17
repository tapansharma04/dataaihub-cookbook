"""Planning runtime — application-managed plan execution.

Goal
  → Create plan
  → Execute step
  → Observe
  → Continue or revise
  → Final answer / termination

This is not an agent loop. The model does not repeatedly select the next
tool. It proposes a plan (or a revision, or a final answer). The application
owns plan state, step transitions, tool execution, and termination.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.executor import ToolExecutor
from agent.model import ModelClient, planning_tools_for
from agent.plan import PlanValidationError, interpret_observation, step_to_public
from agent.schemas import (
    AgentRunMetrics,
    AgentRunResult,
    DecisionKind,
    ModelTurn,
    SequenceEvent,
    ToolCallRequest,
    ToolDefinition,
)
from agent.state import (
    RecordedDecision,
    RecordedObservation,
    RecordedToolCall,
    initial_state,
)
from agent.tools import ToolRegistry

SYSTEM_PROMPT = """\
You are a support assistant. You do not call data tools directly.

First, submit an explicit multi-step plan. Each step is either a tool_call
using get_service_status, get_user_profile, or search_documentation, or a
finalize step.

The application validates the plan, executes tool steps, and tracks progress.
If an observation invalidates remaining steps, submit a revised remaining
plan. Do not repeat completed steps.

When the plan is complete or blocked, produce a final answer from the
observations. Do not invent tool results. Do not claim a blocked plan
succeeded.
"""


def classify_decision(turn: ModelTurn) -> DecisionKind:
    """Map an observable model turn to a runtime planning decision.

    Direct data-tool calls are invalid. The runtime only accepts create_plan,
    revise_plan, or final_answer.
    """
    if turn.decision == "invalid_action":
        return "invalid_action"
    data_tool_names = {
        tc.name
        for tc in turn.tool_calls
        if tc.name not in {"submit_plan", "revise_plan"}
    }
    if data_tool_names:
        return "invalid_action"
    if turn.decision in {"create_plan", "revise_plan", "final_answer"}:
        return turn.decision
    if turn.proposed_steps:
        return "create_plan"
    if turn.content and not turn.tool_calls:
        return "final_answer"
    return "invalid_action"


def _note(state, text: str) -> None:
    state.messages.append({"role": "user", "content": text})


def run_planning_loop(
    *,
    request: str,
    model: ModelClient,
    registry: ToolRegistry,
    executor: ToolExecutor,
    max_turns: int = 6,
    max_tool_calls_per_turn: int = 4,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentRunResult:
    del max_tool_calls_per_turn  # one tool per plan step; cap kept for config parity
    started = time.perf_counter()
    definitions = registry.definitions()

    state = initial_state(
        request=request,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )
    sequence: list[SequenceEvent] = [
        SequenceEvent(kind="user_request", detail={"request": request}),
    ]

    model_ms = 0
    tool_ms = 0
    model_turns = 0
    tool_calls = 0
    successful = 0
    failed = 0

    def record_model_turn(turn: ModelTurn, decision: DecisionKind) -> int:
        nonlocal model_ms, model_turns
        turn_number = state.begin_turn()
        model_turns += 1
        model_ms += turn.latency_ms
        state.record_decision(
            RecordedDecision(
                turn=turn_number,
                decision=decision,
                content=turn.content,
                proposed_steps=list(turn.proposed_steps),
                finish_reason=turn.finish_reason,
                latency_ms=turn.latency_ms,
            )
        )
        sequence.append(
            SequenceEvent(
                kind="model_decision",
                turn=turn_number,
                latency_ms=turn.latency_ms,
                detail={
                    "decision": decision,
                    "content": turn.content,
                    "finishReason": turn.finish_reason,
                    "proposedSteps": list(turn.proposed_steps),
                    "promptTokens": turn.prompt_tokens,
                    "completionTokens": turn.completion_tokens,
                },
            )
        )
        return turn_number

    def stop_invalid(turn_number: int | None, message: str) -> None:
        state.terminate(
            "invalid_action",
            answer=message,
            error={"code": "invalid_action", "message": message},
        )
        sequence.append(
            SequenceEvent(
                kind="error",
                turn=turn_number,
                detail={"code": "invalid_action", "message": message},
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                turn=turn_number if turn_number is not None else state.current_turn,
                detail={"reason": "invalid_action"},
            )
        )

    def stop_max_turns() -> None:
        answer = (
            "Stopped: reached max_turns without completing the plan. "
            "The runtime enforces a hard turn limit to prevent runaway loops."
        )
        state.terminate(
            "max_turns",
            answer=answer,
            error={"code": "max_turns", "message": answer},
        )
        sequence.append(
            SequenceEvent(
                kind="error",
                detail={"code": "max_turns", "message": answer},
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                turn=state.current_turn,
                detail={
                    "reason": "max_turns",
                    "maxTurns": state.max_turns,
                    "currentTurn": state.current_turn,
                },
            )
        )

    def consult(phase: str) -> tuple[int, ModelTurn, DecisionKind] | None:
        if state.current_turn >= state.max_turns:
            stop_max_turns()
            return None
        turn = model.complete(state.messages, planning_tools_for(phase))
        decision = classify_decision(turn)
        turn_number = record_model_turn(turn, decision)
        if decision == "invalid_action":
            stop_invalid(
                turn_number,
                "Stopped: model produced an unrecognized action. "
                "The planning runtime only accepts create_plan, revise_plan, "
                "or final_answer. Data tools run only as validated plan steps.",
            )
            return None
        return turn_number, turn, decision

    def emit_step_started(step) -> None:
        sequence.append(
            SequenceEvent(
                kind="plan_step_started",
                turn=state.current_turn,
                detail={
                    "stepId": step.id,
                    "description": step.description,
                    "actionKind": step.action_kind,
                    "intent": step.intent,
                    "status": step.status,
                    "planVersion": state.plan.version if state.plan else None,
                    "progress": state.progress(),
                },
            )
        )

    def emit_step_finished(step) -> None:
        sequence.append(
            SequenceEvent(
                kind="plan_step_completed",
                turn=state.current_turn,
                detail={
                    "stepId": step.id,
                    "description": step.description,
                    "status": step.status,
                    "planVersion": state.plan.version if state.plan else None,
                    "progress": state.progress(),
                },
            )
        )

    while not state.terminated:
        if state.plan is None:
            consulted = consult("create")
            if consulted is None:
                break
            turn_number, turn, decision = consulted
            if decision != "create_plan":
                stop_invalid(
                    turn_number,
                    "Stopped: expected create_plan before execution.",
                )
                break
            try:
                plan = state.install_plan(turn.proposed_steps, registry=registry)
            except PlanValidationError as exc:
                stop_invalid(turn_number, f"Stopped: invalid plan ({exc}).")
                break
            sequence.append(
                SequenceEvent(
                    kind="plan_created",
                    turn=turn_number,
                    detail={
                        "planId": plan.id,
                        "version": plan.version,
                        "status": plan.status,
                        "steps": [step_to_public(s) for s in plan.steps],
                    },
                )
            )
            _note(
                state,
                "The application accepted the plan and will execute validated steps.",
            )
            continue

        assert state.plan is not None
        next_step = state.plan.next_pending_step()
        if next_step is None:
            consulted = consult("finalize")
            if consulted is None:
                break
            turn_number, turn, decision = consulted
            if decision != "final_answer":
                stop_invalid(
                    turn_number,
                    "Stopped: expected a final answer after the plan stopped.",
                )
                break
            answer = (turn.content or "").strip()
            reason: Any = (
                "plan_failed" if state.plan.status == "failed" else "final_answer"
            )
            if reason == "final_answer":
                try:
                    state.complete_plan()
                except PlanValidationError:
                    reason = "plan_failed"
                    state.fail_plan()
            state.terminate(reason, answer=answer)
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    turn=turn_number,
                    detail={
                        "answer": answer,
                        "planStatus": state.plan.status,
                        "planVersion": state.plan.version,
                    },
                )
            )
            sequence.append(
                SequenceEvent(
                    kind="termination",
                    turn=turn_number,
                    detail={
                        "reason": reason,
                        "planStatus": state.plan.status,
                        "planVersion": state.plan.version,
                        "progress": state.progress(),
                    },
                )
            )
            break

        step = state.start_current_step()
        emit_step_started(step)

        if step.action_kind == "finalize":
            consulted = consult("finalize")
            if consulted is None:
                break
            turn_number, turn, decision = consulted
            if decision != "final_answer":
                stop_invalid(
                    turn_number,
                    "Stopped: expected a final answer for the finalize step.",
                )
                break
            answer = (turn.content or "").strip()
            state.complete_current_step(step.id)
            emit_step_finished(state.plan.step_by_id(step.id))
            state.complete_plan()
            state.terminate("final_answer", answer=answer)
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    turn=turn_number,
                    detail={
                        "answer": answer,
                        "planStatus": state.plan.status,
                        "planVersion": state.plan.version,
                    },
                )
            )
            sequence.append(
                SequenceEvent(
                    kind="termination",
                    turn=turn_number,
                    detail={
                        "reason": "final_answer",
                        "planStatus": state.plan.status,
                        "planVersion": state.plan.version,
                        "progress": state.progress(),
                    },
                )
            )
            break

        call_id = f"call_v{state.plan.version}_{step.id}"
        result = executor.execute(
            ToolCallRequest(
                id=call_id,
                name=step.tool or "",
                arguments=dict(step.arguments),
            )
        )
        tool_calls += 1
        tool_ms += result.latency_ms
        if result.ok:
            successful += 1
        else:
            failed += 1
            state.errors.append(
                {
                    "callId": result.call_id,
                    "name": result.name,
                    "stepId": step.id,
                    "error": result.error,
                }
            )

        state.record_tool_call(
            RecordedToolCall(
                turn=state.current_turn,
                call_id=result.call_id,
                step_id=step.id,
                plan_version=state.plan.version,
                name=result.name,
                arguments=dict(step.arguments),
                validated_arguments=result.validated_arguments,
                latency_ms=result.latency_ms,
            )
        )
        state.record_observation(
            RecordedObservation(
                turn=state.current_turn,
                call_id=result.call_id,
                step_id=step.id,
                plan_version=state.plan.version,
                name=result.name,
                ok=result.ok,
                result=result.result,
                error=result.error,
                latency_ms=result.latency_ms,
            )
        )
        sequence.append(
            SequenceEvent(
                kind="tool_call",
                turn=state.current_turn,
                latency_ms=result.latency_ms,
                detail={
                    "callId": result.call_id,
                    "stepId": step.id,
                    "planVersion": state.plan.version,
                    "name": result.name,
                    "arguments": dict(step.arguments),
                    "validatedArguments": result.validated_arguments,
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="observation",
                turn=state.current_turn,
                latency_ms=result.latency_ms,
                detail={
                    "callId": result.call_id,
                    "stepId": step.id,
                    "planVersion": state.plan.version,
                    "name": result.name,
                    "ok": result.ok,
                    "result": result.result,
                    "error": result.error,
                },
            )
        )
        _note(
            state,
            "Observation for "
            f"{step.id} ({result.name}): {json.dumps(result.result or result.error)}",
        )

        remaining = [
            item
            for item in state.plan.steps
            if item.status == "pending" and item.id != step.id
        ]
        effect = interpret_observation(
            step=step,
            result=result,
            remaining=remaining,
        )

        if effect.kind == "block":
            state.fail_current_step(step.id)
            skipped = state.skip_remaining_steps()
            state.fail_plan()
            emit_step_finished(state.plan.step_by_id(step.id))
            _note(
                state,
                (
                    f"The plan is blocked: {effect.reason}. "
                    f"Skipped remaining steps: {skipped}. "
                    "Produce a final answer that reports this limitation. "
                    "Do not claim the plan succeeded."
                ),
            )
            continue

        state.complete_current_step(step.id)
        emit_step_finished(state.plan.step_by_id(step.id))

        if effect.kind == "revise":
            consulted = consult("revise")
            if consulted is None:
                break
            turn_number, turn, decision = consulted
            if decision != "revise_plan":
                stop_invalid(
                    turn_number,
                    "Stopped: expected revise_plan after the remaining plan "
                    "was invalidated.",
                )
                break
            try:
                revision = state.revise_current_plan(
                    turn.proposed_steps,
                    registry=registry,
                    reason=effect.reason or "remaining plan invalidated",
                    observation_call_id=result.call_id,
                )
            except PlanValidationError as exc:
                stop_invalid(turn_number, f"Stopped: invalid revised plan ({exc}).")
                break
            original = state.plan_history[-1]
            sequence.append(
                SequenceEvent(
                    kind="plan_revised",
                    turn=turn_number,
                    detail={
                        **revision.to_public_dict(),
                        "originalPlan": original.to_public_dict(),
                        "revisedPlan": state.plan.to_public_dict(),
                        "observation": {
                            "callId": result.call_id,
                            "stepId": step.id,
                            "name": result.name,
                            "ok": result.ok,
                            "result": result.result,
                        },
                    },
                )
            )
            _note(
                state,
                (
                    f"The application installed plan version {state.plan.version}, "
                    f"which supersedes version {revision.from_version}."
                ),
            )
            continue

    assert state.termination_reason is not None
    counts = (
        state.plan.counts()
        if state.plan is not None
        else {
            "plan_steps": 0,
            "completed_steps": 0,
            "skipped_steps": 0,
            "failed_steps": 0,
        }
    )
    total_ms = int(round((time.perf_counter() - started) * 1000))
    return AgentRunResult(
        request=request,
        answer=state.final_answer or "",
        model=model.model_name,
        model_driver=model.driver,
        tool_definitions=definitions,
        sequence=sequence,
        metrics=AgentRunMetrics(
            total_ms=total_ms,
            model_ms=model_ms,
            tool_ms=tool_ms,
            model_turns=model_turns,
            tool_calls=tool_calls,
            successful_tool_calls=successful,
            failed_tool_calls=failed,
            termination_reason=state.termination_reason,
            max_turns=max_turns,
            plan_steps=counts["plan_steps"],
            completed_steps=counts["completed_steps"],
            skipped_steps=counts["skipped_steps"],
            failed_steps=counts["failed_steps"],
            plan_revisions=len(state.revisions),
            plan_version=state.plan.version if state.plan is not None else 0,
            plan_status=state.plan.status if state.plan is not None else None,
        ),
        state=state.to_public_dict(),
        errors=list(state.errors),
    )


def tool_definitions_payload(definitions: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [d.model_dump(by_alias=True) for d in definitions]
