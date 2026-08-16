"""Agent loop runtime — application-controlled iteration toward a goal.

User request
  → Model decision
  → Tool call (application-enforced)
  → Observation (appended to state)
  → Model decision
  → …
  → Final answer / termination

Tool calling is a building block. This runtime is what lets an agent act
repeatedly toward a goal under explicit turn limits and termination rules.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.executor import ToolExecutor, parse_tool_arguments_json
from agent.model import ModelClient
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
You are a support assistant with access to internal tools.

Rules:
- Use tools when you need live service or documentation data.
- Prefer canonical tool argument values.
- After observations, decide the next observable action.
- Do not invent tool results.
- When you have enough information, produce a final answer.
"""


def classify_decision(turn: ModelTurn) -> DecisionKind:
    """Map an observable model turn to a runtime decision kind.

    The application owns classification. Unrecognized decisions become
    invalid_action and terminate the loop safely.
    """
    if turn.decision == "invalid_action":
        return "invalid_action"
    if turn.decision == "tool_call" or turn.tool_calls:
        if not turn.tool_calls:
            return "invalid_action"
        return "tool_call"
    if turn.decision == "final_answer" or not turn.tool_calls:
        return "final_answer"
    return "invalid_action"


def run_agent_loop(
    *,
    request: str,
    model: ModelClient,
    registry: ToolRegistry,
    executor: ToolExecutor,
    max_turns: int = 6,
    max_tool_calls_per_turn: int = 4,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentRunResult:
    started = time.perf_counter()
    definitions = registry.definitions()
    openai_tools = registry.openai_tools()

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

    while not state.terminated:
        if state.current_turn >= state.max_turns:
            answer = (
                "Stopped: reached max_turns without a final answer. "
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
            break

        turn_number = state.begin_turn()
        turn = model.complete(state.messages, openai_tools)
        model_turns += 1
        model_ms += turn.latency_ms

        decision = classify_decision(turn)
        tool_call_payload = [
            {
                "id": tc.id,
                "name": tc.name,
                "argumentsJson": tc.arguments_json,
            }
            for tc in turn.tool_calls
        ]
        state.record_decision(
            RecordedDecision(
                turn=turn_number,
                decision=decision,
                content=turn.content,
                tool_calls=tool_call_payload,
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
                    "toolCalls": tool_call_payload,
                    "promptTokens": turn.prompt_tokens,
                    "completionTokens": turn.completion_tokens,
                },
            )
        )

        if decision == "final_answer":
            answer = (turn.content or "").strip()
            state.terminate("final_answer", answer=answer)
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    turn=turn_number,
                    detail={"answer": answer},
                )
            )
            sequence.append(
                SequenceEvent(
                    kind="termination",
                    turn=turn_number,
                    detail={"reason": "final_answer"},
                )
            )
            break

        if decision == "invalid_action":
            message = (
                "Stopped: model produced an unrecognized action. "
                "The runtime only accepts tool_call or final_answer."
            )
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
                    turn=turn_number,
                    detail={"reason": "invalid_action"},
                )
            )
            break

        # decision == tool_call
        state.messages.append(
            {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_json,
                        },
                    }
                    for tc in turn.tool_calls
                ],
            }
        )

        selected = turn.tool_calls[:max_tool_calls_per_turn]
        if len(turn.tool_calls) > max_tool_calls_per_turn:
            cap_error = {
                "code": "tool_call_cap",
                "message": (
                    f"Model requested {len(turn.tool_calls)} tools; "
                    f"executing only first {max_tool_calls_per_turn}"
                ),
            }
            state.errors.append(cap_error)

        for tc in selected:
            raw_args = parse_tool_arguments_json(tc.arguments_json)
            if raw_args.get("__parse_error__"):
                result = executor.execute(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.name,
                        arguments={"__invalid_json__": tc.arguments_json},
                    )
                )
                if result.error and result.error.get("code") == "invalid_arguments":
                    result.error = {
                        "code": "invalid_arguments",
                        "message": "Model produced non-JSON tool arguments",
                        "raw": tc.arguments_json,
                    }
                display_args: dict[str, Any] = {"raw": tc.arguments_json}
            else:
                result = executor.execute(
                    ToolCallRequest(id=tc.id, name=tc.name, arguments=raw_args)
                )
                display_args = raw_args

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
                        "error": result.error,
                    }
                )

            state.record_tool_call(
                RecordedToolCall(
                    turn=turn_number,
                    call_id=result.call_id,
                    name=result.name,
                    arguments=display_args,
                    validated_arguments=result.validated_arguments,
                    latency_ms=result.latency_ms,
                )
            )
            state.record_observation(
                RecordedObservation(
                    turn=turn_number,
                    call_id=result.call_id,
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
                    turn=turn_number,
                    latency_ms=result.latency_ms,
                    detail={
                        "callId": result.call_id,
                        "name": result.name,
                        "arguments": display_args,
                        "validatedArguments": result.validated_arguments,
                    },
                )
            )
            sequence.append(
                SequenceEvent(
                    kind="observation",
                    turn=turn_number,
                    latency_ms=result.latency_ms,
                    detail={
                        "callId": result.call_id,
                        "name": result.name,
                        "ok": result.ok,
                        "result": result.result,
                        "error": result.error,
                    },
                )
            )

            if result.ok:
                observation_payload: dict[str, Any] = result.result or {}
            else:
                observation_payload = {"ok": False, "error": result.error}

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(observation_payload),
                }
            )
        # Loop continues: observation is now state for the next decision.

    assert state.termination_reason is not None
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
        ),
        state=state.to_public_dict(),
        errors=list(state.errors),
    )


def tool_definitions_payload(definitions: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [d.model_dump(by_alias=True) for d in definitions]
