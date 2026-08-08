"""Tool-calling loop — the fundamental model ↔ tool interaction cycle.

User request
  → model decides (answer or tool)
  → tool selection / arguments
  → application executes tool (validate + authorize)
  → observation returned to model
  → model continues or finalizes

This is a building block for agents, not a full autonomous agent runtime.
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
    SequenceEvent,
    ToolCallRequest,
    ToolDefinition,
)
from agent.tools import ToolRegistry

SYSTEM_PROMPT = """\
You are a support assistant with access to internal tools.

Rules:
- Use tools when you need live service, user, or documentation data.
- Prefer canonical tool argument values.
- If a tool returns an error, read the error and recover when possible.
- Do not invent tool results.
- When you can answer without tools, answer directly.
"""


def run_tool_calling_loop(
    *,
    request: str,
    model: ModelClient,
    registry: ToolRegistry,
    executor: ToolExecutor,
    max_model_turns: int = 6,
    max_tool_calls_per_turn: int = 4,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentRunResult:
    started = time.perf_counter()
    definitions = registry.definitions()
    openai_tools = registry.openai_tools()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]
    sequence: list[SequenceEvent] = [
        SequenceEvent(kind="user_request", detail={"request": request}),
    ]
    errors: list[dict[str, Any]] = []

    model_ms = 0
    tool_ms = 0
    model_turns = 0
    tool_calls = 0
    successful = 0
    failed = 0
    answer = ""

    for turn_number in range(1, max_model_turns + 1):
        turn = model.complete(messages, openai_tools)
        model_turns += 1
        model_ms += turn.latency_ms

        decision = "tool_call" if turn.tool_calls else "final_answer"
        sequence.append(
            SequenceEvent(
                kind="model_turn",
                turn=turn_number,
                latency_ms=turn.latency_ms,
                detail={
                    "decision": decision,
                    "content": turn.content,
                    "finishReason": turn.finish_reason,
                    "toolCalls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "argumentsJson": tc.arguments_json,
                        }
                        for tc in turn.tool_calls
                    ],
                    "promptTokens": turn.prompt_tokens,
                    "completionTokens": turn.completion_tokens,
                },
            )
        )

        if not turn.tool_calls:
            answer = (turn.content or "").strip()
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    turn=turn_number,
                    detail={"answer": answer},
                )
            )
            break

        # Append the assistant message in OpenAI tool-call form.
        messages.append(
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
            errors.append(
                {
                    "code": "tool_call_cap",
                    "message": (
                        f"Model requested {len(turn.tool_calls)} tools; "
                        f"executing only first {max_tool_calls_per_turn}"
                    ),
                }
            )

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
            else:
                result = executor.execute(
                    ToolCallRequest(id=tc.id, name=tc.name, arguments=raw_args)
                )

            tool_calls += 1
            tool_ms += result.latency_ms
            if result.ok:
                successful += 1
            else:
                failed += 1
                errors.append(
                    {
                        "callId": result.call_id,
                        "name": result.name,
                        "error": result.error,
                    }
                )

            sequence.append(
                SequenceEvent(
                    kind="tool_call",
                    turn=turn_number,
                    latency_ms=result.latency_ms,
                    detail={
                        "callId": result.call_id,
                        "name": result.name,
                        "arguments": raw_args
                        if not raw_args.get("__parse_error__")
                        else {"raw": tc.arguments_json},
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

            observation_payload: dict[str, Any]
            if result.ok:
                observation_payload = result.result or {}
            else:
                observation_payload = {"ok": False, "error": result.error}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(observation_payload),
                }
            )
    else:
        answer = (
            "Stopped: reached max_model_turns without a final answer. "
            "Tighten the request or raise MAX_MODEL_TURNS."
        )
        sequence.append(
            SequenceEvent(
                kind="error",
                detail={"code": "max_model_turns", "message": answer},
            )
        )
        errors.append({"code": "max_model_turns", "message": answer})

    total_ms = int(round((time.perf_counter() - started) * 1000))
    return AgentRunResult(
        request=request,
        answer=answer,
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
        ),
        errors=errors,
    )


def tool_definitions_payload(definitions: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [d.model_dump(by_alias=True) for d in definitions]
