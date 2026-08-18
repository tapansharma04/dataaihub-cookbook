"""Model clients for the memory runtime.

- OpenAIModelClient: live OpenAI-compatible chat completions
- ScriptedModelClient: deterministic case harness for measured/reproducible runs

Neither client invents hidden chain-of-thought. The harness/model may propose
a memory store, a memory retrieve, or a final answer. The application
validates and performs the memory operation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from agent.schemas import (
    DecisionKind,
    MemoryOperationDefinition,
    ModelToolCall,
    ModelTurn,
)
from config import Settings

STORE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "store_memory",
        "description": (
            "Propose storing a user-provided fact in application memory. "
            "The application validates scope, key, value, and provenance "
            "and performs the write."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Memory owner, typically a user id such as u-1001",
                },
                "key": {
                    "type": "string",
                    "description": "Allowlisted memory key, e.g. notification_channel",
                },
                "value": {
                    "type": "object",
                    "description": 'Stored content, e.g. {"channel": "email"}',
                },
                "source": {
                    "type": "string",
                    "enum": ["user", "system", "tool", "application"],
                    "description": "Provenance of the information to store",
                },
            },
            "required": ["key", "value", "source"],
        },
    },
}

RETRIEVE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_memory",
        "description": (
            "Propose retrieving a memory record by key. The application "
            "enforces the session scope and returns a miss explicitly when "
            "no record exists."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Memory owner, typically a user id such as u-1001",
                },
                "key": {
                    "type": "string",
                    "description": "Allowlisted memory key, e.g. notification_channel",
                },
            },
            "required": ["key"],
        },
    },
}


def memory_tools() -> list[dict[str, Any]]:
    return [STORE_MEMORY_TOOL, RETRIEVE_MEMORY_TOOL]


def memory_operation_definitions() -> list[MemoryOperationDefinition]:
    return [
        MemoryOperationDefinition(
            name="store_memory",
            description=STORE_MEMORY_TOOL["function"]["description"],
            parameters=STORE_MEMORY_TOOL["function"]["parameters"],
        ),
        MemoryOperationDefinition(
            name="retrieve_memory",
            description=RETRIEVE_MEMORY_TOOL["function"]["description"],
            parameters=RETRIEVE_MEMORY_TOOL["function"]["parameters"],
        ),
    ]


class ModelClient(Protocol):
    model_name: str
    driver: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn: ...


def get_openai_client(settings: Settings) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _object_from_arguments(arguments_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decision_from_memory_tools(
    tool_calls: list[ModelToolCall],
) -> tuple[DecisionKind, dict[str, Any] | None, dict[str, Any] | None]:
    if not tool_calls:
        return "final_answer", None, None
    first = tool_calls[0]
    payload = _object_from_arguments(first.arguments_json)
    if first.name == "store_memory":
        return "store_memory", payload, None
    if first.name == "retrieve_memory":
        return "retrieve_memory", None, payload
    return "invalid_action", None, None


class OpenAIModelClient:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model_name = model
        self.driver = "openai-compatible"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ModelToolCall] = []
        for tc in message.tool_calls or []:
            tool_calls.append(
                ModelToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_json=tc.function.arguments or "{}",
                )
            )
        usage = getattr(response, "usage", None)
        decision, memory_write, memory_read = _decision_from_memory_tools(tool_calls)
        return ModelTurn(
            content=message.content,
            tool_calls=tool_calls,
            decision=decision,
            memory_write=memory_write,
            memory_read=memory_read,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
        )


@dataclass
class ScriptedTurn:
    """One predetermined observable memory decision.

    Set `decision` to store_memory, retrieve_memory, final_answer, or an
    unrecognized label (runtime invalid_action).
    """

    content: str | None = None
    decision: DecisionKind | str | None = None
    memory_write: dict[str, Any] | None = None
    memory_read: dict[str, Any] | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ScriptedModelClient:
    """Deterministic model driver for measured cases.

    Memory proposals and final answers are supplied by the case harness.
    Latency is still measured (usually near zero). Provenance is recorded
    as case-harness so traces do not pretend these turns came from a live
    provider.
    """

    def __init__(
        self,
        turns: list[ScriptedTurn],
        *,
        model_name: str = "case-harness",
    ) -> None:
        self.model_name = model_name
        self.driver = "case-harness"
        self._turns = list(turns)
        self._index = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        del messages, tools  # harness does not inspect live prompts
        started = time.perf_counter()
        if self._index >= len(self._turns):
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return ModelTurn(
                content="(case harness exhausted — no further model turns)",
                tool_calls=[],
                decision="final_answer",
                finish_reason="stop",
                latency_ms=latency_ms,
            )
        scripted = self._turns[self._index]
        self._index += 1
        tool_calls = [
            ModelToolCall(
                id=str(item["id"]),
                name=str(item["name"]),
                arguments_json=(
                    item["arguments_json"]
                    if "arguments_json" in item
                    else json.dumps(item.get("arguments", {}))
                ),
            )
            for item in scripted.tool_calls
        ]
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        parsed_write = scripted.memory_write
        parsed_read = scripted.memory_read
        if scripted.decision is not None:
            if scripted.decision in {
                "store_memory",
                "retrieve_memory",
                "final_answer",
                "invalid_action",
            }:
                decision: DecisionKind = scripted.decision  # type: ignore[assignment]
            else:
                decision = "invalid_action"
        elif tool_calls:
            decision, parsed_write, parsed_read = _decision_from_memory_tools(
                tool_calls
            )
        else:
            decision = "final_answer"
        return ModelTurn(
            content=scripted.content,
            tool_calls=tool_calls,
            decision=decision,
            memory_write=parsed_write,
            memory_read=parsed_read,
            finish_reason=scripted.finish_reason,
            latency_ms=latency_ms,
        )
