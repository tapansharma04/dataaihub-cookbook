"""Model clients for the agent loop.

- OpenAIModelClient: live OpenAI-compatible chat completions with tools
- ScriptedModelClient: deterministic case harness for measured/reproducible runs

Neither client invents hidden chain-of-thought; only observable message content
and tool_calls are returned. The runtime classifies the next action.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from agent.schemas import DecisionKind, ModelToolCall, ModelTurn
from config import Settings


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
        decision: DecisionKind = "tool_call" if tool_calls else "final_answer"
        return ModelTurn(
            content=message.content,
            tool_calls=tool_calls,
            decision=decision,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
        )


@dataclass
class ScriptedTurn:
    """One predetermined observable model response.

    Set `decision` explicitly for invalid-action teaching cases. Otherwise the
    harness infers tool_call vs final_answer from `tool_calls`.
    """

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    decision: DecisionKind | str | None = None
    finish_reason: str = "stop"


class ScriptedModelClient:
    """Deterministic model driver for measured teaching cases.

    Tool calls and final answers are supplied by the case harness. Latency is
    still measured (usually near zero). Provenance is recorded as case-harness
    so traces do not pretend these turns came from a live provider.
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
            for item in (scripted.tool_calls or [])
        ]
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        if scripted.decision is not None:
            decision: DecisionKind | None
            if scripted.decision in {"tool_call", "final_answer", "invalid_action"}:
                decision = scripted.decision  # type: ignore[assignment]
            else:
                # Unrecognized harness label → runtime invalid_action.
                decision = "invalid_action"
        else:
            decision = "tool_call" if tool_calls else "final_answer"
        return ModelTurn(
            content=scripted.content,
            tool_calls=tool_calls,
            decision=decision,
            finish_reason=scripted.finish_reason,
            latency_ms=latency_ms,
        )
