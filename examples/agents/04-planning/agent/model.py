"""Model clients for the planning runtime.

- OpenAIModelClient: live OpenAI-compatible chat completions
- ScriptedModelClient: deterministic case harness for measured/reproducible runs

Neither client invents hidden chain-of-thought. The harness/model may propose
a plan, a revision, or a final answer. Data tools are executed by the runtime
from validated plan steps, not selected turn-by-turn.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from agent.schemas import DecisionKind, ModelToolCall, ModelTurn
from config import Settings

SUBMIT_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "Submit an explicit multi-step plan. The application validates "
            "each step and executes tool steps itself."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered plan steps",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "action_kind": {
                                "type": "string",
                                "enum": ["tool_call", "finalize"],
                            },
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "status_check",
                                    "docs_lookup",
                                    "remediation",
                                    "required_docs",
                                    "summarize",
                                ],
                            },
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                            "requires_doc_id": {"type": "string"},
                        },
                        "required": [
                            "id",
                            "description",
                            "action_kind",
                            "intent",
                        ],
                    },
                }
            },
            "required": ["steps"],
        },
    },
}

REVISE_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "revise_plan",
        "description": (
            "Submit a revised remaining plan after an observation invalidated "
            "the previous remaining steps. Do not repeat completed steps."
        ),
        "parameters": SUBMIT_PLAN_TOOL["function"]["parameters"],
    },
}


def planning_tools_for(phase: str) -> list[dict[str, Any]]:
    if phase == "create":
        return [SUBMIT_PLAN_TOOL]
    if phase == "revise":
        return [REVISE_PLAN_TOOL]
    return []


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


def _steps_from_planning_call(name: str, arguments_json: str) -> list[dict[str, Any]]:
    parsed = json.loads(arguments_json or "{}")
    if not isinstance(parsed, dict):
        return []
    steps = parsed.get("steps")
    if isinstance(steps, list):
        return [item for item in steps if isinstance(item, dict)]
    return []


def _decision_from_planning_tools(
    tool_calls: list[ModelToolCall],
) -> tuple[DecisionKind, list[dict[str, Any]]]:
    if not tool_calls:
        return "final_answer", []
    first = tool_calls[0]
    if first.name == "submit_plan":
        return "create_plan", _steps_from_planning_call(
            first.name, first.arguments_json
        )
    if first.name == "revise_plan":
        return "revise_plan", _steps_from_planning_call(
            first.name, first.arguments_json
        )
    return "invalid_action", []


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
        decision, proposed_steps = _decision_from_planning_tools(tool_calls)
        return ModelTurn(
            content=message.content,
            tool_calls=tool_calls,
            decision=decision,
            proposed_steps=proposed_steps,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
        )


@dataclass
class ScriptedTurn:
    """One predetermined observable planning decision.

    Set `decision` to create_plan, revise_plan, final_answer, or an
    unrecognized label (runtime invalid_action). `proposed_steps` is the
    plan/revision proposal. Data tool calls are not scripted here.
    """

    content: str | None = None
    decision: DecisionKind | str | None = None
    proposed_steps: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ScriptedModelClient:
    """Deterministic model driver for measured cases.

    Plan proposals and final answers are supplied by the case harness. Latency
    is still measured (usually near zero). Provenance is recorded as
    case-harness so traces do not pretend these turns came from a live provider.
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
                proposed_steps=[],
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
        if scripted.decision is not None:
            if scripted.decision in {
                "create_plan",
                "revise_plan",
                "final_answer",
                "invalid_action",
            }:
                decision: DecisionKind = scripted.decision  # type: ignore[assignment]
            else:
                decision = "invalid_action"
        elif tool_calls:
            decision, _parsed = _decision_from_planning_tools(tool_calls)
        elif scripted.proposed_steps:
            decision = "create_plan"
        else:
            decision = "final_answer"
        proposed = list(scripted.proposed_steps or [])
        if not proposed and tool_calls:
            _decision, proposed = _decision_from_planning_tools(tool_calls)
        return ModelTurn(
            content=scripted.content,
            tool_calls=tool_calls,
            decision=decision,
            proposed_steps=proposed,
            finish_reason=scripted.finish_reason,
            latency_ms=latency_ms,
        )
