"""Optional final-answer formatting.

Default runs use a deterministic mock that formats the terminating
owner's actual result. This is not coordinator aggregation: it does not
re-run specialists or combine independent delegations.

Live formatting is optional and must call a real provider with that
terminating result. It is never the source of committed traces.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from agent.schemas import AgentResult, TerminationReason
from agent.state import HandoffState
from config import Settings


class Synthesizer(Protocol):
    model_name: str
    driver: str

    def synthesize(self, state: HandoffState) -> tuple[str, int]: ...


def _last_result(state: HandoffState) -> AgentResult | None:
    return state.results[-1] if state.results else None


class MockSynthesizer:
    """Deterministic answer from the terminating owner's actual result."""

    def __init__(self) -> None:
        self.model_name = "mock"
        self.driver = "mock"

    def synthesize(self, state: HandoffState) -> tuple[str, int]:
        reason: TerminationReason | None = state.termination_reason
        last = _last_result(state)
        owner = state.current_owner or "none"
        if reason == "completed" and last is not None:
            summary = last.payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary, 0
            return f"Completed by {owner}.", 0
        if reason == "invalid_handoff" and state.rejections:
            rejection = state.rejections[-1]
            return (
                f"Handoff rejected: {rejection.from_agent} -> "
                f"{rejection.to_agent} ({rejection.code}). "
                f"Ownership remains with {owner}."
            ), 0
        if reason == "agent_failed" and last is not None:
            error = last.error or {}
            return (
                f"Agent {last.agent_id} failed: "
                f"{error.get('code', 'error')}: {error.get('message', '')}. "
                "Not treated as success."
            ), 0
        if reason == "max_handoffs":
            return f"Runtime stopped at the handoff limit. Owner remains {owner}.", 0
        if reason == "invalid_task":
            return "Runtime rejected an empty request.", 0
        if last is not None and last.error:
            error = last.error
            return (
                f"Agent {last.agent_id} error: "
                f"{error.get('code', 'error')}: {error.get('message', '')}."
            ), 0
        return f"Terminated ({reason or 'error'}). Owner remains {owner}.", 0


class LiveSynthesizer:
    """Optional provider formatting of the terminating owner's actual result."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("Live synthesis requires OPENAI_API_KEY")
        self.model_name = settings.chat_model
        self.driver = "openai-compatible"
        self._settings = settings

    def synthesize(self, state: HandoffState) -> tuple[str, int]:
        import time

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self._settings.openai_api_key}
        if self._settings.openai_base_url:
            kwargs["base_url"] = self._settings.openai_base_url
        last = _last_result(state)
        payload = {
            "request": state.request,
            "currentOwner": state.current_owner,
            "previousOwner": state.previous_owner,
            "terminationReason": state.termination_reason,
            "lastResult": last.model_dump(mode="json") if last is not None else None,
            "rejections": [item.model_dump(mode="json") for item in state.rejections],
        }
        started = time.perf_counter()
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self._settings.chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a short operational note from the supplied "
                        "terminating owner result. Do not invent facts. Do not "
                        "include hidden reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True),
                },
            ],
        )
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        text = (response.choices[0].message.content or "").strip()
        return text, latency_ms
