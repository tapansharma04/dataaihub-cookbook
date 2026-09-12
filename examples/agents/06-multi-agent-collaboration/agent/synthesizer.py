"""Final-answer synthesizers.

Measured runs use a deterministic mock that formats actual agent results.
Live synthesis is optional and must call a real provider with those results.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from agent.schemas import AgentResult
from agent.state import TaskState
from config import Settings


class Synthesizer(Protocol):
    model_name: str
    driver: str

    def synthesize(self, state: TaskState) -> tuple[str, int]: ...


def _result_block(result: AgentResult) -> str:
    if not result.ok:
        error = result.error or {}
        return (
            f"- {result.agent_id} ({result.role}) FAILED: "
            f"{error.get('code', 'error')}: {error.get('message', '')}"
        )
    return (
        f"- {result.agent_id} ({result.role}) ok: "
        f"{json.dumps(result.payload, sort_keys=True)}"
    )


class MockSynthesizer:
    """Deterministic aggregation of the actual TaskState results."""

    def __init__(self) -> None:
        self.model_name = "mock"
        self.driver = "mock"

    def synthesize(self, state: TaskState) -> tuple[str, int]:
        lines = [
            f"Collaboration brief for: {state.request}",
            f"Service: {state.service or 'unspecified'}",
            "Agent results:",
        ]
        for result in state.results:
            lines.append(_result_block(result))
        if state.skipped:
            skipped = ", ".join(item.agent_id for item in state.skipped)
            lines.append(f"Skipped: {skipped}")
        failed = [item for item in state.results if not item.ok]
        if failed:
            lines.append(
                "Coordinator: specialist failure recorded; not treated as success."
            )
        return "\n".join(lines), 0


class LiveSynthesizer:
    """Optional provider synthesis from the actual specialist results."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("Live synthesis requires OPENAI_API_KEY")
        self.model_name = settings.chat_model
        self.driver = "openai-compatible"
        self._settings = settings

    def synthesize(self, state: TaskState) -> tuple[str, int]:
        import time

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self._settings.openai_api_key}
        if self._settings.openai_base_url:
            kwargs["base_url"] = self._settings.openai_base_url
        payload = {
            "request": state.request,
            "service": state.service,
            "results": [item.model_dump(mode="json") for item in state.results],
            "skipped": [item.model_dump(mode="json") for item in state.skipped],
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
                        "Write a short operational brief from the supplied "
                        "specialist results. Do not invent facts. Do not "
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
