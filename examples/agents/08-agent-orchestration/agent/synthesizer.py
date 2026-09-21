"""Optional final-answer formatting from actual workflow results.

Default runs use a deterministic mock that formats terminal node results.
This is not coordinator aggregation of independent delegations.

Live formatting is optional and must call a real provider with those
actual results. It is never the source of committed traces.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from agent.schemas import AgentResult, NodeState, TerminationReason
from agent.state import NodeRecord, OrchestrationState
from config import Settings


class Synthesizer(Protocol):
    model_name: str
    driver: str

    def synthesize(self, state: OrchestrationState) -> tuple[str, int]: ...


def _terminal_completed(state: OrchestrationState) -> NodeRecord | None:
    if state.plan is None:
        return None
    depended_on: set[str] = set()
    for node in state.plan.nodes:
        record = state.nodes[node.node_id]
        if record.state in {"COMPLETED", "READY", "RUNNING"}:
            depended_on.update(node.dependencies)
    terminals = [
        state.nodes[node.node_id]
        for node in state.plan.nodes
        if state.nodes[node.node_id].state == "COMPLETED"
        and node.node_id not in depended_on
    ]
    return terminals[-1] if terminals else None


def _failed_record(state: OrchestrationState) -> NodeRecord | None:
    for node in state.plan.nodes if state.plan is not None else []:
        record = state.nodes[node.node_id]
        if record.state == "FAILED":
            return record
    return None


class MockSynthesizer:
    """Deterministic answer from the actual orchestration state."""

    def __init__(self) -> None:
        self.model_name = "mock"
        self.driver = "mock"

    def synthesize(self, state: OrchestrationState) -> tuple[str, int]:
        reason: TerminationReason | None = state.termination_reason
        if reason == "completed":
            terminal = _terminal_completed(state)
            if terminal is not None and terminal.result is not None:
                summary = terminal.result.payload.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary, 0
                return f"Workflow completed at node {terminal.node_id}.", 0
            return "Workflow completed.", 0
        if reason == "workflow_failed":
            failed = _failed_record(state)
            skipped = [
                node_id
                for node_id, record in state.nodes.items()
                if record.state == "SKIPPED"
            ]
            skip_note = f" Skipped: {', '.join(skipped)}." if skipped else ""
            if failed is not None and failed.result is not None:
                error = failed.result.error or {}
                return (
                    f"Node {failed.node_id} failed: "
                    f"{error.get('code', 'error')}: {error.get('message', '')}."
                    f"{skip_note} Not treated as success."
                ), 0
            return f"Workflow failed.{skip_note} Not treated as success.", 0
        if reason == "invalid_task":
            return "Runtime rejected an empty request.", 0
        if reason == "invalid_plan":
            if state.errors:
                message = state.errors[-1].get("message", "invalid plan")
                return f"Invalid orchestration plan: {message}.", 0
            return "Invalid orchestration plan.", 0
        return f"Terminated ({reason or 'error'}).", 0


class LiveSynthesizer:
    """Optional provider formatting of the actual workflow outcome."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("Live synthesis requires OPENAI_API_KEY")
        self.model_name = settings.chat_model
        self.driver = "openai-compatible"
        self._settings = settings

    def synthesize(self, state: OrchestrationState) -> tuple[str, int]:
        import time

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self._settings.openai_api_key}
        if self._settings.openai_base_url:
            kwargs["base_url"] = self._settings.openai_base_url
        last: AgentResult | None = state.results[-1] if state.results else None
        node_states: dict[str, NodeState] = {
            node_id: record.state for node_id, record in state.nodes.items()
        }
        payload = {
            "request": state.request,
            "service": state.service,
            "terminationReason": state.termination_reason,
            "nodeStates": node_states,
            "results": [item.model_dump(mode="json") for item in state.results],
            "skippedNodeIds": [
                node_id
                for node_id, record in state.nodes.items()
                if record.state == "SKIPPED"
            ],
            "lastResult": last.model_dump(mode="json") if last is not None else None,
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
                        "orchestration results. Do not invent facts. Do not "
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
