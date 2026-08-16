"""Tool executor — validate, authorize, time-bound, allowlist, observe.

Authorization and validation live here, outside the model. The model may
propose tool names and arguments; this module decides what actually runs.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.schemas import ToolCallRequest, ToolCallResult
from agent.tools import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_ms: int = 2000,
        allowed_tools: set[str] | None = None,
        caller_roles: set[str] | None = None,
    ) -> None:
        self.registry = registry
        self.timeout_ms = timeout_ms
        self.allowed_tools = (
            allowed_tools if allowed_tools is not None else registry.names()
        )
        # Demo authorization: tools that require an operator role.
        self._role_requirements: dict[str, set[str]] = {
            "get_user_profile": {"operator", "admin"},
        }
        self.caller_roles = caller_roles or {"operator"}

    def execute(self, call: ToolCallRequest) -> ToolCallResult:
        started = time.perf_counter()

        if call.name not in self.allowed_tools:
            return self._fail(
                call,
                started,
                code="tool_not_allowlisted",
                message=f"Tool '{call.name}' is not on the allowlist",
            )

        spec = self.registry.get(call.name)
        if spec is None:
            return self._fail(
                call,
                started,
                code="unknown_tool",
                message=f"Unknown tool '{call.name}'",
            )

        required_roles = self._role_requirements.get(call.name)
        if required_roles and self.caller_roles.isdisjoint(required_roles):
            return self._fail(
                call,
                started,
                code="unauthorized",
                message=(
                    f"Caller roles {sorted(self.caller_roles)} lack permission "
                    f"for '{call.name}' (requires one of {sorted(required_roles)})"
                ),
            )

        try:
            validated = self.registry.parse_arguments(call.name, call.arguments)
        except ValueError as exc:
            return self._fail(
                call,
                started,
                code="invalid_arguments",
                message=str(exc),
            )

        try:
            payload = spec.handler(validated)
        except Exception as exc:  # noqa: BLE001 — surface tool failures
            return self._fail(
                call,
                started,
                code="tool_exception",
                message=str(exc),
                validated=validated.model_dump(),
            )

        latency_ms = _ms(started)
        if latency_ms > self.timeout_ms:
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error={
                    "code": "timeout",
                    "message": f"Tool exceeded timeout of {self.timeout_ms}ms",
                    "latencyMs": latency_ms,
                },
                latency_ms=latency_ms,
                validated_arguments=validated.model_dump(),
            )

        if isinstance(payload, dict) and payload.get("ok") is False:
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=payload.get("error")
                or {"code": "tool_error", "message": "Tool returned ok=false"},
                result=payload,
                latency_ms=latency_ms,
                validated_arguments=validated.model_dump(),
            )

        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            result=payload,
            latency_ms=latency_ms,
            validated_arguments=validated.model_dump(),
        )

    def _fail(
        self,
        call: ToolCallRequest,
        started: float,
        *,
        code: str,
        message: str,
        validated: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            error={"code": code, "message": message},
            latency_ms=_ms(started),
            validated_arguments=validated,
        )


def parse_tool_arguments_json(raw: str) -> dict[str, Any]:
    """Parse model-produced JSON arguments into a dict.

    Invalid JSON becomes a sentinel so the executor can return
    invalid_arguments rather than crashing the loop.
    """
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"__raw__": raw, "__parse_error__": True}
    if not isinstance(value, dict):
        return {"__raw__": raw, "__parse_error__": True}
    if value.get("__parse_error__"):
        return value
    return value


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))
