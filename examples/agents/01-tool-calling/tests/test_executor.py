"""Executor tests — allowlist, auth, validation, errors. No network."""

from __future__ import annotations

from pathlib import Path

from agent.executor import ToolExecutor, parse_tool_arguments_json
from agent.schemas import ToolCallRequest
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_successful_execution_records_latency():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_service_status",
            arguments={"service": "payments"},
        )
    )
    assert result.ok is True
    assert result.result["service"]["status"] == "degraded"
    assert result.latency_ms >= 0
    assert result.validated_arguments == {"service": "payments"}


def test_invalid_arguments():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    result = executor.execute(
        ToolCallRequest(id="c1", name="get_service_status", arguments={})
    )
    assert result.ok is False
    assert result.error["code"] == "invalid_arguments"


def test_unknown_tool():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    result = executor.execute(
        ToolCallRequest(id="c1", name="drop_database", arguments={})
    )
    assert result.ok is False
    assert result.error["code"] in {"tool_not_allowlisted", "unknown_tool"}


def test_allowlist_blocks_registered_tool():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry, allowed_tools={"get_service_status"})
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_user_profile",
            arguments={"user_id": "u-1001"},
        )
    )
    assert result.ok is False
    assert result.error["code"] == "tool_not_allowlisted"


def test_authorization_outside_model():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry, caller_roles={"viewer"})
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_user_profile",
            arguments={"user_id": "u-1001"},
        )
    )
    assert result.ok is False
    assert result.error["code"] == "unauthorized"


def test_tool_soft_error_unknown_entity():
    registry = build_registry(DATA)
    executor = ToolExecutor(registry)
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_service_status",
            arguments={"service": "billing-api"},
        )
    )
    assert result.ok is False
    assert result.error["code"] == "unknown_service"


def test_parse_tool_arguments_json_invalid():
    parsed = parse_tool_arguments_json("{not-json")
    assert parsed["__parse_error__"] is True
