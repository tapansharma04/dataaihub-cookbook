"""Executor tests used by evaluation cases — no network."""

from __future__ import annotations

from pathlib import Path

from agent.executor import ToolExecutor
from agent.schemas import ToolCallRequest
from agent.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_payments_status_executes():
    executor = ToolExecutor(build_registry(DATA))
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


def test_invalid_service_name_is_observable_failure():
    executor = ToolExecutor(build_registry(DATA))
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_service_status",
            arguments={"service": "payments-api"},
        )
    )
    assert result.ok is False
    assert result.error["code"] == "unknown_service"


def test_get_user_profile_executes_for_efficiency_case():
    executor = ToolExecutor(build_registry(DATA))
    result = executor.execute(
        ToolCallRequest(
            id="c1",
            name="get_user_profile",
            arguments={"user_id": "u-1001"},
        )
    )
    assert result.ok is True
    assert result.result["profile"]["userId"] == "u-1001"
