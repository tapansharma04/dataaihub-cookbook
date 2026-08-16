"""Tool schema and deterministic result tests — no network."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.tools import (
    GetServiceStatusArgs,
    SearchDocumentationArgs,
    build_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_registry_exposes_three_tools():
    registry = build_registry(DATA)
    assert registry.names() == {
        "get_service_status",
        "get_user_profile",
        "search_documentation",
    }


def test_get_service_status_payments_is_degraded():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "payments"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["service"]["status"] == "degraded"
    assert "PAY-2041" in result["service"]["incident"]


def test_unknown_service_is_a_tool_level_error():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "payments-api"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_service"


def test_search_documentation_hits_payments_runbook():
    registry = build_registry(DATA)
    args = registry.parse_arguments(
        "search_documentation",
        {"query": "payments degradation"},
    )
    result = registry.get("search_documentation").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["hits"][0]["id"] == "doc-payments-runbook"


def test_argument_validation_rejects_empty_service():
    with pytest.raises(ValidationError):
        GetServiceStatusArgs(service="  ")


def test_argument_validation_rejects_short_query():
    with pytest.raises(ValidationError):
        SearchDocumentationArgs(query="a")
