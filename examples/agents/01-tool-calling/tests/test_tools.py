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
    names = registry.names()
    assert names == {
        "get_service_status",
        "get_user_profile",
        "search_documentation",
    }
    defs = {d.name: d for d in registry.definitions()}
    assert "service" in defs["get_service_status"].parameters.properties
    assert "user_id" in defs["get_user_profile"].parameters.properties
    assert "query" in defs["search_documentation"].parameters.properties


def test_openai_tool_schema_shape():
    tools = build_registry(DATA).openai_tools()
    assert all(t["type"] == "function" for t in tools)
    assert {t["function"]["name"] for t in tools} == {
        "get_service_status",
        "get_user_profile",
        "search_documentation",
    }


def test_get_service_status_deterministic():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "billing"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["service"]["status"] == "operational"
    assert result["service"]["incident"] is None


def test_get_service_status_unknown_lists_valid():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "billing-api"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_service"
    assert "billing" in result["error"]["validServices"]


def test_get_user_profile_deterministic():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_user_profile", {"user_id": "u-1001"})
    result = registry.get("get_user_profile").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["profile"]["displayName"] == "Ada Lovelace"
    assert result["profile"]["plan"] == "enterprise"


def test_search_documentation_deterministic_ranking():
    registry = build_registry(DATA)
    args = registry.parse_arguments(
        "search_documentation",
        {"query": "payments degradation"},
    )
    result = registry.get("search_documentation").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["hitCount"] >= 1
    assert result["hits"][0]["id"] == "doc-payments-runbook"


def test_argument_validation_rejects_empty_service():
    with pytest.raises(ValidationError):
        GetServiceStatusArgs(service="  ")


def test_argument_validation_rejects_short_query():
    with pytest.raises(ValidationError):
        SearchDocumentationArgs(query="a")


def test_parse_arguments_surfaces_validation_errors():
    registry = build_registry(DATA)
    with pytest.raises(ValueError):
        registry.parse_arguments("get_user_profile", {"user_id": ""})
