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


def test_openai_tool_schema_shape():
    tools = build_registry(DATA).openai_tools()
    assert all(t["type"] == "function" for t in tools)


def test_get_service_status_deterministic():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "billing"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["service"]["status"] == "operational"


def test_payments_is_operational():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "payments"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["service"]["status"] == "operational"
    assert result["service"]["incident"] is None


def test_auth_outage_is_deterministic():
    registry = build_registry(DATA)
    args = registry.parse_arguments("get_service_status", {"service": "auth"})
    result = registry.get("get_service_status").handler(args)  # type: ignore[union-attr]
    assert result["service"]["status"] == "major_outage"
    assert "AUTH-881" in result["service"]["incident"]


def test_required_auth_runbook_is_absent():
    registry = build_registry(DATA)
    args = registry.parse_arguments(
        "search_documentation",
        {"query": "AUTH-881 incident runbook"},
    )
    result = registry.get("search_documentation").handler(args)  # type: ignore[union-attr]
    hit_ids = {hit["id"] for hit in result["hits"]}
    assert "doc-auth-881-runbook" not in hit_ids


def test_search_documentation_deterministic_ranking():
    registry = build_registry(DATA)
    args = registry.parse_arguments(
        "search_documentation",
        {"query": "payments recent deployment"},
    )
    result = registry.get("search_documentation").handler(args)  # type: ignore[union-attr]
    assert result["ok"] is True
    assert result["hitCount"] >= 1
    assert result["hits"][0]["id"] == "doc-payments-deploy"


def test_argument_validation_rejects_empty_service():
    with pytest.raises(ValidationError):
        GetServiceStatusArgs(service="  ")


def test_argument_validation_rejects_short_query():
    with pytest.raises(ValidationError):
        SearchDocumentationArgs(query="a")
