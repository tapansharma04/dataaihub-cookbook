"""Server-side fixture and tool handler tests."""

from __future__ import annotations

from pathlib import Path

from server.fixtures import FixtureStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_get_service_status_known_service():
    store = FixtureStore(DATA)
    result = store.get_service_status("billing")
    assert result["ok"] is True
    assert result["service"]["status"] == "operational"


def test_get_service_status_unknown_service():
    store = FixtureStore(DATA)
    result = store.get_service_status("billing-api")
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_service"


def test_search_documentation_finds_payments_runbook():
    store = FixtureStore(DATA)
    result = store.search_documentation("payments degradation")
    assert result["ok"] is True
    assert result["hitCount"] >= 1
    assert any("payments" in hit["title"].lower() for hit in result["hits"])
