"""Server-side fixture tests for MCP resources."""

from __future__ import annotations

from pathlib import Path

from server.fixtures import (
    EXPECTED_URIS,
    URI_KNOWLEDGE_PLATFORM,
    URI_SERVICE_STATUS,
    FixtureStore,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_fixture_store_lists_three_stable_uris():
    store = FixtureStore(DATA)
    resources = store.list_resources()
    assert [resource.uri for resource in resources] == list(EXPECTED_URIS)


def test_knowledge_platform_content_matches_fixture_file():
    store = FixtureStore(DATA)
    text = store.content(URI_KNOWLEDGE_PLATFORM)
    assert "# Knowledge Platform" in text
    assert "knowledge-platform" in text
    assert "billing-api" in text
    assert "identity-api" in text
    on_disk = (DATA / "knowledge-platform.md").read_text(encoding="utf-8")
    assert text == on_disk


def test_service_status_is_json_text():
    store = FixtureStore(DATA)
    resource = store.get(URI_SERVICE_STATUS)
    assert resource.mime_type == "application/json"
    assert '"environment": "acme-ai"' in resource.text
    assert "knowledge-platform" in resource.text


def test_unknown_uri_raises_key_error_locally():
    store = FixtureStore(DATA)
    try:
        store.get("acme://docs/does-not-exist")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "does-not-exist" in str(exc)
