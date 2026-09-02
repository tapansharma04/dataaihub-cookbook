"""Server catalog and no-direct-LLM tests."""

from __future__ import annotations

from pathlib import Path

from server.fixtures import (
    EXPECTED_PROMPT_NAMES,
    EXPECTED_TOOL_NAMES,
    EXPECTED_URIS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_sizes():
    assert len(EXPECTED_URIS) == 3
    assert len(EXPECTED_PROMPT_NAMES) == 2
    assert len(EXPECTED_TOOL_NAMES) == 4


def test_server_does_not_import_openai():
    for path in (ROOT / "server").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "openai" not in source
        assert "OpenAI" not in source


def test_server_requests_sampling_through_mcp():
    source = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
    assert "ctx.session.create_message(" in source
    assert "from openai" not in source


def test_composition_tools_do_not_reload_fixtures():
    source = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
    compose_start = source.index("async def compose_resource_brief")
    compose_src = source[compose_start:]
    assert "store.get(" not in compose_src
    assert "store.get_service_status(" not in compose_src
    assert "store.content(" not in compose_src
    assert "render_prompt_messages(" not in compose_src
    assert "store.doc_uri_for(" not in compose_src
