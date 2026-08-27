"""MCP protocol runner tests — no network / no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from client.cases import get_case
from client.runner import run_case_async
from client.trace import build_signature_view
from config import Settings
from server.app import build_server
from server.fixtures import EXPECTED_URIS, URI_KNOWLEDGE_PLATFORM, FixtureStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def settings() -> Settings:
    return Settings(data_dir=DATA, mcp_client_mode="legacy")


@pytest.fixture
def server():
    return build_server(DATA)


def test_runner_uses_official_mcp_client_apis():
    source = (ROOT / "client" / "runner.py").read_text(encoding="utf-8")
    assert "from mcp import Client" in source
    assert "await client.list_resources()" in source
    assert "await client.read_resource(" in source
    assert "mode=settings.mcp_client_mode" in source
    assert "list_tools" not in source
    assert "call_tool" not in source


def test_client_does_not_ship_hard_coded_resource_catalog():
    for relative in ("client/runner.py", "client/schemas.py", "client/trace.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "RESOURCE_CATALOG" not in source
        assert "EXPECTED_URIS" not in source


@pytest.mark.asyncio
async def test_mcp_initialization_handshake(settings, server):
    case = get_case("discovery")
    result = await run_case_async(case, settings=settings, server=server)
    kinds = [event.kind for event in result.sequence]
    assert kinds[0] == "initialize_request"
    assert kinds[1] == "initialize_response"
    assert result.protocol_version is not None
    assert result.server_name == "dataaihub-cookbook-resources"
    init = next(e for e in result.sequence if e.kind == "initialize_response")
    capabilities = init.detail.get("capabilities") or {}
    assert "resources" in capabilities


@pytest.mark.asyncio
async def test_discovery_lists_resources_via_mcp(settings, server):
    case = get_case("discovery")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.example_class == "DISCOVERY"
    assert result.metrics.resources_discovered == 3
    assert result.metrics.resources_read == 0
    assert result.metrics.successful_reads == 0
    assert result.metrics.model_turns == 0
    assert result.metrics.tool_calls == 0
    uris = [resource.uri for resource in result.discovered_resources]
    assert uris == list(EXPECTED_URIS)
    for resource in result.discovered_resources:
        assert resource.name
        assert resource.description
        assert resource.mime_type
    kinds = [event.kind for event in result.sequence]
    assert "resources_list_request" in kinds
    assert "resources_list_response" in kinds
    assert "resource_read_request" not in kinds
    assert kinds[-1] == "termination"


@pytest.mark.asyncio
async def test_single_resource_read_returns_fixture_content(settings, server):
    case = get_case("single-resource-read-knowledge-platform")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.successful_reads == 1
    assert result.metrics.failed_reads == 0
    read = result.output["reads"][0]
    assert read["requestedUri"] == URI_KNOWLEDGE_PLATFORM
    assert read["returnedUri"] == URI_KNOWLEDGE_PLATFORM
    assert read["mimeType"] == "text/markdown"
    assert read["isError"] is False
    text = read["contents"][0]["text"]
    expected = FixtureStore(DATA).content(URI_KNOWLEDGE_PLATFORM)
    assert text == expected
    assert result.metrics.resource_bytes == len(expected.encode("utf-8"))


@pytest.mark.asyncio
async def test_multi_resource_read_preserves_order(settings, server):
    case = get_case("multi-resource-read-services")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.resources_read == 2
    assert result.metrics.successful_reads == 2
    reads = result.output["reads"]
    assert [item["requestedUri"] for item in reads] == list(case.resource_uris)
    request_uris = [
        event.detail["uri"]
        for event in result.sequence
        if event.kind == "resource_read_request"
    ]
    response_uris = [
        event.detail["uri"]
        for event in result.sequence
        if event.kind == "resource_read_response"
    ]
    assert request_uris == list(case.resource_uris)
    assert response_uris == list(case.resource_uris)


@pytest.mark.asyncio
async def test_invalid_resource_is_protocol_visible(settings, server):
    case = get_case("invalid-resource-uri")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.resources_discovered == 3
    assert result.metrics.resources_read == 1
    assert result.metrics.successful_reads == 0
    assert result.metrics.failed_reads == 1
    assert result.metrics.resource_bytes == 0
    kinds = [event.kind for event in result.sequence]
    assert "resources_list_response" in kinds
    assert "resource_read_request" in kinds
    assert "resource_read_response" in kinds
    assert "error" not in kinds
    response = next(e for e in result.sequence if e.kind == "resource_read_response")
    assert response.detail["isError"] is True
    assert response.detail["uri"] == "acme://docs/does-not-exist"
    assert response.detail.get("error")
    assert "contents" not in response.detail
    read = result.output["reads"][0]
    assert read["isError"] is True
    assert "contents" not in read
    assert result.errors[0]["stage"] == "resources/read"


@pytest.mark.asyncio
async def test_invalid_resource_does_not_fabricate_content(settings, server):
    case = get_case("invalid-resource-uri")
    result = await run_case_async(case, settings=settings, server=server)
    for event in result.sequence:
        if event.kind == "resource_read_response":
            assert event.detail.get("isError") is True
            assert event.detail.get("contents") in (None, [])
    assert result.output["reads"][0].get("contents") in (None, [])


@pytest.mark.asyncio
async def test_discovered_uri_marked_before_valid_read(settings, server):
    case = get_case("single-resource-read-knowledge-platform")
    result = await run_case_async(case, settings=settings, server=server)
    request = next(e for e in result.sequence if e.kind == "resource_read_request")
    assert request.detail["discoveredBeforeRead"] is True


@pytest.mark.asyncio
async def test_invalid_uri_was_not_discovered(settings, server):
    case = get_case("invalid-resource-uri")
    result = await run_case_async(case, settings=settings, server=server)
    request = next(e for e in result.sequence if e.kind == "resource_read_request")
    assert request.detail["discoveredBeforeRead"] is False


def test_signature_view_invalid_resource_flow(settings):
    from client.runner import run_case

    case = get_case("invalid-resource-uri")
    result = run_case(case, settings=settings, server=build_server(DATA))
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "INITIALIZE" in phases
    assert "DISCOVER" in phases
    assert "READ" in phases
    assert "REJECTED" in phases
    assert "CONTENT" not in phases
    assert "ERROR" not in phases
