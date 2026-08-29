"""MCP protocol runner tests — no network / no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from client.cases import get_case
from client.runner import run_case_async
from client.trace import build_signature_view
from config import Settings
from server.app import build_server
from server.fixtures import EXPECTED_PROMPT_NAMES, PROMPT_SUMMARIZE_SERVICE

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return Settings(mcp_client_mode="legacy")


@pytest.fixture
def server():
    return build_server()


def test_runner_uses_official_mcp_client_apis():
    source = (ROOT / "client" / "runner.py").read_text(encoding="utf-8")
    assert "from mcp import Client" in source
    assert "await client.list_prompts()" in source
    assert "await client.get_prompt(" in source
    assert "mode=settings.mcp_client_mode" in source
    assert "list_tools" not in source
    assert "call_tool" not in source
    assert "list_resources" not in source
    assert "read_resource" not in source


def test_client_does_not_ship_hard_coded_prompt_catalog():
    for relative in ("client/runner.py", "client/schemas.py", "client/trace.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "EXPECTED_PROMPT_NAMES" not in source
        assert "CATALOG" not in source


@pytest.mark.asyncio
async def test_mcp_initialization_handshake(settings, server):
    case = get_case("prompt-discovery")
    result = await run_case_async(case, settings=settings, server=server)
    kinds = [event.kind for event in result.sequence]
    assert kinds[0] == "initialize_request"
    assert kinds[1] == "initialize_response"
    assert result.protocol_version is not None
    assert result.server_name == "dataaihub-cookbook-prompts"
    init = next(e for e in result.sequence if e.kind == "initialize_response")
    capabilities = init.detail.get("capabilities") or {}
    assert "prompts" in capabilities


@pytest.mark.asyncio
async def test_discovery_lists_prompts_via_mcp(settings, server):
    case = get_case("prompt-discovery")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.example_class == "PROMPT_DISCOVERY"
    assert result.metrics.prompts_discovered == 3
    assert result.metrics.prompts_requested == 0
    assert result.metrics.successful_gets == 0
    assert result.metrics.model_turns == 0
    assert result.metrics.tool_calls == 0
    assert result.metrics.resources_read == 0
    names = [prompt.name for prompt in result.discovered_prompts]
    assert names == list(EXPECTED_PROMPT_NAMES)
    for prompt in result.discovered_prompts:
        assert prompt.description
    kinds = [event.kind for event in result.sequence]
    assert "prompts_list_request" in kinds
    assert "prompts_list_response" in kinds
    assert "prompt_get_request" not in kinds
    assert kinds[-1] == "termination"


@pytest.mark.asyncio
async def test_discovery_returns_server_owned_prompt_metadata(settings, server):
    case = get_case("prompt-discovery")
    result = await run_case_async(case, settings=settings, server=server)
    summarize = next(
        p for p in result.discovered_prompts if p.name == PROMPT_SUMMARIZE_SERVICE
    )
    assert summarize.description
    arg_names = [arg.name for arg in summarize.arguments]
    assert "service_name" in arg_names
    assert "audience" in arg_names
    service_name_arg = next(a for a in summarize.arguments if a.name == "service_name")
    audience_arg = next(a for a in summarize.arguments if a.name == "audience")
    assert service_name_arg.required is True
    assert audience_arg.required is False


@pytest.mark.asyncio
async def test_single_prompt_get_returns_server_messages(settings, server):
    case = get_case("single-prompt-get-summarize")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.successful_gets == 1
    assert result.metrics.failed_gets == 0
    get = result.output["gets"][0]
    assert get["requestedPrompt"] == PROMPT_SUMMARIZE_SERVICE
    assert get["arguments"] == {"service_name": "knowledge-platform"}
    assert get["isError"] is False
    messages = get["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]["text"]
    assert "knowledge-platform" in text
    assert "engineering" in text


@pytest.mark.asyncio
async def test_prompt_with_arguments_passes_values_through_mcp(settings, server):
    case = get_case("prompt-with-arguments-investigate")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.successful_gets == 1
    get = result.output["gets"][0]
    assert get["arguments"] == {"service": "billing-api", "incident": "INC-2048"}
    request = next(e for e in result.sequence if e.kind == "prompt_get_request")
    assert request.detail["arguments"] == get["arguments"]


@pytest.mark.asyncio
async def test_multi_message_prompt_preserves_order(settings, server):
    case = get_case("prompt-with-arguments-investigate")
    result = await run_case_async(case, settings=settings, server=server)
    messages = result.output["gets"][0]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert "INC-2048" in messages[0]["content"]["text"]
    assert "billing-api" in messages[0]["content"]["text"]
    assert messages[1]["role"] == "assistant"
    assert "billing-api" in messages[1]["content"]["text"]
    assert messages[2]["role"] == "user"
    assert "INC-2048" in messages[2]["content"]["text"]


@pytest.mark.asyncio
async def test_invalid_prompt_is_protocol_visible(settings, server):
    case = get_case("invalid-prompt-name")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.prompts_discovered == 3
    assert result.metrics.prompts_requested == 1
    assert result.metrics.successful_gets == 0
    assert result.metrics.failed_gets == 1
    assert result.metrics.message_count == 0
    assert result.metrics.message_bytes == 0
    kinds = [event.kind for event in result.sequence]
    assert "prompts_list_response" in kinds
    assert "prompt_get_request" in kinds
    assert "prompt_get_response" in kinds
    assert "error" not in kinds
    response = next(e for e in result.sequence if e.kind == "prompt_get_response")
    assert response.detail["isError"] is True
    assert response.detail["name"] == "does-not-exist"
    assert response.detail.get("error")
    assert "messages" not in response.detail
    get = result.output["gets"][0]
    assert get["isError"] is True
    assert "messages" not in get
    assert result.errors[0]["stage"] == "prompts/get"


@pytest.mark.asyncio
async def test_invalid_prompt_does_not_fabricate_messages(settings, server):
    case = get_case("invalid-prompt-name")
    result = await run_case_async(case, settings=settings, server=server)
    for event in result.sequence:
        if event.kind == "prompt_get_response":
            assert event.detail.get("isError") is True
            assert event.detail.get("messages") in (None, [])
    assert result.output["gets"][0].get("messages") in (None, [])


@pytest.mark.asyncio
async def test_invalid_prompt_does_not_mutate_catalog(settings, server):
    discovery = await run_case_async(
        get_case("prompt-discovery"), settings=settings, server=server
    )
    before = [prompt.name for prompt in discovery.discovered_prompts]
    invalid = await run_case_async(
        get_case("invalid-prompt-name"), settings=settings, server=server
    )
    after = [prompt.name for prompt in invalid.discovered_prompts]
    assert before == after == list(EXPECTED_PROMPT_NAMES)


@pytest.mark.asyncio
async def test_discovered_prompt_marked_before_valid_get(settings, server):
    case = get_case("single-prompt-get-summarize")
    result = await run_case_async(case, settings=settings, server=server)
    request = next(e for e in result.sequence if e.kind == "prompt_get_request")
    assert request.detail["discoveredBeforeGet"] is True


@pytest.mark.asyncio
async def test_invalid_prompt_was_not_discovered(settings, server):
    case = get_case("invalid-prompt-name")
    result = await run_case_async(case, settings=settings, server=server)
    request = next(e for e in result.sequence if e.kind == "prompt_get_request")
    assert request.detail["discoveredBeforeGet"] is False


@pytest.mark.asyncio
async def test_no_tools_or_resources_used(settings, server):
    for trace_id in (
        "prompt-discovery",
        "single-prompt-get-summarize",
        "prompt-with-arguments-investigate",
        "invalid-prompt-name",
    ):
        result = await run_case_async(
            get_case(trace_id), settings=settings, server=server
        )
        kinds = [event.kind for event in result.sequence]
        assert "tools_list_request" not in kinds
        assert "resource_read_request" not in kinds
        assert result.metrics.tool_calls == 0
        assert result.metrics.resources_read == 0


def test_signature_view_invalid_prompt_flow(settings):
    from client.runner import run_case

    case = get_case("invalid-prompt-name")
    result = run_case(case, settings=settings, server=build_server())
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "INITIALIZE" in phases
    assert "DISCOVER" in phases
    assert "GET" in phases
    assert "REJECTED" in phases
    assert "MESSAGES" not in phases
    assert "ERROR" not in phases


def test_discovery_signature_has_no_get_phase(settings):
    from client.runner import run_case

    case = get_case("prompt-discovery")
    result = run_case(case, settings=settings, server=build_server())
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "GET" not in phases
    assert "MESSAGES" not in phases
    assert "PROMPTS" in phases
