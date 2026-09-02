"""Regression: composition consumes prior MCP results, not fixture reloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mcp_types as types
import pytest
from mcp import Client

from client.sampling import mock_complete
from config import Settings
from server.app import build_server
from server.fixtures import (
    TOOL_COMPOSE_FROM_PROMPT,
    TOOL_COMPOSE_INCIDENT_BRIEF,
    TOOL_COMPOSE_RESOURCE_BRIEF,
    URI_KNOWLEDGE_PLATFORM,
)

DATA = Path(__file__).resolve().parents[1] / "data"

SENTINEL_RESOURCE = "SENTINEL-RESOURCE-CONTENT-NOT-IN-FIXTURE"
SENTINEL_PROMPT = "SENTINEL-PROMPT-MESSAGE-NOT-IN-TEMPLATE"
SENTINEL_TOOL = {
    "ok": True,
    "service": {
        "service": "billing-api",
        "status": "sentinel-status",
        "incident": "SENTINEL-TOOL-RESULT-NOT-IN-FIXTURE",
    },
}
FIXTURE_RESOURCE_PHRASE = "indexes internal documents"
FIXTURE_PROMPT_PHRASE = "knowledge-platform service for a engineering audience"
FIXTURE_INCIDENT = "BILL-2048"


@pytest.fixture
def settings() -> Settings:
    return Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")


@pytest.fixture
def server():
    return build_server(DATA)


def _sampling_text(params: types.CreateMessageRequestParams) -> str:
    parts: list[str] = []
    for message in params.messages:
        for block in message.content_as_list:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


async def _capture_sampling(
    server,
    settings: Settings,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    captured: list[types.CreateMessageRequestParams] = []

    async def sampling_callback(_context, params):
        captured.append(params)
        return mock_complete(params)

    client_info = types.Implementation(
        name=settings.client_name,
        version=settings.client_version,
    )
    async with Client(
        server,
        mode=settings.mcp_client_mode,
        client_info=client_info,
        sampling_callback=sampling_callback,
    ) as client:
        result = await client.call_tool(tool_name, arguments)
        assert result.is_error is False
    assert captured, "server did not request sampling"
    return _sampling_text(captured[0])


@pytest.mark.asyncio
async def test_resource_composition_uses_argument_not_fixture(settings, server):
    text = await _capture_sampling(
        server,
        settings,
        TOOL_COMPOSE_RESOURCE_BRIEF,
        {
            "uri": URI_KNOWLEDGE_PLATFORM,
            "content": SENTINEL_RESOURCE,
            "mime_type": "text/plain",
            "name": "sentinel",
        },
    )
    assert SENTINEL_RESOURCE in text
    assert FIXTURE_RESOURCE_PHRASE not in text


@pytest.mark.asyncio
async def test_prompt_composition_uses_argument_not_rerender(settings, server):
    text = await _capture_sampling(
        server,
        settings,
        TOOL_COMPOSE_FROM_PROMPT,
        {
            "prompt_name": "summarize-service",
            "service_name": "knowledge-platform",
            "audience": "engineering",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": SENTINEL_PROMPT},
                }
            ],
        },
    )
    assert SENTINEL_PROMPT in text
    assert FIXTURE_PROMPT_PHRASE not in text


@pytest.mark.asyncio
async def test_incident_composition_uses_arguments_not_fixture(settings, server):
    text = await _capture_sampling(
        server,
        settings,
        TOOL_COMPOSE_INCIDENT_BRIEF,
        {
            "service": "billing-api",
            "tool_result": SENTINEL_TOOL,
            "resource_uri": "acme://docs/billing-portal",
            "resource_content": SENTINEL_RESOURCE,
            "resource_mime_type": "text/plain",
            "prompt_name": "draft-status-update",
            "prompt_messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": SENTINEL_PROMPT},
                }
            ],
        },
    )
    assert SENTINEL_TOOL["service"]["incident"] in text
    assert SENTINEL_RESOURCE in text
    assert SENTINEL_PROMPT in text
    assert FIXTURE_INCIDENT not in text
    assert FIXTURE_RESOURCE_PHRASE not in text
    assert FIXTURE_PROMPT_PHRASE not in text


def test_sampling_messages_require_prompt_dicts():
    from server.app import _sampling_messages_from_prompt_dump

    with pytest.raises(TypeError, match="prompts/get message dict"):
        _sampling_messages_from_prompt_dump(["not-a-prompt-message"])


def test_bind_resource_brief_matches_uri_not_last_read():
    from client.runner import _bind_composition_arguments

    bound = _bind_composition_arguments(
        "compose_resource_brief",
        {"uri": "acme://docs/knowledge-platform"},
        output={
            "reads": [
                {
                    "requestedUri": "acme://docs/knowledge-platform",
                    "isError": False,
                    "contents": [
                        {"text": "FIRST-BLOCK", "mimeType": "text/markdown"},
                        {"text": "SECOND-BLOCK", "mimeType": "text/markdown"},
                    ],
                },
                {
                    "requestedUri": "acme://docs/billing-portal",
                    "isError": False,
                    "contents": [{"text": "LAST-READ", "mimeType": "text/markdown"}],
                },
            ]
        },
        discovered_resources=[],
    )
    assert bound["content"] == "FIRST-BLOCK\nSECOND-BLOCK"
    assert "LAST-READ" not in bound["content"]


def test_bind_prompt_matches_name_not_last_get():
    from client.runner import _bind_composition_arguments

    bound = _bind_composition_arguments(
        "compose_from_prompt",
        {"prompt_name": "summarize-service", "service_name": "knowledge-platform"},
        output={
            "gets": [
                {
                    "requestedPrompt": "summarize-service",
                    "arguments": {"service_name": "knowledge-platform"},
                    "isError": False,
                    "messages": [{"role": "user", "content": {"text": "MATCHED"}}],
                },
                {
                    "requestedPrompt": "draft-status-update",
                    "arguments": {"service": "billing-api", "status": "degraded"},
                    "isError": False,
                    "messages": [{"role": "user", "content": {"text": "LAST-GET"}}],
                },
            ]
        },
        discovered_resources=[],
    )
    assert bound["messages"][0]["content"]["text"] == "MATCHED"


def test_bind_incident_uses_prompts_get_arguments():
    from client.runner import _bind_composition_arguments

    get_arguments = {"service": "billing-api", "status": "degraded"}
    bound = _bind_composition_arguments(
        "compose_incident_brief",
        {"service": "billing-api"},
        output={
            "invocations": [
                {
                    "tool": "get_service_status",
                    "arguments": {"service": "identity-api"},
                    "isError": False,
                    "result": {"ok": True, "service": {"service": "identity-api"}},
                },
                {
                    "tool": "get_service_status",
                    "arguments": {"service": "billing-api"},
                    "isError": False,
                    "result": {"ok": True, "service": {"service": "billing-api"}},
                },
            ],
            "reads": [
                {
                    "requestedUri": "acme://docs/billing-portal",
                    "isError": False,
                    "contents": [{"text": "BILLING-DOC", "mimeType": "text/markdown"}],
                }
            ],
            "gets": [
                {
                    "requestedPrompt": "draft-status-update",
                    "arguments": get_arguments,
                    "isError": False,
                    "messages": [{"role": "user", "content": {"text": "PROMPT"}}],
                }
            ],
        },
        discovered_resources=[],
    )
    assert bound["tool_result"]["service"]["service"] == "billing-api"
    assert bound["prompt_arguments"] == get_arguments
    assert bound["resource_content"] == "BILLING-DOC"
