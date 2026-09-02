"""MCP server exposing composed Acme AI tools, resources, and prompts.

The server does not call a model provider. Composition tools request
generation through MCP Sampling (`sampling/createMessage`) so the client
owns the model interaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import SamplingMessage, TextContent
from pydantic import Field

from server.fixtures import (
    PROMPT_DRAFT_STATUS_UPDATE,
    PROMPT_SUMMARIZE_SERVICE,
    TOOL_COMPOSE_FROM_PROMPT,
    TOOL_COMPOSE_INCIDENT_BRIEF,
    TOOL_COMPOSE_RESOURCE_BRIEF,
    TOOL_GET_SERVICE_STATUS,
    URI_BILLING_PORTAL,
    URI_KNOWLEDGE_PLATFORM,
    URI_SERVICE_STATUS,
    FixtureStore,
)
from server.prompts import render_draft_status_update, render_summarize_service

SERVER_NAME = "dataaihub-cookbook-composition"
SAMPLING_MAX_TOKENS = 200
RESOURCE_SYSTEM_PROMPT = (
    "Use only the supplied MCP resource content. Do not invent facts."
)
PROMPT_SYSTEM_PROMPT = (
    "Follow the supplied MCP prompt messages. Do not invent a different task."
)
COMPOSITION_SYSTEM_PROMPT = (
    "Use only the tool result, resource content, and prompt messages supplied. "
    "Do not invent facts."
)


def _sampling_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if content is not None and getattr(content, "type", None) == "text":
        return str(content.text)
    return ""


def _prompt_message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise TypeError(
        f"Expected a prompts/get message dict, got {type(message).__name__}"
    )


def _sampling_messages_from_prompt_dump(
    messages: list[Any],
) -> list[SamplingMessage]:
    """Build sampling messages from prompts/get payloads, not by re-rendering."""
    sampling: list[SamplingMessage] = []
    for message in messages:
        payload = _prompt_message_dict(message)
        role: Literal["user", "assistant"] = (
            "assistant" if payload.get("role") == "assistant" else "user"
        )
        content = payload.get("content") or {}
        if isinstance(content, dict):
            text = str(content.get("text") or "")
        else:
            text = str(content)
        sampling.append(
            SamplingMessage(
                role=role,
                content=TextContent(type="text", text=text),
            )
        )
    return sampling


def _sampling_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": getattr(result, "model", None),
        "role": getattr(result, "role", None),
        "text": _sampling_text(result),
    }
    stop_reason = getattr(result, "stop_reason", None)
    if stop_reason is not None:
        payload["stopReason"] = stop_reason
    return payload


def build_server(data_dir: Path) -> MCPServer:
    """Create an MCP server wired to Acme AI fixtures and composition tools."""
    store = FixtureStore(data_dir)
    mcp = MCPServer(SERVER_NAME)

    @mcp.resource(
        URI_KNOWLEDGE_PLATFORM,
        name="knowledge-platform",
        description="Service documentation for the Knowledge Platform.",
        mime_type="text/markdown",
    )
    def knowledge_platform() -> str:
        return store.content(URI_KNOWLEDGE_PLATFORM)

    @mcp.resource(
        URI_BILLING_PORTAL,
        name="billing-portal",
        description="Service documentation for the Billing Portal.",
        mime_type="text/markdown",
    )
    def billing_portal() -> str:
        return store.content(URI_BILLING_PORTAL)

    @mcp.resource(
        URI_SERVICE_STATUS,
        name="service-status",
        description="Service statuses for the Acme AI environment.",
        mime_type="application/json",
    )
    def service_status() -> str:
        return store.content(URI_SERVICE_STATUS)

    @mcp.prompt(
        PROMPT_SUMMARIZE_SERVICE,
        description="Produce a concise service summary tailored to a target audience.",
    )
    def summarize_service(
        service_name: Annotated[
            str,
            Field(description="Canonical Acme service name, e.g. knowledge-platform"),
        ],
        audience: Annotated[
            str,
            Field(description="Target audience for the summary"),
        ] = "engineering",
    ) -> str:
        return render_summarize_service(service_name, audience)

    @mcp.prompt(
        PROMPT_DRAFT_STATUS_UPDATE,
        description="Draft a stakeholder status update for a service.",
    )
    def draft_status_update(
        service: Annotated[
            str,
            Field(description="Service receiving the status update"),
        ],
        status: Annotated[
            str,
            Field(description="Current status label: operational, degraded, or outage"),
        ],
    ) -> str:
        return render_draft_status_update(service, status)

    @mcp.tool(name=TOOL_GET_SERVICE_STATUS)
    def get_service_status(
        service: Annotated[
            str,
            Field(
                description=(
                    "Canonical service name: knowledge-platform, billing-api, "
                    "identity-api"
                )
            ),
        ],
    ) -> dict:
        """Return the current status for a canonical Acme AI service name."""
        return store.get_service_status(service.strip())

    @mcp.tool(name=TOOL_COMPOSE_RESOURCE_BRIEF)
    async def compose_resource_brief(
        uri: Annotated[
            str,
            Field(description="Resource URI whose resources/read content is supplied"),
        ],
        content: Annotated[
            str,
            Field(description="Text returned by resources/read for this URI"),
        ],
        ctx: Context,
        mime_type: Annotated[
            str,
            Field(description="MIME type returned by resources/read"),
        ] = "text/markdown",
        name: Annotated[
            str,
            Field(description="Resource name from resources/list, if known"),
        ] = "",
    ) -> dict:
        """Ground sampling in the supplied resources/read content."""
        user_text = (
            f"Resource URI: {uri}\n"
            f"Name: {name or uri}\n"
            f"MIME type: {mime_type}\n"
            f"Content:\n{content}\n\n"
            "Write a two-sentence operational brief using only this resource."
        )
        sampled = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=user_text),
                )
            ],
            max_tokens=SAMPLING_MAX_TOKENS,
            system_prompt=RESOURCE_SYSTEM_PROMPT,
        )
        return {
            "ok": True,
            "workflow": "resource_to_sampling",
            "resourceUri": uri,
            "resourceName": name or uri,
            "mimeType": mime_type,
            "sampling": _sampling_payload(sampled),
        }

    @mcp.tool(name=TOOL_COMPOSE_FROM_PROMPT)
    async def compose_from_prompt(
        prompt_name: Annotated[
            str,
            Field(description="MCP prompt name retrieved via prompts/get"),
        ],
        messages: Annotated[
            list[dict[str, Any]],
            Field(description="PromptMessage objects returned by prompts/get"),
        ],
        service_name: Annotated[
            str,
            Field(description="Service name argument originally passed to prompts/get"),
        ],
        ctx: Context,
        audience: Annotated[
            str,
            Field(description="Audience argument originally passed to prompts/get"),
        ] = "engineering",
    ) -> dict:
        """Request sampling from the supplied prompts/get messages."""
        sampling_messages = _sampling_messages_from_prompt_dump(messages)
        if not sampling_messages:
            return {
                "ok": False,
                "workflow": "prompt_to_sampling",
                "error": {"code": "missing_prompt_messages", "promptName": prompt_name},
            }
        sampled = await ctx.session.create_message(
            messages=sampling_messages,
            max_tokens=SAMPLING_MAX_TOKENS,
            system_prompt=PROMPT_SYSTEM_PROMPT,
        )
        return {
            "ok": True,
            "workflow": "prompt_to_sampling",
            "promptName": prompt_name,
            "arguments": {
                "service_name": service_name,
                "audience": audience,
            },
            "sampling": _sampling_payload(sampled),
        }

    @mcp.tool(name=TOOL_COMPOSE_INCIDENT_BRIEF)
    async def compose_incident_brief(
        service: Annotated[
            str,
            Field(description="Affected Acme service name"),
        ],
        tool_result: Annotated[
            dict[str, Any],
            Field(description="Structured result returned by get_service_status"),
        ],
        resource_uri: Annotated[
            str,
            Field(description="Resource URI returned by resources/read"),
        ],
        resource_content: Annotated[
            str,
            Field(description="Text returned by resources/read"),
        ],
        prompt_name: Annotated[
            str,
            Field(description="Prompt name retrieved via prompts/get"),
        ],
        prompt_messages: Annotated[
            list[dict[str, Any]],
            Field(description="PromptMessage objects returned by prompts/get"),
        ],
        ctx: Context,
        resource_mime_type: Annotated[
            str,
            Field(description="MIME type returned by resources/read"),
        ] = "text/markdown",
        prompt_arguments: Annotated[
            dict[str, Any] | None,
            Field(description="Arguments originally supplied to prompts/get"),
        ] = None,
    ) -> dict:
        """Compose sampling from prior tool, resource, and prompt MCP results."""
        prompt_sampling = _sampling_messages_from_prompt_dump(prompt_messages)
        sampling_messages = [
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "Tool result from get_service_status:\n"
                        + json.dumps(tool_result, indent=2, sort_keys=True)
                    ),
                ),
            ),
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        f"Resource URI: {resource_uri}\n"
                        f"MIME type: {resource_mime_type}\n"
                        f"Content:\n{resource_content}"
                    ),
                ),
            ),
            *prompt_sampling,
        ]
        sampled = await ctx.session.create_message(
            messages=sampling_messages,
            max_tokens=SAMPLING_MAX_TOKENS,
            system_prompt=COMPOSITION_SYSTEM_PROMPT,
        )
        return {
            "ok": True,
            "workflow": "tool_resource_prompt_composition",
            "tool": TOOL_GET_SERVICE_STATUS,
            "toolResult": tool_result,
            "resourceUri": resource_uri,
            "promptName": prompt_name,
            "promptArguments": prompt_arguments or {},
            "sampling": _sampling_payload(sampled),
        }

    return mcp
