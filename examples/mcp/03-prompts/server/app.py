"""MCP server exposing deterministic reusable prompt templates.

Prompt names, descriptions, argument contracts, and rendered message content
live on the server side of the protocol boundary. Clients discover templates
through prompts/list and retrieve messages through prompts/get — they do not
import these handlers as an application catalog.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.prompts.base import AssistantMessage, UserMessage
from pydantic import Field

from server.fixtures import (
    PROMPT_DRAFT_STATUS_UPDATE,
    PROMPT_INVESTIGATE_INCIDENT,
    PROMPT_SUMMARIZE_SERVICE,
)

SERVER_NAME = "dataaihub-cookbook-prompts"


def build_server() -> MCPServer:
    """Create an MCP server wired to the Acme AI prompt catalog."""
    mcp = MCPServer(SERVER_NAME)

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
        return (
            f"Provide a concise summary of the {service_name} service for a "
            f"{audience} audience. Cover purpose, upstream/downstream "
            f"dependencies, and current operational posture."
        )

    @mcp.prompt(
        PROMPT_INVESTIGATE_INCIDENT,
        description="Start an incident investigation workflow for an Acme AI service.",
    )
    def investigate_incident(
        service: Annotated[
            str,
            Field(description="Affected Acme service name"),
        ],
        incident: Annotated[
            str,
            Field(description="Incident identifier, e.g. INC-2048"),
        ],
    ) -> list[UserMessage | AssistantMessage]:
        return [
            UserMessage(
                f"Investigate incident {incident} affecting the {service} "
                f"service in the Acme AI environment."
            ),
            AssistantMessage(
                "I will review recent deploys, error-rate dashboards, and "
                f"dependency health for {service}."
            ),
            UserMessage(
                f"Prioritize logs and traces tagged with {incident} and "
                f"compare {service} latency against the prior 24 hours."
            ),
        ]

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
            Field(description="Current status label: healthy, degraded, or outage"),
        ],
    ) -> str:
        return (
            f"Draft a brief stakeholder status update for {service}. "
            f"Current status: {status}. Include impact, mitigation steps taken, "
            f"and the next checkpoint time."
        )

    return mcp
