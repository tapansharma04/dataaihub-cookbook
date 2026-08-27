"""MCP server exposing deterministic local resources.

Resource definitions, metadata, MIME types, and contents live on the server
side of the protocol boundary. Clients discover contracts through
resources/list and read contents through resources/read — they do not import
these handlers or fixtures as an application catalog.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from server.fixtures import (
    URI_BILLING_PORTAL,
    URI_KNOWLEDGE_PLATFORM,
    URI_SERVICE_STATUS,
    FixtureStore,
)

SERVER_NAME = "dataaihub-cookbook-resources"


def build_server(data_dir: Path) -> MCPServer:
    """Create an MCP server wired to local resource fixtures."""
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

    return mcp
