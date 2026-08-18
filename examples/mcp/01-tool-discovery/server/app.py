"""MCP server exposing deterministic local tools.

Tool definitions, validation, and execution live on the server side of the
protocol boundary. Clients discover contracts through tools/list and invoke
through tools/call — they do not import these handlers directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from server.fixtures import FixtureStore

SERVER_NAME = "dataaihub-cookbook-fixtures"


def build_server(data_dir: Path) -> MCPServer:
    """Create an MCP server wired to local JSON fixtures."""
    store = FixtureStore(data_dir)
    mcp = MCPServer(SERVER_NAME)

    @mcp.tool()
    def get_service_status(
        service: Annotated[
            str,
            Field(description="Canonical service name: billing, payments, auth"),
        ],
    ) -> dict:
        """Return the current status for a canonical service name."""
        cleaned = service.strip().lower()
        return store.get_service_status(cleaned)

    @mcp.tool()
    def get_user_profile(
        user_id: Annotated[str, Field(description="User id, e.g. u-1001")],
    ) -> dict:
        """Return a user profile by user id."""
        cleaned = user_id.strip()
        return store.get_user_profile(cleaned)

    @mcp.tool()
    def search_documentation(
        query: Annotated[
            str,
            Field(description="Short keyword query over internal docs"),
        ],
    ) -> dict:
        """Search internal documentation snippets by keyword query."""
        cleaned = query.strip()
        if len(cleaned) < 2:
            return {
                "ok": False,
                "error": {
                    "code": "invalid_query",
                    "message": "query must be at least 2 characters",
                },
            }
        return store.search_documentation(cleaned)

    return mcp
