"""Deterministic local fixtures for MCP server resources.

Resource URIs, metadata, and contents are owned by the server. Clients
discover them through resources/list and read them through resources/read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

URI_KNOWLEDGE_PLATFORM = "acme://docs/knowledge-platform"
URI_BILLING_PORTAL = "acme://docs/billing-portal"
URI_SERVICE_STATUS = "acme://status/services"

EXPECTED_URIS = (
    URI_KNOWLEDGE_PLATFORM,
    URI_BILLING_PORTAL,
    URI_SERVICE_STATUS,
)


@dataclass(frozen=True)
class ResourceFixture:
    uri: str
    name: str
    description: str
    mime_type: str
    text: str


class FixtureStore:
    """Read-only fixture access for MCP resource handlers."""

    def __init__(self, data_dir: Path) -> None:
        knowledge = (data_dir / "knowledge-platform.md").read_text(encoding="utf-8")
        billing = (data_dir / "billing-portal.md").read_text(encoding="utf-8")
        services = json.dumps(
            json.loads((data_dir / "services.json").read_text(encoding="utf-8")),
            indent=2,
            sort_keys=True,
        )
        self._by_uri: dict[str, ResourceFixture] = {
            URI_KNOWLEDGE_PLATFORM: ResourceFixture(
                uri=URI_KNOWLEDGE_PLATFORM,
                name="knowledge-platform",
                description="Service documentation for the Knowledge Platform.",
                mime_type="text/markdown",
                text=knowledge,
            ),
            URI_BILLING_PORTAL: ResourceFixture(
                uri=URI_BILLING_PORTAL,
                name="billing-portal",
                description="Service documentation for the Billing Portal.",
                mime_type="text/markdown",
                text=billing,
            ),
            URI_SERVICE_STATUS: ResourceFixture(
                uri=URI_SERVICE_STATUS,
                name="service-status",
                description="Service statuses for the Acme AI environment.",
                mime_type="application/json",
                text=services + "\n",
            ),
        }

    def list_resources(self) -> list[ResourceFixture]:
        return [self._by_uri[uri] for uri in EXPECTED_URIS]

    def get(self, uri: str) -> ResourceFixture:
        try:
            return self._by_uri[uri]
        except KeyError as exc:
            raise KeyError(f"Unknown resource URI '{uri}'") from exc

    def content(self, uri: str) -> str:
        return self.get(uri).text
