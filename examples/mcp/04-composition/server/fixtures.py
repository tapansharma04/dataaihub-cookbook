"""Deterministic Acme AI fixtures for composed MCP workflows.

Resource URIs, prompt names, and tool data are owned by the server.
Clients discover them through the MCP protocol; they do not import this
module as an application catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

URI_KNOWLEDGE_PLATFORM = "acme://docs/knowledge-platform"
URI_BILLING_PORTAL = "acme://docs/billing-portal"
URI_SERVICE_STATUS = "acme://status/services"

EXPECTED_URIS = (
    URI_KNOWLEDGE_PLATFORM,
    URI_BILLING_PORTAL,
    URI_SERVICE_STATUS,
)

PROMPT_SUMMARIZE_SERVICE = "summarize-service"
PROMPT_DRAFT_STATUS_UPDATE = "draft-status-update"

EXPECTED_PROMPT_NAMES = (
    PROMPT_SUMMARIZE_SERVICE,
    PROMPT_DRAFT_STATUS_UPDATE,
)

TOOL_GET_SERVICE_STATUS = "get_service_status"
TOOL_COMPOSE_RESOURCE_BRIEF = "compose_resource_brief"
TOOL_COMPOSE_FROM_PROMPT = "compose_from_prompt"
TOOL_COMPOSE_INCIDENT_BRIEF = "compose_incident_brief"

EXPECTED_TOOL_NAMES = (
    TOOL_GET_SERVICE_STATUS,
    TOOL_COMPOSE_RESOURCE_BRIEF,
    TOOL_COMPOSE_FROM_PROMPT,
    TOOL_COMPOSE_INCIDENT_BRIEF,
)


@dataclass(frozen=True)
class ResourceFixture:
    uri: str
    name: str
    description: str
    mime_type: str
    text: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class FixtureStore:
    """Read-only fixture access for MCP resource, prompt, and tool handlers."""

    def __init__(self, data_dir: Path) -> None:
        knowledge = (data_dir / "knowledge-platform.md").read_text(encoding="utf-8")
        billing = (data_dir / "billing-portal.md").read_text(encoding="utf-8")
        self.services: dict[str, dict[str, Any]] = load_json(data_dir / "services.json")
        status_doc = {
            "environment": "acme-ai",
            "description": "Service statuses for the Acme AI environment.",
            "services": [
                {
                    "name": record["service"],
                    "status": record["status"],
                    "detail": record.get("incident")
                    or f"{record['service']} is {record['status']}.",
                    "region": record.get("region"),
                    "latencyP99Ms": record.get("latencyP99Ms"),
                }
                for record in self.services.values()
            ],
        }
        status_text = json.dumps(status_doc, indent=2, sort_keys=True) + "\n"
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
                text=status_text,
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

    def get_service_status(self, service: str) -> dict[str, Any]:
        record = self.services.get(service)
        if record is None:
            return {
                "ok": False,
                "error": {
                    "code": "unknown_service",
                    "message": f"Unknown service '{service}'",
                    "validServices": sorted(self.services),
                },
            }
        return {"ok": True, "service": record}
