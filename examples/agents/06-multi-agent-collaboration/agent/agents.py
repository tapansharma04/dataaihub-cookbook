"""Specialized agents. Each has a role and communicates only via messages.

Status and docs agents read their own catalogs. The analysis agent uses
only the payload on the inbound AgentMessage — it does not reload
service fixtures.
"""

from __future__ import annotations

import copy
from typing import Protocol

from agent.catalog import Catalog
from agent.schemas import AgentMessage, AgentResult, AgentRole


class Agent(Protocol):
    agent_id: str
    role: AgentRole

    def handle(self, message: AgentMessage) -> AgentResult: ...


class StatusAgent:
    """Looks up canonical service status from the status catalog."""

    agent_id = "status_agent"
    role: AgentRole = "status"

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def handle(self, message: AgentMessage) -> AgentResult:
        service = str(message.payload.get("service") or "").strip()
        if not service:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                in_reply_to=message.message_id,
                error={
                    "code": "missing_service",
                    "message": "status_agent requires payload.service",
                },
            )
        record = self._catalog.get_service(service)
        if record is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                in_reply_to=message.message_id,
                error={
                    "code": "unknown_service",
                    "message": f"Unknown service '{service}'",
                    "validServices": sorted(self._catalog.services),
                },
            )
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            in_reply_to=message.message_id,
            payload={"service": record},
        )


class DocsAgent:
    """Retrieves a documentation record from the docs catalog."""

    agent_id = "docs_agent"
    role: AgentRole = "docs"

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def handle(self, message: AgentMessage) -> AgentResult:
        doc_id = str(message.payload.get("doc_id") or "").strip()
        if not doc_id:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                in_reply_to=message.message_id,
                error={
                    "code": "missing_doc_id",
                    "message": "docs_agent requires payload.doc_id",
                },
            )
        record = self._catalog.get_doc(doc_id)
        if record is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                in_reply_to=message.message_id,
                error={
                    "code": "unknown_doc",
                    "message": f"Unknown document '{doc_id}'",
                    "validDocIds": sorted(self._catalog.docs),
                },
            )
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            in_reply_to=message.message_id,
            payload={"doc": record},
        )


class AnalysisAgent:
    """Interprets a prior status-agent result carried on the message.

    This agent must not read the status catalog. Downstream analysis is
    grounded in the inbound payload only.
    """

    agent_id = "analysis_agent"
    role: AgentRole = "analysis"

    def handle(self, message: AgentMessage) -> AgentResult:
        inbound = message.payload.get("prior_result")
        if not isinstance(inbound, dict) or not inbound:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                in_reply_to=message.message_id,
                error={
                    "code": "missing_prior_result",
                    "message": "analysis_agent requires payload.prior_result",
                },
            )
        prior = copy.deepcopy(inbound)
        service = prior.get("service") if isinstance(prior.get("service"), dict) else {}
        status = str(service.get("status") or "unknown")
        incident = service.get("incident")
        if status == "operational" and not incident:
            summary = (
                f"{service.get('service', 'service')} is operational "
                "with no open incident."
            )
            severity = "none"
        elif status == "major_outage":
            summary = (
                f"{service.get('service', 'service')} is in major outage. "
                f"Incident: {incident or 'unspecified'}."
            )
            severity = "critical"
        else:
            summary = (
                f"{service.get('service', 'service')} is {status}. "
                f"Incident: {incident or 'unspecified'}."
            )
            severity = "elevated"
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            in_reply_to=message.message_id,
            payload={
                "basedOn": prior,
                "severity": severity,
                "summary": summary,
                "incident": incident,
                "status": status,
            },
        )


def default_agents(catalog: Catalog) -> dict[str, Agent]:
    return {
        StatusAgent.agent_id: StatusAgent(catalog),
        DocsAgent.agent_id: DocsAgent(catalog),
        AnalysisAgent.agent_id: AnalysisAgent(),
    }
