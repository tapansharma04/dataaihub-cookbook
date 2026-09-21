"""Specialist agents. Each node executes one agent against its inbound context.

Agents do not call one another and do not write orchestration state.
Downstream agents consume actual upstream results, not fixture catalogs.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

from agent.catalog import Catalog, DocsStore, StatusStore
from agent.schemas import AgentResult, AgentRole, NodeContext


class Agent(Protocol):
    agent_id: str
    role: AgentRole

    def handle(self, context: NodeContext) -> AgentResult: ...


def _upstream_payloads(context: NodeContext) -> dict[str, dict[str, Any]]:
    return {
        node_id: copy.deepcopy(result.payload)
        for node_id, result in context.upstream_results.items()
    }


def _result_for_role(context: NodeContext, role: AgentRole) -> AgentResult | None:
    for result in context.upstream_results.values():
        if result.role == role:
            return result
    return None


def _status_record(payload: dict[str, Any]) -> dict[str, Any]:
    service = payload.get("service")
    return service if isinstance(service, dict) else {}


class StatusAgent:
    """Reads service status from the status store only."""

    agent_id = "status_agent"
    role: AgentRole = "status"

    def __init__(self, store: StatusStore) -> None:
        self._store = store

    def handle(self, context: NodeContext) -> AgentResult:
        service = str(context.assignment.get("service") or "").strip()
        if not service:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                error={
                    "code": "missing_service",
                    "message": "status_agent requires assignment.service",
                },
            )
        record = self._store.get_service(service)
        if record is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"service_id": service},
                error={
                    "code": "unknown_service",
                    "message": f"Unknown service '{service}'",
                    "serviceId": service,
                    "validServices": self._store.ids,
                },
            )
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={"service": record},
        )


class DocsAgent:
    """Reads operational documentation from the docs store only."""

    agent_id = "docs_agent"
    role: AgentRole = "docs"

    def __init__(self, store: DocsStore) -> None:
        self._store = store

    def handle(self, context: NodeContext) -> AgentResult:
        doc_id = str(context.assignment.get("doc_id") or "").strip()
        if not doc_id:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                error={
                    "code": "missing_doc_id",
                    "message": "docs_agent requires assignment.doc_id",
                },
            )
        record = self._store.get_doc(doc_id)
        if record is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"doc_id": doc_id},
                error={
                    "code": "unknown_doc",
                    "message": f"Unknown document '{doc_id}'",
                    "docId": doc_id,
                    "validDocIds": self._store.ids,
                },
            )
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={"doc": record},
        )


class AnalysisAgent:
    """Interprets actual upstream status/docs results.

    This agent must not read the status or docs catalogs. Downstream
    analysis is grounded in inbound upstream_results only.
    """

    agent_id = "analysis_agent"
    role: AgentRole = "analysis"

    def handle(self, context: NodeContext) -> AgentResult:
        if not context.upstream_results:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                error={
                    "code": "missing_upstream",
                    "message": "analysis_agent requires upstream_results",
                },
            )
        status_result = _result_for_role(context, "status")
        if status_result is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"basedOn": _upstream_payloads(context)},
                error={
                    "code": "missing_upstream_status",
                    "message": "analysis_agent requires an upstream status result",
                },
            )
        service = _status_record(status_result.payload)
        status = str(service.get("status") or "unknown")
        incident_id = service.get("incident")
        has_incident = bool(incident_id)
        docs_result = _result_for_role(context, "docs")
        doc_title = None
        if docs_result is not None:
            doc = docs_result.payload.get("doc")
            if isinstance(doc, dict):
                doc_title = doc.get("title")
        if status == "operational" and not has_incident:
            summary = (
                f"{service.get('service', 'service')} is operational "
                "with no open incident."
            )
            severity = "none"
        elif status == "major_outage":
            summary = (
                f"{service.get('service', 'service')} is in major outage. "
                f"Incident: {incident_id or 'unspecified'}."
            )
            severity = "critical"
        else:
            summary = (
                f"{service.get('service', 'service')} is {status}. "
                f"Incident: {incident_id or 'unspecified'}."
            )
            severity = "elevated"
        if doc_title:
            summary = f"{summary} Docs: {doc_title}."
        token = f"analysis:{status}:{incident_id}"
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "basedOn": _upstream_payloads(context),
                "incident": has_incident,
                "incident_id": incident_id,
                "status": status,
                "severity": severity,
                "summary": summary,
                "analysis_token": token,
            },
        )


class DecisionAgent:
    """Determines the operational outcome from the actual analysis result."""

    agent_id = "decision_agent"
    role: AgentRole = "decision"

    def handle(self, context: NodeContext) -> AgentResult:
        analysis = _result_for_role(context, "analysis")
        if analysis is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"basedOn": _upstream_payloads(context)},
                error={
                    "code": "missing_upstream_analysis",
                    "message": "decision_agent requires an upstream analysis result",
                },
            )
        based_on = copy.deepcopy(analysis.payload)
        incident = bool(based_on.get("incident"))
        outcome = "escalate" if incident else "steady_state"
        summary = based_on.get("summary") or "No analysis summary."
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "basedOn": based_on,
                "outcome": outcome,
                "analysis_token": based_on.get("analysis_token"),
                "incident": incident,
                "incident_id": based_on.get("incident_id"),
                "summary": f"Decision: {outcome}. {summary}",
            },
        )


class RemediationAgent:
    """Takes action when analysis reports an incident."""

    agent_id = "remediation_agent"
    role: AgentRole = "remediation"

    def handle(self, context: NodeContext) -> AgentResult:
        analysis = _result_for_role(context, "analysis")
        if analysis is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"basedOn": _upstream_payloads(context)},
                error={
                    "code": "missing_upstream_analysis",
                    "message": "remediation_agent requires an upstream analysis result",
                },
            )
        based_on = copy.deepcopy(analysis.payload)
        incident_id = based_on.get("incident_id")
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "basedOn": based_on,
                "action": "open_incident_bridge",
                "analysis_token": based_on.get("analysis_token"),
                "incident": True,
                "incident_id": incident_id,
                "summary": (
                    f"Remediation started for {incident_id or 'unspecified incident'}."
                ),
            },
        )


class NoActionAgent:
    """Records that no operational action is required."""

    agent_id = "no_action_agent"
    role: AgentRole = "no_action"

    def handle(self, context: NodeContext) -> AgentResult:
        analysis = _result_for_role(context, "analysis")
        if analysis is None:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=False,
                payload={"basedOn": _upstream_payloads(context)},
                error={
                    "code": "missing_upstream_analysis",
                    "message": "no_action_agent requires an upstream analysis result",
                },
            )
        based_on = copy.deepcopy(analysis.payload)
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "basedOn": based_on,
                "action": "no_action",
                "analysis_token": based_on.get("analysis_token"),
                "incident": False,
                "summary": "No incident; no operational action.",
            },
        )


def default_agents(catalog: Catalog) -> dict[str, Agent]:
    return {
        StatusAgent.agent_id: StatusAgent(catalog.status_store()),
        DocsAgent.agent_id: DocsAgent(catalog.docs_store()),
        AnalysisAgent.agent_id: AnalysisAgent(),
        DecisionAgent.agent_id: DecisionAgent(),
        RemediationAgent.agent_id: RemediationAgent(),
        NoActionAgent.agent_id: NoActionAgent(),
    }
