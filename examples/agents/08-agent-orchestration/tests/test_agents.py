"""Specialist agent unit tests."""

from __future__ import annotations

from pathlib import Path

from agent.agents import (
    AnalysisAgent,
    DecisionAgent,
    DocsAgent,
    NoActionAgent,
    RemediationAgent,
    StatusAgent,
)
from agent.catalog import Catalog
from agent.schemas import AgentResult, NodeContext

DATA = Path(__file__).resolve().parents[1] / "data"


def _status_context(service: str) -> NodeContext:
    return NodeContext(
        task_id="t",
        node_id="status",
        agent_id="status_agent",
        assignment={"service": service},
    )


def test_status_reads_payments_fixture():
    agent = StatusAgent(Catalog(DATA).status_store())
    result = agent.handle(_status_context("payments"))
    assert result.ok is True
    assert result.payload["service"]["incident"] == (
        "PAY-2041: elevated card-auth latency"
    )


def test_status_unknown_service():
    agent = StatusAgent(Catalog(DATA).status_store())
    result = agent.handle(_status_context("does-not-exist"))
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "unknown_service"


def test_docs_reads_runbook():
    agent = DocsAgent(Catalog(DATA).docs_store())
    result = agent.handle(
        NodeContext(
            task_id="t",
            node_id="docs",
            agent_id="docs_agent",
            assignment={"doc_id": "doc-payments-runbook"},
        )
    )
    assert result.ok is True
    assert result.payload["doc"]["id"] == "doc-payments-runbook"


def test_analysis_requires_upstream_status():
    result = AnalysisAgent().handle(
        NodeContext(task_id="t", node_id="analysis", agent_id="analysis_agent")
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "missing_upstream"


def test_decision_requires_upstream_analysis():
    result = DecisionAgent().handle(
        NodeContext(task_id="t", node_id="decision", agent_id="decision_agent")
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "missing_upstream_analysis"


def test_remediation_and_no_action_require_analysis():
    remediation = RemediationAgent().handle(
        NodeContext(task_id="t", node_id="remediation", agent_id="remediation_agent")
    )
    no_action = NoActionAgent().handle(
        NodeContext(task_id="t", node_id="no_action", agent_id="no_action_agent")
    )
    assert remediation.error is not None
    assert no_action.error is not None
    assert remediation.error["code"] == "missing_upstream_analysis"
    assert no_action.error["code"] == "missing_upstream_analysis"


def test_analysis_uses_status_and_docs_payloads():
    status = AgentResult(
        result_id="result-1",
        node_id="status",
        agent_id="status_agent",
        role="status",
        ok=True,
        payload={
            "service": {
                "service": "payments",
                "status": "degraded",
                "incident": "PAY-2041: elevated card-auth latency",
            }
        },
    )
    docs = AgentResult(
        result_id="result-2",
        node_id="docs",
        agent_id="docs_agent",
        role="docs",
        ok=True,
        payload={"doc": {"id": "doc-payments-runbook", "title": "Payments runbook"}},
    )
    result = AnalysisAgent().handle(
        NodeContext(
            task_id="t",
            node_id="analysis",
            agent_id="analysis_agent",
            upstream_results={"status": status, "docs": docs},
            upstream_result_ids=["result-1", "result-2"],
        )
    )
    assert result.ok is True
    assert result.payload["incident"] is True
    assert result.payload["basedOn"]["status"] == status.payload
    assert result.payload["basedOn"]["docs"] == docs.payload
    assert "Payments runbook" in result.payload["summary"]
