"""Specialist agent unit tests."""

from __future__ import annotations

from pathlib import Path

from agent.agents import AnalysisAgent, DocsAgent, StatusAgent
from agent.catalog import Catalog
from agent.schemas import AgentMessage

DATA = Path(__file__).resolve().parents[1] / "data"


def _message(to: str, payload: dict) -> AgentMessage:
    return AgentMessage(
        message_id="msg-test",
        to_agent=to,
        task_id="task-test",
        payload=payload,
    )


def test_status_agent_returns_catalog_record():
    agent = StatusAgent(Catalog(DATA))
    result = agent.handle(_message("status_agent", {"service": "payments"}))
    assert result.ok is True
    assert result.role == "status"
    assert result.payload["service"]["incident"] == (
        "PAY-2041: elevated card-auth latency"
    )


def test_status_agent_unknown_service():
    agent = StatusAgent(Catalog(DATA))
    result = agent.handle(_message("status_agent", {"service": "not-a-service"}))
    assert result.ok is False
    assert result.error["code"] == "unknown_service"


def test_status_agent_missing_service():
    agent = StatusAgent(Catalog(DATA))
    result = agent.handle(_message("status_agent", {}))
    assert result.ok is False
    assert result.error["code"] == "missing_service"


def test_docs_agent_returns_document():
    agent = DocsAgent(Catalog(DATA))
    result = agent.handle(_message("docs_agent", {"doc_id": "doc-payments-runbook"}))
    assert result.ok is True
    assert result.payload["doc"]["title"] == "Payments degradation runbook"


def test_docs_agent_unknown_doc():
    agent = DocsAgent(Catalog(DATA))
    result = agent.handle(_message("docs_agent", {"doc_id": "doc-missing"}))
    assert result.ok is False
    assert result.error["code"] == "unknown_doc"


def test_analysis_uses_inbound_payload_not_catalog():
    agent = AnalysisAgent()
    sentinel = {
        "service": {
            "service": "payments",
            "status": "degraded",
            "incident": "SENTINEL-NOT-IN-FIXTURE",
        }
    }
    result = agent.handle(_message("analysis_agent", {"prior_result": sentinel}))
    assert result.ok is True
    assert result.payload["basedOn"] == sentinel
    assert result.payload["incident"] == "SENTINEL-NOT-IN-FIXTURE"
    assert "PAY-2041" not in result.payload["summary"]


def test_analysis_requires_prior_result():
    agent = AnalysisAgent()
    result = agent.handle(_message("analysis_agent", {}))
    assert result.ok is False
    assert result.error["code"] == "missing_prior_result"


def test_analysis_agent_source_does_not_load_status_catalog():
    source = (Path(__file__).resolve().parents[1] / "agent" / "agents.py").read_text(
        encoding="utf-8"
    )
    analysis_start = source.index("class AnalysisAgent")
    analysis_src = source[analysis_start:]
    next_def = analysis_src.find("\ndef default_agents")
    analysis_src = analysis_src if next_def < 0 else analysis_src[:next_def]
    assert "Catalog" not in analysis_src
    assert "services.json" not in analysis_src
    assert "self._catalog" not in analysis_src
