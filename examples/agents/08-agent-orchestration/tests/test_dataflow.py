"""Data-flow tests: downstream nodes consume actual upstream runtime results."""

from __future__ import annotations

from pathlib import Path

from agent.agents import AnalysisAgent, DecisionAgent, default_agents
from agent.cases import get_case
from agent.catalog import Catalog
from agent.runtime import run_orchestration
from agent.schemas import AgentResult, NodeContext

DATA = Path(__file__).resolve().parents[1] / "data"
SENTINEL_INCIDENT = "SENTINEL-INCIDENT-NOT-IN-FIXTURE"
SENTINEL_TOKEN = "SENTINEL-ANALYSIS-TOKEN-NOT-IN-FIXTURE"


class SentinelStatusAgent:
    agent_id = "status_agent"
    role = "status"

    def handle(self, context: NodeContext) -> AgentResult:
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "service": {
                    "service": context.assignment.get("service"),
                    "status": "degraded",
                    "incident": SENTINEL_INCIDENT,
                }
            },
        )


class SentinelAnalysisAgent:
    agent_id = "analysis_agent"
    role = "analysis"

    def handle(self, context: NodeContext) -> AgentResult:
        status = context.upstream_results["status"]
        return AgentResult(
            result_id="",
            node_id=context.node_id,
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            payload={
                "basedOn": {"status": status.payload},
                "incident": True,
                "incident_id": SENTINEL_INCIDENT,
                "summary": f"Analysis of {SENTINEL_INCIDENT}.",
                "analysis_token": SENTINEL_TOKEN,
            },
        )


def test_analysis_uses_injected_status_sentinel_not_fixture():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SentinelStatusAgent()
    result = run_orchestration(
        get_case("payments-status-dependency-chain"),
        catalog=catalog,
        agents=agents,
    )
    analysis = result.state["results"][1]
    decision = result.state["results"][2]
    assert analysis["payload"]["incident_id"] == SENTINEL_INCIDENT
    assert analysis["payload"]["basedOn"]["status"]["service"]["incident"] == (
        SENTINEL_INCIDENT
    )
    assert SENTINEL_INCIDENT in analysis["payload"]["summary"]
    assert SENTINEL_INCIDENT in result.answer
    assert "PAY-2041" not in analysis["payload"]["summary"]
    assert "PAY-2041" not in result.answer
    assert decision["payload"]["basedOn"]["incident_id"] == SENTINEL_INCIDENT
    fixture = catalog.get_service("payments")
    assert fixture is not None
    assert fixture["incident"] != SENTINEL_INCIDENT
    assert fixture["incident"] not in analysis["payload"]["summary"]


def test_decision_uses_injected_analysis_token_not_fixture():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["analysis_agent"] = SentinelAnalysisAgent()
    result = run_orchestration(
        get_case("payments-status-dependency-chain"),
        catalog=catalog,
        agents=agents,
    )
    decision = result.state["results"][2]
    assert decision["agent_id"] == "decision_agent"
    assert decision["payload"]["analysis_token"] == SENTINEL_TOKEN
    assert decision["payload"]["basedOn"]["analysis_token"] == SENTINEL_TOKEN
    assert SENTINEL_TOKEN not in str(catalog.services)
    assert SENTINEL_TOKEN not in str(catalog.docs)
    assert SENTINEL_INCIDENT in result.answer
    assert "PAY-2041" not in decision["payload"]["basedOn"]["summary"]


def test_basic_analysis_sentinel_does_not_reload_docs_or_status_catalog():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SentinelStatusAgent()
    result = run_orchestration(
        get_case("payments-incident-basic-orchestration"),
        catalog=catalog,
        agents=agents,
    )
    analysis = next(
        item for item in result.state["results"] if item["agent_id"] == "analysis_agent"
    )
    assert analysis["payload"]["basedOn"]["status"]["service"]["incident"] == (
        SENTINEL_INCIDENT
    )
    assert analysis["payload"]["basedOn"]["docs"]["doc"]["id"] == "doc-payments-runbook"
    assert SENTINEL_INCIDENT in analysis["payload"]["summary"]
    assert (
        "PAY-2041"
        not in analysis["payload"]["basedOn"]["status"]["service"]["incident"]
    )


def test_analysis_direct_handle_ignores_fixture_catalog():
    agent = AnalysisAgent()
    result = agent.handle(
        NodeContext(
            task_id="t",
            node_id="analysis",
            agent_id="analysis_agent",
            upstream_results={
                "status": AgentResult(
                    result_id="result-1",
                    node_id="status",
                    agent_id="status_agent",
                    role="status",
                    ok=True,
                    payload={
                        "service": {
                            "service": "payments",
                            "status": "major_outage",
                            "incident": SENTINEL_INCIDENT,
                        }
                    },
                )
            },
            upstream_result_ids=["result-1"],
        )
    )
    fixture_incident = Catalog(DATA).get_service("payments")["incident"]
    assert fixture_incident != SENTINEL_INCIDENT
    assert result.payload["incident_id"] == SENTINEL_INCIDENT
    assert fixture_incident not in result.payload["summary"]
    assert SENTINEL_INCIDENT in result.payload["analysis_token"]


def test_decision_direct_handle_consumes_analysis_token():
    agent = DecisionAgent()
    result = agent.handle(
        NodeContext(
            task_id="t",
            node_id="decision",
            agent_id="decision_agent",
            upstream_results={
                "analysis": AgentResult(
                    result_id="result-2",
                    node_id="analysis",
                    agent_id="analysis_agent",
                    role="analysis",
                    ok=True,
                    payload={
                        "incident": True,
                        "incident_id": SENTINEL_INCIDENT,
                        "summary": "sentinel analysis",
                        "analysis_token": SENTINEL_TOKEN,
                    },
                )
            },
            upstream_result_ids=["result-2"],
        )
    )
    assert result.payload["analysis_token"] == SENTINEL_TOKEN
    assert result.payload["basedOn"]["analysis_token"] == SENTINEL_TOKEN
    assert SENTINEL_TOKEN not in str(Catalog(DATA).services)


def test_live_synthesizer_sends_actual_workflow_results(monkeypatch):
    import json

    from agent.state import OrchestrationState
    from agent.synthesizer import LiveSynthesizer
    from config import Settings

    captured: dict = {}

    class FakeMessage:
        content = "note from actual orchestration results"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    state = OrchestrationState(
        task_id="t",
        request="sentinel brief",
        service="payments",
        termination_reason="completed",
    )
    state.record_result(
        AgentResult(
            result_id="result-1",
            node_id="decision",
            agent_id="decision_agent",
            role="decision",
            ok=True,
            payload={"summary": "escalate", "analysis_token": SENTINEL_TOKEN},
        )
    )
    text, _latency = LiveSynthesizer(
        Settings(openai_api_key="sk-test", data_dir=DATA)
    ).synthesize(state)
    user_payload = json.loads(captured["messages"][1]["content"])
    assert text == "note from actual orchestration results"
    assert user_payload["results"][0]["payload"]["analysis_token"] == SENTINEL_TOKEN
    raw = captured["messages"][1]["content"]
    assert SENTINEL_TOKEN in raw
    assert "PAY-2041" not in raw
    assert "services.json" not in raw
