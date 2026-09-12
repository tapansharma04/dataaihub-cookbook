"""Data-flow tests: downstream work uses actual upstream runtime results."""

from __future__ import annotations

from pathlib import Path

from agent.agents import AnalysisAgent, StatusAgent, default_agents
from agent.cases import CoordinatorStep, MeasuredCase
from agent.catalog import Catalog
from agent.runtime import run_collaboration
from agent.schemas import AgentMessage, AgentResult

DATA = Path(__file__).resolve().parents[1] / "data"
SENTINEL = "SENTINEL-INCIDENT-NOT-IN-FIXTURE"


class SentinelStatusAgent:
    agent_id = "status_agent"
    role = "status"

    def handle(self, message: AgentMessage) -> AgentResult:
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            ok=True,
            in_reply_to=message.message_id,
            payload={
                "service": {
                    "service": message.payload.get("service"),
                    "status": "degraded",
                    "incident": SENTINEL,
                }
            },
        )


def test_analysis_grounded_in_injected_status_result_not_fixture():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SentinelStatusAgent()
    result = run_collaboration(
        MeasuredCase(
            trace_id="sentinel-sequential",
            example_class="SEQUENTIAL_COLLABORATION",
            request="Analyze using the status agent's runtime result.",
            service="payments",
            selection_note="Test-only sentinel data flow.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="status_agent",
                    assignment={"service": "payments"},
                ),
                CoordinatorStep(
                    kind="delegate",
                    agent_id="analysis_agent",
                    input_from_agent="status_agent",
                ),
                CoordinatorStep(kind="aggregate"),
            ),
        ),
        catalog=catalog,
        agents=agents,
    )
    analysis = result.state["results"][1]
    assert analysis["payload"]["incident"] == SENTINEL
    assert analysis["payload"]["basedOn"]["service"]["incident"] == SENTINEL
    assert SENTINEL in result.answer
    assert "PAY-2041" not in analysis["payload"]["summary"]
    assert "PAY-2041" not in result.answer


def test_coordinator_aggregates_actual_results_not_reloaded_fixtures():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SentinelStatusAgent()
    result = run_collaboration(
        MeasuredCase(
            trace_id="sentinel-basic",
            example_class="BASIC_COLLABORATION",
            request="Packet from actual specialist results.",
            service="payments",
            selection_note="Test-only aggregation data flow.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="status_agent",
                    assignment={"service": "payments"},
                ),
                CoordinatorStep(
                    kind="delegate",
                    agent_id="docs_agent",
                    assignment={"doc_id": "doc-payments-runbook"},
                ),
                CoordinatorStep(kind="aggregate"),
            ),
        ),
        catalog=catalog,
        agents=agents,
    )
    status_payload = result.state["results"][0]["payload"]
    assert status_payload["service"]["incident"] == SENTINEL
    assert SENTINEL in result.answer
    real_status = StatusAgent(catalog).handle(
        AgentMessage(
            message_id="probe",
            to_agent="status_agent",
            task_id="probe",
            payload={"service": "payments"},
        )
    )
    assert real_status.payload["service"]["incident"] != SENTINEL
    assert (
        real_status.payload["service"]["incident"]
        not in (result.state["results"][0]["payload"]["service"]["incident"])
    )


def test_analysis_direct_handle_ignores_fixture_catalog():
    agent = AnalysisAgent()
    result = agent.handle(
        AgentMessage(
            message_id="m",
            to_agent="analysis_agent",
            task_id="t",
            payload={
                "prior_result": {
                    "service": {
                        "service": "payments",
                        "status": "major_outage",
                        "incident": SENTINEL,
                    }
                }
            },
        )
    )
    fixture_incident = Catalog(DATA).get_service("payments")["incident"]
    assert fixture_incident != SENTINEL
    assert result.payload["incident"] == SENTINEL
    assert fixture_incident not in result.payload["summary"]


def test_live_synthesizer_sends_actual_taskstate_results(monkeypatch):
    import json

    from agent.schemas import AgentResult
    from agent.state import TaskState
    from agent.synthesizer import LiveSynthesizer
    from config import Settings

    captured: dict = {}

    class FakeMessage:
        content = "brief from actual specialist results"

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
    state = TaskState(task_id="t", request="sentinel brief", service="payments")
    state.record_result(
        AgentResult(
            result_id="result-1",
            agent_id="status_agent",
            role="status",
            ok=True,
            in_reply_to="msg-1",
            payload={
                "service": {
                    "service": "payments",
                    "incident": SENTINEL,
                }
            },
        )
    )
    text, _latency = LiveSynthesizer(
        Settings(openai_api_key="sk-test", data_dir=DATA)
    ).synthesize(state)
    user_payload = json.loads(captured["messages"][1]["content"])
    assert text == "brief from actual specialist results"
    assert user_payload["results"][0]["payload"]["service"]["incident"] == SENTINEL
    raw = captured["messages"][1]["content"]
    assert SENTINEL in raw
    assert "PAY-2041" not in raw
    assert "services.json" not in raw
