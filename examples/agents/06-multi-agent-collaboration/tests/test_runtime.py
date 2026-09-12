"""Coordinator runtime tests — real specialists, no paid APIs."""

from __future__ import annotations

from pathlib import Path

from agent.agents import default_agents
from agent.cases import CoordinatorStep, MeasuredCase, get_case
from agent.catalog import Catalog
from agent.runtime import run_collaboration
from agent.schemas import AgentResult
from agent.synthesizer import MockSynthesizer

DATA = Path(__file__).resolve().parents[1] / "data"


def _run(trace_id: str, **kwargs):
    return run_collaboration(
        get_case(trace_id),
        catalog=Catalog(DATA),
        synthesizer=MockSynthesizer(),
        max_delegations=kwargs.get("max_delegations", 6),
        agents=kwargs.get("agents"),
    )


def test_basic_collaboration_invokes_two_independent_agents():
    result = _run("independent-status-and-docs")
    assert result.ok is True
    assert result.metrics.termination_reason == "aggregated"
    assert result.metrics.agents_invoked == 2
    assert result.metrics.successful_agent_results == 2
    assert result.metrics.failed_agent_results == 0
    ids = result.state["invokedAgentIds"]
    assert ids == ["status_agent", "docs_agent"]
    status = result.state["results"][0]
    docs = result.state["results"][1]
    assert status["role"] == "status"
    assert docs["role"] == "docs"
    assert status["payload"]["service"]["incident"] == (
        "PAY-2041: elevated card-auth latency"
    )
    assert docs["payload"]["doc"]["id"] == "doc-payments-runbook"
    docs_msg = next(
        event
        for event in result.sequence
        if event.kind == "agent_message" and event.detail.get("to") == "docs_agent"
    )
    assert "prior_result" not in docs_msg.detail["payload"]
    assert "latencyP99Ms" not in docs_msg.detail["payload"]
    assert "PAY-2041" in result.answer
    assert "Payments degradation runbook" in result.answer


def test_sequential_analysis_receives_actual_status_result():
    result = _run("sequential-status-then-analysis")
    assert result.ok is True
    status = result.state["results"][0]
    analysis = result.state["results"][1]
    assert status["agent_id"] == "status_agent"
    assert analysis["agent_id"] == "analysis_agent"
    assert analysis["payload"]["basedOn"] == status["payload"]
    assert analysis["payload"]["incident"] == status["payload"]["service"]["incident"]
    message = next(
        event
        for event in result.sequence
        if event.kind == "agent_message" and event.detail.get("to") == "analysis_agent"
    )
    assert message.detail["parentResultId"] == status["result_id"]
    assert message.detail["payload"]["prior_result"] == status["payload"]
    status_result_idx = next(
        i
        for i, event in enumerate(result.sequence)
        if (
            event.kind == "agent_result"
            and event.detail.get("agentId") == "status_agent"
        )
    )
    analysis_msg_idx = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "agent_message" and event.detail.get("to") == "analysis_agent"
    )
    assert status_result_idx < analysis_msg_idx


def test_failed_docs_agent_does_not_become_success():
    result = _run("docs-agent-failure")
    assert result.ok is False
    assert result.metrics.termination_reason == "agent_failed"
    assert result.metrics.failed_agent_results == 1
    assert result.metrics.successful_agent_results == 1
    docs = next(
        item for item in result.state["results"] if item["agent_id"] == "docs_agent"
    )
    assert docs["ok"] is False
    assert docs["error"]["code"] == "unknown_doc"
    assert "analysis_agent" not in result.state["invokedAgentIds"]
    kinds = [event.kind for event in result.sequence]
    assert "failure" in kinds
    assert kinds.count("termination") == 1
    assert kinds.index("failure") < kinds.index("aggregation")
    assert kinds.index("aggregation") < kinds.index("termination")
    assert kinds[-1] == "termination"
    assert "not treated as success" in result.answer
    assert result.errors


def test_delegate_if_incident_invokes_analysis_when_incident_present():
    result = run_collaboration(
        MeasuredCase(
            trace_id="incident-then-analysis",
            example_class="SEQUENTIAL_COLLABORATION",
            request="Analyze payments if an incident is open.",
            service="payments",
            selection_note="Test-only incident branch of delegate_if_incident.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="status_agent",
                    assignment={"service": "payments"},
                ),
                CoordinatorStep(
                    kind="delegate_if_incident",
                    agent_id="analysis_agent",
                    input_from_agent="status_agent",
                ),
                CoordinatorStep(kind="aggregate"),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is True
    assert result.metrics.termination_reason == "aggregated"
    assert result.state["invokedAgentIds"] == ["status_agent", "analysis_agent"]
    assert result.state["skippedAgentIds"] == []
    status_payload = result.state["results"][0]["payload"]
    analysis_payload = result.state["results"][1]["payload"]
    assert analysis_payload["basedOn"] == status_payload


def test_mutating_agent_cannot_rewrite_recorded_state_or_trace():
    class MutatingAnalysisAgent:
        agent_id = "analysis_agent"
        role = "analysis"

        def handle(self, message):
            message.payload["tampered"] = True
            prior = message.payload["prior_result"]
            prior["service"]["incident"] = "MUTATED-AFTER-RECEIPT"
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                ok=True,
                in_reply_to=message.message_id,
                payload={"basedOn": prior, "incident": prior["service"]["incident"]},
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["analysis_agent"] = MutatingAnalysisAgent()
    result = run_collaboration(
        get_case("sequential-status-then-analysis"),
        catalog=catalog,
        agents=agents,
    )
    recorded_message = result.state["messages"][1]
    assert recorded_message["payload"].get("tampered") is None
    assert (
        recorded_message["payload"]["prior_result"]["service"]["incident"]
        == "PAY-2041: elevated card-auth latency"
    )
    status = result.state["results"][0]
    assert status["payload"]["service"]["incident"] == (
        "PAY-2041: elevated card-auth latency"
    )
    analysis_msg = next(
        event
        for event in result.sequence
        if event.kind == "agent_message" and event.detail.get("to") == "analysis_agent"
    )
    assert "tampered" not in analysis_msg.detail["payload"]
    assert analysis_msg.detail["payload"]["prior_result"]["service"]["incident"] == (
        "PAY-2041: elevated card-auth latency"
    )


def test_operational_status_skips_analysis():
    result = _run("operational-short-circuit")
    assert result.ok is True
    assert result.metrics.termination_reason == "no_further_work"
    assert result.state["invokedAgentIds"] == ["status_agent"]
    assert result.state["skippedAgentIds"] == ["analysis_agent"]
    assert result.metrics.skipped_delegations == 1
    assert result.metrics.agents_invoked == 1
    kinds = [event.kind for event in result.sequence]
    assert "agent_message" in kinds
    analysis_messages = [
        event
        for event in result.sequence
        if event.kind == "agent_message" and event.detail.get("to") == "analysis_agent"
    ]
    assert analysis_messages == []
    skip = next(
        event
        for event in result.sequence
        if (
            event.kind == "coordinator_decision"
            and event.detail.get("decision") == "skip"
        )
    )
    assert skip.detail["reason"] == "no_incident"
    assert "Skipped: analysis_agent" in result.answer


def test_empty_request_is_invalid_task():
    result = run_collaboration(
        MeasuredCase(
            trace_id="empty-request",
            example_class="BASIC_COLLABORATION",
            request="   ",
            service="payments",
            selection_note="Test-only empty request.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="status_agent",
                    assignment={"service": "payments"},
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_task"
    assert result.metrics.agents_invoked == 0
    assert result.sequence[-1].kind == "termination"


def test_unknown_agent_is_error():
    result = run_collaboration(
        MeasuredCase(
            trace_id="unknown-agent",
            example_class="BASIC_COLLABORATION",
            request="Delegate to a missing specialist.",
            service="payments",
            selection_note="Test-only unknown agent.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="ghost_agent",
                    assignment={"service": "payments"},
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "error"
    assert result.metrics.agents_invoked == 0
    assert result.errors[0]["code"] == "unknown_agent"


def test_max_delegations_stops_additional_work():
    result = _run("independent-status-and-docs", max_delegations=1)
    assert result.metrics.termination_reason == "max_delegations"
    assert result.metrics.delegations == 1
    assert result.state["invokedAgentIds"] == ["status_agent"]
    assert "docs_agent" not in result.state["invokedAgentIds"]


def test_missing_prior_result_is_recorded():
    result = run_collaboration(
        MeasuredCase(
            trace_id="analysis-without-status",
            example_class="SEQUENTIAL_COLLABORATION",
            request="Analyze without a prior status result.",
            service="payments",
            selection_note="Test-only missing upstream result.",
            steps=(
                CoordinatorStep(
                    kind="delegate",
                    agent_id="analysis_agent",
                    input_from_agent="status_agent",
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "error"
    assert result.errors[0]["code"] == "missing_prior_result"
    assert result.metrics.agents_invoked == 0


def test_repeated_runs_are_semantically_stable():
    first = _run("sequential-status-then-analysis")
    second = _run("sequential-status-then-analysis")
    assert [event.kind for event in first.sequence] == [
        event.kind for event in second.sequence
    ]
    assert first.state["results"][1]["payload"] == second.state["results"][1]["payload"]
    assert first.answer == second.answer


def test_event_order_starts_with_task_and_ends_with_termination():
    result = _run("independent-status-and-docs")
    assert result.sequence[0].kind == "task_created"
    assert result.sequence[-1].kind == "termination"
    kinds = [event.kind for event in result.sequence]
    assert kinds.count("termination") == 1


def test_main_live_requires_api_key(monkeypatch):
    from main import main

    monkeypatch.setattr(
        "main.get_settings",
        lambda: type(
            "S",
            (),
            {"openai_api_key": "", "data_dir": DATA, "max_delegations": 6},
        )(),
    )
    assert main(["--case", "independent-status-and-docs", "--live"]) == 1
