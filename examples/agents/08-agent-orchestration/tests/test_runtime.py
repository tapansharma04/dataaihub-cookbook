"""Orchestration runtime tests — real specialists, no paid APIs."""

from __future__ import annotations

from pathlib import Path

from agent.agents import default_agents
from agent.cases import MeasuredCase, NodeSpec, get_case
from agent.catalog import Catalog
from agent.runtime import run_orchestration
from agent.schemas import AgentResult, NodeContext
from agent.synthesizer import MockSynthesizer

DATA = Path(__file__).resolve().parents[1] / "data"


def _run(trace_id: str, **kwargs):
    return run_orchestration(
        get_case(trace_id),
        catalog=Catalog(DATA),
        synthesizer=MockSynthesizer(),
        max_nodes=kwargs.get("max_nodes", 12),
        agents=kwargs.get("agents"),
    )


def _kinds(result) -> list[str]:
    return [event.kind for event in result.sequence]


def _ready_ids(result) -> list[str]:
    return [
        event.detail["nodeId"]
        for event in result.sequence
        if event.kind == "node_ready"
    ]


def test_basic_independent_nodes_ready_before_either_executes():
    result = _run("payments-incident-basic-orchestration")
    assert result.ok is True
    assert result.metrics.termination_reason == "completed"
    assert result.state["nodeStates"] == {
        "status": "COMPLETED",
        "docs": "COMPLETED",
        "analysis": "COMPLETED",
        "decision": "COMPLETED",
    }
    kinds = _kinds(result)
    first_result = kinds.index("node_result")
    ready_before_exec = [
        event.detail["nodeId"]
        for event in result.sequence[:first_result]
        if event.kind == "node_ready"
    ]
    assert ready_before_exec == ["status", "docs"]
    analysis_ready = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_ready" and event.detail["nodeId"] == "analysis"
    )
    status_done = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_completed" and event.detail["nodeId"] == "status"
    )
    docs_done = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_completed" and event.detail["nodeId"] == "docs"
    )
    assert analysis_ready > status_done
    assert analysis_ready > docs_done
    assert _ready_ids(result) == ["status", "docs", "analysis", "decision"]
    assert result.state["executionOrder"] == ["status", "docs", "analysis", "decision"]


def test_basic_analysis_consumes_both_upstream_results():
    result = _run("payments-incident-basic-orchestration")
    status = result.state["results"][0]
    docs = result.state["results"][1]
    analysis = result.state["results"][2]
    decision = result.state["results"][3]
    assert analysis["payload"]["basedOn"]["status"] == status["payload"]
    assert analysis["payload"]["basedOn"]["docs"] == docs["payload"]
    assert analysis["upstream_result_ids"] == [status["result_id"], docs["result_id"]]
    assert decision["payload"]["basedOn"] == analysis["payload"]
    assert decision["upstream_result_ids"] == [analysis["result_id"]]
    ready_analysis = next(
        event
        for event in result.sequence
        if event.kind == "node_ready" and event.detail["nodeId"] == "analysis"
    )
    assert ready_analysis.detail["reason"] == "dependencies_completed"
    assert ready_analysis.detail["dependencyIds"] == ["status", "docs"]
    assert ready_analysis.detail["upstreamResultIds"] == [
        status["result_id"],
        docs["result_id"],
    ]


def test_dependency_chain_is_strictly_ordered():
    result = _run("payments-status-dependency-chain")
    assert result.ok is True
    assert result.state["executionOrder"] == ["status", "analysis", "decision"]
    kinds = _kinds(result)
    ready_idxs = [i for i, kind in enumerate(kinds) if kind == "node_ready"]
    result_idxs = [i for i, kind in enumerate(kinds) if kind == "node_result"]
    assert len(ready_idxs) == 3
    assert len(result_idxs) == 3
    assert ready_idxs[0] < result_idxs[0] < ready_idxs[1]
    assert ready_idxs[1] < result_idxs[1] < ready_idxs[2]
    assert ready_idxs[2] < result_idxs[2]
    assert _ready_ids(result) == ["status", "analysis", "decision"]
    first_result = kinds.index("node_result")
    ready_before_first_agent = [
        event.detail["nodeId"]
        for event in result.sequence[:first_result]
        if event.kind == "node_ready"
    ]
    assert ready_before_first_agent == ["status"]
    analysis_ready = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_ready" and event.detail["nodeId"] == "analysis"
    )
    status_result = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_result" and event.detail["nodeId"] == "status"
    )
    decision_ready = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_ready" and event.detail["nodeId"] == "decision"
    )
    analysis_result = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "node_result" and event.detail["nodeId"] == "analysis"
    )
    assert analysis_ready > status_result
    assert decision_ready > analysis_result


def test_conditional_branch_skips_non_selected_node():
    result = _run("payments-incident-conditional-branch")
    assert result.ok is True
    assert result.metrics.termination_reason == "completed"
    assert result.state["nodeStates"]["remediation"] == "COMPLETED"
    assert result.state["nodeStates"]["no_action"] == "SKIPPED"
    assert "no_action" in result.state["skippedNodeIds"]
    assert "no_action_agent" not in result.state["invokedAgentIds"]
    skipped = next(event for event in result.sequence if event.kind == "node_skipped")
    assert skipped.detail["nodeId"] == "no_action"
    assert skipped.detail["reason"] == "condition_not_selected"
    conditions = [
        event for event in result.sequence if event.kind == "condition_evaluated"
    ]
    assert [
        (event.detail["nodeId"], event.detail["selected"]) for event in conditions
    ] == [
        ("remediation", True),
        ("no_action", False),
    ]
    assert conditions[0].detail["actual"] is True
    assert conditions[1].detail["actual"] is True
    analysis = result.state["results"][1]
    remediation = result.state["results"][2]
    assert remediation["payload"]["basedOn"] == analysis["payload"]
    assert result.state["executionOrder"] == ["status", "analysis", "remediation"]


def test_failure_propagates_to_dependents():
    result = _run("unknown-service-orchestration-failure")
    assert result.ok is False
    assert result.metrics.termination_reason == "workflow_failed"
    assert result.metrics.nodes_failed == 1
    assert result.metrics.nodes_skipped == 2
    assert result.state["nodeStates"] == {
        "status": "FAILED",
        "analysis": "SKIPPED",
        "decision": "SKIPPED",
    }
    assert result.state["executionOrder"] == ["status"]
    assert "analysis_agent" not in result.state["invokedAgentIds"]
    assert "decision_agent" not in result.state["invokedAgentIds"]
    skips = [event for event in result.sequence if event.kind == "node_skipped"]
    assert [event.detail["nodeId"] for event in skips] == ["analysis", "decision"]
    assert skips[0].detail["reason"] == "dependency_failed"
    assert skips[1].detail["reason"] == "dependency_skipped"
    kinds = _kinds(result)
    assert "completed" not in kinds
    assert kinds.index("node_failed") < kinds.index("node_skipped")
    assert kinds[-1] == "termination"
    assert "not treated as success" in result.answer.lower()
    status = result.state["results"][0]
    assert status["ok"] is False
    assert status["error"]["code"] == "unknown_service"


def test_empty_request_is_invalid_task():
    result = run_orchestration(
        MeasuredCase(
            trace_id="empty-request",
            example_class="BASIC_ORCHESTRATION",
            request="   ",
            service="payments",
            selection_note="Test-only empty request.",
            nodes=(
                NodeSpec(
                    node_id="status",
                    agent_id="status_agent",
                    assignment={"service": "payments"},
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_task"
    assert result.metrics.nodes_executed == 0
    assert result.sequence[-1].kind == "termination"
    assert "plan_created" not in _kinds(result)


def test_invalid_plan_terminates_without_execution():
    result = run_orchestration(
        MeasuredCase(
            trace_id="cycle-plan",
            example_class="DEPENDENCY_CHAIN",
            request="A cyclic plan should not execute.",
            service="payments",
            selection_note="Test-only invalid plan.",
            nodes=(
                NodeSpec(
                    node_id="a",
                    agent_id="status_agent",
                    dependencies=("b",),
                    assignment={"service": "payments"},
                ),
                NodeSpec(
                    node_id="b",
                    agent_id="docs_agent",
                    dependencies=("a",),
                    assignment={"doc_id": "doc-payments-runbook"},
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_plan"
    assert result.metrics.nodes_executed == 0
    assert "node_ready" not in _kinds(result)


def test_condition_follows_runtime_incident_not_case_class():
    class FalseIncidentAnalysis:
        agent_id = "analysis_agent"
        role = "analysis"

        def handle(self, context: NodeContext) -> AgentResult:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=True,
                payload={
                    "incident": False,
                    "incident_id": None,
                    "summary": "Injected no-incident analysis.",
                    "analysis_token": "analysis:operational:None",
                },
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["analysis_agent"] = FalseIncidentAnalysis()
    result = run_orchestration(
        get_case("payments-incident-conditional-branch"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is True
    assert result.state["nodeStates"]["no_action"] == "COMPLETED"
    assert result.state["nodeStates"]["remediation"] == "SKIPPED"
    skipped = next(event for event in result.sequence if event.kind == "node_skipped")
    assert skipped.detail["nodeId"] == "remediation"
    assert skipped.detail["reason"] == "condition_not_selected"
    assert skipped.detail["actual"] is False
    assert "remediation_agent" not in result.state["invokedAgentIds"]
    assert result.state["executionOrder"] == ["status", "analysis", "no_action"]


def test_mutating_agent_cannot_rewrite_recorded_upstream():
    class MutatingAnalysis:
        agent_id = "analysis_agent"
        role = "analysis"

        def handle(self, context: NodeContext) -> AgentResult:
            status = context.upstream_results["status"]
            status.payload["service"]["incident"] = "MUTATED-AFTER-RECEIPT"
            context.assignment["tampered"] = True
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id=self.agent_id,
                role=self.role,
                ok=True,
                payload={
                    "basedOn": {"status": status.payload},
                    "incident": True,
                    "summary": "mutated",
                    "analysis_token": "mutated",
                },
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["analysis_agent"] = MutatingAnalysis()
    result = _run("payments-status-dependency-chain", agents=agents)
    recorded_status = result.state["results"][0]
    assert (
        recorded_status["payload"]["service"]["incident"]
        == "PAY-2041: elevated card-auth latency"
    )
    status_event = next(
        event
        for event in result.sequence
        if event.kind == "node_result" and event.detail["nodeId"] == "status"
    )
    assert "MUTATED-AFTER-RECEIPT" not in str(status_event.detail["payload"])


def test_repeated_runs_are_semantically_stable():
    first = _run("payments-incident-basic-orchestration")
    second = _run("payments-incident-basic-orchestration")
    assert _kinds(first) == _kinds(second)
    assert first.state["nodeStates"] == second.state["nodeStates"]
    assert first.state["results"][2]["payload"] == second.state["results"][2]["payload"]
    assert first.answer == second.answer
    assert first.state["executionOrder"] == second.state["executionOrder"]


def test_event_order_starts_with_task_and_ends_with_termination():
    result = _run("payments-incident-basic-orchestration")
    assert result.sequence[0].kind == "task_created"
    assert result.sequence[1].kind == "plan_created"
    assert result.sequence[-1].kind == "termination"
    assert _kinds(result).count("termination") == 1


def test_independent_ready_reason_is_no_dependencies():
    result = _run("payments-incident-basic-orchestration")
    ready = {
        event.detail["nodeId"]: event.detail["reason"]
        for event in result.sequence
        if event.kind == "node_ready"
    }
    assert ready["status"] == "no_dependencies"
    assert ready["docs"] == "no_dependencies"
    assert ready["analysis"] == "dependencies_completed"
    assert ready["decision"] == "dependencies_completed"


def test_completed_nodes_do_not_reexecute():
    result = _run("payments-status-dependency-chain")
    executed = [
        event.detail["nodeId"]
        for event in result.sequence
        if event.kind == "node_started"
    ]
    assert executed == ["status", "analysis", "decision"]
    assert executed.count("status") == 1
    assert executed.count("analysis") == 1
    assert executed.count("decision") == 1


def test_agent_mismatch_is_not_rewritten():
    class SpoofingStatus:
        agent_id = "status_agent"
        role = "status"

        def handle(self, context: NodeContext) -> AgentResult:
            return AgentResult(
                result_id="",
                node_id=context.node_id,
                agent_id="docs_agent",
                role=self.role,
                ok=True,
                payload={"service": {"service": "payments", "incident": "spoof"}},
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SpoofingStatus()
    result = run_orchestration(
        get_case("payments-status-dependency-chain"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "workflow_failed"
    recorded = result.state["results"][0]
    assert recorded["agent_id"] == "docs_agent"
    assert recorded["error"]["code"] == "agent_mismatch"
    assert recorded["error"]["expectedAgentId"] == "status_agent"
    assert result.state["nodeStates"] == {
        "status": "FAILED",
        "analysis": "SKIPPED",
        "decision": "SKIPPED",
    }
    failed = next(event for event in result.sequence if event.kind == "node_failed")
    assert failed.detail["actualAgentId"] == "docs_agent"
    assert failed.detail["agentId"] == "status_agent"
    assert "analysis_agent" not in result.state["invokedAgentIds"]


def test_node_id_mismatch_is_not_rewritten():
    class SpoofingStatus:
        agent_id = "status_agent"
        role = "status"

        def handle(self, context: NodeContext) -> AgentResult:
            return AgentResult(
                result_id="",
                node_id="decision",
                agent_id=self.agent_id,
                role=self.role,
                ok=True,
                payload={"service": {"service": "payments", "incident": "spoof"}},
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["status_agent"] = SpoofingStatus()
    result = run_orchestration(
        get_case("payments-status-dependency-chain"),
        catalog=catalog,
        agents=agents,
    )
    recorded = result.state["results"][0]
    assert recorded["node_id"] == "decision"
    assert recorded["error"]["code"] == "node_mismatch"
    assert recorded["error"]["expectedNodeId"] == "status"
    assert result.state["nodes"]["status"]["result"]["node_id"] == "decision"
    assert result.state["executionOrder"] == ["status"]
    assert result.state["nodeStates"]["status"] == "FAILED"
    assert result.ok is False


def test_unknown_agent_fails_without_fabricated_role():
    result = run_orchestration(
        MeasuredCase(
            trace_id="unknown-agent",
            example_class="ORCHESTRATION_FAILURE",
            request="Run a missing specialist.",
            service="payments",
            selection_note="Test-only unknown agent.",
            nodes=(
                NodeSpec(
                    node_id="status",
                    agent_id="missing_agent",
                    assignment={"service": "payments"},
                ),
                NodeSpec(
                    node_id="analysis",
                    agent_id="analysis_agent",
                    dependencies=("status",),
                ),
            ),
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "workflow_failed"
    assert result.state["nodeStates"] == {
        "status": "FAILED",
        "analysis": "SKIPPED",
    }
    assert result.state["results"] == []
    assert result.state["executionOrder"] == ["status"]
    assert "node_result" not in _kinds(result)
    failed = next(event for event in result.sequence if event.kind == "node_failed")
    assert failed.detail["error"]["code"] == "unknown_agent"
    assert failed.detail["agentId"] == "missing_agent"
    assert "role" not in failed.detail
    assert result.errors[0]["code"] == "unknown_agent"


def test_max_nodes_leaves_ready_as_error_not_success():
    result = _run("payments-status-dependency-chain", max_nodes=1)
    assert result.ok is False
    assert result.metrics.termination_reason == "error"
    assert result.errors[0]["code"] == "max_nodes"
    assert result.errors[0]["readyNodeIds"] == ["analysis"]
    assert result.state["nodeStates"] == {
        "status": "COMPLETED",
        "analysis": "READY",
        "decision": "PENDING",
    }
    assert result.state["executionOrder"] == ["status"]
    assert "completed" not in _kinds(result)
    assert result.sequence[-1].kind == "termination"


def test_main_live_requires_api_key(monkeypatch):
    from main import main

    monkeypatch.setattr(
        "main.get_settings",
        lambda: type(
            "S",
            (),
            {"openai_api_key": "", "data_dir": DATA, "max_nodes": 12},
        )(),
    )
    assert main(["--case", "payments-incident-basic-orchestration", "--live"]) == 1
