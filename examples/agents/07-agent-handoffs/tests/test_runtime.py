"""Handoff runtime tests — real specialists, no paid APIs."""

from __future__ import annotations

from pathlib import Path

from agent.agents import default_agents
from agent.cases import MeasuredCase, get_case
from agent.catalog import Catalog
from agent.runtime import run_handoffs
from agent.schemas import AgentContext, AgentResult, HandoffRequest
from agent.synthesizer import MockSynthesizer

DATA = Path(__file__).resolve().parents[1] / "data"


def _run(trace_id: str, **kwargs):
    return run_handoffs(
        get_case(trace_id),
        catalog=Catalog(DATA),
        synthesizer=MockSynthesizer(),
        max_handoffs=kwargs.get("max_handoffs", 6),
        agents=kwargs.get("agents"),
    )


def _kinds(result) -> list[str]:
    return [event.kind for event in result.sequence]


def _assert_no_ticket_id(obj) -> None:
    if isinstance(obj, dict):
        assert "ticket_id" not in obj
        for value in obj.values():
            _assert_no_ticket_id(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_ticket_id(item)


def test_basic_handoff_transfers_ownership_to_billing():
    result = _run("invoice-duplicate-basic-handoff")
    assert result.ok is True
    assert result.metrics.termination_reason == "completed"
    assert result.metrics.handoffs_requested == 1
    assert result.metrics.handoffs_accepted == 1
    assert result.metrics.handoffs_rejected == 0
    assert result.metrics.ownership_transfers == 1
    assert result.state["invokedAgentIds"] == ["triage_agent", "billing_agent"]
    assert result.state["currentOwner"] == "billing_agent"
    assert result.state["previousOwner"] == "triage_agent"
    billing = result.state["results"][1]
    assert billing["kind"] == "complete"
    assert billing["payload"]["basedOn"]["invoice_id"] == "INV-1001"
    assert "INV-1001" in result.answer
    activated = [
        event.detail["agentId"]
        for event in result.sequence
        if event.kind == "agent_activated"
    ]
    assert activated == ["triage_agent", "billing_agent"]
    assert activated.count("triage_agent") == 1


def test_previous_owner_is_not_reactivated_after_handoff():
    result = _run("invoice-duplicate-basic-handoff")
    owners_after_transfer = [
        event.detail.get("currentOwner")
        for event in result.sequence
        if event.kind in {"agent_activated", "agent_result", "completed"}
    ]
    transfer_idx = next(
        i
        for i, event in enumerate(result.sequence)
        if event.kind == "ownership_transferred"
    )
    later = result.sequence[transfer_idx + 1 :]
    triage_later = [
        event
        for event in later
        if event.kind == "agent_result"
        and event.detail.get("agentId") == "triage_agent"
    ]
    assert triage_later == []
    assert result.sequence[transfer_idx].detail["currentOwner"] == "billing_agent"
    assert owners_after_transfer[-1] == "billing_agent"


def test_multi_hop_transfers_billing_then_technical():
    result = _run("refund-blocked-multi-hop")
    assert result.ok is True
    assert result.metrics.termination_reason == "completed"
    assert result.metrics.handoffs_accepted == 2
    assert result.metrics.ownership_transfers == 2
    assert result.state["invokedAgentIds"] == [
        "triage_agent",
        "billing_agent",
        "technical_agent",
    ]
    assert result.state["currentOwner"] == "technical_agent"
    assert result.state["previousOwner"] == "billing_agent"
    billing = result.state["results"][1]
    technical = result.state["results"][2]
    assert billing["kind"] == "handoff"
    assert technical["kind"] == "complete"
    assert technical["payload"]["basedOn"] == billing["handoff"]["payload"]
    _assert_no_ticket_id(technical["payload"])
    _assert_no_ticket_id(billing["handoff"]["payload"])
    _assert_no_ticket_id(billing["payload"])
    transfers = [
        event for event in result.sequence if event.kind == "ownership_transferred"
    ]
    assert [(event.detail["from"], event.detail["to"]) for event in transfers] == [
        ("triage_agent", "billing_agent"),
        ("billing_agent", "technical_agent"),
    ]
    assert "CHK-4401" in result.answer


def test_invalid_target_is_rejected_without_reroute():
    result = _run("legal-target-invalid-handoff")
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_handoff"
    assert result.metrics.handoffs_requested == 1
    assert result.metrics.handoffs_accepted == 0
    assert result.metrics.handoffs_rejected == 1
    assert result.metrics.ownership_transfers == 0
    assert result.state["currentOwner"] == "triage_agent"
    assert result.state["previousOwner"] is None
    assert result.state["invokedAgentIds"] == ["triage_agent"]
    assert "billing_agent" not in result.state["invokedAgentIds"]
    assert "technical_agent" not in result.state["invokedAgentIds"]
    rejection = result.state["rejections"][0]
    assert rejection["to_agent"] == "legal_agent"
    assert rejection["code"] == "unknown_target"
    kinds = _kinds(result)
    assert "handoff_requested" in kinds
    assert "handoff_rejected" in kinds
    assert "handoff_accepted" not in kinds
    assert "ownership_transferred" not in kinds
    assert kinds.index("handoff_requested") < kinds.index("handoff_rejected")
    assert kinds[-1] == "termination"
    assert "legal_agent" in result.answer
    assert result.errors[0]["code"] == "unknown_target"


def test_handoff_failure_stays_a_failure():
    result = _run("unknown-system-handoff-failure")
    assert result.ok is False
    assert result.metrics.termination_reason == "agent_failed"
    assert result.metrics.failed_agent_results == 1
    assert result.metrics.handoffs_accepted == 2
    assert result.state["currentOwner"] == "technical_agent"
    technical = result.state["results"][2]
    assert technical["ok"] is False
    assert technical["kind"] == "failure"
    assert technical["error"]["code"] == "unknown_system"
    kinds = _kinds(result)
    assert "failure" in kinds
    assert "completed" not in kinds
    assert kinds.index("failure") < kinds.index("termination")
    assert kinds[-1] == "termination"
    assert "not treated as success" in result.answer.lower()
    assert result.errors


def test_empty_request_is_invalid_task():
    result = run_handoffs(
        MeasuredCase(
            trace_id="empty-request",
            example_class="BASIC_HANDOFF",
            request="   ",
            ticket_id="tkt-duplicate-charge",
            selection_note="Test-only empty request.",
        ),
        catalog=Catalog(DATA),
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_task"
    assert result.metrics.agents_activated == 0
    assert result.sequence[-1].kind == "termination"


def test_self_handoff_is_rejected():
    class SelfHandoffTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload={},
                handoff=HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent="triage_agent",
                    reason="self",
                    payload={},
                ),
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = SelfHandoffTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_handoff"
    assert result.state["rejections"][0]["code"] == "self_handoff"
    assert result.state["currentOwner"] == "triage_agent"
    assert result.metrics.ownership_transfers == 0


def test_spoofed_from_agent_is_rejected():
    class SpoofingTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload={"invoice_id": "INV-1001"},
                handoff=HandoffRequest(
                    from_agent="billing_agent",
                    to_agent="technical_agent",
                    reason="spoofed owner",
                    payload={"invoice_id": "INV-1001"},
                ),
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = SpoofingTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_handoff"
    assert result.state["rejections"][0]["code"] == "not_current_owner"
    assert result.state["currentOwner"] == "triage_agent"
    assert result.metrics.ownership_transfers == 0
    assert "technical_agent" not in result.state["invokedAgentIds"]


def test_empty_target_is_rejected():
    class EmptyTargetTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload={},
                handoff=HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent="   ",
                    reason="empty",
                    payload={},
                ),
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = EmptyTargetTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.state["rejections"][0]["code"] == "empty_target"
    assert result.metrics.ownership_transfers == 0


def test_empty_from_agent_is_rejected_not_repaired():
    class EmptyFromTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload={"invoice_id": "INV-1001"},
                handoff=HandoffRequest(
                    from_agent="   ",
                    to_agent="billing_agent",
                    reason="empty from",
                    payload={"invoice_id": "INV-1001"},
                ),
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = EmptyFromTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "invalid_handoff"
    assert result.state["rejections"][0]["code"] == "not_current_owner"
    assert result.state["rejections"][0]["from_agent"] == ""
    assert result.state["currentOwner"] == "triage_agent"
    assert result.metrics.ownership_transfers == 0
    assert "billing_agent" not in result.state["invokedAgentIds"]


def test_max_handoffs_stops_additional_transfers():
    result = _run("refund-blocked-multi-hop", max_handoffs=1)
    assert result.metrics.termination_reason == "max_handoffs"
    assert result.ok is False
    assert result.metrics.handoffs_accepted == 1
    assert result.metrics.handoffs_rejected == 1
    assert result.state["rejections"][0]["code"] == "max_handoffs"
    assert result.state["invokedAgentIds"] == ["triage_agent", "billing_agent"]
    assert "technical_agent" not in result.state["invokedAgentIds"]
    assert result.state["currentOwner"] == "billing_agent"
    kinds = _kinds(result)
    assert "handoff_rejected" in kinds
    assert kinds.count("error") == 0
    assert kinds.index("handoff_requested") < kinds.index("handoff_rejected")


def test_padded_handoff_target_is_normalized():
    class PaddedTargetTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload={"invoice_id": "INV-1001"},
                handoff=HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent="  billing_agent  ",
                    reason="padded target",
                    payload={"invoice_id": "INV-1001"},
                ),
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = PaddedTargetTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is True
    assert result.state["currentOwner"] == "billing_agent"
    assert result.state["acceptedHandoffs"][0]["to_agent"] == "billing_agent"
    requested = next(
        event for event in result.sequence if event.kind == "handoff_requested"
    )
    assert requested.detail["to"] == "billing_agent"


def test_result_agent_id_must_match_current_owner():
    class MismatchedTriage:
        agent_id = "triage_agent"
        role = "triage"

        def handle(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                result_id="",
                agent_id="billing_agent",
                role="billing",
                kind="complete",
                ok=True,
                payload={"summary": "spoofed completion"},
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = MismatchedTriage()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is False
    assert result.metrics.termination_reason == "error"
    assert result.errors[0]["code"] == "owner_mismatch"
    assert result.errors[0]["agentId"] == "billing_agent"
    assert result.errors[0]["currentOwner"] == "triage_agent"
    assert result.state["currentOwner"] == "triage_agent"
    assert result.metrics.ownership_transfers == 0
    assert "completed" not in _kinds(result)
    activated = [
        event.detail["agentId"]
        for event in result.sequence
        if event.kind == "agent_activated"
    ]
    assert activated == ["triage_agent"]


def test_downstream_context_omits_ticket_and_request():
    captured: list[AgentContext] = []

    class CapturingBilling:
        agent_id = "billing_agent"
        role = "billing"

        def handle(self, context: AgentContext) -> AgentResult:
            captured.append(context.model_copy(deep=True))
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="complete",
                ok=True,
                payload={"summary": "captured"},
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["billing_agent"] = CapturingBilling()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is True
    assert len(captured) == 1
    assert captured[0].ticket_id is None
    assert captured[0].request is None
    assert captured[0].inbound_handoff is not None
    assert "ticket_id" not in captured[0].inbound_handoff.payload
    _assert_no_ticket_id(captured[0].inbound_handoff.payload)
    _assert_no_ticket_id(captured[0].inbound_handoff.context)
    assert captured[0].inbound_handoff.payload["invoice_id"] == "INV-1001"


def test_multi_hop_downstream_contexts_omit_ticket_id():
    captured: dict[str, list[AgentContext]] = {
        "billing_agent": [],
        "technical_agent": [],
    }

    class Capturing:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.agent_id = inner.agent_id
            self.role = inner.role

        def handle(self, context: AgentContext) -> AgentResult:
            captured[self.agent_id].append(context.model_copy(deep=True))
            return self._inner.handle(context)

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["billing_agent"] = Capturing(agents["billing_agent"])
    agents["technical_agent"] = Capturing(agents["technical_agent"])
    result = run_handoffs(
        get_case("refund-blocked-multi-hop"),
        catalog=catalog,
        agents=agents,
    )
    assert result.ok is True
    assert len(captured["billing_agent"]) == 1
    assert len(captured["technical_agent"]) == 1
    for context in (*captured["billing_agent"], *captured["technical_agent"]):
        assert context.ticket_id is None
        assert context.request is None
        assert context.inbound_handoff is not None
        assert "ticket_id" not in context.inbound_handoff.payload
        _assert_no_ticket_id(context.inbound_handoff.payload)
        _assert_no_ticket_id(context.inbound_handoff.context)
    technical = result.state["results"][2]
    assert technical["agent_id"] == "technical_agent"
    _assert_no_ticket_id(technical["payload"])


def test_mutating_agent_cannot_rewrite_recorded_handoff():
    class MutatingBillingAgent:
        agent_id = "billing_agent"
        role = "billing"

        def handle(self, context: AgentContext) -> AgentResult:
            inbound = context.inbound_handoff
            assert inbound is not None
            inbound.payload["tampered"] = True
            inbound.payload["invoice_id"] = "MUTATED-AFTER-RECEIPT"
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="complete",
                ok=True,
                payload={
                    "basedOn": inbound.payload,
                    "summary": "mutated",
                },
            )

    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["billing_agent"] = MutatingBillingAgent()
    result = run_handoffs(
        get_case("invoice-duplicate-basic-handoff"),
        catalog=catalog,
        agents=agents,
    )
    recorded = result.state["acceptedHandoffs"][0]
    assert recorded["payload"].get("tampered") is None
    assert recorded["payload"]["invoice_id"] == "INV-1001"
    requested = next(
        event for event in result.sequence if event.kind == "handoff_requested"
    )
    assert "tampered" not in requested.detail["payload"]
    assert requested.detail["payload"]["invoice_id"] == "INV-1001"


def test_parent_result_relationship_on_handoff():
    result = _run("invoice-duplicate-basic-handoff")
    triage = result.state["results"][0]
    handoff = result.state["acceptedHandoffs"][0]
    assert handoff["parent_result_id"] == triage["result_id"]
    assert handoff["from_agent"] == "triage_agent"
    assert handoff["to_agent"] == "billing_agent"
    requested = next(
        event for event in result.sequence if event.kind == "handoff_requested"
    )
    assert requested.detail["parentResultId"] == triage["result_id"]


def test_repeated_runs_are_semantically_stable():
    first = _run("refund-blocked-multi-hop")
    second = _run("refund-blocked-multi-hop")
    assert _kinds(first) == _kinds(second)
    assert first.state["results"][2]["payload"] == second.state["results"][2]["payload"]
    assert first.answer == second.answer
    assert first.state["currentOwner"] == second.state["currentOwner"]


def test_event_order_starts_with_task_and_ends_with_termination():
    result = _run("invoice-duplicate-basic-handoff")
    assert result.sequence[0].kind == "task_created"
    assert result.sequence[-1].kind == "termination"
    assert _kinds(result).count("termination") == 1


def test_main_live_requires_api_key(monkeypatch):
    from main import main

    monkeypatch.setattr(
        "main.get_settings",
        lambda: type(
            "S",
            (),
            {"openai_api_key": "", "data_dir": DATA, "max_handoffs": 6},
        )(),
    )
    assert main(["--case", "invoice-duplicate-basic-handoff", "--live"]) == 1
