"""Data-flow tests: downstream owners consume the actual handoff payload."""

from __future__ import annotations

from pathlib import Path

from agent.agents import BillingAgent, TechnicalAgent, default_agents
from agent.cases import MeasuredCase, get_case
from agent.catalog import Catalog
from agent.runtime import run_handoffs
from agent.schemas import AgentContext, AgentResult, HandoffRequest

DATA = Path(__file__).resolve().parents[1] / "data"
SENTINEL_INVOICE = "SENTINEL-INVOICE-NOT-IN-FIXTURE"
SENTINEL_SYSTEM = "SENTINEL-SYSTEM-NOT-IN-FIXTURE"
SENTINEL_NOTE = "SENTINEL-NOTE-NOT-IN-FIXTURE"


class SentinelTriageAgent:
    agent_id = "triage_agent"
    role = "triage"

    def handle(self, context: AgentContext) -> AgentResult:
        payload = {
            "department": "billing",
            "invoice_id": SENTINEL_INVOICE,
            "note": SENTINEL_NOTE,
        }
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            kind="handoff",
            ok=True,
            payload=payload,
            handoff=HandoffRequest(
                from_agent=self.agent_id,
                to_agent="billing_agent",
                reason="Sentinel classification.",
                payload=payload,
            ),
        )


class SentinelBillingAgent:
    agent_id = "billing_agent"
    role = "billing"

    def handle(self, context: AgentContext) -> AgentResult:
        inbound = context.inbound_handoff
        assert inbound is not None
        payload = {
            "basedOn": inbound.payload,
            "invoice_id": inbound.payload.get("invoice_id"),
            "system_id": SENTINEL_SYSTEM,
            "note": SENTINEL_NOTE,
        }
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            kind="handoff",
            ok=True,
            payload=payload,
            handoff=HandoffRequest(
                from_agent=self.agent_id,
                to_agent="technical_agent",
                reason="Sentinel technical handoff.",
                payload=payload,
            ),
        )


def test_billing_uses_injected_handoff_invoice_not_fixture():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["triage_agent"] = SentinelTriageAgent()
    result = run_handoffs(
        MeasuredCase(
            trace_id="sentinel-basic",
            example_class="BASIC_HANDOFF",
            request="Invoice INV-1001 was charged twice.",
            ticket_id="tkt-duplicate-charge",
            selection_note="Test-only sentinel data flow.",
        ),
        catalog=catalog,
        agents=agents,
    )
    billing = result.state["results"][1]
    assert billing["agent_id"] == "billing_agent"
    assert billing["payload"]["basedOn"]["invoice_id"] == SENTINEL_INVOICE
    assert billing["payload"]["basedOn"]["note"] == SENTINEL_NOTE
    assert billing["kind"] == "failure"
    assert billing["error"]["code"] == "unknown_invoice"
    assert SENTINEL_INVOICE in billing["error"]["message"]
    assert "INV-1001" not in billing["payload"]["basedOn"]["invoice_id"]
    assert SENTINEL_INVOICE in result.answer
    assert "duplicate_charge" not in result.answer
    fixture_invoice = catalog.get_invoice("INV-1001")
    assert fixture_invoice is not None
    assert fixture_invoice["invoice_id"] != SENTINEL_INVOICE


def test_technical_uses_injected_system_id_not_fixture():
    catalog = Catalog(DATA)
    agents = default_agents(catalog)
    agents["billing_agent"] = SentinelBillingAgent()
    result = run_handoffs(
        get_case("refund-blocked-multi-hop"),
        catalog=catalog,
        agents=agents,
    )
    technical = result.state["results"][2]
    assert technical["agent_id"] == "technical_agent"
    assert technical["payload"]["basedOn"]["system_id"] == SENTINEL_SYSTEM
    assert technical["payload"]["basedOn"]["note"] == SENTINEL_NOTE
    assert technical["kind"] == "failure"
    assert technical["error"]["code"] == "unknown_system"
    assert SENTINEL_SYSTEM in technical["error"]["message"]
    assert "CHK-4401" not in result.answer
    assert "checkout" not in (technical["payload"].get("system") or {})
    fixture = catalog.get_system("checkout")
    assert fixture is not None
    assert fixture["incident"] not in result.answer


def test_billing_direct_handle_ignores_ticket_fixture():
    agent = BillingAgent(Catalog(DATA).invoice_store())
    result = agent.handle(
        AgentContext(
            task_id="t",
            request="Invoice INV-1001 was charged twice.",
            ticket_id="tkt-duplicate-charge",
            current_owner="billing_agent",
            inbound_handoff=HandoffRequest(
                from_agent="triage_agent",
                to_agent="billing_agent",
                reason="sentinel",
                payload={"invoice_id": SENTINEL_INVOICE, "note": SENTINEL_NOTE},
            ),
        )
    )
    fixture_invoice = Catalog(DATA).get_invoice("INV-1001")
    assert fixture_invoice is not None
    assert fixture_invoice["invoice_id"] != SENTINEL_INVOICE
    assert result.payload["basedOn"]["invoice_id"] == SENTINEL_INVOICE
    assert result.payload["basedOn"]["note"] == SENTINEL_NOTE
    assert result.error is not None
    assert result.error["invoiceId"] == SENTINEL_INVOICE


def test_technical_direct_handle_ignores_invoice_fixture():
    agent = TechnicalAgent(Catalog(DATA).system_store())
    result = agent.handle(
        AgentContext(
            task_id="t",
            request=(
                "Refund for INV-2002 is blocked because checkout capture is failing."
            ),
            ticket_id="tkt-refund-blocked",
            current_owner="technical_agent",
            inbound_handoff=HandoffRequest(
                from_agent="billing_agent",
                to_agent="technical_agent",
                reason="sentinel",
                payload={"system_id": SENTINEL_SYSTEM, "invoice_id": "INV-2002"},
            ),
        )
    )
    fixture = Catalog(DATA).get_system("checkout")
    assert fixture is not None
    assert result.payload["basedOn"]["system_id"] == SENTINEL_SYSTEM
    assert result.error is not None
    assert result.error["systemId"] == SENTINEL_SYSTEM
    assert fixture["incident"] not in str(result.payload)


def test_live_synthesizer_sends_terminating_owner_result(monkeypatch):
    import json

    from agent.state import HandoffState
    from agent.synthesizer import LiveSynthesizer
    from config import Settings

    captured: dict = {}

    class FakeMessage:
        content = "note from terminating owner result"

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
    state = HandoffState(
        task_id="t",
        request="sentinel brief",
        ticket_id="tkt-duplicate-charge",
        current_owner="billing_agent",
        termination_reason="completed",
    )
    state.record_result(
        AgentResult(
            result_id="result-1",
            agent_id="billing_agent",
            role="billing",
            kind="complete",
            ok=True,
            payload={"summary": "refund", "note": SENTINEL_NOTE},
        )
    )
    text, _latency = LiveSynthesizer(
        Settings(openai_api_key="sk-test", data_dir=DATA)
    ).synthesize(state)
    user_payload = json.loads(captured["messages"][1]["content"])
    assert text == "note from terminating owner result"
    assert user_payload["lastResult"]["payload"]["note"] == SENTINEL_NOTE
    assert user_payload["currentOwner"] == "billing_agent"
    raw = captured["messages"][1]["content"]
    assert SENTINEL_NOTE in raw
    assert "INV-1001" not in raw
    assert "invoices.json" not in raw
