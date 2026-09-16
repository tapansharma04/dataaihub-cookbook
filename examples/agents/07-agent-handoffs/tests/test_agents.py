"""Specialist agent unit tests."""

from __future__ import annotations

from pathlib import Path

from agent.agents import BillingAgent, TechnicalAgent, TriageAgent
from agent.catalog import Catalog, InvoiceStore, SystemStore, TicketStore
from agent.schemas import AgentContext, HandoffRequest

DATA = Path(__file__).resolve().parents[1] / "data"


def _catalog() -> Catalog:
    return Catalog(DATA)


def _tickets() -> TicketStore:
    return _catalog().ticket_store()


def _invoices() -> InvoiceStore:
    return _catalog().invoice_store()


def _systems() -> SystemStore:
    return _catalog().system_store()


def _context(
    *,
    owner: str,
    ticket_id: str | None = None,
    inbound: HandoffRequest | None = None,
) -> AgentContext:
    return AgentContext(
        task_id="task-test",
        request="test request",
        ticket_id=ticket_id,
        current_owner=owner,
        inbound_handoff=inbound,
    )


def test_triage_hands_off_billing_duplicate_charge():
    agent = TriageAgent(_tickets())
    result = agent.handle(
        _context(owner="triage_agent", ticket_id="tkt-duplicate-charge")
    )
    assert result.kind == "handoff"
    assert result.ok is True
    assert result.handoff is not None
    assert result.handoff.from_agent == "triage_agent"
    assert result.handoff.to_agent == "billing_agent"
    assert result.handoff.payload["invoice_id"] == "INV-1001"
    assert "ticket_id" not in result.handoff.payload
    assert "ticket_id" not in result.payload
    assert "ticket_id" not in result.handoff.context


def test_triage_hands_off_unregistered_legal_target():
    agent = TriageAgent(_tickets())
    result = agent.handle(_context(owner="triage_agent", ticket_id="tkt-legal-dispute"))
    assert result.kind == "handoff"
    assert result.handoff is not None
    assert result.handoff.to_agent == "legal_agent"


def test_triage_unknown_ticket():
    agent = TriageAgent(_tickets())
    result = agent.handle(_context(owner="triage_agent", ticket_id="tkt-missing"))
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "unknown_ticket"


def test_triage_missing_ticket_id():
    agent = TriageAgent(_tickets())
    result = agent.handle(_context(owner="triage_agent"))
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "missing_ticket_id"


def test_billing_completes_duplicate_charge_from_handoff_payload():
    agent = BillingAgent(_invoices())
    result = agent.handle(
        _context(
            owner="billing_agent",
            inbound=HandoffRequest(
                from_agent="triage_agent",
                to_agent="billing_agent",
                reason="classified",
                payload={"invoice_id": "INV-1001", "department": "billing"},
            ),
        )
    )
    assert result.kind == "complete"
    assert result.payload["basedOn"]["invoice_id"] == "INV-1001"
    assert result.payload["invoice"]["status"] == "duplicate_charge"
    assert "INV-1001" in result.payload["summary"]


def test_billing_hands_off_when_invoice_requires_technical():
    agent = BillingAgent(_invoices())
    result = agent.handle(
        _context(
            owner="billing_agent",
            inbound=HandoffRequest(
                from_agent="triage_agent",
                to_agent="billing_agent",
                reason="classified",
                payload={"invoice_id": "INV-2002"},
            ),
        )
    )
    assert result.kind == "handoff"
    assert result.handoff is not None
    assert result.handoff.to_agent == "technical_agent"
    assert result.handoff.payload["system_id"] == "checkout"
    assert result.handoff.payload["basedOn"]["invoice_id"] == "INV-2002"


def test_billing_requires_inbound_handoff():
    agent = BillingAgent(_invoices())
    result = agent.handle(_context(owner="billing_agent"))
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "missing_handoff"


def test_billing_unknown_invoice_uses_payload_id():
    agent = BillingAgent(_invoices())
    result = agent.handle(
        _context(
            owner="billing_agent",
            inbound=HandoffRequest(
                from_agent="triage_agent",
                to_agent="billing_agent",
                reason="classified",
                payload={"invoice_id": "INV-MISSING"},
            ),
        )
    )
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "unknown_invoice"
    assert result.payload["invoice_id"] == "INV-MISSING"


def test_technical_completes_from_system_id_on_handoff():
    agent = TechnicalAgent(_systems())
    result = agent.handle(
        _context(
            owner="technical_agent",
            inbound=HandoffRequest(
                from_agent="billing_agent",
                to_agent="technical_agent",
                reason="blocked",
                payload={"system_id": "checkout", "invoice_id": "INV-2002"},
            ),
        )
    )
    assert result.kind == "complete"
    assert result.payload["basedOn"]["system_id"] == "checkout"
    assert result.payload["system"]["incident"] == (
        "CHK-4401: payment capture timeouts"
    )


def test_technical_unknown_system_is_failure():
    agent = TechnicalAgent(_systems())
    result = agent.handle(
        _context(
            owner="technical_agent",
            inbound=HandoffRequest(
                from_agent="billing_agent",
                to_agent="technical_agent",
                reason="blocked",
                payload={"system_id": "SYS-MISSING"},
            ),
        )
    )
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "unknown_system"
    assert result.payload["system_id"] == "SYS-MISSING"


def test_technical_requires_inbound_handoff():
    agent = TechnicalAgent(_systems())
    result = agent.handle(_context(owner="technical_agent"))
    assert result.kind == "failure"
    assert result.error is not None
    assert result.error["code"] == "missing_handoff"


def test_billing_source_does_not_load_ticket_catalog():
    source = (Path(__file__).resolve().parents[1] / "agent" / "agents.py").read_text(
        encoding="utf-8"
    )
    start = source.index("class BillingAgent")
    end = source.index("class TechnicalAgent")
    billing_src = source[start:end]
    assert "get_ticket" not in billing_src
    assert "tickets.json" not in billing_src
    assert "TicketStore" not in billing_src
    assert "Catalog" not in billing_src
    assert "context.request" not in billing_src
    assert "context.ticket_id" not in billing_src


def test_technical_source_does_not_load_ticket_or_invoice_catalogs():
    source = (Path(__file__).resolve().parents[1] / "agent" / "agents.py").read_text(
        encoding="utf-8"
    )
    start = source.index("class TechnicalAgent")
    end = source.index("def default_agents")
    technical_src = source[start:end]
    assert "get_ticket" not in technical_src
    assert "get_invoice" not in technical_src
    assert "tickets.json" not in technical_src
    assert "invoices.json" not in technical_src
    assert "TicketStore" not in technical_src
    assert "InvoiceStore" not in technical_src
    assert "Catalog" not in technical_src
    assert "context.request" not in technical_src
    assert "context.ticket_id" not in technical_src


def test_invoice_store_cannot_read_tickets_or_systems():
    store = _invoices()
    assert not hasattr(store, "get_ticket")
    assert not hasattr(store, "get_system")
    assert not hasattr(store, "tickets")
    assert store.get_invoice("INV-1001") is not None


def test_system_store_cannot_read_tickets_or_invoices():
    store = _systems()
    assert not hasattr(store, "get_ticket")
    assert not hasattr(store, "get_invoice")
    assert not hasattr(store, "invoices")
    assert store.get_system("checkout") is not None


def test_ticket_store_cannot_read_invoices_or_systems():
    store = _tickets()
    assert not hasattr(store, "get_invoice")
    assert not hasattr(store, "get_system")
    assert store.get_ticket("tkt-duplicate-charge") is not None
