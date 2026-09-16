"""Specialist agents. Each may complete, fail, or request a handoff.

Agents do not call one another. They return an AgentResult; the runtime
validates handoffs and transfers ownership. Each specialist receives only
the store it is allowed to read.
"""

from __future__ import annotations

import copy
from typing import Protocol

from agent.catalog import Catalog, InvoiceStore, SystemStore, TicketStore
from agent.schemas import AgentContext, AgentResult, AgentRole, HandoffRequest

DEPARTMENT_TARGETS: dict[str, str] = {
    "billing": "billing_agent",
    "technical": "technical_agent",
    "legal": "legal_agent",
}


class Agent(Protocol):
    agent_id: str
    role: AgentRole

    def handle(self, context: AgentContext) -> AgentResult: ...


class TriageAgent:
    """Classifies a ticket and transfers ownership. Never completes the ticket."""

    agent_id = "triage_agent"
    role: AgentRole = "triage"

    def __init__(self, tickets: TicketStore) -> None:
        self._tickets = tickets

    def handle(self, context: AgentContext) -> AgentResult:
        ticket_id = (context.ticket_id or "").strip()
        if not ticket_id:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                error={
                    "code": "missing_ticket_id",
                    "message": "triage_agent requires context.ticket_id",
                },
            )
        ticket = self._tickets.get_ticket(ticket_id)
        if ticket is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                error={
                    "code": "unknown_ticket",
                    "message": f"Unknown ticket '{ticket_id}'",
                    "validTicketIds": self._tickets.ids,
                },
            )
        department = str(ticket.get("department") or "").strip()
        target = DEPARTMENT_TARGETS.get(department)
        if target is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                error={
                    "code": "unroutable_department",
                    "message": f"No handoff target for department '{department}'",
                    "department": department,
                },
            )
        payload = {
            "department": department,
            "account_id": ticket.get("account_id"),
            "invoice_id": ticket.get("invoice_id"),
            "subject": ticket.get("subject"),
        }
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            kind="handoff",
            ok=True,
            payload=copy.deepcopy(payload),
            handoff=HandoffRequest(
                from_agent=self.agent_id,
                to_agent=target,
                reason=(
                    f"Ticket classified as {department}; "
                    f"transferring ownership to {target}."
                ),
                context={"department": department},
                payload=copy.deepcopy(payload),
            ),
        )


class BillingAgent:
    """Resolves invoice work from the inbound handoff payload.

    This agent must not read tickets or parse the original request for
    invoice identity. Invoice identity comes from the handoff.
    """

    agent_id = "billing_agent"
    role: AgentRole = "billing"

    def __init__(self, invoices: InvoiceStore) -> None:
        self._invoices = invoices

    def handle(self, context: AgentContext) -> AgentResult:
        inbound = context.inbound_handoff
        if inbound is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                error={
                    "code": "missing_handoff",
                    "message": "billing_agent requires inbound_handoff",
                },
            )
        based_on = copy.deepcopy(inbound.payload)
        invoice_id = str(based_on.get("invoice_id") or "").strip()
        if not invoice_id:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                payload={"basedOn": based_on},
                error={
                    "code": "missing_invoice_id",
                    "message": "billing_agent requires payload.invoice_id",
                },
            )
        record = self._invoices.get_invoice(invoice_id)
        if record is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                payload={
                    "basedOn": based_on,
                    "invoice_id": invoice_id,
                },
                error={
                    "code": "unknown_invoice",
                    "message": f"Unknown invoice '{invoice_id}'",
                    "invoiceId": invoice_id,
                    "validInvoiceIds": self._invoices.ids,
                },
            )
        if record.get("requires_technical"):
            payload = {
                "basedOn": based_on,
                "invoice": record,
                "invoice_id": record["invoice_id"],
                "system_id": record.get("system_id"),
                "blocker": record.get("blocker"),
            }
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="handoff",
                ok=True,
                payload=copy.deepcopy(payload),
                handoff=HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent="technical_agent",
                    reason=(
                        "Invoice refund is blocked by a system incident; "
                        "transferring ownership to technical_agent."
                    ),
                    context={
                        "invoice_id": record["invoice_id"],
                        "system_id": record.get("system_id"),
                    },
                    payload=copy.deepcopy(payload),
                ),
            )
        amount = record.get("amount")
        currency = record.get("currency") or "USD"
        summary = (
            f"Invoice {record['invoice_id']} has a {record['status']}. "
            f"Issue a refund of {amount} {currency}."
        )
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            kind="complete",
            ok=True,
            payload={
                "basedOn": based_on,
                "invoice": record,
                "resolution": "issue_refund",
                "summary": summary,
            },
        )


class TechnicalAgent:
    """Diagnoses a system incident from the inbound handoff payload.

    This agent must not read tickets or invoices. System identity comes
    from the billing handoff payload.
    """

    agent_id = "technical_agent"
    role: AgentRole = "technical"

    def __init__(self, systems: SystemStore) -> None:
        self._systems = systems

    def handle(self, context: AgentContext) -> AgentResult:
        inbound = context.inbound_handoff
        if inbound is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                error={
                    "code": "missing_handoff",
                    "message": "technical_agent requires inbound_handoff",
                },
            )
        based_on = copy.deepcopy(inbound.payload)
        system_id = str(based_on.get("system_id") or "").strip()
        if not system_id:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                payload={"basedOn": based_on},
                error={
                    "code": "missing_system_id",
                    "message": "technical_agent requires payload.system_id",
                },
            )
        record = self._systems.get_system(system_id)
        if record is None:
            return AgentResult(
                result_id="",
                agent_id=self.agent_id,
                role=self.role,
                kind="failure",
                ok=False,
                payload={
                    "basedOn": based_on,
                    "system_id": system_id,
                },
                error={
                    "code": "unknown_system",
                    "message": f"Unknown system '{system_id}'",
                    "systemId": system_id,
                    "validSystemIds": self._systems.ids,
                },
            )
        summary = (
            f"System {record['system_id']} is {record['status']}. "
            f"Incident: {record.get('incident') or 'none'}. "
            f"Action: {record.get('action')}."
        )
        return AgentResult(
            result_id="",
            agent_id=self.agent_id,
            role=self.role,
            kind="complete",
            ok=True,
            payload={
                "basedOn": based_on,
                "system": record,
                "resolution": record.get("action"),
                "summary": summary,
            },
        )


def default_agents(catalog: Catalog) -> dict[str, Agent]:
    return {
        TriageAgent.agent_id: TriageAgent(catalog.ticket_store()),
        BillingAgent.agent_id: BillingAgent(catalog.invoice_store()),
        TechnicalAgent.agent_id: TechnicalAgent(catalog.system_store()),
    }
