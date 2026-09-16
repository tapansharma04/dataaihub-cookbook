"""Deterministic catalogs owned by specialist agents.

Triage receives only tickets, billing only invoices, technical only
systems. Downstream agents cannot reach another specialist's store.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class TicketStore:
    """Ticket records. No invoice or system access."""

    def __init__(self, tickets: dict[str, dict[str, Any]]) -> None:
        self._tickets = tickets

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        record = self._tickets.get(ticket_id)
        return copy.deepcopy(record) if record is not None else None

    @property
    def ids(self) -> list[str]:
        return sorted(self._tickets)


class InvoiceStore:
    """Invoice records. No ticket or system access."""

    def __init__(self, invoices: dict[str, dict[str, Any]]) -> None:
        self._invoices = invoices

    def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        record = self._invoices.get(invoice_id)
        return copy.deepcopy(record) if record is not None else None

    @property
    def ids(self) -> list[str]:
        return sorted(self._invoices)


class SystemStore:
    """System records. No ticket or invoice access."""

    def __init__(self, systems: dict[str, dict[str, Any]]) -> None:
        self._systems = systems

    def get_system(self, system_id: str) -> dict[str, Any] | None:
        record = self._systems.get(system_id)
        return copy.deepcopy(record) if record is not None else None

    @property
    def ids(self) -> list[str]:
        return sorted(self._systems)


class Catalog:
    def __init__(self, data_dir: Path) -> None:
        self.tickets: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "tickets.json").read_text(encoding="utf-8")
        )
        self.invoices: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "invoices.json").read_text(encoding="utf-8")
        )
        self.systems: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "systems.json").read_text(encoding="utf-8")
        )

    def ticket_store(self) -> TicketStore:
        return TicketStore(self.tickets)

    def invoice_store(self) -> InvoiceStore:
        return InvoiceStore(self.invoices)

    def system_store(self) -> SystemStore:
        return SystemStore(self.systems)

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return self.ticket_store().get_ticket(ticket_id)

    def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        return self.invoice_store().get_invoice(invoice_id)

    def get_system(self, system_id: str) -> dict[str, Any] | None:
        return self.system_store().get_system(system_id)
