"""Measured handoff cases.

Cases supply the ticket and request. Agents themselves request handoffs;
the runtime validates and transfers ownership. Routing is not a
coordinator script of delegate steps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    ticket_id: str | None
    selection_note: str
    initial_agent: str = "triage_agent"


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="invoice-duplicate-basic-handoff",
        example_class="BASIC_HANDOFF",
        request="Invoice INV-1001 was charged twice.",
        ticket_id="tkt-duplicate-charge",
        selection_note=(
            "Measured case: triage classifies a duplicate-charge ticket and "
            "transfers ownership to billing. Billing completes the ticket. "
            "Control does not return to triage."
        ),
    ),
    MeasuredCase(
        trace_id="refund-blocked-multi-hop",
        example_class="MULTI_HOP_HANDOFF",
        request=("Refund for INV-2002 is blocked because checkout capture is failing."),
        ticket_id="tkt-refund-blocked",
        selection_note=(
            "Measured case: triage hands off to billing, then billing hands "
            "off to technical because the refund is blocked by a system "
            "incident. Both ownership transfers are explicit."
        ),
    ),
    MeasuredCase(
        trace_id="legal-target-invalid-handoff",
        example_class="INVALID_HANDOFF",
        request="Forward this billing dispute to the legal department.",
        ticket_id="tkt-legal-dispute",
        selection_note=(
            "Measured case: triage requests a handoff to legal_agent, which "
            "is not a registered owner. The runtime rejects the target and "
            "does not silently reroute."
        ),
    ),
    MeasuredCase(
        trace_id="unknown-system-handoff-failure",
        example_class="HANDOFF_FAILURE",
        request="Refund for INV-3003 is blocked by processor SYS-MISSING.",
        ticket_id="tkt-unknown-processor",
        selection_note=(
            "Measured case: triage hands off to billing, billing hands off "
            "to technical, and technical fails on an unknown system. The "
            "failure remains a failure."
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(item.trace_id for item in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
