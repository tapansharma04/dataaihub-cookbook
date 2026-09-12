"""Measured collaboration cases.

Coordinator routing is explicit and deterministic. Specialist agents still
execute for real against local catalogs or inbound messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.schemas import StepKind

PAYMENTS_RUNBOOK = "doc-payments-runbook"


@dataclass(frozen=True)
class CoordinatorStep:
    kind: StepKind
    agent_id: str | None = None
    assignment: dict[str, Any] = field(default_factory=dict)
    input_from_agent: str | None = None


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    service: str | None
    selection_note: str
    steps: tuple[CoordinatorStep, ...]
    stop_on_failure: bool = False


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="independent-status-and-docs",
        example_class="BASIC_COLLABORATION",
        request="Prepare an incident packet for the payments service.",
        service="payments",
        selection_note=(
            "Measured case: the coordinator delegates independent work to "
            "the status agent and the docs agent, then aggregates their "
            "actual results. Neither specialist receives the other's output."
        ),
        steps=(
            CoordinatorStep(
                kind="delegate",
                agent_id="status_agent",
                assignment={"service": "payments"},
            ),
            CoordinatorStep(
                kind="delegate",
                agent_id="docs_agent",
                assignment={"doc_id": PAYMENTS_RUNBOOK, "service": "payments"},
            ),
            CoordinatorStep(kind="aggregate"),
        ),
    ),
    MeasuredCase(
        trace_id="sequential-status-then-analysis",
        example_class="SEQUENTIAL_COLLABORATION",
        request="Analyze the payments degradation using the status agent's result.",
        service="payments",
        selection_note=(
            "Measured case: the coordinator sends the status agent's actual "
            "result to the analysis agent. Analysis does not reload the "
            "status catalog."
        ),
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
    MeasuredCase(
        trace_id="docs-agent-failure",
        example_class="AGENT_FAILURE",
        request="Prepare an incident packet, including a missing runbook.",
        service="payments",
        selection_note=(
            "Measured case: the docs agent fails on an unknown document. "
            "The coordinator records the failure, does not treat it as "
            "success, and stops remaining work."
        ),
        stop_on_failure=True,
        steps=(
            CoordinatorStep(
                kind="delegate",
                agent_id="status_agent",
                assignment={"service": "payments"},
            ),
            CoordinatorStep(
                kind="delegate",
                agent_id="docs_agent",
                assignment={"doc_id": "doc-does-not-exist", "service": "payments"},
            ),
            CoordinatorStep(
                kind="delegate",
                agent_id="analysis_agent",
                input_from_agent="status_agent",
            ),
            CoordinatorStep(kind="aggregate"),
        ),
    ),
    MeasuredCase(
        trace_id="operational-short-circuit",
        example_class="COLLABORATION_TERMINATION",
        request="Check billing and analyze only if an incident is open.",
        service="billing",
        selection_note=(
            "Measured case: billing is operational with no incident. The "
            "coordinator skips analysis and terminates without additional "
            "specialist work."
        ),
        steps=(
            CoordinatorStep(
                kind="delegate",
                agent_id="status_agent",
                assignment={"service": "billing"},
            ),
            CoordinatorStep(
                kind="delegate_if_incident",
                agent_id="analysis_agent",
                input_from_agent="status_agent",
            ),
            CoordinatorStep(kind="aggregate"),
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(item.trace_id for item in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
