"""Measured orchestration cases.

Cases supply the inspectable workflow plan. The runtime calculates
readiness from dependencies and conditions; routing is not a script of
delegate or handoff steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAYMENTS_RUNBOOK = "doc-payments-runbook"


@dataclass(frozen=True)
class ConditionSpec:
    source_node_id: str
    field: str
    equals: Any


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    agent_id: str
    dependencies: tuple[str, ...] = ()
    condition: ConditionSpec | None = None
    assignment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    service: str | None
    selection_note: str
    nodes: tuple[NodeSpec, ...]


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="payments-incident-basic-orchestration",
        example_class="BASIC_ORCHESTRATION",
        request=(
            "Prepare an operational decision for the payments service "
            "using status and documentation."
        ),
        service="payments",
        selection_note=(
            "Measured case: status and docs have no dependency on each "
            "other and become READY independently. Analysis becomes READY "
            "only after both complete. Decision consumes the analysis result."
        ),
        nodes=(
            NodeSpec(
                node_id="status",
                agent_id="status_agent",
                assignment={"service": "payments"},
            ),
            NodeSpec(
                node_id="docs",
                agent_id="docs_agent",
                assignment={"doc_id": PAYMENTS_RUNBOOK},
            ),
            NodeSpec(
                node_id="analysis",
                agent_id="analysis_agent",
                dependencies=("status", "docs"),
            ),
            NodeSpec(
                node_id="decision",
                agent_id="decision_agent",
                dependencies=("analysis",),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="payments-status-dependency-chain",
        example_class="DEPENDENCY_CHAIN",
        request="Analyze payments status and decide the operational outcome.",
        service="payments",
        selection_note=(
            "Measured case: a strict chain status → analysis → decision. "
            "Analysis cannot execute before status completes. Decision "
            "cannot execute before analysis completes."
        ),
        nodes=(
            NodeSpec(
                node_id="status",
                agent_id="status_agent",
                assignment={"service": "payments"},
            ),
            NodeSpec(
                node_id="analysis",
                agent_id="analysis_agent",
                dependencies=("status",),
            ),
            NodeSpec(
                node_id="decision",
                agent_id="decision_agent",
                dependencies=("analysis",),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="payments-incident-conditional-branch",
        example_class="CONDITIONAL_BRANCH",
        request="Analyze payments and remediate only if an incident is open.",
        service="payments",
        selection_note=(
            "Measured case: after analysis, the runtime evaluates the "
            "incident condition from the actual analysis result. "
            "Remediation executes; no_action is explicitly SKIPPED."
        ),
        nodes=(
            NodeSpec(
                node_id="status",
                agent_id="status_agent",
                assignment={"service": "payments"},
            ),
            NodeSpec(
                node_id="analysis",
                agent_id="analysis_agent",
                dependencies=("status",),
            ),
            NodeSpec(
                node_id="remediation",
                agent_id="remediation_agent",
                dependencies=("analysis",),
                condition=ConditionSpec(
                    source_node_id="analysis",
                    field="incident",
                    equals=True,
                ),
            ),
            NodeSpec(
                node_id="no_action",
                agent_id="no_action_agent",
                dependencies=("analysis",),
                condition=ConditionSpec(
                    source_node_id="analysis",
                    field="incident",
                    equals=False,
                ),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="unknown-service-orchestration-failure",
        example_class="ORCHESTRATION_FAILURE",
        request="Analyze an unknown service and decide the operational outcome.",
        service="does-not-exist",
        selection_note=(
            "Measured case: status fails on an unknown service. Analysis "
            "and decision are SKIPPED because their required upstream "
            "results are missing. The workflow remains a failure."
        ),
        nodes=(
            NodeSpec(
                node_id="status",
                agent_id="status_agent",
                assignment={"service": "does-not-exist"},
            ),
            NodeSpec(
                node_id="analysis",
                agent_id="analysis_agent",
                dependencies=("status",),
            ),
            NodeSpec(
                node_id="decision",
                agent_id="decision_agent",
                dependencies=("analysis",),
            ),
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(item.trace_id for item in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
