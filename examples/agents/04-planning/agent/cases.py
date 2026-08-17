"""Measured cases for the agent-planning example.

Case harness turns are predetermined so sequences are reproducible. Tool
execution is real against local fixtures; timings are measured. The harness
proposes plans and revisions; the runtime owns plan state and execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.model import ScriptedModelClient, ScriptedTurn

SIMPLE_PLAN_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-1",
        "description": "Check billing service status",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "billing"},
    },
    {
        "id": "step-2",
        "description": "Inspect billing documentation",
        "action_kind": "tool_call",
        "intent": "docs_lookup",
        "tool": "search_documentation",
        "arguments": {"query": "billing operations"},
    },
    {
        "id": "step-3",
        "description": "Summarize findings",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]

PLAN_EXECUTION_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-1",
        "description": "Check payments service status",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "payments"},
    },
    {
        "id": "step-2",
        "description": "Inspect payments documentation",
        "action_kind": "tool_call",
        "intent": "docs_lookup",
        "tool": "search_documentation",
        "arguments": {"query": "payments status"},
    },
    {
        "id": "step-3",
        "description": "Check related billing service status",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "billing"},
    },
    {
        "id": "step-4",
        "description": "Summarize investigation progress",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]

PLAN_REVISION_INITIAL_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-1",
        "description": "Check payments service status",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "payments"},
    },
    {
        "id": "step-2",
        "description": "Inspect payments remediation documentation",
        "action_kind": "tool_call",
        "intent": "remediation",
        "tool": "search_documentation",
        "arguments": {"query": "payments remediation"},
    },
    {
        "id": "step-3",
        "description": "Recommend remediation",
        "action_kind": "finalize",
        "intent": "remediation",
    },
]

PLAN_REVISION_REVISED_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-2-revised",
        "description": "Check recent payments deployment information",
        "action_kind": "tool_call",
        "intent": "docs_lookup",
        "tool": "search_documentation",
        "arguments": {"query": "payments recent deployment"},
    },
    {
        "id": "step-3-revised",
        "description": "Summarize current status",
        "action_kind": "finalize",
        "intent": "summarize",
    },
]

PLAN_FAILURE_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-1",
        "description": "Check auth service status",
        "action_kind": "tool_call",
        "intent": "status_check",
        "tool": "get_service_status",
        "arguments": {"service": "auth"},
    },
    {
        "id": "step-2",
        "description": "Find the AUTH-881 incident runbook",
        "action_kind": "tool_call",
        "intent": "required_docs",
        "tool": "search_documentation",
        "arguments": {"query": "AUTH-881 incident runbook"},
        "requires_doc_id": "doc-auth-881-runbook",
    },
    {
        "id": "step-3",
        "description": "Recommend remediation",
        "action_kind": "finalize",
        "intent": "remediation",
    },
]


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    selection_note: str
    turns: tuple[ScriptedTurn, ...]
    max_turns: int | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="simple-plan-billing-docs",
        example_class="SIMPLE_PLAN",
        request=(
            "Investigate the billing service and summarize current status from "
            "service data and documentation."
        ),
        selection_note=(
            "Measured case: a three-step plan is created, then executed in "
            "order. All steps complete with zero revisions. Shows that "
            "planning makes the intended sequence of work explicit before "
            "execution."
        ),
        turns=(
            ScriptedTurn(
                content="plan proposed with 3 steps",
                decision="create_plan",
                proposed_steps=SIMPLE_PLAN_STEPS,
            ),
            ScriptedTurn(
                content=(
                    "Billing is operational in us-east-1 (p99 latency 120ms, "
                    "no active incident). Per the billing operations overview, "
                    "invoice generation and dunning run on schedule."
                ),
                decision="final_answer",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="plan-execution-payments-progress",
        example_class="PLAN_EXECUTION",
        request=(
            "Investigate the payments service: check status, inspect "
            "documentation, then check the related billing service before "
            "summarizing."
        ),
        selection_note=(
            "Measured case: a four-step plan executes step-by-step while the "
            "runtime records pending, in_progress, completed, and remaining "
            "state. Shows that a plan is useful only when execution state "
            "can be tracked against it."
        ),
        turns=(
            ScriptedTurn(
                content="plan proposed with 4 steps",
                decision="create_plan",
                proposed_steps=PLAN_EXECUTION_STEPS,
            ),
            ScriptedTurn(
                content=(
                    "Payments is operational in us-east-1 (p99 latency 95ms, "
                    "no active incident). Payments current status notes confirm "
                    "checkout should complete at normal card-auth latency. "
                    "Billing, the adjacent financial service, is also "
                    "operational in us-east-1."
                ),
                decision="final_answer",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="plan-revision-payments-healthy",
        example_class="PLAN_REVISION",
        request=(
            "Investigate the payments service and determine what should be "
            "checked before recommending remediation."
        ),
        selection_note=(
            "Measured case: the initial plan assumes a remediation path. "
            "Payments is operational, which invalidates the remaining "
            "remediation steps. The runtime records plan v1, the invalidating "
            "observation, and plan v2, then continues. Shows that a plan is "
            "adaptive, not a fixed script."
        ),
        turns=(
            ScriptedTurn(
                content="plan proposed with 3 steps",
                decision="create_plan",
                proposed_steps=PLAN_REVISION_INITIAL_STEPS,
            ),
            ScriptedTurn(
                content="revised remaining plan proposed with 2 steps",
                decision="revise_plan",
                proposed_steps=PLAN_REVISION_REVISED_STEPS,
            ),
            ScriptedTurn(
                content=(
                    "Payments is operational in us-east-1 with no active "
                    "incident, so the remediation path is not appropriate. "
                    "The latest payments deployment was a configuration "
                    "rollback with no open incident. Current status: healthy; "
                    "no remediation recommended."
                ),
                decision="final_answer",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="plan-failure-auth-runbook-missing",
        example_class="PLAN_FAILURE",
        request=(
            "Investigate the auth outage and find the AUTH-881 incident "
            "runbook before recommending remediation."
        ),
        selection_note=(
            "Measured case: auth status is gathered, but the required "
            "AUTH-881 incident runbook is unavailable. The remaining "
            "remediation step cannot run. The plan status is failed and "
            "termination is plan_failed. Shows that a planning system must "
            "stop without claiming an incomplete plan succeeded."
        ),
        turns=(
            ScriptedTurn(
                content="plan proposed with 3 steps",
                decision="create_plan",
                proposed_steps=PLAN_FAILURE_STEPS,
            ),
            ScriptedTurn(
                content=(
                    "Auth is in major_outage in us-west-2 (AUTH-881: token "
                    "issuer unreachable). The required AUTH-881 incident "
                    "runbook was not found, so the remaining remediation step "
                    "cannot be completed. The plan failed; no remediation is "
                    "recommended without that documentation."
                ),
                decision="final_answer",
            ),
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")


def scripted_client_for(case: MeasuredCase) -> ScriptedModelClient:
    return ScriptedModelClient(list(case.turns), model_name="case-harness")
