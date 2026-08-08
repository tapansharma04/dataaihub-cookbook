"""Measured teaching cases for the tool-calling example.

Case harness turns are predetermined so sequences are reproducible. Tool
execution is real against local fixtures; timings are measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.model import ScriptedModelClient, ScriptedTurn


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    selection_note: str
    turns: tuple[ScriptedTurn, ...]


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="direct-answer",
        example_class="DIRECT_ANSWER",
        request=(
            "In one sentence, what is the difference between a calculator and "
            "a weather lookup when answering a user question?"
        ),
        selection_note=(
            "Measured case: model finalizes without tool calls. Demonstrates "
            "User → Model → Answer when no external action is required."
        ),
        turns=(
            ScriptedTurn(
                content=(
                    "A calculator can be answered from general knowledge, while "
                    "a weather lookup needs a live tool because conditions change."
                ),
                finish_reason="stop",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="single-tool-service-status",
        example_class="SINGLE_TOOL",
        request="What is the current status of the billing service?",
        selection_note=(
            "Measured case: one get_service_status call, then a final answer "
            "grounded in the tool observation."
        ),
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_billing_1",
                        "name": "get_service_status",
                        "arguments": {"service": "billing"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=(
                    "Billing is operational in us-east-1 "
                    "(p99 latency 120ms, no active incident)."
                ),
                finish_reason="stop",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="multi-step-user-and-payments",
        example_class="MULTI_STEP",
        request=(
            "For user u-1001, report their plan/region and whether the payments "
            "service issue might affect them. Use tools; do not guess."
        ),
        selection_note=(
            "Measured case: get_user_profile then get_service_status, then a "
            "final answer that combines both observations."
        ),
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_user_1",
                        "name": "get_user_profile",
                        "arguments": {"user_id": "u-1001"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_payments_1",
                        "name": "get_service_status",
                        "arguments": {"service": "payments"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=(
                    "User u-1001 (Ada Lovelace) is on the enterprise plan in "
                    "us-east-1. Payments is degraded there (PAY-2041: elevated "
                    "card-auth latency), so checkout confirmations may be slower "
                    "for this user."
                ),
                finish_reason="stop",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="recovery-invalid-service-name",
        example_class="ERROR_RECOVERY",
        request="Check status for billing-api and tell me if it is healthy.",
        selection_note=(
            "Measured case: first tool call uses invalid service name "
            "'billing-api' and fails; second call uses canonical 'billing' and "
            "succeeds. Demonstrates an error → corrected tool-call loop; the "
            "correction is supplied by the reproducible case harness — not a "
            "claim that a live model independently discovered it. Failure is "
            "preserved in the trace."
        ),
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_bad_1",
                        "name": "get_service_status",
                        "arguments": {"service": "billing-api"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_ok_1",
                        "name": "get_service_status",
                        "arguments": {"service": "billing"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=(
                    "The name 'billing-api' is invalid. Using canonical service "
                    "'billing', status is operational in us-east-1."
                ),
                finish_reason="stop",
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
