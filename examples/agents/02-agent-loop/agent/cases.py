"""Measured teaching cases for the agent-loop example.

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
    max_turns: int | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="simple-loop-payments-docs",
        example_class="SIMPLE_LOOP",
        request=(
            "Check the payments service. If it is degraded, inspect the relevant "
            "documentation and summarize what the user should know."
        ),
        selection_note=(
            "Measured case: at least two loop iterations — status check, then "
            "documentation lookup, then final answer. Teaches persistent state "
            "across model → action → observation cycles."
        ),
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_payments_status",
                        "name": "get_service_status",
                        "arguments": {"service": "payments"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_payments_docs",
                        "name": "search_documentation",
                        "arguments": {"query": "payments degradation"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=(
                    "Payments is degraded in us-east-1 (PAY-2041: elevated "
                    "card-auth latency). Per the payments degradation runbook, "
                    "enterprise operators may see slower checkout confirmations."
                ),
                finish_reason="stop",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="termination-after-status",
        example_class="TERMINATION",
        request="What is the current status of the billing service?",
        selection_note=(
            "Measured case: one useful tool observation is enough. The loop "
            "terminates with reason final_answer as soon as the model decides "
            "the task is complete — no extra iterations."
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
        trace_id="max-turns-safety-boundary",
        example_class="MAX_TURNS",
        request=(
            "Keep investigating the payments service until you are completely sure."
        ),
        selection_note=(
            "Measured case: the harness keeps requesting another tool action. "
            "The runtime stops at max_turns. Teaches that an agent runtime needs "
            "an explicit loop boundary — the model does not enforce it."
        ),
        max_turns=3,
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_loop_1",
                        "name": "get_service_status",
                        "arguments": {"service": "payments"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_loop_2",
                        "name": "search_documentation",
                        "arguments": {"query": "payments degradation"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_loop_3",
                        "name": "get_service_status",
                        "arguments": {"service": "payments"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            # Extra scripted turn would never run: runtime stops at max_turns=3.
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_loop_4_unused",
                        "name": "get_service_status",
                        "arguments": {"service": "auth"},
                    }
                ],
                finish_reason="tool_calls",
            ),
        ),
    ),
    MeasuredCase(
        trace_id="invalid-action-rejected",
        example_class="INVALID_ACTION",
        request="Check payments and keep going forever if needed.",
        selection_note=(
            "Measured case: the harness emits an unrecognized action kind. The "
            "runtime rejects it and terminates with reason invalid_action — "
            "without allowing uncontrolled execution. Distinct from tool-level "
            "error recovery in the tool-calling example."
        ),
        turns=(
            ScriptedTurn(
                content="I will keep working indefinitely without a concrete action.",
                tool_calls=None,
                decision="continue",  # unrecognized → invalid_action
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
