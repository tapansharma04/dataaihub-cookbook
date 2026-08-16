"""Measured teaching cases for the agent-evaluation example.

Case harness turns are predetermined so sequences are reproducible. Tool
execution is real against local fixtures; timings are measured. Evaluation
criteria are explicit observable constraints, not hidden reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.model import ScriptedModelClient, ScriptedTurn
from evaluation.criteria import (
    PAYMENTS_INVESTIGATION_CRITERIA,
    RECOVERY_CRITERIA,
    CaseCriteria,
)

PAYMENTS_TASK = (
    "Check the payments service. If it is degraded, inspect the relevant "
    "documentation and summarize what the user should know."
)

SUCCESS_ANSWER = (
    "Payments is degraded in us-east-1 (PAY-2041: elevated card-auth latency). "
    "Per the payments degradation runbook, enterprise operators may see slower "
    "checkout confirmations."
)

GOAL_MISS_ANSWER = (
    "The payments service is operational in us-east-1. Users are unaffected "
    "and no further action is needed."
)


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    request: str
    selection_note: str
    turns: tuple[ScriptedTurn, ...]
    criteria: CaseCriteria
    max_turns: int | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="task-success-payments-docs",
        example_class="TASK_SUCCESS",
        request=PAYMENTS_TASK,
        selection_note=(
            "Measured case: status check, documentation lookup, then a final "
            "answer that uses both observations. Outcome and trajectory both "
            "satisfy the case criteria."
        ),
        criteria=PAYMENTS_INVESTIGATION_CRITERIA,
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
            ScriptedTurn(content=SUCCESS_ANSWER, finish_reason="stop"),
        ),
    ),
    MeasuredCase(
        trace_id="partial-success-extra-profile",
        example_class="PARTIAL_SUCCESS",
        request=PAYMENTS_TASK,
        selection_note=(
            "Measured case: the agent reaches a correct answer after an "
            "unnecessary get_user_profile call. Teaches that a correct outcome "
            "does not necessarily mean a good trajectory. Case constraint: "
            "expected maximum useful tool calls is 2."
        ),
        criteria=PAYMENTS_INVESTIGATION_CRITERIA,
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
                content=None,
                tool_calls=[
                    {
                        "id": "call_extra_profile",
                        "name": "get_user_profile",
                        "arguments": {"user_id": "u-1001"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(content=SUCCESS_ANSWER, finish_reason="stop"),
        ),
    ),
    MeasuredCase(
        trace_id="tool-error-recovery-payments",
        example_class="TOOL_ERROR_RECOVERY",
        request=PAYMENTS_TASK,
        selection_note=(
            "Measured case: first get_service_status uses invalid name "
            "'payments-api' and fails; the next call uses canonical "
            "'payments' and succeeds, then documentation is inspected. The "
            "trace preserves the failure. Task success does not treat the "
            "recovered failure as an overall task failure."
        ),
        criteria=RECOVERY_CRITERIA,
        turns=(
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_bad_status",
                        "name": "get_service_status",
                        "arguments": {"service": "payments-api"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(
                content=None,
                tool_calls=[
                    {
                        "id": "call_ok_status",
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
            ScriptedTurn(content=SUCCESS_ANSWER, finish_reason="stop"),
        ),
    ),
    MeasuredCase(
        trace_id="goal-miss-wrong-answer",
        example_class="GOAL_MISS",
        request=PAYMENTS_TASK,
        selection_note=(
            "Measured case: the agent gathers the required evidence and "
            "terminates with a final answer, but the answer does not satisfy "
            "the task. Teaches that final-answer presence is not task success, "
            "and that a normal termination can still fail evaluation."
        ),
        criteria=PAYMENTS_INVESTIGATION_CRITERIA,
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
            ScriptedTurn(content=GOAL_MISS_ANSWER, finish_reason="stop"),
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
