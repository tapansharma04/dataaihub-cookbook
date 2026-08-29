"""Measured cases for MCP prompt discovery and retrieval.

Client-side prompt selection is deterministic and explicit — not agent reasoning.
The server remains authoritative for prompt metadata and rendered messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CaseAction = Literal["discover_only", "get"]


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    selection_note: str
    action: CaseAction
    prompt_name: str | None = None
    prompt_arguments: dict[str, Any] | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="prompt-discovery",
        example_class="PROMPT_DISCOVERY",
        selection_note=(
            "Measured case: client initializes with the server and discovers "
            "available prompt templates through prompts/list without calling "
            "prompts/get."
        ),
        action="discover_only",
    ),
    MeasuredCase(
        trace_id="single-prompt-get-summarize",
        example_class="SINGLE_PROMPT_GET",
        selection_note=(
            "Measured case: after discovery, the client explicitly retrieves "
            "the summarize-service prompt with required arguments and records "
            "the MCP-rendered messages."
        ),
        action="get",
        prompt_name="summarize-service",
        prompt_arguments={"service_name": "knowledge-platform"},
    ),
    MeasuredCase(
        trace_id="prompt-with-arguments-investigate",
        example_class="PROMPT_WITH_ARGUMENTS",
        selection_note=(
            "Measured case: after discovery, the client retrieves "
            "investigate-incident with multiple arguments. The returned "
            "multi-message prompt depends on both service and incident values."
        ),
        action="get",
        prompt_name="investigate-incident",
        prompt_arguments={
            "service": "billing-api",
            "incident": "INC-2048",
        },
    ),
    MeasuredCase(
        trace_id="invalid-prompt-name",
        example_class="INVALID_PROMPT",
        selection_note=(
            "Measured case: after discovery, the client requests a prompt name "
            "the server does not expose. The failure is recorded from the MCP "
            "prompts/get boundary."
        ),
        action="get",
        prompt_name="does-not-exist",
        prompt_arguments={},
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
