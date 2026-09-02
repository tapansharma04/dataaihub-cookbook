"""Measured cases for composed MCP workflows with Sampling.

Client-side step selection is deterministic and explicit — not agent reasoning.
The server remains authoritative for resources, prompts, tools, and sampling
requests. The client owns the sampling callback (mock, reject, or live model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SamplingMode = Literal["mock", "reject", "live"]
StepKind = Literal[
    "list_resources",
    "read_resource",
    "list_prompts",
    "get_prompt",
    "list_tools",
    "call_tool",
]


@dataclass(frozen=True)
class ProtocolStep:
    kind: StepKind
    uri: str | None = None
    prompt_name: str | None = None
    prompt_arguments: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    selection_note: str
    steps: tuple[ProtocolStep, ...]
    sampling_mode: SamplingMode = "mock"


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="resource-to-sampling",
        example_class="RESOURCE_TO_SAMPLING",
        selection_note=(
            "Measured case: the client reads an MCP resource, then calls a "
            "composition tool. The server grounds a sampling request in that "
            "resource and the client sampling callback completes generation."
        ),
        steps=(
            ProtocolStep(kind="list_resources"),
            ProtocolStep(
                kind="read_resource",
                uri="acme://docs/knowledge-platform",
            ),
            ProtocolStep(
                kind="call_tool",
                tool_name="compose_resource_brief",
                tool_arguments={"uri": "acme://docs/knowledge-platform"},
            ),
        ),
        sampling_mode="mock",
    ),
    MeasuredCase(
        trace_id="prompt-to-sampling",
        example_class="PROMPT_TO_SAMPLING",
        selection_note=(
            "Measured case: the client retrieves an MCP prompt template, then "
            "calls a composition tool. The server requests sampling using the "
            "same rendered prompt messages."
        ),
        steps=(
            ProtocolStep(kind="list_prompts"),
            ProtocolStep(
                kind="get_prompt",
                prompt_name="summarize-service",
                prompt_arguments={
                    "service_name": "knowledge-platform",
                    "audience": "engineering",
                },
            ),
            ProtocolStep(
                kind="call_tool",
                tool_name="compose_from_prompt",
                tool_arguments={
                    "prompt_name": "summarize-service",
                    "service_name": "knowledge-platform",
                    "audience": "engineering",
                },
            ),
        ),
        sampling_mode="mock",
    ),
    MeasuredCase(
        trace_id="tool-resource-prompt-composition",
        example_class="TOOL_RESOURCE_PROMPT_COMPOSITION",
        selection_note=(
            "Measured case: the client calls a status tool, reads the related "
            "resource, retrieves a prompt template, then calls a composition "
            "tool. The server requests sampling from the combined tool result, "
            "resource content, and prompt messages."
        ),
        steps=(
            ProtocolStep(kind="list_tools"),
            ProtocolStep(
                kind="call_tool",
                tool_name="get_service_status",
                tool_arguments={"service": "billing-api"},
            ),
            ProtocolStep(kind="list_resources"),
            ProtocolStep(
                kind="read_resource",
                uri="acme://docs/billing-portal",
            ),
            ProtocolStep(kind="list_prompts"),
            ProtocolStep(
                kind="get_prompt",
                prompt_name="draft-status-update",
                prompt_arguments={
                    "service": "billing-api",
                    "status": "degraded",
                },
            ),
            ProtocolStep(
                kind="call_tool",
                tool_name="compose_incident_brief",
                tool_arguments={"service": "billing-api"},
            ),
        ),
        sampling_mode="mock",
    ),
    MeasuredCase(
        trace_id="sampling-failure",
        example_class="SAMPLING_FAILURE",
        selection_note=(
            "Measured case: the server requests sampling after a resource-"
            "grounded composition tool call. The client sampling callback "
            "returns a protocol ErrorData rejection. No model output is produced."
        ),
        steps=(
            ProtocolStep(kind="list_resources"),
            ProtocolStep(
                kind="read_resource",
                uri="acme://docs/knowledge-platform",
            ),
            ProtocolStep(
                kind="call_tool",
                tool_name="compose_resource_brief",
                tool_arguments={"uri": "acme://docs/knowledge-platform"},
            ),
        ),
        sampling_mode="reject",
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
