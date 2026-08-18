"""Measured teaching cases for MCP tool discovery and invocation.

Client-side tool selection is deterministic and explicit — not agent reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CaseAction = Literal["discover_only", "invoke"]


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    selection_note: str
    action: CaseAction
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    select_tool_by: str | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="discovery",
        example_class="DISCOVERY",
        selection_note=(
            "Measured case: client initializes with the server and discovers "
            "available tool contracts through tools/list without invoking a tool."
        ),
        action="discover_only",
    ),
    MeasuredCase(
        trace_id="single-tool-service-status",
        example_class="SINGLE_TOOL_CALL",
        selection_note=(
            "Measured case: after discovery, the client invokes get_service_status "
            "with a deterministic fixture argument and receives a structured result."
        ),
        action="invoke",
        tool_name="get_service_status",
        tool_arguments={"service": "payments"},
    ),
    MeasuredCase(
        trace_id="multi-tool-search-docs",
        example_class="MULTI_TOOL_DISCOVERY",
        selection_note=(
            "Measured case: client discovers all server tools, then explicitly "
            "selects search_documentation (not autonomous agent reasoning)."
        ),
        action="invoke",
        tool_name="search_documentation",
        tool_arguments={"query": "payments degradation"},
        select_tool_by="search_documentation",
    ),
    MeasuredCase(
        trace_id="invalid-arguments-service-type",
        example_class="INVALID_ARGUMENTS",
        selection_note=(
            "Measured case: client discovers get_service_status, then calls it "
            "with an invalid argument type. The server rejects the request at "
            "the MCP tool boundary with a structured protocol-visible error."
        ),
        action="invoke",
        tool_name="get_service_status",
        tool_arguments={"service": 123},
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
