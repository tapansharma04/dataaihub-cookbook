"""Measured cases for MCP resource discovery and reading.

Client-side URI selection is deterministic and explicit — not agent reasoning.
The server remains authoritative for resource metadata and content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CaseAction = Literal["discover_only", "read"]


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    selection_note: str
    action: CaseAction
    resource_uris: tuple[str, ...] = ()


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="discovery",
        example_class="DISCOVERY",
        selection_note=(
            "Measured case: client initializes with the server and discovers "
            "available resources through resources/list without reading content."
        ),
        action="discover_only",
    ),
    MeasuredCase(
        trace_id="single-resource-read-knowledge-platform",
        example_class="SINGLE_RESOURCE_READ",
        selection_note=(
            "Measured case: after discovery, the client explicitly reads "
            "acme://docs/knowledge-platform and records the MCP resource content."
        ),
        action="read",
        resource_uris=("acme://docs/knowledge-platform",),
    ),
    MeasuredCase(
        trace_id="multi-resource-read-services",
        example_class="MULTI_RESOURCE_READ",
        selection_note=(
            "Measured case: after discovery, the client sequentially reads "
            "acme://docs/knowledge-platform and acme://status/services."
        ),
        action="read",
        resource_uris=(
            "acme://docs/knowledge-platform",
            "acme://status/services",
        ),
    ),
    MeasuredCase(
        trace_id="invalid-resource-uri",
        example_class="INVALID_RESOURCE",
        selection_note=(
            "Measured case: after discovery, the client requests a URI the "
            "server does not expose. The failure is recorded from the MCP "
            "resources/read boundary."
        ),
        action="read",
        resource_uris=("acme://docs/does-not-exist",),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
