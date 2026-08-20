"""Explicit types for SPARQL query runs and normalized bindings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "user_request",
    "query_started",
    "query_executed",
    "result_bindings",
    "query_completed",
    "termination",
]
TerminationReason = Literal[
    "completed",
    "no_match",
    "query_rejected",
    "row_limit",
    "query_failed",
]


class SparqlError(ValueError):
    """Application-owned SPARQL validation or execution failure."""

    def __init__(self, code: TerminationReason, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BindingValue(BaseModel):
    iri: str
    label: str | None = None
    literal: str | None = None
    datatype: str | None = None

    def public(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {"iri": self.iri}
        if self.label is not None:
            out["label"] = self.label
        if self.literal is not None:
            out["literal"] = self.literal
        if self.datatype is not None:
            out["datatype"] = self.datatype
        return out


class BindingRow(BaseModel):
    variables: dict[str, BindingValue] = Field(default_factory=dict)

    def public(self) -> dict[str, dict[str, str | None]]:
        return {name: value.public() for name, value in self.variables.items()}


class SequenceEvent(BaseModel):
    """Ordered observable SPARQL operation."""

    kind: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class QueryRunMetrics(BaseModel):
    query_execution_ms: int
    result_rows: int
    triple_patterns: int
    filter_count: int
    variables: list[str]
    query_case: str
    bindings_returned: int
    termination_reason: TerminationReason
    provenance: Literal["measured"] = "measured"


class QueryRunResult(BaseModel):
    case_id: str
    example_class: str
    question: str
    query_name: str
    query: str
    prefixes: dict[str, str]
    patterns: list[str]
    bindings: list[BindingRow] = Field(default_factory=list)
    matches: list[dict[str, dict[str, str | None]]] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: QueryRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
