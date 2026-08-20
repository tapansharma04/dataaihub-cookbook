"""Explicit types for SPARQL UPDATE runs and graph state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "user_request",
    "update_started",
    "update_executed",
    "graph_state",
    "verification_started",
    "verification_result",
    "update_completed",
    "termination",
]
TerminationReason = Literal[
    "completed",
    "update_rejected",
    "update_failed",
    "verification_failed",
    "row_limit",
]
UpdateType = Literal[
    "INSERT_DATA",
    "INSERT_WHERE",
    "DELETE_DATA",
    "UPDATE_AND_VERIFY",
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


class TripleTerm(BaseModel):
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


class TripleState(BaseModel):
    subject: TripleTerm
    predicate: TripleTerm
    object: TripleTerm

    def public(self) -> dict[str, dict[str, str | None]]:
        return {
            "subject": self.subject.public(),
            "predicate": self.predicate.public(),
            "object": self.object.public(),
        }

    def sort_key(self) -> tuple[str, str, str]:
        return (self.subject.iri, self.predicate.iri, self.object.iri)


class SequenceEvent(BaseModel):
    """Ordered observable SPARQL UPDATE operation."""

    kind: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class UpdateRunMetrics(BaseModel):
    update_execution_ms: int
    verification_execution_ms: int
    inserted_triple_count: int
    deleted_triple_count: int
    before_triple_count: int
    after_triple_count: int
    verification_rows: int
    update_type: UpdateType
    verification_query_count: int = 1
    termination_reason: TerminationReason
    provenance: Literal["measured"] = "measured"


class UpdateRunResult(BaseModel):
    case_id: str
    example_class: UpdateType
    question: str
    update_name: str
    update_query: str
    verification_query: str
    prefixes: dict[str, str]
    before_state: list[TripleState] = Field(default_factory=list)
    after_state: list[TripleState] = Field(default_factory=list)
    inserted_triples: list[TripleState] = Field(default_factory=list)
    deleted_triples: list[TripleState] = Field(default_factory=list)
    verification_bindings: list[BindingRow] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: UpdateRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
