"""Explicit graph types: entities, triples, hops, and measured run results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["outgoing", "incoming"]
EventKind = Literal[
    "user_request",
    "graph_lookup",
    "traversal_started",
    "traversal_step",
    "relationship_match",
    "traversal_completed",
    "result",
    "termination",
]
TerminationReason = Literal[
    "completed",
    "no_path",
    "invalid_entity",
    "invalid_relationship",
    "depth_limit",
]


class GraphError(ValueError):
    """Application-owned graph validation or lookup failure."""

    def __init__(self, code: TerminationReason, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Entity(BaseModel):
    id: str
    label: str
    type: str

    def public(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "type": self.type}


class Triple(BaseModel):
    """One directed relationship: (subject, predicate, object)."""

    subject: str
    predicate: str
    object: str

    def public(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


class Hop(BaseModel):
    """One explicit traversal hop. Not a query string."""

    predicate: str
    direction: Direction = "outgoing"


class TraversalRequest(BaseModel):
    start_id: str
    hops: list[Hop] = Field(default_factory=list)
    max_depth: int | None = None


class SequenceEvent(BaseModel):
    """Ordered observable graph operation."""

    kind: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class MatchedPath(BaseModel):
    entities: list[Entity]
    relationships: list[Triple]
    depth: int


class GraphRunMetrics(BaseModel):
    entities_visited: int
    relationships_visited: int
    traversal_depth: int
    matched_relationships: int
    path_found: bool
    execution_ms: int
    termination_reason: TerminationReason
    max_depth: int
    provenance: Literal["measured"] = "measured"


class GraphRunResult(BaseModel):
    case_id: str
    example_class: str
    question: str
    start: Entity | None = None
    hops: list[Hop] = Field(default_factory=list)
    paths: list[MatchedPath] = Field(default_factory=list)
    answers: list[Entity] = Field(default_factory=list)
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: GraphRunMetrics
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
