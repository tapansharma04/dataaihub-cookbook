"""Explicit types for extraction proposals, validation, runs, and traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["structured", "llm_assisted"]

EventKind = Literal[
    "source_loaded",
    "extraction_started",
    "entity_proposed",
    "relationship_proposed",
    "validation_started",
    "validation_passed",
    "validation_rejected",
    "entity_resolved",
    "triple_created",
    "graph_committed",
    "graph_verified",
    "result",
    "termination",
]

TerminationReason = Literal[
    "completed",
    "validation_rejected",
    "unresolved_entity",
    "model_unavailable",
    "model_failed",
]

ExampleClass = Literal[
    "ENTITY_EXTRACTION",
    "RELATIONSHIP_EXTRACTION",
    "ENTITY_LINKING",
    "INVALID_FACT",
]

StartGraph = Literal["empty", "seed"]


class EntityProposal(BaseModel):
    label: str
    entity_type: str


class RelationshipProposal(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractionProposal(BaseModel):
    entities: list[EntityProposal] = Field(default_factory=list)
    relationships: list[RelationshipProposal] = Field(default_factory=list)


class ResolvedEntity(BaseModel):
    label: str
    iri: str
    entity_type: str


class ValidatedRelationship(BaseModel):
    subject_iri: str
    predicate_local: str
    predicate_iri: str
    object_iri: str
    subject_label: str
    object_label: str


class RejectedRelationship(BaseModel):
    subject: str
    predicate: str
    object: str
    reason: str


class ValidationResult(BaseModel):
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    unresolved_labels: list[str] = Field(default_factory=list)
    accepted_relationships: list[ValidatedRelationship] = Field(default_factory=list)
    rejected_relationships: list[RejectedRelationship] = Field(default_factory=list)
    entity_type_errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.unresolved_labels
            and not self.rejected_relationships
            and not self.entity_type_errors
        )


class TripleRef(BaseModel):
    subject: str
    predicate: str
    object: str
    kind: Literal["type", "label", "relationship"] = "relationship"


class SequenceEvent(BaseModel):
    kind: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class ConstructionMetrics(BaseModel):
    source_characters: int
    entities_proposed: int
    entities_resolved: int
    relationships_proposed: int
    relationships_accepted: int
    relationships_rejected: int
    triples_created: int
    triples_rejected: int
    graph_triple_count: int
    validation_errors: int
    model_turns: int
    total_ms: int
    model_ms: int
    termination_reason: TerminationReason


class ConstructionResult(BaseModel):
    case_id: str
    mode: Mode
    example_class: ExampleClass
    source_text: str
    proposal: ExtractionProposal
    validation: ValidationResult
    triples_created: list[TripleRef] = Field(default_factory=list)
    graph_before_count: int
    graph_after_count: int
    graph_unchanged: bool
    termination: TerminationReason = "completed"
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: ConstructionMetrics
    provenance: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    model_latency_ms: int | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
