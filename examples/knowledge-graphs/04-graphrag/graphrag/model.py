"""Explicit types for GraphRAG runs, traces, and subgraph facts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["graph_grounded", "graphrag_llm"]

EventKind = Literal[
    "user_request",
    "entity_resolution",
    "retrieval_started",
    "retrieval_step",
    "subgraph_retrieved",
    "context_assembled",
    "model_request",
    "model_response",
    "final_answer",
    "termination",
]

TerminationReason = Literal[
    "completed",
    "no_entity_match",
    "no_relevant_subgraph",
    "model_unavailable",
    "model_failed",
]


class GraphEntityRef(BaseModel):
    iri: str
    label: str


class GraphPredicateRef(BaseModel):
    iri: str
    label: str


class GraphFact(BaseModel):
    subject: GraphEntityRef
    predicate: GraphPredicateRef
    object: GraphEntityRef

    def sort_key(self) -> tuple[str, str, str]:
        return (self.subject.iri, self.predicate.iri, self.object.iri)

    def public(self) -> dict[str, dict[str, str]]:
        return {
            "subject": self.subject.model_dump(),
            "predicate": self.predicate.model_dump(),
            "object": self.object.model_dump(),
        }


class PathStep(BaseModel):
    subject: GraphEntityRef
    predicate: GraphPredicateRef
    object: GraphEntityRef


class RetrievalPath(BaseModel):
    steps: list[PathStep] = Field(default_factory=list)

    def endpoint_iris(self) -> set[str]:
        if not self.steps:
            return set()
        return {self.steps[-1].object.iri}


class ResolvedEntity(BaseModel):
    iri: str
    label: str
    matchSpan: str


class SequenceEvent(BaseModel):
    kind: EventKind
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class GraphRunMetrics(BaseModel):
    entity_candidates: int
    resolved_entity_count: int
    retrieval_hops: int
    entities_retrieved: int
    relationships_retrieved: int
    subgraph_triple_count: int
    context_fact_count: int
    retrieval_execution_ms: int
    context_assembly_ms: int
    answer_generation_ms: int
    total_ms: int
    model_turns: int
    termination_reason: TerminationReason


class GraphRunResult(BaseModel):
    case_id: str
    mode: Mode
    example_class: str
    user_request: str
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    subgraph: list[GraphFact] = Field(default_factory=list)
    paths: list[RetrievalPath] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    answer: str = ""
    termination: TerminationReason = "completed"
    sequence: list[SequenceEvent] = Field(default_factory=list)
    metrics: GraphRunMetrics
    provenance: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    model_latency_ms: int | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
