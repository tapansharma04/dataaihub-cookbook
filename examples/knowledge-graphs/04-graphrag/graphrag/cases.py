"""Four measured GraphRAG teaching cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["forward", "inverse"]


@dataclass(frozen=True)
class TraversalStep:
    predicate: str
    direction: Direction


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    question: str
    selection_note: str
    traversal_steps: tuple[TraversalStep, ...]
    max_hops: int
    require_direct_only: bool = False


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="entity-retrieval-knowledge-platform",
        example_class="ENTITY_RETRIEVAL",
        question="Who works on the Knowledge Platform?",
        selection_note=(
            "Measured case: resolve Knowledge Platform, traverse inverse worksOn "
            "to retrieve Alice and Bob before answer generation."
        ),
        traversal_steps=(TraversalStep(predicate="worksOn", direction="inverse"),),
        max_hops=1,
    ),
    MeasuredCase(
        trace_id="multi-hop-alice-technologies",
        example_class="MULTI_HOP_RETRIEVAL",
        question="Which technologies are used by projects Alice works on?",
        selection_note=(
            "Measured case: resolve Alice, follow worksOn then uses to retrieve "
            "PostgreSQL and Redis through Knowledge Platform."
        ),
        traversal_steps=(
            TraversalStep(predicate="worksOn", direction="forward"),
            TraversalStep(predicate="uses", direction="forward"),
        ),
        max_hops=2,
    ),
    MeasuredCase(
        trace_id="relationship-grounded-alice-employer-project",
        example_class="RELATIONSHIP_GROUNDED_ANSWER",
        question="Which company employs Alice and what project does she work on?",
        selection_note=(
            "Measured case: resolve Alice, retrieve inverse employs and forward "
            "worksOn relationships as explicit graph facts."
        ),
        traversal_steps=(
            TraversalStep(predicate="employs", direction="inverse"),
            TraversalStep(predicate="worksOn", direction="forward"),
        ),
        max_hops=1,
    ),
    MeasuredCase(
        trace_id="no-relevant-subgraph-alice-direct-uses",
        example_class="NO_RELEVANT_SUBGRAPH",
        question="What technology does Alice directly use?",
        selection_note=(
            "Measured case: resolve Alice but require a direct uses relationship. "
            "The graph has no Alice → uses edge, so retrieval returns no subgraph."
        ),
        traversal_steps=(TraversalStep(predicate="uses", direction="forward"),),
        max_hops=1,
        require_direct_only=True,
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(case.trace_id for case in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
