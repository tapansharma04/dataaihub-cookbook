"""Four measured SPARQL teaching cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    question: str
    query_name: str
    selection_note: str


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="basic-select-knowledge-platform-people",
        example_class="BASIC_SELECT",
        question="Which people work on the Knowledge Platform?",
        query_name="BASIC_SELECT",
        selection_note=(
            "Measured case: one triple pattern with a fixed object "
            "(ex:knowledgePlatform). Variable ?person binds to Alice and Bob."
        ),
    ),
    MeasuredCase(
        trace_id="multi-pattern-alice-technologies",
        example_class="MULTI_PATTERN_QUERY",
        question="Which technologies are used by projects Alice works on?",
        query_name="MULTI_PATTERN_QUERY",
        selection_note=(
            "Measured case: two joined triple patterns share ?project. "
            "Alice → worksOn → Knowledge Platform → uses → PostgreSQL."
        ),
    ),
    MeasuredCase(
        trace_id="filter-platform-team-people",
        example_class="FILTER_QUERY",
        question="Which people work on platform-team projects?",
        query_name="FILTER_QUERY",
        selection_note=(
            "Measured case: triple patterns joined on ?project plus "
            'FILTER(?team = "platform") inside SPARQL. Carol is excluded '
            'because Billing Portal has ex:team "billing".'
        ),
    ),
    MeasuredCase(
        trace_id="no-match-quantum-platform",
        example_class="NO_MATCH",
        question="Which people work on the Quantum Computing Platform?",
        query_name="NO_MATCH",
        selection_note=(
            "Measured case: legitimate query for ex:quantumComputingPlatform "
            "which does not exist in the graph. Zero bindings is a valid result."
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(case.trace_id for case in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
