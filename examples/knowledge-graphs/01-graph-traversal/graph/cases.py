"""Four measured RDF graph-traversal teaching cases.

Each case names an explicit hop list. The runner does not infer missing edges.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.model import Hop
from graph.vocab import EX


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    question: str
    start_id: str
    hops: tuple[Hop, ...]
    selection_note: str


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="direct-relationship-employs",
        example_class="DIRECT_RELATIONSHIP",
        question="Who does Acme AI employ?",
        start_id=str(EX.acmeAI),
        hops=(Hop(predicate="employs", direction="outgoing"),),
        selection_note=(
            "Measured case: one outgoing employs hop from Acme AI. "
            "Direct neighbors are Alice, Bob, and Carol."
        ),
    ),
    MeasuredCase(
        trace_id="multi-hop-alice-technologies",
        example_class="MULTI_HOP_TRAVERSAL",
        question="Which technologies are used by projects that Alice works on?",
        start_id=str(EX.alice),
        hops=(
            Hop(predicate="worksOn", direction="outgoing"),
            Hop(predicate="uses", direction="outgoing"),
        ),
        selection_note=(
            "Measured case: two explicit hops from Alice "
            "(worksOn then uses). PostgreSQL is reached only because both "
            "RDF triples exist."
        ),
    ),
    MeasuredCase(
        trace_id="relationship-filter-project-people",
        example_class="RELATIONSHIP_FILTER",
        question="Which people work on the Knowledge Platform project?",
        start_id=str(EX.knowledgePlatform),
        hops=(Hop(predicate="worksOn", direction="incoming"),),
        selection_note=(
            "Measured case: incoming worksOn edges into Knowledge Platform. "
            "The stored RDF triple remains Alice --worksOn--> Knowledge Platform; "
            "only the walk direction is incoming. Carol is excluded because she "
            "works on Billing Portal."
        ),
    ),
    MeasuredCase(
        trace_id="no-path-alice-uses",
        example_class="NO_PATH",
        question="What technology does Alice use directly?",
        start_id=str(EX.alice),
        hops=(Hop(predicate="uses", direction="outgoing"),),
        selection_note=(
            "Measured case: a direct Alice --uses--> Technology triple does not "
            "exist. The walker reports no path and does not invent "
            "Alice → Project → Technology."
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(case.trace_id for case in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
