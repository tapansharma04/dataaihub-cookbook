"""Four measured graph-construction cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_SOURCES_PATH
from graph.model import (
    EntityProposal,
    ExampleClass,
    ExtractionProposal,
    RelationshipProposal,
    StartGraph,
)


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: ExampleClass
    source_text: str
    selection_note: str
    expected_proposal: ExtractionProposal
    expected_resolved: tuple[tuple[str, str], ...]
    expected_committed_predicates: tuple[str, ...]
    start_graph: StartGraph
    expect_graph_unchanged: bool


def _load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["sources"])


def _proposal_from_dict(data: dict[str, Any]) -> ExtractionProposal:
    return ExtractionProposal(
        entities=[
            EntityProposal(label=item["label"], entity_type=item["entity_type"])
            for item in data.get("entities", [])
        ],
        relationships=[
            RelationshipProposal(
                subject=item["subject"],
                predicate=item["predicate"],
                object=item["object"],
            )
            for item in data.get("relationships", [])
        ],
    )


def load_cases(path: Path | None = None) -> tuple[MeasuredCase, ...]:
    sources = _load_sources(path or DEFAULT_SOURCES_PATH)
    cases: list[MeasuredCase] = []
    for item in sources:
        expected = item["expectedProposal"]
        resolved = tuple(
            (entry["label"], entry["iri"]) for entry in item.get("expectedResolved", [])
        )
        cases.append(
            MeasuredCase(
                trace_id=item["id"],
                example_class=item["exampleClass"],
                source_text=item["text"],
                selection_note=item["selectionNote"],
                expected_proposal=_proposal_from_dict(expected),
                expected_resolved=resolved,
                expected_committed_predicates=tuple(
                    item.get("expectedCommittedPredicates", [])
                ),
                start_graph=item.get("startGraph", "empty"),
                expect_graph_unchanged=bool(item.get("expectGraphUnchanged", False)),
            )
        )
    return tuple(cases)


CASES: tuple[MeasuredCase, ...] = load_cases()


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(case.trace_id for case in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
