"""Pydantic model and proposal schema tests."""

from __future__ import annotations

from graph.model import (
    EntityProposal,
    ExtractionProposal,
    RelationshipProposal,
    ValidationResult,
)


def test_extraction_proposal_round_trip():
    proposal = ExtractionProposal(
        entities=[EntityProposal(label="Alice", entity_type="Person")],
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="works_on",
                object="Knowledge Platform",
            )
        ],
    )
    restored = ExtractionProposal.model_validate(proposal.model_dump())
    assert restored.entities[0].label == "Alice"
    assert restored.relationships[0].predicate == "works_on"


def test_proposal_has_no_cot_fields():
    fields = set(ExtractionProposal.model_fields)
    fields |= set(EntityProposal.model_fields)
    fields |= set(RelationshipProposal.model_fields)
    forbidden = {
        "reasoning",
        "thought",
        "chain_of_thought",
        "cot",
        "scratchpad",
        "hidden_reasoning",
    }
    assert forbidden.isdisjoint(fields)


def test_validation_result_ok_property():
    ok = ValidationResult()
    assert ok.ok is True
    bad = ValidationResult(unresolved_labels=["Unknown"])
    assert bad.ok is False
