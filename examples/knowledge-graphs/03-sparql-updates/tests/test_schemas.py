"""Pydantic model schema tests."""

from __future__ import annotations

from sparql.model import (
    BindingRow,
    BindingValue,
    TripleState,
    TripleTerm,
    UpdateRunMetrics,
)


def test_binding_value_public_shape():
    value = BindingValue(iri="https://example.org/alice", label="Alice")
    assert value.public() == {"iri": "https://example.org/alice", "label": "Alice"}


def test_binding_row_public_shape():
    row = BindingRow(
        variables={
            "person": BindingValue(
                iri="https://dataaihub.co/example/kg/alice",
                label="Alice",
            )
        }
    )
    public = row.public()
    assert public["person"]["label"] == "Alice"


def test_triple_state_public_shape():
    triple = TripleState(
        subject=TripleTerm(
            iri="https://dataaihub.co/example/kg/billingPortal",
            label="Billing Portal",
        ),
        predicate=TripleTerm(
            iri="https://dataaihub.co/example/kg/uses",
            label=None,
        ),
        object=TripleTerm(
            iri="https://dataaihub.co/example/kg/redis",
            label="Redis",
        ),
    )
    public = triple.public()
    assert public["subject"]["label"] == "Billing Portal"
    assert public["object"]["label"] == "Redis"


def test_metrics_provenance():
    metrics = UpdateRunMetrics(
        update_execution_ms=1,
        verification_execution_ms=1,
        inserted_triple_count=1,
        deleted_triple_count=0,
        before_triple_count=1,
        after_triple_count=2,
        verification_rows=2,
        update_type="INSERT_DATA",
        termination_reason="completed",
    )
    assert metrics.provenance == "measured"
