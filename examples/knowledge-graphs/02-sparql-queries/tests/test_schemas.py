"""Pydantic model schema tests."""

from __future__ import annotations

from sparql.model import BindingRow, BindingValue, QueryRunMetrics


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


def test_metrics_provenance():
    metrics = QueryRunMetrics(
        query_execution_ms=1,
        result_rows=2,
        triple_patterns=2,
        filter_count=0,
        variables=["person"],
        query_case="BASIC_SELECT",
        bindings_returned=2,
        termination_reason="completed",
    )
    assert metrics.provenance == "measured"
