"""Schema serialization smoke tests."""

from __future__ import annotations

from graph.model import Entity, GraphRunMetrics, Hop, Triple
from graph.vocab import EX


def test_entity_keeps_iri_and_label_separate():
    entity = Entity(id=str(EX.alice), label="Alice", type="Person")
    payload = entity.public()
    assert payload["id"] == str(EX.alice)
    assert payload["label"] == "Alice"
    again = Entity.model_validate(payload)
    assert again.id == entity.id
    assert again.label == entity.label


def test_triple_is_subject_predicate_object_iris():
    triple = Triple(
        subject=str(EX.alice),
        predicate=str(EX.worksOn),
        object=str(EX.knowledgePlatform),
    )
    payload = triple.public()
    assert list(payload.keys()) == ["subject", "predicate", "object"]
    assert Triple.model_validate(payload).predicate == str(EX.worksOn)


def test_hop_records_direction():
    hop = Hop(predicate="worksOn", direction="incoming")
    assert hop.model_dump() == {"predicate": "worksOn", "direction": "incoming"}


def test_metrics_are_operational_not_scores():
    metrics = GraphRunMetrics(
        entities_visited=4,
        relationships_visited=3,
        traversal_depth=1,
        matched_relationships=3,
        path_found=True,
        execution_ms=1,
        termination_reason="completed",
        max_depth=8,
    )
    payload = metrics.model_dump()
    assert metrics.provenance == "measured"
    assert "graph_intelligence_score" not in payload
    assert "benchmark_score" not in payload
