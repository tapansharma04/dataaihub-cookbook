"""Graph loading and snapshot tests."""

from __future__ import annotations

from pathlib import Path

from sparql.graph import RdfGraphStore
from sparql.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def test_turtle_loads_into_rdflib_graph():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    assert len(store.rdf) > 0


def test_graph_has_expected_entities():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    ids = {entity["id"] for entity in store.entities()}
    assert str(EX.alice) in ids
    assert str(EX.bob) in ids
    assert str(EX.knowledgePlatform) in ids
    assert str(EX.postgresql) in ids


def test_graph_has_team_property_for_filter_case():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    team = store.rdf.value(EX.knowledgePlatform, EX.team)
    assert team is not None
    assert str(team) == "platform"


def test_snapshot_shape():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    snapshot = store.snapshot()
    assert snapshot["format"] == "rdf"
    assert snapshot["namespace"] == str(EX)
    assert snapshot["entityCount"] == 8
    assert "team" in snapshot["predicates"]
