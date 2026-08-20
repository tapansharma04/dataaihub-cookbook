"""Graph loading and fresh-graph isolation tests."""

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
    assert str(EX.carol) in ids
    assert str(EX.acmeAI) in ids
    assert str(EX.knowledgePlatform) in ids
    assert str(EX.billingPortal) in ids
    assert str(EX.postgresql) in ids
    assert str(EX.redis) in ids


def test_billing_portal_uses_postgresql_not_redis():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    assert (EX.billingPortal, EX.uses, EX.postgresql) in store.rdf
    assert (EX.billingPortal, EX.uses, EX.redis) not in store.rdf


def test_fresh_graph_isolation():
    first = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    first.rdf.update(
        """
        PREFIX ex: <https://dataaihub.co/example/kg/>
        INSERT DATA { ex:billingPortal ex:uses ex:redis . }
        """
    )
    assert (EX.billingPortal, EX.uses, EX.redis) in first.rdf

    second = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    assert (EX.billingPortal, EX.uses, EX.redis) not in second.rdf
    assert (EX.billingPortal, EX.uses, EX.postgresql) in second.rdf


def test_snapshot_shape():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    snapshot = store.snapshot()
    assert snapshot["format"] == "rdf"
    assert snapshot["namespace"] == str(EX)
    assert snapshot["entityCount"] == 8
    assert "uses" in snapshot["predicates"]
