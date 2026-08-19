"""Entity and RDF relationship store tests — no network / no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph

from graph.model import GraphError
from graph.store import GraphStore
from graph.vocab import EX, RDFS

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def fixture_store() -> GraphStore:
    return GraphStore.from_path(GRAPH_PATH)


def test_entity_creation():
    store = GraphStore()
    entity = store.add_entity(str(EX.northstar), "Northstar", "Company")
    assert entity.id == str(EX.northstar)
    assert entity.label == "Northstar"
    assert entity.type == "Company"
    assert store.lookup(str(EX.northstar)).label == "Northstar"
    assert str(store.rdf.value(EX.northstar, RDFS.label)) == "Northstar"


def test_relationship_creation():
    store = GraphStore()
    store.add_entity(str(EX.northstar), "Northstar", "Company")
    store.add_entity(str(EX.dee), "Dee", "Person")
    triple = store.add_relationship(str(EX.northstar), "employs", str(EX.dee))
    assert triple.subject == str(EX.northstar)
    assert triple.predicate == str(EX.employs)
    assert triple.object == str(EX.dee)
    assert (EX.northstar, EX.employs, EX.dee) in store.rdf


def test_lookup_returns_iri_not_just_label():
    store = fixture_store()
    alice = store.lookup(str(EX.alice))
    assert alice.id == str(EX.alice)
    assert alice.label == "Alice"
    assert alice.id != alice.label
    assert alice.id.startswith("https://")


def test_get_neighbors_outgoing_employs():
    store = fixture_store()
    neighbors = store.get_neighbors(
        str(EX.acmeAI),
        predicate="employs",
        direction="outgoing",
    )
    assert [triple.object for triple in neighbors] == [
        str(EX.alice),
        str(EX.bob),
        str(EX.carol),
    ]


def test_get_neighbors_incoming_works_on():
    store = fixture_store()
    neighbors = store.get_neighbors(
        str(EX.knowledgePlatform),
        predicate="worksOn",
        direction="incoming",
    )
    assert [triple.subject for triple in neighbors] == [str(EX.alice), str(EX.bob)]
    assert str(EX.carol) not in [triple.subject for triple in neighbors]


def test_duplicate_entity_rejected():
    store = GraphStore()
    store.add_entity(str(EX.alice), "Alice", "Person")
    with pytest.raises(GraphError) as exc:
        store.add_entity(str(EX.alice), "Alice", "Person")
    assert exc.value.code == "invalid_entity"


def test_invalid_entity_lookup():
    store = fixture_store()
    with pytest.raises(GraphError) as exc:
        store.lookup(str(EX.nobody))
    assert exc.value.code == "invalid_entity"


def test_invalid_relationship_predicate():
    store = GraphStore()
    store.add_entity(str(EX.alice), "Alice", "Person")
    store.add_entity(str(EX.knowledgePlatform), "Knowledge Platform", "Project")
    with pytest.raises(GraphError) as exc:
        store.add_relationship(str(EX.alice), "likes", str(EX.knowledgePlatform))
    assert exc.value.code == "invalid_relationship"


def test_relationship_requires_existing_entities():
    store = GraphStore()
    store.add_entity(str(EX.alice), "Alice", "Person")
    with pytest.raises(GraphError) as exc:
        store.add_relationship(str(EX.alice), "worksOn", str(EX.missing))
    assert exc.value.code == "invalid_entity"


def test_fixture_graph_is_inspectable():
    store = fixture_store()
    snapshot = store.snapshot()
    assert snapshot["format"] == "rdf"
    assert snapshot["entityCount"] == 8
    assert snapshot["relationshipCount"] == 8
    assert snapshot["predicates"] == ["employs", "uses", "worksOn"]
    assert isinstance(store.rdf, Graph)
