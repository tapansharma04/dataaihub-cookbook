"""RDF fixture and rdflib graph tests — no SPARQL / no LLM."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from graph.store import GraphStore
from graph.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
TTL_PATH = ROOT / "data" / "graph.ttl"
JSON_PATH = ROOT / "data" / "graph.json"


def test_turtle_fixture_is_the_source_of_truth():
    assert TTL_PATH.exists()
    assert not JSON_PATH.exists()


def test_turtle_parses_into_rdflib_graph():
    graph = Graph()
    graph.parse(TTL_PATH, format="turtle")
    assert isinstance(graph, Graph)
    assert len(graph) > 0


def test_expected_rdf_triples_exist():
    graph = Graph()
    graph.parse(TTL_PATH, format="turtle")
    assert (EX.acmeAI, EX.employs, EX.alice) in graph
    assert (EX.acmeAI, EX.employs, EX.bob) in graph
    assert (EX.acmeAI, EX.employs, EX.carol) in graph
    assert (EX.alice, EX.worksOn, EX.knowledgePlatform) in graph
    assert (EX.bob, EX.worksOn, EX.knowledgePlatform) in graph
    assert (EX.carol, EX.worksOn, EX.billingPortal) in graph
    assert (EX.knowledgePlatform, EX.uses, EX.postgresql) in graph
    assert (EX.billingPortal, EX.uses, EX.redis) in graph
    assert (EX.alice, EX.uses, EX.postgresql) not in graph
    assert (EX.knowledgePlatform, EX.worksOn, EX.alice) not in graph


def test_rdfs_labels_live_in_the_rdf_graph():
    graph = Graph()
    graph.parse(TTL_PATH, format="turtle")
    assert graph.value(EX.alice, RDFS.label) == Literal("Alice")
    assert graph.value(EX.acmeAI, RDFS.label) == Literal("Acme AI")
    assert graph.value(EX.knowledgePlatform, RDFS.label) == Literal(
        "Knowledge Platform"
    )
    assert graph.value(EX.postgresql, RDFS.label) == Literal("PostgreSQL")


def test_rdf_types_are_iris_not_blank_nodes():
    graph = Graph()
    graph.parse(TTL_PATH, format="turtle")
    assert (EX.alice, RDF.type, EX.Person) in graph
    assert (EX.acmeAI, RDF.type, EX.Company) in graph
    assert isinstance(EX.alice, URIRef)
    assert not any(
        subject is None for subject, _, _ in graph.triples((EX.alice, None, None))
    )


def test_store_uses_rdflib_graph_as_authority():
    store = GraphStore.from_path(TTL_PATH)
    assert isinstance(store.rdf, Graph)
    assert store.rdf is not None
    assert (EX.alice, EX.worksOn, EX.knowledgePlatform) in store.rdf
    objects = sorted(store.rdf.objects(EX.acmeAI, EX.employs), key=str)
    assert objects == [EX.alice, EX.bob, EX.carol]
    subjects = sorted(store.rdf.subjects(EX.worksOn, EX.knowledgePlatform), key=str)
    assert subjects == [EX.alice, EX.bob]


def test_outgoing_uses_graph_objects():
    store = GraphStore.from_path(TTL_PATH)
    rdf_objects = {str(node) for node in store.rdf.objects(EX.alice, EX.worksOn)}
    neighbors = store.get_neighbors(
        str(EX.alice), predicate="worksOn", direction="outgoing"
    )
    assert {triple.object for triple in neighbors} == rdf_objects
    assert rdf_objects == {str(EX.knowledgePlatform)}


def test_incoming_uses_graph_subjects():
    store = GraphStore.from_path(TTL_PATH)
    rdf_subjects = {
        str(node) for node in store.rdf.subjects(EX.worksOn, EX.knowledgePlatform)
    }
    neighbors = store.get_neighbors(
        str(EX.knowledgePlatform), predicate="worksOn", direction="incoming"
    )
    assert {triple.subject for triple in neighbors} == rdf_subjects
    assert str(EX.carol) not in rdf_subjects
