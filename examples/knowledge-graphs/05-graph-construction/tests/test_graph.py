"""RDF graph construction and vocabulary enforcement tests."""

from __future__ import annotations

from pathlib import Path

from rdflib import Literal, URIRef
from rdflib.namespace import RDFS

from config import Settings
from graph.builder import RdfGraphStore
from graph.cases import get_case
from graph.extractor import StructuredExtractor
from graph.model import ResolvedEntity, ValidatedRelationship
from graph.runner import run_case
from graph.vocabulary import EX, PREDICATE_BY_LOCAL, RDF

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH, openai_api_key="")


def test_rdflib_graph_is_authority():
    store = RdfGraphStore.empty()
    assert store.triple_count() == 0
    store.rdf.add((EX.alice, RDF.type, EX.Person))
    assert store.triple_count() == 1


def test_commit_entities_adds_type_and_label():
    store = RdfGraphStore.empty()
    created = store.commit_entities(
        [
            ResolvedEntity(
                label="Alice",
                iri=str(EX.alice),
                entity_type="Person",
            )
        ]
    )
    assert len(created) == 2
    assert (EX.alice, RDF.type, EX.Person) in store.rdf
    assert (EX.alice, RDFS.label, Literal("Alice")) in store.rdf
    assert store.label_for(EX.alice) == "Alice"


def test_commit_relationship_uses_vocabulary_iri():
    store = RdfGraphStore.empty()
    created = store.commit_relationships(
        [
            ValidatedRelationship(
                subject_iri=str(EX.alice),
                predicate_local="worksOn",
                predicate_iri=str(EX.worksOn),
                object_iri=str(EX.knowledgePlatform),
                subject_label="Alice",
                object_label="Knowledge Platform",
            )
        ]
    )
    assert len(created) == 1
    assert (EX.alice, EX.worksOn, EX.knowledgePlatform) in store.rdf


def test_fresh_graph_per_case_isolation():
    settings = _settings()
    extractor = StructuredExtractor()
    case_a = get_case("entity-extraction-alice")
    case_b = get_case("invalid-fact-unsupported-predicate")
    store_a = RdfGraphStore.fresh(start=case_a.start_graph, seed_path=GRAPH_PATH)
    store_b = RdfGraphStore.fresh(start=case_b.start_graph, seed_path=GRAPH_PATH)
    before_b = store_b.triple_count()
    run_case(case_a, settings, mode="structured", extractor=extractor, store=store_a)
    assert store_b.triple_count() == before_b
    assert store_a is not store_b


def test_labels_separate_from_iris():
    store = RdfGraphStore.empty()
    store.commit_entities(
        [
            ResolvedEntity(
                label="Knowledge Platform",
                iri=str(EX.knowledgePlatform),
                entity_type="Project",
            )
        ]
    )
    assert store.label_for(EX.knowledgePlatform) == "Knowledge Platform"
    assert str(EX.knowledgePlatform) != "Knowledge Platform"


def test_no_predicate_outside_vocabulary_in_seed():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    allowed = set(PREDICATE_BY_LOCAL.values()) | {RDF.type, RDFS.label}
    for _, pred, _ in store.rdf:
        if isinstance(pred, URIRef):
            assert pred in allowed, f"unexpected predicate {pred}"


def test_graph_verification_detects_missing_rel():
    store = RdfGraphStore.empty()
    verification = store.verify_committed(
        entities=[],
        relationships=[
            ValidatedRelationship(
                subject_iri=str(EX.alice),
                predicate_local="worksOn",
                predicate_iri=str(EX.worksOn),
                object_iri=str(EX.knowledgePlatform),
                subject_label="Alice",
                object_label="Knowledge Platform",
            )
        ],
        triples=[],
    )
    assert verification["ok"] is False
    assert verification["missing"]
