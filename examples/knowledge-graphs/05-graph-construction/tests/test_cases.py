"""Measured case behavior tests."""

from __future__ import annotations

from pathlib import Path

from rdflib.namespace import RDFS

from config import Settings
from graph.builder import RdfGraphStore
from graph.cases import CASES, get_case
from graph.extractor import MockLLMExtractor, StructuredExtractor
from graph.model import ExtractionProposal, RelationshipProposal
from graph.runner import run_case
from graph.vocabulary import EX, RDF

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH, openai_api_key="")


def _run(trace_id: str, *, mode: str = "structured", extractor=None):
    case = get_case(trace_id)
    store = RdfGraphStore.fresh(start=case.start_graph, seed_path=GRAPH_PATH)
    if extractor is None:
        extractor = StructuredExtractor()
    result = run_case(
        case,
        _settings(),
        mode=mode,  # type: ignore[arg-type]
        extractor=extractor,
        store=store,
    )
    return result, store


def test_four_cases_defined():
    assert len(CASES) == 4
    assert {c.example_class for c in CASES} == {
        "ENTITY_EXTRACTION",
        "RELATIONSHIP_EXTRACTION",
        "ENTITY_LINKING",
        "INVALID_FACT",
    }


def test_entity_extraction():
    result, store = _run("entity-extraction-alice")
    assert result.termination == "completed"
    assert {e.label for e in result.validation.resolved_entities} == {
        "Alice",
        "Knowledge Platform",
    }
    assert (EX.alice, RDF.type, EX.Person) in store.rdf
    assert (EX.knowledgePlatform, RDF.type, EX.Project) in store.rdf
    assert store.label_for(EX.alice) == "Alice"
    assert store.label_for(EX.knowledgePlatform) == "Knowledge Platform"
    assert result.metrics.relationships_accepted == 0
    assert result.metrics.triples_created == 4  # 2 type + 2 label


def test_relationship_extraction():
    result, store = _run("relationship-extraction-alice-platform")
    assert result.termination == "completed"
    assert (EX.alice, EX.worksOn, EX.knowledgePlatform) in store.rdf
    assert result.metrics.relationships_accepted == 1
    assert result.metrics.triples_created == 1
    kinds = [e.kind for e in result.sequence]
    assert "relationship_proposed" in kinds
    assert "validation_passed" in kinds
    assert "triple_created" in kinds


def test_entity_linking():
    result, store = _run("entity-linking-known-entities")
    assert result.termination == "completed"
    by_label = {e.label: e.iri for e in result.validation.resolved_entities}
    assert by_label["Knowledge Platform"] == str(EX.knowledgePlatform)
    assert by_label["PostgreSQL"] == str(EX.postgresql)
    assert by_label["Knowledge Platform"] != "Knowledge Platform"
    assert (EX.knowledgePlatform, EX.uses, EX.postgresql) in store.rdf


def test_invalid_fact_rejects_and_does_not_mutate_graph():
    result, store = _run("invalid-fact-unsupported-predicate")
    assert result.termination == "validation_rejected"
    assert result.graph_before_count == result.graph_after_count
    assert result.graph_unchanged is True
    assert result.metrics.relationships_rejected == 1
    assert result.metrics.triples_created == 0
    rejected = result.validation.rejected_relationships[0]
    assert rejected.predicate == "supervises"
    assert rejected.reason == "unsupported_predicate"
    # Must not rewrite supervises → worksOn
    assert not store.has_predicate_iri("http://evil.example/supervises")
    # Seed already has alice worksOn knowledgePlatform — that is fine.
    # Ensure no supervises-like predicate exists.
    for _, pred, _ in store.rdf:
        assert "supervises" not in str(pred).lower()


def test_invalid_fact_sequence_shape():
    result, _ = _run("invalid-fact-unsupported-predicate")
    kinds = [e.kind for e in result.sequence]
    assert kinds[0] == "source_loaded"
    assert "extraction_started" in kinds
    assert "relationship_proposed" in kinds
    assert "validation_started" in kinds
    assert "validation_rejected" in kinds
    assert "triple_created" not in kinds
    assert "graph_committed" not in kinds
    assert kinds[-2] == "result"
    assert kinds[-1] == "termination"


def test_malicious_predicate_iri_never_committed():
    case = get_case("relationship-extraction-alice-platform")
    store = RdfGraphStore.fresh(start="empty", seed_path=GRAPH_PATH)
    before = store.triple_count()
    evil = ExtractionProposal(
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="http://evil.example/supervises",
                object="Knowledge Platform",
            )
        ]
    )
    mock = MockLLMExtractor(proposals={case.trace_id: evil})
    result = run_case(
        case,
        _settings(),
        mode="llm_assisted",
        extractor=mock,
        store=store,
    )
    assert result.termination == "validation_rejected"
    assert result.validation.accepted_relationships == []
    assert result.validation.rejected_relationships
    assert store.triple_count() == before
    assert not store.has_predicate_iri("http://evil.example/supervises")
    for _, pred, _ in store.rdf:
        assert str(pred) != "http://evil.example/supervises"


def test_structured_provenance():
    result, _ = _run("entity-extraction-alice")
    assert result.provenance == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert result.metrics.model_turns == 0


def test_mock_llm_extraction():
    mock = MockLLMExtractor()
    result, store = _run(
        "relationship-extraction-alice-platform",
        mode="llm_assisted",
        extractor=mock,
    )
    assert mock.last_source_text == "Alice works on the Knowledge Platform."
    assert result.metrics.model_turns == 1
    assert result.provenance["model"] == "mock"
    assert (EX.alice, EX.worksOn, EX.knowledgePlatform) in store.rdf


def test_entity_extraction_sequence_core_events():
    result, _ = _run("entity-extraction-alice")
    kinds = [e.kind for e in result.sequence]
    assert kinds[0] == "source_loaded"
    assert "extraction_started" in kinds
    assert kinds.count("entity_proposed") == 2
    assert "validation_started" in kinds
    assert kinds.count("entity_resolved") == 2
    assert "triple_created" in kinds
    assert "graph_committed" in kinds
    assert "graph_verified" in kinds
    assert kinds[-1] == "termination"


def test_rdfs_label_not_used_as_iri():
    result, store = _run("entity-extraction-alice")
    for entity in result.validation.resolved_entities:
        assert entity.label != entity.iri
        assert entity.iri.startswith(str(EX))
    for subject, pred, obj in store.rdf:
        if pred == RDFS.label:
            assert str(obj) in {"Alice", "Knowledge Platform"}
            assert str(subject).startswith(str(EX))
