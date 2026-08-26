"""Application validation boundary tests."""

from __future__ import annotations

from graph.model import EntityProposal, ExtractionProposal, RelationshipProposal
from graph.validator import validate_proposal
from graph.vocabulary import EX, PREDICATE_BY_LOCAL


def test_works_on_alias_maps_to_worksOn():
    proposal = ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="works_on",
                object="Knowledge Platform",
            )
        ],
    )
    result = validate_proposal(proposal)
    assert len(result.accepted_relationships) == 1
    assert result.accepted_relationships[0].predicate_local == "worksOn"
    assert result.accepted_relationships[0].predicate_iri == str(EX.worksOn)


def test_unsupported_predicate_supervises_rejected():
    proposal = ExtractionProposal(
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="supervises",
                object="Knowledge Platform",
            )
        ]
    )
    result = validate_proposal(proposal)
    assert result.accepted_relationships == []
    assert len(result.rejected_relationships) == 1
    assert result.rejected_relationships[0].reason == "unsupported_predicate"
    assert result.rejected_relationships[0].predicate == "supervises"


def test_arbitrary_predicate_iri_rejected():
    proposal = ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="http://evil.example/supervises",
                object="Knowledge Platform",
            )
        ],
    )
    result = validate_proposal(proposal)
    assert result.accepted_relationships == []
    assert result.rejected_relationships[0].reason == "unsupported_predicate"
    assert "evil.example" in result.rejected_relationships[0].predicate


def test_unknown_entity_label_not_invented():
    proposal = ExtractionProposal(
        entities=[EntityProposal(label="Zed Unknown", entity_type="Person")]
    )
    result = validate_proposal(proposal)
    assert result.resolved_entities == []
    assert result.unresolved_labels == ["Zed Unknown"]


def test_known_entity_mismatched_type_rejected():
    """Registry type is authoritative — Alice/Company must not resolve as Person."""
    proposal = ExtractionProposal(
        entities=[EntityProposal(label="Alice", entity_type="Company")]
    )
    result = validate_proposal(proposal)
    assert result.ok is False
    assert result.resolved_entities == []
    assert result.entity_type_errors == ["entity_type_mismatch:Alice:Company:Person"]
    assert any(
        err.startswith("entity_type_mismatch") for err in result.entity_type_errors
    )


def test_entity_type_mismatch_does_not_commit_triples():
    from pathlib import Path

    from config import Settings
    from graph.builder import RdfGraphStore
    from graph.cases import get_case
    from graph.extractor import MockLLMExtractor
    from graph.runner import run_case
    from graph.vocabulary import EX, RDF

    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data" / "graph.ttl"
    case = get_case("entity-extraction-alice")
    store = RdfGraphStore.fresh(start="empty", seed_path=graph_path)
    before = store.triple_count()
    mock = MockLLMExtractor(
        proposals={
            case.trace_id: ExtractionProposal(
                entities=[EntityProposal(label="Alice", entity_type="Company")]
            )
        }
    )
    result = run_case(
        case,
        Settings(graph_path=graph_path, openai_api_key=""),
        mode="llm_assisted",
        extractor=mock,
        store=store,
    )
    assert result.termination == "validation_rejected"
    assert result.validation.entity_type_errors == [
        "entity_type_mismatch:Alice:Company:Person"
    ]
    assert result.triples_created == []
    assert store.triple_count() == before
    assert (EX.alice, RDF.type, EX.Person) not in store.rdf
    assert (EX.alice, RDF.type, EX.Company) not in store.rdf
    rejected = [
        event
        for event in result.sequence
        if event.kind == "validation_rejected"
        and event.detail.get("reason") == "entity_type_mismatch"
    ]
    assert rejected
    assert rejected[0].detail["proposedType"] == "Company"
    assert rejected[0].detail["registryType"] == "Person"


def test_entity_linking_type_mismatch_trace_has_no_accepted_relationship():
    """ENTITY_LINKING mismatch: reject, no accepted/committed uses edge."""
    from pathlib import Path

    from config import Settings
    from graph.builder import RdfGraphStore
    from graph.cases import get_case
    from graph.extractor import MockLLMExtractor
    from graph.runner import run_case
    from graph.trace import build_trace
    from graph.vocabulary import EX

    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data" / "graph.ttl"
    case = get_case("entity-linking-known-entities")
    store = RdfGraphStore.fresh(start=case.start_graph, seed_path=graph_path)
    before = store.triple_count()
    settings = Settings(graph_path=graph_path, openai_api_key="")
    mock = MockLLMExtractor(
        proposals={
            case.trace_id: ExtractionProposal(
                entities=[
                    EntityProposal(label="Knowledge Platform", entity_type="Company"),
                    EntityProposal(label="PostgreSQL", entity_type="Technology"),
                ],
                relationships=[
                    RelationshipProposal(
                        subject="Knowledge Platform",
                        predicate="uses",
                        object="PostgreSQL",
                    )
                ],
            )
        }
    )
    result = run_case(
        case,
        settings,
        mode="llm_assisted",
        extractor=mock,
        store=store,
    )
    assert result.termination == "validation_rejected"
    assert result.validation.accepted_relationships == []
    assert result.validation.rejected_relationships[0].reason == (
        "entity_validation_failed"
    )
    assert result.triples_created == []
    assert result.graph_unchanged is True
    assert store.triple_count() == before
    assert (EX.knowledgePlatform, EX.uses, EX.postgresql) not in store.rdf

    trace = build_trace(case=case, result=result, settings=settings, store=store)
    assert trace["validation"]["acceptedRelationships"] == []
    assert trace["triplesCreated"] == []
    assert trace["graphUnchanged"] is True
    assert "validation_passed" not in [e["kind"] for e in trace["sequence"]]


def test_valid_entity_type_still_resolves_after_mismatch_rule():
    proposal = ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ]
    )
    result = validate_proposal(proposal)
    assert result.ok is True
    assert result.entity_type_errors == []
    by_label = {e.label: e.entity_type for e in result.resolved_entities}
    assert by_label == {"Alice": "Person", "Knowledge Platform": "Project"}


def test_entity_type_mismatch_does_not_accept_relationship():
    """Failed entity endpoints must not appear as accepted relationships."""
    proposal = ExtractionProposal(
        entities=[
            EntityProposal(label="Knowledge Platform", entity_type="Company"),
            EntityProposal(label="PostgreSQL", entity_type="Technology"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Knowledge Platform",
                predicate="uses",
                object="PostgreSQL",
            )
        ],
    )
    result = validate_proposal(proposal)
    assert result.ok is False
    assert result.entity_type_errors == [
        "entity_type_mismatch:Knowledge Platform:Company:Project"
    ]
    assert result.accepted_relationships == []
    assert len(result.rejected_relationships) == 1
    rejected = result.rejected_relationships[0]
    assert rejected.predicate == "uses"
    assert rejected.reason == "entity_validation_failed"
    assert {e.label for e in result.resolved_entities} == {"PostgreSQL"}


def test_invalid_fact_extraction_contract_preserves_supervises():
    """Extraction contract instructs preserving source verbs like supervises."""
    from graph.extractor import SYSTEM_PROMPT, build_extraction_user_prompt

    source = "Alice supervises the Knowledge Platform."
    user = build_extraction_user_prompt(source)
    assert "supervises" in user
    assert 'source "supervises" → predicate "supervises"' in user
    assert "Do not substitute an allowed predicate" in user
    assert source in user
    assert "supervises" in SYSTEM_PROMPT
    assert "not employs or works_on" in SYSTEM_PROMPT.replace("\n", " ")


def test_invalid_fact_supervises_rejected_graph_unchanged():
    from pathlib import Path

    from config import Settings
    from graph.builder import RdfGraphStore
    from graph.cases import get_case
    from graph.extractor import StructuredExtractor
    from graph.runner import run_case

    root = Path(__file__).resolve().parents[1]
    graph_path = root / "data" / "graph.ttl"
    case = get_case("invalid-fact-unsupported-predicate")
    assert case.source_text == "Alice supervises the Knowledge Platform."
    store = RdfGraphStore.fresh(start=case.start_graph, seed_path=graph_path)
    before = store.triple_count()
    result = run_case(
        case,
        Settings(graph_path=graph_path, openai_api_key=""),
        mode="structured",
        extractor=StructuredExtractor(),
        store=store,
    )
    assert result.proposal.relationships[0].predicate == "supervises"
    assert result.termination == "validation_rejected"
    assert result.validation.accepted_relationships == []
    assert result.validation.rejected_relationships[0].reason == "unsupported_predicate"
    assert result.triples_created == []
    assert result.graph_unchanged is True
    assert store.triple_count() == before


def test_entity_labels_resolve_to_stable_iris():
    proposal = ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
            EntityProposal(label="PostgreSQL", entity_type="Technology"),
        ]
    )
    result = validate_proposal(proposal)
    by_label = {e.label: e.iri for e in result.resolved_entities}
    assert by_label["Alice"] == str(EX.alice)
    assert by_label["Knowledge Platform"] == str(EX.knowledgePlatform)
    assert by_label["PostgreSQL"] == str(EX.postgresql)


def test_only_allowed_predicates_accepted():
    for local in PREDICATE_BY_LOCAL:
        proposal = ExtractionProposal(
            entities=[
                EntityProposal(label="Acme AI", entity_type="Company"),
                EntityProposal(label="Alice", entity_type="Person"),
            ],
            relationships=[
                RelationshipProposal(
                    subject="Acme AI" if local == "employs" else "Alice",
                    predicate=local,
                    object="Alice" if local == "employs" else "Knowledge Platform",
                )
            ],
        )
        # Ensure endpoints exist for worksOn/uses.
        if local != "employs":
            proposal.entities.append(
                EntityProposal(label="Knowledge Platform", entity_type="Project")
            )
            if local == "uses":
                proposal.entities = [
                    EntityProposal(label="Knowledge Platform", entity_type="Project"),
                    EntityProposal(label="PostgreSQL", entity_type="Technology"),
                ]
                proposal.relationships = [
                    RelationshipProposal(
                        subject="Knowledge Platform",
                        predicate="uses",
                        object="PostgreSQL",
                    )
                ]
        result = validate_proposal(proposal)
        assert result.accepted_relationships, f"expected accept for {local}"
        assert result.accepted_relationships[0].predicate_local == local
