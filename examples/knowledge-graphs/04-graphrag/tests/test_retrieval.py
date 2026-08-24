"""Graph retrieval tests."""

from pathlib import Path

from graphrag.cases import get_case
from graphrag.entity_resolution import resolve_entities
from graphrag.graph import RdfGraphStore
from graphrag.retrieval import retrieve_subgraph
from graphrag.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _store() -> RdfGraphStore:
    return RdfGraphStore.from_path(GRAPH_PATH)


def test_entity_retrieval_finds_alice_and_bob():
    store = _store()
    case = get_case("entity-retrieval-knowledge-platform")
    resolved, _ = resolve_entities(case.question, store)
    result = retrieve_subgraph(store, resolved, case)
    subjects = {fact.subject.iri for fact in result.facts}
    assert subjects == {str(EX.alice), str(EX.bob)}
    assert all(fact.object.iri == str(EX.knowledgePlatform) for fact in result.facts)


def test_multi_hop_reaches_postgresql_and_redis():
    store = _store()
    case = get_case("multi-hop-alice-technologies")
    resolved, _ = resolve_entities(case.question, store)
    result = retrieve_subgraph(store, resolved, case)
    technologies = {
        fact.object.iri for fact in result.facts if fact.predicate.iri == str(EX.uses)
    }
    assert technologies == {str(EX.postgresql), str(EX.redis)}
    assert result.hops_used == 2
    assert len(result.paths) == 2


def test_relationship_grounded_retrieves_employer_and_project():
    store = _store()
    case = get_case("relationship-grounded-alice-employer-project")
    resolved, _ = resolve_entities(case.question, store)
    result = retrieve_subgraph(store, resolved, case)
    predicates = {fact.predicate.iri for fact in result.facts}
    assert str(EX.employs) in predicates
    assert str(EX.worksOn) in predicates


def test_no_relevant_subgraph_for_direct_uses():
    store = _store()
    case = get_case("no-relevant-subgraph-alice-direct-uses")
    resolved, _ = resolve_entities(case.question, store)
    result = retrieve_subgraph(store, resolved, case)
    assert result.facts == []
    assert result.paths == []


def test_retrieval_respects_max_hops_setting():
    store = _store()
    case = get_case("multi-hop-alice-technologies")
    resolved, _ = resolve_entities(case.question, store)
    # Case requires 2 hops — validation happens in runner; direct call still bounded.
    result = retrieve_subgraph(store, resolved, case)
    assert result.hops_used <= case.max_hops


def test_retrieval_is_deterministic():
    store = _store()
    case = get_case("multi-hop-alice-technologies")
    resolved, _ = resolve_entities(case.question, store)
    first = retrieve_subgraph(store, resolved, case)
    second = retrieve_subgraph(store, resolved, case)
    assert [fact.sort_key() for fact in first.facts] == [
        fact.sort_key() for fact in second.facts
    ]
