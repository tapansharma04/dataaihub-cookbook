"""Entity resolution tests."""

from pathlib import Path

from graphrag.entity_resolution import resolve_entities
from graphrag.graph import RdfGraphStore
from graphrag.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def test_resolves_knowledge_platform_label():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    resolved, _ = resolve_entities("Who works on the Knowledge Platform?", store)
    assert len(resolved) == 1
    assert resolved[0].iri == str(EX.knowledgePlatform)
    assert resolved[0].label == "Knowledge Platform"


def test_resolves_alice_label():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    resolved, _ = resolve_entities(
        "Which technologies are used by projects Alice works on?",
        store,
    )
    assert len(resolved) == 1
    assert resolved[0].iri == str(EX.alice)


def test_no_match_returns_empty():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    resolved, candidates = resolve_entities("Tell me about Quantum Computing", store)
    assert resolved == []
    assert candidates > 0


def test_resolution_is_deterministic():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    question = "Which company employs Alice and what project does she work on?"
    first, _ = resolve_entities(question, store)
    second, _ = resolve_entities(question, store)
    assert first == second
