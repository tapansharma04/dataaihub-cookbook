"""Context assembly tests."""

from pathlib import Path

from graphrag.context import assemble_context
from graphrag.graph import RdfGraphStore
from graphrag.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def test_context_from_facts():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    facts = [
        store.make_fact(EX.alice, EX.worksOn, EX.knowledgePlatform),
        store.make_fact(EX.knowledgePlatform, EX.uses, EX.postgresql),
    ]
    context, ms = assemble_context(facts)
    assert context == [
        "Alice works on Knowledge Platform.",
        "Knowledge Platform uses PostgreSQL.",
    ]
    assert ms >= 0


def test_unrelated_triples_not_in_context():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    facts = [store.make_fact(EX.carol, EX.worksOn, EX.billingPortal)]
    context, _ = assemble_context(facts)
    assert all("Carol" in line for line in context)
    assert all("Billing Portal" in line for line in context)
    assert not any("Redis" in line for line in context)
