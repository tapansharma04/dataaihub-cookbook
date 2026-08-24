"""Graph store tests."""

from pathlib import Path

from graphrag.graph import RdfGraphStore
from graphrag.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def test_graph_loads_entities_and_relationships():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    snapshot = store.snapshot()
    assert snapshot["entityCount"] == 8
    assert snapshot["relationshipCount"] == 9
    assert snapshot["namespace"] == str(EX)


def test_knowledge_platform_uses_postgresql_and_redis():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    kp = str(EX.knowledgePlatform)
    uses = {str(obj) for obj in store.rdf.objects(EX.knowledgePlatform, EX.uses)}
    assert str(EX.postgresql) in uses
    assert str(EX.redis) in uses
    assert store.label_for(kp) == "Knowledge Platform"
