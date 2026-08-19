"""Traversal tests — explicit RDF hops, direction, no-path, and depth limit."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from graph.model import GraphError, Hop, TraversalRequest
from graph.store import GraphStore
from graph.traversal import traverse
from graph.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _store() -> GraphStore:
    return GraphStore.from_path(GRAPH_PATH)


def _settings(**kwargs: object) -> Settings:
    return Settings(graph_path=GRAPH_PATH, **kwargs)


def _run(start_id: str, hops: list[Hop], settings: Settings | None = None):
    settings = settings or _settings()
    request = TraversalRequest(start_id=start_id, hops=hops)
    return traverse(_store(), request, settings=settings)


def test_direct_traversal_employs():
    paths, answers, start, entities_visited, relationships_visited = _run(
        str(EX.acmeAI),
        [Hop(predicate="employs", direction="outgoing")],
    )
    assert start.id == str(EX.acmeAI)
    assert [entity.id for entity in answers] == [
        str(EX.alice),
        str(EX.bob),
        str(EX.carol),
    ]
    assert all(path.depth == 1 for path in paths)
    assert entities_visited == 4
    assert relationships_visited == 3


def test_multi_hop_traversal_alice_to_technology():
    paths, answers, _, _, _ = _run(
        str(EX.alice),
        [
            Hop(predicate="worksOn", direction="outgoing"),
            Hop(predicate="uses", direction="outgoing"),
        ],
    )
    assert [entity.id for entity in answers] == [str(EX.postgresql)]
    assert paths[0].depth == 2
    assert [entity.id for entity in paths[0].entities] == [
        str(EX.alice),
        str(EX.knowledgePlatform),
        str(EX.postgresql),
    ]
    assert [triple.predicate for triple in paths[0].relationships] == [
        str(EX.worksOn),
        str(EX.uses),
    ]


def test_relationship_direction_incoming_works_on():
    _, answers, _, _, _ = _run(
        str(EX.knowledgePlatform),
        [Hop(predicate="worksOn", direction="incoming")],
    )
    assert [entity.id for entity in answers] == [str(EX.alice), str(EX.bob)]
    assert str(EX.carol) not in [entity.id for entity in answers]


def test_outgoing_works_on_from_project_is_not_people():
    paths, answers, _, _, _ = _run(
        str(EX.knowledgePlatform),
        [Hop(predicate="worksOn", direction="outgoing")],
    )
    assert paths == []
    assert answers == []


def test_no_path_does_not_invent_multi_hop():
    paths, answers, _, entities_visited, relationships_visited = _run(
        str(EX.alice),
        [Hop(predicate="uses", direction="outgoing")],
    )
    assert paths == []
    assert answers == []
    assert entities_visited == 1
    assert relationships_visited == 0
    assert all(entity.id != str(EX.postgresql) for entity in answers)


def test_employs_is_not_works_on():
    paths, answers, _, _, _ = _run(
        str(EX.alice),
        [Hop(predicate="employs", direction="outgoing")],
    )
    assert paths == []
    assert answers == []


def test_traversal_depth_limit():
    settings = _settings(max_traversal_depth=1)
    with pytest.raises(GraphError) as exc:
        _run(
            str(EX.alice),
            [
                Hop(predicate="worksOn", direction="outgoing"),
                Hop(predicate="uses", direction="outgoing"),
            ],
            settings=settings,
        )
    assert exc.value.code == "depth_limit"


def test_empty_hop_list_rejected():
    with pytest.raises(GraphError) as exc:
        _run(str(EX.alice), [])
    assert exc.value.code == "invalid_relationship"


def test_invalid_entity_start():
    with pytest.raises(GraphError) as exc:
        _run(str(EX.nobody), [Hop(predicate="uses", direction="outgoing")])
    assert exc.value.code == "invalid_entity"


def test_invalid_relationship_hop():
    with pytest.raises(GraphError) as exc:
        _run(str(EX.alice), [Hop(predicate="likes", direction="outgoing")])
    assert exc.value.code == "invalid_relationship"
