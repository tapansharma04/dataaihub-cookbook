"""Vector store tests — no network / no paid APIs."""

import pytest

from rag.chunker import Chunk
from rag.store import InMemoryVectorStore


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c-{i}", text=text, start=0, end=len(text))


def test_search_returns_nearest_neighbor():
    store = InMemoryVectorStore()
    store.add(
        [_chunk(0, "alpha"), _chunk(1, "beta"), _chunk(2, "gamma")],
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c-0"
    assert results[0].score > results[1].score


def test_search_empty_store():
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0], top_k=3) == []


def test_add_length_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([_chunk(0, "x")], [[1.0], [2.0]])


def test_cosine_of_identical_vectors_is_one():
    store = InMemoryVectorStore()
    vec = [0.2, 0.5, 0.8]
    store.add([_chunk(0, "same")], [vec])
    results = store.search(vec, top_k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-9)
