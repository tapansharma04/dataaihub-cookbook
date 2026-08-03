"""Reranker logic tests — deterministic scores, no model download."""

from rag.chunker import Chunk
from rag.fusion import FusedChunk
from rag.reranker import (
    build_pairs,
    order_by_scores,
    rank_movement,
    rerank_candidates,
)


def _fused(
    chunk_id: str,
    rank: int,
    text: str | None = None,
    *,
    rrf: float = 0.03,
) -> FusedChunk:
    return FusedChunk(
        chunk=Chunk(id=chunk_id, text=text or chunk_id, start=0, end=1),
        rrf_score=rrf,
        rank=rank,
        dense_rank=rank,
        lexical_rank=None,
        dense_score=0.5,
        lexical_score=None,
    )


def test_build_pairs_preserves_candidate_order():
    candidates = [_fused("A", 1, "alpha"), _fused("B", 2, "beta")]
    pairs = build_pairs("q", candidates)
    assert pairs == [("q", "alpha"), ("q", "beta")]


def test_order_by_scores_reorders_and_keeps_provenance():
    candidates = [
        _fused("A", 1, rrf=0.04),
        _fused("B", 2, rrf=0.03),
        _fused("C", 3, rrf=0.02),
    ]
    # Mock: B best, C mid, A worst.
    ranked = order_by_scores(candidates, [0.4, 0.9, 0.6], top_k=3)
    assert [item.chunk.id for item in ranked] == ["B", "C", "A"]
    assert ranked[0].rank == 1
    assert ranked[0].previous_rank == 2
    assert ranked[0].reranker_score == 0.9
    assert ranked[0].rrf_score == 0.03
    assert ranked[2].previous_rank == 1


def test_rerank_candidates_respects_final_top_k():
    candidates = [_fused("A", 1), _fused("B", 2), _fused("C", 3)]

    def score_fn(_query: str, texts: list[str]) -> list[float]:
        assert texts == ["A", "B", "C"]
        return [0.4, 0.9, 0.6]

    ranked = rerank_candidates("q", candidates, top_k=2, score_fn=score_fn)
    assert [item.chunk.id for item in ranked] == ["B", "C"]


def test_rank_movement_labels():
    assert rank_movement(3, 1) == "↑ 3 → 1"
    assert rank_movement(1, 2) == "↓ 1 → 2"
    assert rank_movement(2, 2) == "—"


def test_order_by_scores_rejects_length_mismatch():
    try:
        order_by_scores([_fused("A", 1)], [0.1, 0.2], top_k=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
