"""RRF fusion tests — deterministic, no paid APIs."""

from rag.chunker import Chunk
from rag.fusion import reciprocal_rank_fusion
from rag.store import RankedChunk


def _rc(chunk_id: str, rank: int, score: float = 1.0) -> RankedChunk:
    chunk = Chunk(id=chunk_id, text=chunk_id, start=0, end=1)
    return RankedChunk(chunk=chunk, score=score, rank=rank)


def test_rrf_prefers_chunk_strong_in_both_lists():
    # Dense: A, B, C
    # Lexical: B, D, A
    # B = 1/62 + 1/61
    # A = 1/61 + 1/63
    # D = 1/62
    # C = 1/63
    dense = [_rc("A", 1), _rc("B", 2), _rc("C", 3)]
    lexical = [_rc("B", 1), _rc("D", 2), _rc("A", 3)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=4)

    assert [item.chunk.id for item in fused] == ["B", "A", "D", "C"]
    assert fused[0].dense_rank == 2
    assert fused[0].lexical_rank == 1
    assert fused[1].dense_rank == 1
    assert fused[1].lexical_rank == 3


def test_rrf_handles_candidate_only_in_one_list():
    dense = [_rc("A", 1)]
    lexical = [_rc("B", 1)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=2)
    # Both only appear once → equal RRF; stable tie-break by chunk id
    assert {item.chunk.id for item in fused} == {"A", "B"}
    assert fused[0].rrf_score == fused[1].rrf_score


def test_rrf_formula_is_visible():
    dense = [_rc("X", 1)]
    lexical = [_rc("X", 2)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert fused[0].rrf_score == expected


def test_rrf_top_k_and_empty():
    assert reciprocal_rank_fusion([], [], top_k=3) == []
    dense = [_rc("A", 1), _rc("B", 2)]
    fused = reciprocal_rank_fusion(dense, [], k=60, top_k=1)
    assert len(fused) == 1
    assert fused[0].chunk.id == "A"
    assert fused[0].lexical_rank is None
