"""Prompt / generation-context tests — no paid APIs."""

from rag.chunker import Chunk
from rag.fusion import FusedChunk
from rag.generator import build_prompt
from rag.reranker import RerankedChunk, order_by_scores
from rag.store import RankedChunk


def _fused(chunk_id: str, rank: int, text: str) -> FusedChunk:
    return FusedChunk(
        chunk=Chunk(id=chunk_id, text=text, start=0, end=len(text)),
        rrf_score=0.03,
        rank=rank,
        dense_rank=rank,
        lexical_rank=None,
        dense_score=0.5,
        lexical_score=None,
    )


def test_build_prompt_for_ranked_chunks():
    scored = [
        RankedChunk(
            chunk=Chunk(id="sample-0", text="Restart the router.", start=0, end=18),
            score=0.81,
            rank=1,
        )
    ]
    prompt = build_prompt("How do I restore connectivity?", scored)
    assert "Restart the router." in prompt
    assert "sample-0" in prompt


def test_build_prompt_for_reranked_chunks_includes_provenance():
    reranked = [
        RerankedChunk(
            chunk=Chunk(id="sample-1", text="E_CONN_42 remediation.", start=0, end=22),
            reranker_score=0.91,
            rank=1,
            previous_rank=3,
            rrf_score=0.0328,
            dense_rank=2,
            lexical_rank=1,
            dense_score=0.55,
            lexical_score=4.2,
        )
    ]
    prompt = build_prompt("How do I fix E_CONN_42?", reranked)
    assert "rerank=0.9100" in prompt
    assert "prev_rank=3" in prompt
    assert "E_CONN_42 remediation." in prompt


def test_prompt_uses_reranked_order_not_hybrid_order():
    """Critical boundary: generation context follows reranked top-k."""
    hybrid = [
        _fused("A", 1, "hybrid-first somewhat relevant"),
        _fused("B", 2, "hybrid-second related"),
        _fused("C", 3, "BEST ANSWER for the query"),
    ]
    # Reranker promotes C over A/B.
    reranked = order_by_scores(hybrid, [0.4, 0.5, 0.95], top_k=2)
    assert [item.chunk.id for item in reranked] == ["C", "B"]

    prompt = build_prompt("Need the best answer", reranked)
    # Prompt context order must match reranked order, not RRF order.
    pos_c = prompt.index("BEST ANSWER for the query")
    pos_b = prompt.index("hybrid-second related")
    assert pos_c < pos_b
    assert "hybrid-first somewhat relevant" not in prompt
    assert "prev_rank=3" in prompt


def test_build_prompt_handles_empty_retrieval():
    prompt = build_prompt("Anything?", [])
    assert "no documents retrieved" in prompt
