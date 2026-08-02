"""Prompt smoke tests — no paid APIs."""

from rag.chunker import Chunk
from rag.fusion import FusedChunk
from rag.generator import build_prompt
from rag.store import RankedChunk


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


def test_build_prompt_for_fused_chunks_includes_provenance():
    fused = [
        FusedChunk(
            chunk=Chunk(id="sample-1", text="E_CONN_42 remediation.", start=0, end=22),
            rrf_score=0.0328,
            rank=1,
            dense_rank=2,
            lexical_rank=1,
            dense_score=0.55,
            lexical_score=4.2,
        )
    ]
    prompt = build_prompt("How do I fix E_CONN_42?", fused)
    assert "dense_rank=2" in prompt
    assert "lexical_rank=1" in prompt
    assert "E_CONN_42 remediation." in prompt


def test_build_prompt_handles_empty_retrieval():
    prompt = build_prompt("Anything?", [])
    assert "no documents retrieved" in prompt
