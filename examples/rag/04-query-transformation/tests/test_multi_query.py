"""Multi-query retrieval and aggregation tests with injection."""

from rag.chunker import Chunk
from rag.fusion import FusedChunk
from rag.retriever import QueryRun, aggregate_multi_query_rrf, retrieve_all
from rag.store import RankedChunk


def _ranked(chunk_id: str, rank: int, score: float) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(id=chunk_id, text=chunk_id, start=0, end=1), score=score, rank=rank
    )


def _fused(chunk_id: str, rank: int) -> FusedChunk:
    return FusedChunk(
        chunk=Chunk(id=chunk_id, text=chunk_id, start=0, end=1),
        rrf_score=1 / (60 + rank),
        rank=rank,
        dense_rank=rank,
        lexical_rank=None,
        dense_score=0.5,
        lexical_score=None,
    )


def test_dedup_and_provenance_union_across_queries():
    q1 = QueryRun(
        query="q1",
        dense=[_ranked("A", 1, 0.9)],
        lexical=[],
        fused=[_fused("A", 1), _fused("B", 2)],
        latency_ms=2,
    )
    q2 = QueryRun(
        query="q2",
        dense=[_ranked("A", 2, 0.8)],
        lexical=[],
        fused=[_fused("C", 1), _fused("A", 2)],
        latency_ms=3,
    )
    merged = aggregate_multi_query_rrf([q1, q2], rrf_k=60, top_k=10)
    by_id = {m.chunk.id: m for m in merged}
    assert set(by_id.keys()) == {"A", "B", "C"}
    assert len(by_id["A"].provenance) == 2


def test_multi_query_aggregation_prefers_consistent_hits():
    q1 = QueryRun("q1", [], [], [_fused("A", 1), _fused("B", 2)], 1)
    q2 = QueryRun("q2", [], [], [_fused("A", 2), _fused("C", 1)], 1)
    merged = aggregate_multi_query_rrf([q1, q2], rrf_k=60, top_k=3)
    assert merged[0].chunk.id == "A"


def test_reranking_uses_original_question(monkeypatch):
    def fake_single(_client, _store, _bm25, query: str, **_kwargs):
        return QueryRun(query, [], [], [_fused("A", 1)], 1)

    captured = {"query": ""}

    def score_fn(query: str, _texts: list[str]) -> list[float]:
        captured["query"] = query
        return [0.9]

    monkeypatch.setattr("rag.retriever._run_single_query", fake_single)
    views = retrieve_all(
        client=None,
        store=None,
        bm25=None,
        query="ORIGINAL",
        max_alternative_queries=2,
        embedding_model="unused",
        dense_top_k=1,
        lexical_top_k=1,
        candidate_k=1,
        final_context_k=1,
        rrf_k=60,
        merge_top_k=1,
        transform_fn=lambda _q, _m: ["ALT1", "ALT2"],
        score_fn=score_fn,
    )
    assert captured["query"] == "ORIGINAL"
    assert views.transformed_queries[0] == "ORIGINAL"
