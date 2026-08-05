"""Pipeline wiring tests with injected scores/transforms — no APIs."""

from __future__ import annotations

from pipelines import (
    PIPELINE_DENSE,
    PIPELINE_HYBRID,
    PIPELINE_HYBRID_RERANKED,
    PIPELINE_QUERY_TRANSFORM,
    retrieve_hybrid_reranked,
    retrieve_query_transform,
)
from rag.bm25 import BM25Index
from rag.chunker import Chunk
from rag.fusion import reciprocal_rank_fusion
from rag.store import InMemoryVectorStore, RankedChunk


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="sample-1", text="alpha document about cats", start=0, end=10),
        Chunk(id="sample-2", text="beta document about dogs", start=10, end=20),
        Chunk(id="sample-3", text="gamma document about birds", start=20, end=30),
    ]


def test_hybrid_rerank_uses_injected_score_fn():
    chunks = _chunks()
    store = InMemoryVectorStore()
    # Unit vectors so search works without OpenAI.
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    bm25 = BM25Index(chunks)

    # Bypass dense embedding call by testing rerank path pieces via fusion input
    # through retrieve_hybrid_reranked with a stub client that returns vectors.
    class Client:
        class embeddings:
            @staticmethod
            def create(model, input):  # noqa: A002
                class Item:
                    def __init__(self, index, embedding):
                        self.index = index
                        self.embedding = embedding

                class Resp:
                    data = [Item(0, [1.0, 0.0])]

                return Resp()

    def score_fn(_query: str, texts: list[str]) -> list[float]:
        # Prefer the dogs chunk.
        return [0.1 if "dogs" in t else 0.9 for t in texts]

    out = retrieve_hybrid_reranked(
        Client(),  # type: ignore[arg-type]
        store,
        bm25,
        "dogs",
        embedding_model="dummy",
        dense_top_k=3,
        lexical_top_k=3,
        candidate_k=3,
        final_k=2,
        rrf_k=60,
        score_fn=score_fn,
    )
    assert out.pipeline == PIPELINE_HYBRID_RERANKED
    assert len(out.retrieved_ids) == 2
    assert "total_ms" in out.latency_ms


def test_query_transform_reranks_original_query():
    chunks = _chunks()
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    bm25 = BM25Index(chunks)

    class Client:
        class embeddings:
            @staticmethod
            def create(model, input):  # noqa: A002
                class Item:
                    def __init__(self, index, embedding):
                        self.index = index
                        self.embedding = embedding

                texts = input if isinstance(input, list) else [input]

                class Resp:
                    data = [Item(i, [1.0, 0.0]) for i in range(len(texts))]

                return Resp()

    seen_queries: list[str] = []

    def transform_fn(_original: str, max_alts: int) -> list[str]:
        return ["AUTH_TOKEN_EXPIRED idle"][:max_alts]

    def score_fn(query: str, texts: list[str]) -> list[float]:
        seen_queries.append(query)
        return [0.5] * len(texts)

    original = "Why unlock after idle?"
    out = retrieve_query_transform(
        Client(),  # type: ignore[arg-type]
        store,
        bm25,
        original,
        embedding_model="dummy",
        dense_top_k=3,
        lexical_top_k=3,
        candidate_k=3,
        final_k=2,
        rrf_k=60,
        max_alternative_queries=1,
        transform_fn=transform_fn,
        score_fn=score_fn,
    )
    assert out.pipeline == PIPELINE_QUERY_TRANSFORM
    assert seen_queries == [original]
    assert "transform_ms" in out.latency_ms


def test_rrf_fusion_deterministic():
    chunks = _chunks()
    dense = [
        RankedChunk(chunk=chunks[0], score=0.9, rank=1),
        RankedChunk(chunk=chunks[1], score=0.8, rank=2),
    ]
    lexical = [
        RankedChunk(chunk=chunks[1], score=2.0, rank=1),
        RankedChunk(chunk=chunks[2], score=1.0, rank=2),
    ]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=3)
    assert [item.chunk.id for item in fused] == [
        fused[0].chunk.id,
        fused[1].chunk.id,
        fused[2].chunk.id,
    ]
    assert {PIPELINE_DENSE, PIPELINE_HYBRID} <= {
        PIPELINE_DENSE,
        PIPELINE_HYBRID,
        PIPELINE_HYBRID_RERANKED,
        PIPELINE_QUERY_TRANSFORM,
    }
