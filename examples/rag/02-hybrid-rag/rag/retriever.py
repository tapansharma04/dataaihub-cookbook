"""Orchestrate dense, lexical, and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from rag.bm25 import BM25Index
from rag.embeddings import embed_query
from rag.fusion import FusedChunk, reciprocal_rank_fusion
from rag.store import InMemoryVectorStore, RankedChunk


@dataclass(frozen=True)
class RetrievalViews:
    """Side-by-side retrieval results for teaching and CLI --compare."""

    dense: list[RankedChunk]
    lexical: list[RankedChunk]
    hybrid: list[FusedChunk]


def retrieve_dense(
    client: OpenAI,
    store: InMemoryVectorStore,
    query: str,
    *,
    embedding_model: str,
    top_k: int,
) -> list[RankedChunk]:
    query_vector = embed_query(client, query, embedding_model)
    return store.search(query_vector, top_k=top_k)


def retrieve_lexical(
    bm25: BM25Index,
    query: str,
    *,
    top_k: int,
) -> list[RankedChunk]:
    return bm25.search(query, top_k=top_k)


def retrieve_all(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    hybrid_top_k: int,
    rrf_k: int,
) -> RetrievalViews:
    dense = retrieve_dense(
        client,
        store,
        query,
        embedding_model=embedding_model,
        top_k=dense_top_k,
    )
    lexical = retrieve_lexical(bm25, query, top_k=lexical_top_k)
    hybrid = reciprocal_rank_fusion(
        dense,
        lexical,
        k=rrf_k,
        top_k=hybrid_top_k,
    )
    return RetrievalViews(dense=dense, lexical=lexical, hybrid=hybrid)
