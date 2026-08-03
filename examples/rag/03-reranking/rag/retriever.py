"""Orchestrate dense, lexical, hybrid, and reranked retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from rag.bm25 import BM25Index
from rag.embeddings import embed_query
from rag.fusion import FusedChunk, reciprocal_rank_fusion
from rag.reranker import CrossEncoderReranker, RerankedChunk, ScoreFn, rerank_candidates
from rag.store import InMemoryVectorStore, RankedChunk


@dataclass(frozen=True)
class RetrievalViews:
    """Side-by-side retrieval results for teaching and CLI compare output."""

    dense: list[RankedChunk]
    lexical: list[RankedChunk]
    hybrid: list[FusedChunk]
    reranked: list[RerankedChunk]


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
    candidate_k: int,
    final_context_k: int,
    rrf_k: int,
    reranker: CrossEncoderReranker | None = None,
    score_fn: ScoreFn | None = None,
) -> RetrievalViews:
    """Hybrid candidate generation, then local cross-encoder refinement.

    Provide either ``reranker`` or ``score_fn``. Tests inject ``score_fn`` to
    avoid downloading model weights.
    """
    if reranker is None and score_fn is None:
        raise ValueError("provide reranker or score_fn")

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
        top_k=candidate_k,
    )

    if score_fn is not None:
        reranked = rerank_candidates(
            query,
            hybrid,
            top_k=final_context_k,
            score_fn=score_fn,
        )
    else:
        assert reranker is not None
        reranked = reranker.rerank(query, hybrid, top_k=final_context_k)

    return RetrievalViews(
        dense=dense,
        lexical=lexical,
        hybrid=hybrid,
        reranked=reranked,
    )
