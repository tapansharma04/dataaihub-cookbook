"""Retrieval pipelines representing Cookbook examples 01–04.

Each pipeline returns a ranked list of chunk IDs (and measured timings).
Relevance judgments always attach to the original information need — never
to transformed query strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from openai import OpenAI

from rag.bm25 import BM25Index
from rag.embeddings import embed_query
from rag.fusion import FusedChunk, reciprocal_rank_fusion
from rag.query_transformer import TransformFn, build_multi_queries
from rag.reranker import CrossEncoderReranker, ScoreFn, rerank_candidates
from rag.store import InMemoryVectorStore

PIPELINE_DENSE = "dense"
PIPELINE_HYBRID = "hybrid"
PIPELINE_HYBRID_RERANKED = "hybrid-reranked"
PIPELINE_QUERY_TRANSFORM = "query-transform"

ALL_PIPELINES = (
    PIPELINE_DENSE,
    PIPELINE_HYBRID,
    PIPELINE_HYBRID_RERANKED,
    PIPELINE_QUERY_TRANSFORM,
)


@dataclass(frozen=True)
class RetrievalOutput:
    """Ranked retrieval result for one query / pipeline."""

    pipeline: str
    query: str
    retrieved_ids: list[str]
    latency_ms: dict[str, int]


def _ms(start: float) -> int:
    return int(round((perf_counter() - start) * 1000))


def retrieve_dense(
    client: OpenAI,
    store: InMemoryVectorStore,
    query: str,
    *,
    embedding_model: str,
    top_k: int,
) -> RetrievalOutput:
    t0 = perf_counter()
    vector = embed_query(client, query, embedding_model)
    ranked = store.search(vector, top_k=top_k)
    latency = _ms(t0)
    return RetrievalOutput(
        pipeline=PIPELINE_DENSE,
        query=query,
        retrieved_ids=[item.chunk.id for item in ranked],
        latency_ms={"retrieval_ms": latency, "total_ms": latency},
    )


def retrieve_hybrid(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    top_k: int,
    rrf_k: int,
) -> RetrievalOutput:
    t0 = perf_counter()
    dense = store.search(embed_query(client, query, embedding_model), top_k=dense_top_k)
    lexical = bm25.search(query, top_k=lexical_top_k)
    fused = reciprocal_rank_fusion(dense, lexical, k=rrf_k, top_k=top_k)
    latency = _ms(t0)
    return RetrievalOutput(
        pipeline=PIPELINE_HYBRID,
        query=query,
        retrieved_ids=[item.chunk.id for item in fused],
        latency_ms={"retrieval_ms": latency, "total_ms": latency},
    )


def retrieve_hybrid_reranked(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    final_k: int,
    rrf_k: int,
    reranker: CrossEncoderReranker | None = None,
    score_fn: ScoreFn | None = None,
) -> RetrievalOutput:
    if reranker is None and score_fn is None:
        raise ValueError("provide reranker or score_fn")

    t_ret = perf_counter()
    dense = store.search(embed_query(client, query, embedding_model), top_k=dense_top_k)
    lexical = bm25.search(query, top_k=lexical_top_k)
    hybrid = reciprocal_rank_fusion(dense, lexical, k=rrf_k, top_k=candidate_k)
    retrieval_ms = _ms(t_ret)

    t_rr = perf_counter()
    if score_fn is not None:
        reranked = rerank_candidates(query, hybrid, top_k=final_k, score_fn=score_fn)
    else:
        assert reranker is not None
        reranked = reranker.rerank(query, hybrid, top_k=final_k)
    rerank_ms = _ms(t_rr)

    return RetrievalOutput(
        pipeline=PIPELINE_HYBRID_RERANKED,
        query=query,
        retrieved_ids=[item.chunk.id for item in reranked],
        latency_ms={
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "total_ms": retrieval_ms + rerank_ms,
        },
    )


def _aggregate_multi_query_rrf(
    per_query_fused: list[list[FusedChunk]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[FusedChunk]:
    """Cross-query RRF aggregation matching example 04 semantics."""
    if top_k <= 0:
        return []
    score_by_id: dict[str, float] = {}
    chunk_by_id: dict[str, object] = {}
    meta_by_id: dict[str, FusedChunk] = {}
    for fused in per_query_fused:
        for item in fused:
            cid = item.chunk.id
            chunk_by_id[cid] = item.chunk
            score_by_id[cid] = score_by_id.get(cid, 0.0) + (1.0 / (rrf_k + item.rank))
            # Keep first-seen provenance fields for FusedChunk construction.
            meta_by_id.setdefault(cid, item)
    ranked = sorted(score_by_id.items(), key=lambda pair: (-pair[1], pair[0]))
    out: list[FusedChunk] = []
    for rank, (cid, score) in enumerate(ranked[:top_k], start=1):
        base = meta_by_id[cid]
        out.append(
            FusedChunk(
                chunk=base.chunk,
                rrf_score=score,
                rank=rank,
                dense_rank=base.dense_rank,
                lexical_rank=base.lexical_rank,
                dense_score=base.dense_score,
                lexical_score=base.lexical_score,
            )
        )
    return out


def retrieve_query_transform(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    final_k: int,
    rrf_k: int,
    max_alternative_queries: int,
    transform_fn: TransformFn,
    reranker: CrossEncoderReranker | None = None,
    score_fn: ScoreFn | None = None,
) -> RetrievalOutput:
    """Multi-query hybrid retrieval + rerank against the ORIGINAL query.

    Transformed queries are retrieval aids only. Relevance for evaluation
    remains attached to ``query`` (the original information need).
    """
    if reranker is None and score_fn is None:
        raise ValueError("provide reranker or score_fn")

    t_tx = perf_counter()
    queries = build_multi_queries(
        query,
        max_alternative_queries=max_alternative_queries,
        transform_fn=transform_fn,
    )
    transform_ms = _ms(t_tx)

    t_ret = perf_counter()
    per_query_fused: list[list[FusedChunk]] = []
    for q in queries:
        dense = store.search(embed_query(client, q, embedding_model), top_k=dense_top_k)
        lexical = bm25.search(q, top_k=lexical_top_k)
        fused = reciprocal_rank_fusion(dense, lexical, k=rrf_k, top_k=candidate_k)
        per_query_fused.append(fused)
    merged = _aggregate_multi_query_rrf(per_query_fused, rrf_k=rrf_k, top_k=candidate_k)
    retrieval_ms = _ms(t_ret)

    # Rerank with the ORIGINAL information need (example 04 semantics).
    t_rr = perf_counter()
    if score_fn is not None:
        reranked = rerank_candidates(query, merged, top_k=final_k, score_fn=score_fn)
    else:
        assert reranker is not None
        reranked = reranker.rerank(query, merged, top_k=final_k)
    rerank_ms = _ms(t_rr)

    return RetrievalOutput(
        pipeline=PIPELINE_QUERY_TRANSFORM,
        query=query,
        retrieved_ids=[item.chunk.id for item in reranked],
        latency_ms={
            "transform_ms": transform_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "total_ms": transform_ms + retrieval_ms + rerank_ms,
        },
    )


RetrieverFn = Callable[[str], RetrievalOutput]


def build_pipeline_runners(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    eval_k: int,
    rrf_k: int,
    max_alternative_queries: int,
    transform_fn: TransformFn | None = None,
    reranker: CrossEncoderReranker | None = None,
    score_fn: ScoreFn | None = None,
    pipelines: tuple[str, ...] = ALL_PIPELINES,
) -> dict[str, RetrieverFn]:
    """Build callable retrievers for the selected pipeline names."""
    runners: dict[str, RetrieverFn] = {}

    if PIPELINE_DENSE in pipelines:

        def dense(q: str) -> RetrievalOutput:
            return retrieve_dense(
                client,
                store,
                q,
                embedding_model=embedding_model,
                top_k=eval_k,
            )

        runners[PIPELINE_DENSE] = dense

    if PIPELINE_HYBRID in pipelines:

        def hybrid(q: str) -> RetrievalOutput:
            return retrieve_hybrid(
                client,
                store,
                bm25,
                q,
                embedding_model=embedding_model,
                dense_top_k=dense_top_k,
                lexical_top_k=lexical_top_k,
                top_k=eval_k,
                rrf_k=rrf_k,
            )

        runners[PIPELINE_HYBRID] = hybrid

    if PIPELINE_HYBRID_RERANKED in pipelines:

        def hybrid_reranked(q: str) -> RetrievalOutput:
            return retrieve_hybrid_reranked(
                client,
                store,
                bm25,
                q,
                embedding_model=embedding_model,
                dense_top_k=dense_top_k,
                lexical_top_k=lexical_top_k,
                candidate_k=candidate_k,
                final_k=eval_k,
                rrf_k=rrf_k,
                reranker=reranker,
                score_fn=score_fn,
            )

        runners[PIPELINE_HYBRID_RERANKED] = hybrid_reranked

    if PIPELINE_QUERY_TRANSFORM in pipelines:
        if transform_fn is None:
            raise ValueError("query-transform pipeline requires transform_fn")

        def query_transform(q: str) -> RetrievalOutput:
            return retrieve_query_transform(
                client,
                store,
                bm25,
                q,
                embedding_model=embedding_model,
                dense_top_k=dense_top_k,
                lexical_top_k=lexical_top_k,
                candidate_k=candidate_k,
                final_k=eval_k,
                rrf_k=rrf_k,
                max_alternative_queries=max_alternative_queries,
                transform_fn=transform_fn,
                reranker=reranker,
                score_fn=score_fn,
            )

        runners[PIPELINE_QUERY_TRANSFORM] = query_transform

    return runners
