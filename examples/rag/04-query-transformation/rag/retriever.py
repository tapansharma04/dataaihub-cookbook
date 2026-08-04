"""Orchestrate multi-query retrieval, aggregation, and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from openai import OpenAI

from rag.bm25 import BM25Index
from rag.chunker import Chunk
from rag.embeddings import embed_query
from rag.fusion import FusedChunk, reciprocal_rank_fusion
from rag.query_transformer import TransformFn, build_multi_queries
from rag.reranker import CrossEncoderReranker, RerankedChunk, ScoreFn, rerank_candidates
from rag.store import InMemoryVectorStore, RankedChunk


@dataclass(frozen=True)
class QueryRun:
    query: str
    dense: list[RankedChunk]
    lexical: list[RankedChunk]
    fused: list[FusedChunk]
    latency_ms: int


@dataclass(frozen=True)
class CandidateProvenance:
    query: str
    dense_rank: int | None
    lexical_rank: int | None
    fused_rank: int | None
    dense_score: float | None
    lexical_score: float | None
    fused_rrf_score: float | None


@dataclass(frozen=True)
class MergedCandidate:
    chunk: Chunk
    rrf_score: float
    rank: int
    provenance: list[CandidateProvenance]


@dataclass(frozen=True)
class RetrievalViews:
    """Multi-query retrieval views for CLI and trace export."""

    original_query: str
    transformed_queries: list[str]
    per_query: list[QueryRun]
    merged: list[MergedCandidate]
    reranked: list[RerankedChunk]
    transform_latency_ms: int
    retrieval_latency_ms: int
    rerank_latency_ms: int


def _run_single_query(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    rrf_k: int,
) -> QueryRun:
    t0 = perf_counter()
    dense = store.search(embed_query(client, query, embedding_model), top_k=dense_top_k)
    lexical = bm25.search(query, top_k=lexical_top_k)
    fused = reciprocal_rank_fusion(dense, lexical, k=rrf_k, top_k=candidate_k)
    return QueryRun(
        query=query,
        dense=dense,
        lexical=lexical,
        fused=fused,
        latency_ms=int(round((perf_counter() - t0) * 1000)),
    )


def aggregate_multi_query_rrf(
    per_query_fused: list[QueryRun],
    *,
    rrf_k: int,
    top_k: int,
) -> list[MergedCandidate]:
    if top_k <= 0:
        return []
    score_by_id: dict[str, float] = {}
    chunk_by_id: dict[str, object] = {}
    prov_by_id: dict[str, list[CandidateProvenance]] = {}
    for run in per_query_fused:
        dense_by_id = {x.chunk.id: x for x in run.dense}
        lexical_by_id = {x.chunk.id: x for x in run.lexical}
        for item in run.fused:
            chunk_id = item.chunk.id
            chunk_by_id[chunk_id] = item.chunk
            score_by_id[chunk_id] = score_by_id.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + item.rank)
            )
            d = dense_by_id.get(chunk_id)
            lexical_item = lexical_by_id.get(chunk_id)
            prov_by_id.setdefault(chunk_id, []).append(
                CandidateProvenance(
                    query=run.query,
                    dense_rank=item.dense_rank,
                    lexical_rank=item.lexical_rank,
                    fused_rank=item.rank,
                    dense_score=d.score if d else None,
                    lexical_score=lexical_item.score if lexical_item else None,
                    fused_rrf_score=item.rrf_score,
                )
            )
    ranked_ids = sorted(score_by_id.items(), key=lambda pair: (-pair[1], pair[0]))
    out: list[MergedCandidate] = []
    for idx, (chunk_id, score) in enumerate(ranked_ids[:top_k], start=1):
        out.append(
            MergedCandidate(
                chunk=chunk_by_id[chunk_id],
                rrf_score=score,
                rank=idx,
                provenance=prov_by_id.get(chunk_id, []),
            )
        )
    return out


def retrieve_all(
    client: OpenAI,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    query: str,
    *,
    max_alternative_queries: int,
    embedding_model: str,
    dense_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    final_context_k: int,
    rrf_k: int,
    merge_top_k: int,
    transform_fn: TransformFn,
    reranker: CrossEncoderReranker | None = None,
    score_fn: ScoreFn | None = None,
) -> RetrievalViews:
    """Multi-query retrieve, aggregate, then rerank with original query."""
    if reranker is None and score_fn is None:
        raise ValueError("provide reranker or score_fn")

    t0 = perf_counter()
    transformed_queries = build_multi_queries(
        query,
        max_alternative_queries=max_alternative_queries,
        transform_fn=transform_fn,
    )
    transform_latency_ms = int(round((perf_counter() - t0) * 1000))

    per_query: list[QueryRun] = []
    for q in transformed_queries:
        per_query.append(
            _run_single_query(
                client,
                store,
                bm25,
                q,
                embedding_model=embedding_model,
                dense_top_k=dense_top_k,
                lexical_top_k=lexical_top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
        )
    retrieval_latency_ms = sum(run.latency_ms for run in per_query)

    merged = aggregate_multi_query_rrf(
        per_query,
        rrf_k=rrf_k,
        top_k=merge_top_k,
    )
    rerank_input: list[FusedChunk] = [
        FusedChunk(
            chunk=item.chunk,
            rrf_score=item.rrf_score,
            rank=item.rank,
            dense_rank=None,
            lexical_rank=None,
            dense_score=None,
            lexical_score=None,
        )
        for item in merged
    ]

    t_rerank = perf_counter()
    if score_fn is not None:
        reranked = rerank_candidates(
            query,
            rerank_input,
            top_k=final_context_k,
            score_fn=score_fn,
        )
    else:
        assert reranker is not None
        reranked = reranker.rerank(query, rerank_input, top_k=final_context_k)
    rerank_latency_ms = int(round((perf_counter() - t_rerank) * 1000))

    return RetrievalViews(
        original_query=query,
        transformed_queries=transformed_queries,
        per_query=per_query,
        merged=merged,
        reranked=reranked,
        transform_latency_ms=transform_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
    )
