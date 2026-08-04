"""query-transformation — multi-query retrieval + reranking."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from time import perf_counter

from config import EXAMPLE_ID, Settings, get_settings
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.generator import generate_answer
from rag.loader import load_document
from rag.query_transformer import LLMQueryTransformer
from rag.reranker import CrossEncoderReranker, RerankedChunk, rank_movement
from rag.retriever import RetrievalViews, retrieve_all
from rag.store import InMemoryVectorStore


@dataclass
class Indexes:
    store: InMemoryVectorStore
    bm25: BM25Index


def build_indexes(settings: Settings) -> Indexes:
    text = load_document(settings.data_path)
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source=settings.data_path.stem,
    )
    if not chunks:
        raise RuntimeError(f"No chunks produced from {settings.data_path}")

    client = get_client(settings)
    vectors = embed_texts(
        client,
        [chunk.text for chunk in chunks],
        settings.embedding_model,
    )

    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    bm25 = BM25Index(chunks)
    return Indexes(store=store, bm25=bm25)


def _preview(text: str, width: int = 90) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def print_reranked(items: list[RerankedChunk]) -> None:
    print("\nRERANKED CONTEXT")
    if not items:
        print("  (none)")
        return
    for item in items:
        move = rank_movement(item.previous_rank, item.rank)
        print(
            f"  #{item.rank}  {item.chunk.id:<12}  {move:<12}  "
            f"reranker={item.reranker_score:.4f}  "
            f"prev=#{item.previous_rank}  rrf={item.rrf_score:.6f}  "
            f"{_preview(item.chunk.text)}"
        )


def print_queries(views: RetrievalViews) -> None:
    print(f"\nORIGINAL QUERY\n  {views.original_query}")
    print("\nRETRIEVAL QUERY SET (original + alternatives for this example)")
    for i, q in enumerate(views.transformed_queries):
        marker = "Q0 original" if i == 0 else f"Q{i} alternative"
        print(f"  [{marker}] {q}")


def print_per_query(views: RetrievalViews) -> None:
    print("\nPER-QUERY RETRIEVAL SUMMARY")
    for run in views.per_query:
        print(
            f"- {run.query} | dense={len(run.dense)} lexical={len(run.lexical)} "
            f"fused={len(run.fused)} latency={run.latency_ms}ms"
        )


def print_merge_summary(views: RetrievalViews) -> None:
    before = sum(len(run.fused) for run in views.per_query)
    after = len(views.merged)
    dup = max(before - after, 0)
    rate = (dup / before) if before else 0.0
    print("\nMERGE / DEDUP SUMMARY")
    print(f"  fused candidates before dedup: {before}")
    print(f"  unique candidates after dedup: {after}")
    print(f"  duplicates: {dup} ({rate:.1%})")
    for item in views.merged[: min(5, len(views.merged))]:
        print(
            f"  #{item.rank} {item.chunk.id:<12} agg_rrf={item.rrf_score:.6f} "
            f"seen_in={len(item.provenance)} queries"
        )


def print_timings(
    views: RetrievalViews, rerank_ms: int, gen_ms: int, total_ms: int
) -> None:
    print("\nTIMINGS")
    print(f"  transform: {views.transform_latency_ms}ms")
    print(f"  retrieval total (all queries): {views.retrieval_latency_ms}ms")
    print(f"  rerank: {rerank_ms}ms")
    print(f"  generation: {gen_ms}ms")
    print(f"  total: {total_ms}ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "Why does everyone have to unlock the app again after being idle "
            "half a day?"
        ),
        help="Question to answer with multi-query hybrid retrieval + reranking",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.openai_api_key:
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.",
            file=sys.stderr,
        )
        return 1

    print(f"[{EXAMPLE_ID}] indexing {settings.data_path.name} …")
    indexes = build_indexes(settings)
    print(
        f"[{EXAMPLE_ID}] indexed {len(indexes.store)} chunks "
        f"(dense + BM25); loading reranker {settings.reranker_model} …"
    )
    client = get_client(settings)
    reranker = CrossEncoderReranker(settings.reranker_model)
    transformer = LLMQueryTransformer(client, settings.query_transformer_model)
    print(
        f"[{EXAMPLE_ID}] reranker ready (candidate_k={settings.candidate_k}, "
        f"final_context_k={settings.final_context_k})"
    )

    t_total = perf_counter()
    views = retrieve_all(
        client,
        indexes.store,
        indexes.bm25,
        args.question,
        max_alternative_queries=settings.max_alternative_queries,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        lexical_top_k=settings.lexical_top_k,
        candidate_k=settings.candidate_k,
        final_context_k=settings.final_context_k,
        rrf_k=settings.rrf_k,
        merge_top_k=settings.candidate_k,
        transform_fn=transformer.transform,
        reranker=reranker,
    )
    rerank_ms = views.rerank_latency_ms
    context = views.reranked
    t_gen = perf_counter()
    answer = generate_answer(
        client,
        args.question,
        context,
        model=settings.chat_model,
    )
    gen_ms = int(round((perf_counter() - t_gen) * 1000))
    total_ms = int(round((perf_counter() - t_total) * 1000))

    print_queries(views)
    print_per_query(views)
    print_merge_summary(views)
    print_reranked(views.reranked)
    print_timings(views, rerank_ms, gen_ms, total_ms)

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
