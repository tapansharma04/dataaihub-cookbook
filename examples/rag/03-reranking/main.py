"""reranking — hybrid candidates refined by a local cross-encoder.

Example ID: reranking

Pipeline:
  Documents → Chunking → Embeddings + BM25 index
  Question  → Dense + Lexical → RRF (candidate_k)
           → Cross-Encoder rerank → Final top-k → Prompt → LLM → Answer
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from config import EXAMPLE_ID, Settings, get_settings
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.fusion import FusedChunk
from rag.generator import generate_answer
from rag.loader import load_document
from rag.reranker import CrossEncoderReranker, RerankedChunk, rank_movement
from rag.retriever import RetrievalViews, retrieve_all
from rag.store import InMemoryVectorStore, RankedChunk


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


def print_ranked(title: str, items: list[RankedChunk]) -> None:
    print(f"\n{title}")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(
            f"  #{item.rank}  {item.chunk.id:<12}  score={item.score:.4f}  "
            f"{_preview(item.chunk.text)}"
        )


def print_hybrid(items: list[FusedChunk]) -> None:
    print("\nHYBRID CANDIDATES (RRF)")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(
            f"  #{item.rank}  {item.chunk.id:<12}  rrf={item.rrf_score:.6f}  "
            f"dense_rank={item.dense_rank}  lexical_rank={item.lexical_rank}  "
            f"{_preview(item.chunk.text)}"
        )


def print_reranked(items: list[RerankedChunk]) -> None:
    print("\nRERANKED")
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


def print_compare(views: RetrievalViews) -> None:
    print_ranked("DENSE (cosine)", views.dense)
    print_ranked("LEXICAL (BM25)", views.lexical)
    print_hybrid(views.hybrid)
    print_reranked(views.reranked)


def selected_context(
    mode: str,
    views: RetrievalViews,
) -> list[RankedChunk] | list[FusedChunk] | list[RerankedChunk]:
    if mode == "dense":
        return views.dense[:3]
    if mode == "lexical":
        return views.lexical[:3]
    if mode == "hybrid":
        return views.hybrid[:3]
    return views.reranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Is calendar certificate rotation the fix for E_CONN_42?",
        help="Question to answer with hybrid retrieval + reranking",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "lexical", "hybrid", "reranked"),
        default="reranked",
        help="Context used for generation (default: reranked)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print dense, lexical, hybrid candidates, and reranked order",
    )
    parser.add_argument(
        "--show-ranking",
        action="store_true",
        help="Print hybrid candidates and reranked order (default teaching view)",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the chunks used for generation",
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
    reranker = CrossEncoderReranker(settings.reranker_model)
    print(
        f"[{EXAMPLE_ID}] reranker ready (candidate_k={settings.candidate_k}, "
        f"final_context_k={settings.final_context_k})"
    )

    client = get_client(settings)
    views = retrieve_all(
        client,
        indexes.store,
        indexes.bm25,
        args.question,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        lexical_top_k=settings.lexical_top_k,
        candidate_k=settings.candidate_k,
        final_context_k=settings.final_context_k,
        rrf_k=settings.rrf_k,
        reranker=reranker,
    )

    # Default teaching view already shows before/after ranking.
    if args.compare:
        print_compare(views)
    elif args.show_ranking or args.mode == "reranked":
        print_hybrid(views.hybrid)
        print_reranked(views.reranked)
    elif args.show_chunks:
        chosen = selected_context(args.mode, views)
        if args.mode == "hybrid":
            print_hybrid(chosen)  # type: ignore[arg-type]
        elif args.mode == "reranked":
            print_reranked(chosen)  # type: ignore[arg-type]
        else:
            print_ranked(args.mode.upper(), chosen)  # type: ignore[arg-type]

    context = selected_context(args.mode, views)
    answer = generate_answer(
        client,
        args.question,
        context,
        model=settings.chat_model,
    )

    print(f"\nMode:     {args.mode}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
