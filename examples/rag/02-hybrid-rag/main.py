"""hybrid-rag — dense + BM25 retrieval fused with Reciprocal Rank Fusion.

Example ID: hybrid-rag

Pipeline:
  Documents → Chunking → Embeddings + BM25 index
  Question  → Dense search + Lexical search → RRF → Prompt → LLM → Answer
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
    print("\nHYBRID (RRF)")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(
            f"  #{item.rank}  {item.chunk.id:<12}  rrf={item.rrf_score:.6f}  "
            f"dense_rank={item.dense_rank}  lexical_rank={item.lexical_rank}  "
            f"{_preview(item.chunk.text)}"
        )


def print_compare(views: RetrievalViews) -> None:
    print_ranked("DENSE (cosine)", views.dense)
    print_ranked("LEXICAL (BM25)", views.lexical)
    print_hybrid(views.hybrid)


def selected_context(
    mode: str,
    views: RetrievalViews,
) -> list[RankedChunk] | list[FusedChunk]:
    if mode == "dense":
        return views.dense
    if mode == "lexical":
        return views.lexical
    return views.hybrid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="How do I fix error E_CONN_42?",
        help="Question to answer with hybrid RAG",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "lexical", "hybrid"),
        default="hybrid",
        help="Retrieval mode used for generation (default: hybrid)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print dense, lexical, and hybrid rankings before generation",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the chunks used for generation (implied by --compare)",
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
    print(f"[{EXAMPLE_ID}] indexed {len(indexes.store)} chunks (dense + BM25)")

    client = get_client(settings)
    views = retrieve_all(
        client,
        indexes.store,
        indexes.bm25,
        args.question,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        lexical_top_k=settings.lexical_top_k,
        hybrid_top_k=settings.hybrid_top_k,
        rrf_k=settings.rrf_k,
    )

    if args.compare:
        print_compare(views)
    elif args.show_chunks:
        chosen = selected_context(args.mode, views)
        if args.mode == "hybrid":
            print_hybrid(chosen)  # type: ignore[arg-type]
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
