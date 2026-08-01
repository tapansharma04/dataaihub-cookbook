"""basic-rag — minimal retrieval-augmented generation pipeline.

Example ID: basic-rag

Pipeline:
  Documents → Chunking → Embeddings → Vector store
  Question  → Embed → Similarity search → Prompt → LLM → Answer
"""

from __future__ import annotations

import argparse
import sys

from config import EXAMPLE_ID, get_settings
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.generator import generate_answer
from rag.loader import load_document
from rag.retriever import retrieve
from rag.store import InMemoryVectorStore


def build_index(settings) -> InMemoryVectorStore:
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
    return store


def answer_question(question: str, store: InMemoryVectorStore, settings) -> str:
    client = get_client(settings)
    scored = retrieve(
        client,
        store,
        question,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )
    return generate_answer(
        client,
        question,
        scored,
        model=settings.chat_model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is retrieval-augmented generation?",
        help="Question to answer with RAG",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved chunks before the answer",
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
    store = build_index(settings)
    print(f"[{EXAMPLE_ID}] indexed {len(store)} chunks")

    client = get_client(settings)
    scored = retrieve(
        client,
        store,
        args.question,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )

    if args.show_chunks:
        print("\nRetrieved chunks:")
        for i, item in enumerate(scored, start=1):
            preview = item.chunk.text.replace("\n", " ")[:120]
            print(f"  {i}. {item.chunk.id}  score={item.score:.3f}  {preview}…")

    answer = generate_answer(
        client,
        args.question,
        scored,
        model=settings.chat_model,
    )

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
