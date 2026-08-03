"""One-shot: run hybrid-rag and emit guided Lab traces as JSON (measured timings)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import EXAMPLE_ID, get_settings
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_query, embed_texts, get_client
from rag.fusion import reciprocal_rank_fusion
from rag.generator import SYSTEM_PROMPT, build_prompt
from rag.loader import load_document
from rag.store import InMemoryVectorStore

QUESTIONS = [
    {
        "traceId": "dense-friendly-connectivity",
        "question": (
            "What should I do when a customer's internet keeps cutting out "
            "unexpectedly?"
        ),
    },
    {
        "traceId": "lexical-friendly-error-code",
        "question": "How do I fix error E_CONN_42?",
    },
    {
        "traceId": "hybrid-mixed-api-version",
        "question": (
            "How do I load user profiles with NebulaAPI v2.3 without blocking?"
        ),
    },
]


def ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def ranked_detail(items) -> list[dict]:
    return [
        {
            "id": item.chunk.id,
            "text": item.chunk.text,
            "score": round(item.score, 4),
            "rank": item.rank,
            "start": item.chunk.start,
            "end": item.chunk.end,
        }
        for item in items
    ]


def fused_detail(items) -> list[dict]:
    return [
        {
            "id": item.chunk.id,
            "text": item.chunk.text,
            "rrfScore": round(item.rrf_score, 6),
            "rank": item.rank,
            "denseRank": item.dense_rank,
            "lexicalRank": item.lexical_rank,
            "denseScore": (
                round(item.dense_score, 4) if item.dense_score is not None else None
            ),
            "lexicalScore": (
                round(item.lexical_score, 4) if item.lexical_score is not None else None
            ),
            "start": item.chunk.start,
            "end": item.chunk.end,
        }
        for item in items
    ]


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY")

    text = load_document(settings.data_path)
    t0 = time.perf_counter()
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source=settings.data_path.stem,
    )
    chunk_ms = ms(t0)

    client = get_client(settings)

    t0 = time.perf_counter()
    vectors = embed_texts(
        client,
        [c.text for c in chunks],
        settings.embedding_model,
    )
    index_embed_ms = ms(t0)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    t0 = time.perf_counter()
    bm25 = BM25Index(chunks)
    bm25_index_ms = ms(t0)

    corpus = {
        "source": "data/sample.md",
        "chunkCount": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "text": c.text,
                "start": c.start,
                "end": c.end,
            }
            for c in chunks
        ],
    }

    traces = []
    for q in QUESTIONS:
        question = q["question"]

        t0 = time.perf_counter()
        query_vector = embed_query(client, question, settings.embedding_model)
        query_embed_ms = ms(t0)

        t0 = time.perf_counter()
        dense = store.search(query_vector, top_k=settings.dense_top_k)
        dense_ms = ms(t0)

        t0 = time.perf_counter()
        lexical = bm25.search(question, top_k=settings.lexical_top_k)
        lexical_ms = ms(t0)

        t0 = time.perf_counter()
        hybrid = reciprocal_rank_fusion(
            dense,
            lexical,
            k=settings.rrf_k,
            top_k=settings.hybrid_top_k,
        )
        fusion_ms = ms(t0)

        user_prompt = build_prompt(question, hybrid)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        gen_ms = ms(t0)
        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        trace = {
            "labId": EXAMPLE_ID,
            "traceId": q["traceId"],
            "executionMode": "guided",
            "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metricsProvenance": "measured",
            "input": {
                "question": question,
                "config": {
                    "chunkSize": settings.chunk_size,
                    "chunkOverlap": settings.chunk_overlap,
                    "denseTopK": settings.dense_top_k,
                    "lexicalTopK": settings.lexical_top_k,
                    "hybridTopK": settings.hybrid_top_k,
                    "rrfK": settings.rrf_k,
                    "embeddingModel": settings.embedding_model,
                    "chatModel": settings.chat_model,
                },
            },
            "corpus": {
                "source": corpus["source"],
                "chunkCount": corpus["chunkCount"],
            },
            "steps": [
                {
                    "id": "chunking",
                    "type": "chunking",
                    "title": "Chunk documents",
                    "status": "ok",
                    "detail": {
                        "source": corpus["source"],
                        "chunkCount": corpus["chunkCount"],
                        "chunkSize": settings.chunk_size,
                        "chunkOverlap": settings.chunk_overlap,
                        "chunks": corpus["chunks"],
                    },
                    "metrics": {
                        "latencyMs": chunk_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "embed-corpus",
                    "type": "embedding",
                    "title": "Embed chunks",
                    "status": "ok",
                    "detail": {
                        "model": settings.embedding_model,
                        "inputCount": len(chunks),
                        "vectorDimensions": len(vectors[0]) if vectors else 0,
                    },
                    "metrics": {
                        "latencyMs": index_embed_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "bm25-index",
                    "type": "index",
                    "title": "Build BM25 index",
                    "status": "ok",
                    "detail": {
                        "documentCount": len(bm25),
                        "k1": bm25.k1,
                        "b": bm25.b,
                    },
                    "metrics": {
                        "latencyMs": bm25_index_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "embed-query",
                    "type": "embedding",
                    "title": "Embed query",
                    "status": "ok",
                    "detail": {
                        "model": settings.embedding_model,
                        "query": question,
                        "vectorDimensions": len(query_vector),
                    },
                    "metrics": {
                        "latencyMs": query_embed_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "retrieve-dense",
                    "type": "retrieval",
                    "title": "Dense similarity search",
                    "status": "ok",
                    "detail": {
                        "topK": settings.dense_top_k,
                        "similarity": "cosine",
                        "chunks": ranked_detail(dense),
                    },
                    "metrics": {
                        "latencyMs": dense_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "retrieve-lexical",
                    "type": "retrieval",
                    "title": "Lexical BM25 search",
                    "status": "ok",
                    "detail": {
                        "topK": settings.lexical_top_k,
                        "algorithm": "bm25",
                        "chunks": ranked_detail(lexical),
                    },
                    "metrics": {
                        "latencyMs": lexical_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "fuse-rrf",
                    "type": "fusion",
                    "title": "Reciprocal Rank Fusion",
                    "status": "ok",
                    "detail": {
                        "formula": "RRF(d) = Σ 1 / (k + rank(d))",
                        "rrfK": settings.rrf_k,
                        "topK": settings.hybrid_top_k,
                        "inputs": {
                            "denseRanks": [
                                {"id": item.chunk.id, "rank": item.rank}
                                for item in dense
                            ],
                            "lexicalRanks": [
                                {"id": item.chunk.id, "rank": item.rank}
                                for item in lexical
                            ],
                        },
                        "chunks": fused_detail(hybrid),
                    },
                    "metrics": {
                        "latencyMs": fusion_ms,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "prompt",
                    "type": "prompt",
                    "title": "Prompt construction",
                    "status": "ok",
                    "detail": {
                        "system": SYSTEM_PROMPT.strip(),
                        "user": user_prompt,
                        "contextSource": "hybrid",
                    },
                    "metrics": {
                        "latencyMs": 0,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "llm",
                    "type": "llm",
                    "title": "Generation",
                    "status": "ok",
                    "detail": {
                        "model": settings.chat_model,
                        "temperature": 0,
                    },
                    "metrics": {
                        "latencyMs": gen_ms,
                        "promptTokens": prompt_tokens,
                        "completionTokens": completion_tokens,
                        "provenance": "measured",
                    },
                },
                {
                    "id": "output",
                    "type": "output",
                    "title": "Answer",
                    "status": "ok",
                    "detail": {"answer": answer},
                    "metrics": {"provenance": "measured"},
                },
            ],
            "output": {"answer": answer},
            "metrics": {
                "chunkingMs": chunk_ms,
                "corpusEmbeddingMs": index_embed_ms,
                "bm25IndexMs": bm25_index_ms,
                "queryEmbeddingMs": query_embed_ms,
                "denseRetrievalMs": dense_ms,
                "lexicalRetrievalMs": lexical_ms,
                "fusionMs": fusion_ms,
                "generationMs": gen_ms,
                "totalMs": (
                    chunk_ms
                    + index_embed_ms
                    + bm25_index_ms
                    + query_embed_ms
                    + dense_ms
                    + lexical_ms
                    + fusion_ms
                    + gen_ms
                ),
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "provenance": "measured",
            },
            "relatedEntities": ["openai", "bm25", "rrf"],
            "relatedContent": ["rag", "hybrid-retrieval", "embeddings"],
            "cookbook": {"path": "examples/rag/02-hybrid-rag"},
        }
        traces.append(trace)

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
