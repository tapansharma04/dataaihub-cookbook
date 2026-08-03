"""One-shot: run basic-rag and emit guided Lab traces as JSON (measured timings)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import EXAMPLE_ID, get_settings
from rag.chunker import chunk_text
from rag.embeddings import embed_query, embed_texts, get_client
from rag.generator import SYSTEM_PROMPT, build_prompt, generate_answer
from rag.loader import load_document
from rag.retriever import retrieve
from rag.store import InMemoryVectorStore

QUESTIONS = [
    {
        "traceId": "what-problem-does-rag-solve",
        "question": "What problem does RAG solve?",
    },
    {
        "traceId": "why-embeddings",
        "question": "Why are embeddings needed?",
    },
    {
        "traceId": "how-retrieval-works",
        "question": "How does retrieval work?",
    },
]


def ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


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
    # usage not always available consistently; leave tokens only when present

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

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
        scored = store.search(query_vector, top_k=settings.top_k)
        retrieve_ms = ms(t0)

        user_prompt = build_prompt(question, scored)

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
                    "topK": settings.top_k,
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
                    "id": "retrieve",
                    "type": "retrieval",
                    "title": "Similarity search",
                    "status": "ok",
                    "detail": {
                        "topK": settings.top_k,
                        "similarity": "cosine",
                        "chunks": [
                            {
                                "id": item.chunk.id,
                                "text": item.chunk.text,
                                "score": round(item.score, 4),
                                "start": item.chunk.start,
                                "end": item.chunk.end,
                            }
                            for item in scored
                        ],
                    },
                    "metrics": {
                        "latencyMs": retrieve_ms,
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
                "queryEmbeddingMs": query_embed_ms,
                "retrievalMs": retrieve_ms,
                "generationMs": gen_ms,
                "totalMs": chunk_ms
                + index_embed_ms
                + query_embed_ms
                + retrieve_ms
                + gen_ms,
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "provenance": "measured",
            },
            "relatedEntities": ["openai"],
            "relatedContent": ["rag", "embeddings"],
            "cookbook": {"path": "examples/rag/01-basic-rag"},
        }
        traces.append(trace)

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
