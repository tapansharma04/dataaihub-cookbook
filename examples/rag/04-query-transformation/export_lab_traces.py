"""One-shot export for query-transformation lab traces."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import EXAMPLE_ID, get_settings
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.generator import SYSTEM_PROMPT, build_prompt
from rag.loader import load_document
from rag.query_transformer import LLMQueryTransformer
from rag.reranker import CrossEncoderReranker
from rag.retriever import RetrievalViews, retrieve_all
from rag.store import InMemoryVectorStore

# Selected after measured experimental runs — see README measured results.
QUESTIONS = [
    {
        "traceId": "transformation-bridges-idle-auth",
        "question": (
            "Why does everyone have to unlock the app again after being idle "
            "half a day?"
        ),
        "teachingClass": "TRANSFORM_HELPS",
    },
    {
        "traceId": "original-query-already-strong-econn42",
        "question": "How do I fix error E_CONN_42 on the EdgeGateway?",
        "teachingClass": "REDUNDANT",
    },
    {
        "traceId": "expansion-adds-platform-noise",
        "question": (
            "Something feels off with the edge platform today — profiles or network?"
        ),
        "teachingClass": "ADDS_NOISE",
    },
]


def ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def reranked_detail(items) -> list[dict]:
    return [
        {
            "id": item.chunk.id,
            "text": item.chunk.text,
            "rerankerScore": round(item.reranker_score, 4),
            "rank": item.rank,
            "previousRank": item.previous_rank,
            "rrfScore": round(item.rrf_score, 6),
            "start": item.chunk.start,
            "end": item.chunk.end,
        }
        for item in items
    ]


def run_path(
    views: RetrievalViews,
    *,
    label: str,
) -> dict:
    before = sum(len(run.fused) for run in views.per_query)
    after = len(views.merged)
    duplicate_count = max(before - after, 0)
    duplicate_rate = (duplicate_count / before) if before else 0.0
    query_labels = {query: f"Q{i}" for i, query in enumerate(views.transformed_queries)}
    return {
        "path": label,
        "queries": [
            {"id": query_labels[q], "text": q, "isOriginal": i == 0}
            for i, q in enumerate(views.transformed_queries)
        ],
        "perQueryRetrievalLatencyMs": [
            {
                "queryId": query_labels[run.query],
                "query": run.query,
                "latencyMs": run.latency_ms,
                "provenance": "measured",
            }
            for run in views.per_query
        ],
        "candidateCounts": {
            "beforeDedup": before,
            "afterDedup": after,
            "duplicates": duplicate_count,
            "duplicateRate": round(duplicate_rate, 4),
            "provenance": {
                "beforeDedup": "derived",
                "afterDedup": "measured",
                "duplicates": "derived",
                "duplicateRate": "derived",
            },
        },
        "mergedCandidates": [
            {
                "id": item.chunk.id,
                "rank": item.rank,
                "aggregateRrfScore": round(item.rrf_score, 6),
                "foundBy": [query_labels[p.query] for p in item.provenance],
                "perQueryProvenance": [
                    {
                        "queryId": query_labels[p.query],
                        "query": p.query,
                        "denseRank": p.dense_rank,
                        "lexicalRank": p.lexical_rank,
                        "rrfRank": p.fused_rank,
                        "denseScore": p.dense_score,
                        "lexicalScore": p.lexical_score,
                        "rrfScore": p.fused_rrf_score,
                    }
                    for p in item.provenance
                ],
            }
            for item in views.merged
        ],
        "rerankedContext": reranked_detail(views.reranked),
        "latency": {
            "transformMs": views.transform_latency_ms,
            "totalRetrievalMs": views.retrieval_latency_ms,
            "rerankMs": views.rerank_latency_ms,
            "provenance": "measured",
        },
    }


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

    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(settings.reranker_model)
    reranker_load_ms = ms(t0)

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
    transformer = LLMQueryTransformer(client, settings.query_transformer_model)
    for q in QUESTIONS:
        question = q["question"]
        baseline = retrieve_all(
            client,
            store,
            bm25,
            question,
            max_alternative_queries=0,
            embedding_model=settings.embedding_model,
            dense_top_k=settings.dense_top_k,
            lexical_top_k=settings.lexical_top_k,
            candidate_k=settings.candidate_k,
            final_context_k=settings.final_context_k,
            rrf_k=settings.rrf_k,
            merge_top_k=settings.candidate_k,
            transform_fn=lambda _q, _m: [],
            reranker=reranker,
        )
        multi = retrieve_all(
            client,
            store,
            bm25,
            question,
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
        user_prompt = build_prompt(question, multi.reranked)

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

        baseline_ids = {item.chunk.id for item in baseline.merged}
        multi_ids = {item.chunk.id for item in multi.merged}
        discovered = sorted(multi_ids - baseline_ids)
        multi_detail = run_path(multi, label="multi-query")
        baseline_detail = run_path(baseline, label="original-only")

        trace = {
            "labId": EXAMPLE_ID,
            "traceId": q["traceId"],
            "executionMode": "guided",
            "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metricsProvenance": "measured",
            "teachingClass": q["teachingClass"],
            "input": {
                "question": question,
                "rerankerQuery": question,
                "generationQuestion": question,
                "config": {
                    "chunkSize": settings.chunk_size,
                    "chunkOverlap": settings.chunk_overlap,
                    "denseTopK": settings.dense_top_k,
                    "lexicalTopK": settings.lexical_top_k,
                    "candidateK": settings.candidate_k,
                    "finalContextK": settings.final_context_k,
                    "rrfK": settings.rrf_k,
                    "maxAlternativeQueries": settings.max_alternative_queries,
                    "embeddingModel": settings.embedding_model,
                    "chatModel": settings.chat_model,
                    "rerankerModel": settings.reranker_model,
                    "queryTransformerModel": settings.query_transformer_model,
                },
            },
            "corpus": {
                "source": corpus["source"],
                "chunkCount": corpus["chunkCount"],
            },
            "comparison": {
                "originalOnlyCandidateIds": [item.chunk.id for item in baseline.merged],
                "multiQueryCandidateIds": [item.chunk.id for item in multi.merged],
                "candidatesDiscoveredByTransform": discovered,
                "originalOnlyFinalContextIds": [
                    item.chunk.id for item in baseline.reranked
                ],
                "multiQueryFinalContextIds": [item.chunk.id for item in multi.reranked],
                "provenance": "derived",
            },
            "steps": [
                {
                    "id": "chunking",
                    "metrics": {"latencyMs": chunk_ms, "provenance": "measured"},
                },
                {
                    "id": "embed-corpus",
                    "metrics": {"latencyMs": index_embed_ms, "provenance": "measured"},
                },
                {
                    "id": "bm25-index",
                    "metrics": {"latencyMs": bm25_index_ms, "provenance": "measured"},
                },
                {
                    "id": "query-transform",
                    "detail": multi_detail,
                },
                {
                    "id": "query-baseline",
                    "detail": baseline_detail,
                },
                {
                    "id": "prompt",
                    "detail": {
                        "system": SYSTEM_PROMPT.strip(),
                        "user": user_prompt,
                        "contextSource": "reranked",
                        "generationQuestion": question,
                        "finalContextIds": [item.chunk.id for item in multi.reranked],
                    },
                },
                {
                    "id": "llm",
                    "metrics": {
                        "latencyMs": gen_ms,
                        "promptTokens": prompt_tokens,
                        "completionTokens": completion_tokens,
                        "provenance": "measured",
                    },
                },
                {"id": "output", "detail": {"answer": answer}},
            ],
            "output": {"answer": answer},
            "metrics": {
                "chunkingMs": chunk_ms,
                "corpusEmbeddingMs": index_embed_ms,
                "bm25IndexMs": bm25_index_ms,
                "rerankerLoadMs": reranker_load_ms,
                "transformMs": multi.transform_latency_ms,
                "retrievalMs": multi.retrieval_latency_ms,
                "rerankingMs": multi.rerank_latency_ms,
                "generationMs": gen_ms,
                "totalMs": multi.transform_latency_ms
                + multi.retrieval_latency_ms
                + multi.rerank_latency_ms
                + gen_ms,
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "provenance": "measured",
            },
            "relatedEntities": [
                "openai",
                "bm25",
                "rrf",
                "cross-encoder",
                "ms-marco-MiniLM-L6-v2",
            ],
            "relatedContent": [
                "rag",
                "hybrid-retrieval",
                "query-transformation",
                "embeddings",
            ],
            "cookbook": {"path": "examples/rag/04-query-transformation"},
        }
        traces.append(trace)

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")


if __name__ == "__main__":
    main()
