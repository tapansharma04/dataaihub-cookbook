"""Export measured retrieval-evaluation traces for Interactive Lab #5."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import EXAMPLE_ID, get_settings
from evaluation.dataset import load_eval_dataset
from evaluation.evaluator import (
    PipelineEvalResult,
    aggregate_pipeline,
    evaluate_ranking,
)
from pipelines import ALL_PIPELINES, build_pipeline_runners
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.loader import load_document
from rag.query_transformer import LLMQueryTransformer
from rag.reranker import CrossEncoderReranker
from rag.store import InMemoryVectorStore

# Selected after measured runs — see README / evaluation_report.json.
# Categories describe observed behavior; they are not manufactured targets.
TEACHING_TRACE_SPECS = [
    {
        "traceId": "discover-auth-idle-timeout",
        "queryId": "auth-idle-timeout",
        "teachingClass": "DISCOVER",
        "selectionNote": (
            "Measured: dense retrieves sample-12 (relevant) at rank 3, while "
            "hybrid / hybrid-reranked / query-transform miss it in final top-K. "
            "Example 04 can show candidate discovery of sample-12; example 05 "
            "evaluates final top-K=3 retention — different cuts, not a "
            "contradiction. Lesson: candidate discovery ≠ final top-K quality."
        ),
    },
    {
        "traceId": "ranking-sensitivity-profile-async",
        "queryId": "profile-async",
        "teachingClass": "RANKING_SENSITIVITY",
        "selectionNote": (
            "Measured: Recall@3 stays 0.667 across pipelines (primary sample-4 "
            "at rank 1), while nDCG@3 moves (~0.947 → ~0.798) when secondary "
            "graded evidence is reordered. Not a ranking improvement — ranking "
            "sensitivity: same Recall, different nDCG."
        ),
    },
    {
        "traceId": "regression-econn-42-remediation",
        "queryId": "econn-42-remediation",
        "teachingClass": "REGRESSION",
        "selectionNote": (
            "Measured: dense/hybrid Recall@3=0.667 (sample-2 + sample-8); "
            "hybrid-reranked/query-transform drop to 0.333 and lower nDCG "
            "while RR stays 1.0. An advanced pipeline can regress a specific "
            "information need on this teaching case."
        ),
    },
]


def ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def _query_payload(result) -> dict:
    return {
        "queryId": result.query_id,
        "query": result.query,
        "pipeline": result.pipeline,
        "k": result.k,
        "retrieved": [
            {
                "chunkId": h.chunk_id,
                "rank": h.rank,
                "relevanceGrade": h.relevance_grade,
                "isRelevant": h.is_relevant,
                "title": h.title,
                "text": h.text,
            }
            for h in result.retrieved
        ],
        "metrics": {
            "recallAtK": round(result.recall_at_k, 6),
            "reciprocalRank": round(result.reciprocal_rank, 6),
            "firstRelevantRank": result.first_relevant_rank,
            "dcgAtK": round(result.dcg_at_k, 6),
            "idcgAtK": round(result.idcg_at_k, 6),
            "ndcgAtK": round(result.ndcg_at_k, 6),
            "provenance": "computed",
        },
        "failure": {
            "label": result.failure.label,
            "firstRelevantRank": result.failure.first_relevant_rank,
            "missedIds": result.failure.missed_ids,
            "lateIds": result.failure.late_ids,
            "goodIds": result.failure.good_ids,
            "explanation": result.failure.explanation,
        },
        "latencyMs": result.latency_ms,
    }


def _aggregate_payload(result: PipelineEvalResult) -> dict:
    return {
        "pipeline": result.pipeline,
        "k": result.k,
        "queryCount": result.query_count,
        "meanRecallAtK": round(result.mean_recall_at_k, 6),
        "mrr": round(result.mrr, 6),
        "meanNdcgAtK": round(result.mean_ndcg_at_k, 6),
        "totalLatencyMs": result.total_latency_ms,
        "provenance": "computed",
    }


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY")

    dataset = load_eval_dataset(settings.eval_path)
    k = settings.eval_k

    text = load_document(settings.data_path)
    t0 = time.perf_counter()
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source=settings.data_path.stem,
    )
    chunk_ms = ms(t0)
    chunk_text_by_id = {c.id: c.text for c in chunks}

    client = get_client(settings)
    t0 = time.perf_counter()
    vectors = embed_texts(client, [c.text for c in chunks], settings.embedding_model)
    embed_ms = ms(t0)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    t0 = time.perf_counter()
    bm25 = BM25Index(chunks)
    bm25_ms = ms(t0)

    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(settings.reranker_model)
    reranker_load_ms = ms(t0)
    transformer = LLMQueryTransformer(client, settings.query_transformer_model)

    runners = build_pipeline_runners(
        client,
        store,
        bm25,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        lexical_top_k=settings.lexical_top_k,
        candidate_k=settings.candidate_k,
        eval_k=k,
        rrf_k=settings.rrf_k,
        max_alternative_queries=settings.max_alternative_queries,
        transform_fn=transformer.transform,
        reranker=reranker,
        pipelines=ALL_PIPELINES,
    )

    # Full evaluation over all queries × all pipelines (measured).
    pipeline_results: list[PipelineEvalResult] = []
    per_pipeline_by_query: dict[str, dict[str, object]] = {
        name: {} for name in ALL_PIPELINES
    }
    for name in ALL_PIPELINES:
        per_query = []
        for case in dataset.queries:
            output = runners[name](case.query)
            q_result = evaluate_ranking(
                case,
                output.retrieved_ids,
                pipeline=name,
                k=k,
                chunk_text_by_id=chunk_text_by_id,
                latency_ms=output.latency_ms,
            )
            per_query.append(q_result)
            per_pipeline_by_query[name][case.id] = q_result
        pipeline_results.append(aggregate_pipeline(name, k, per_query))

    aggregates = [_aggregate_payload(r) for r in pipeline_results]

    # Build teaching traces for selected queries with all pipeline comparisons.
    traces = []
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for spec in TEACHING_TRACE_SPECS:
        case = dataset.get(spec["queryId"])
        pipeline_details = [
            _query_payload(per_pipeline_by_query[name][case.id])  # type: ignore[index]
            for name in ALL_PIPELINES
        ]
        # Derive teaching observations from measured per-pipeline metrics.
        recalls = {p["pipeline"]: p["metrics"]["recallAtK"] for p in pipeline_details}
        ndcgs = {p["pipeline"]: p["metrics"]["ndcgAtK"] for p in pipeline_details}
        first_ranks = {
            p["pipeline"]: p["metrics"]["firstRelevantRank"] for p in pipeline_details
        }

        relevant_chunks = [
            {
                "chunkId": cid,
                "grade": grade,
                "rationale": case.rationale[cid],
                "text": chunk_text_by_id.get(cid, ""),
            }
            for cid, grade in sorted(
                case.relevance.items(), key=lambda item: (-item[1], item[0])
            )
        ]

        trace = {
            "labId": EXAMPLE_ID,
            "traceId": spec["traceId"],
            "executionMode": "guided",
            "recordedAt": recorded_at,
            "metricsProvenance": "measured",
            "teachingClass": spec["teachingClass"],
            "selectionNote": spec["selectionNote"],
            "input": {
                "queryId": case.id,
                "query": case.query,
                "k": k,
                "relevanceJudgments": case.relevance,
                "rationale": case.rationale,
                "config": {
                    "chunkSize": settings.chunk_size,
                    "chunkOverlap": settings.chunk_overlap,
                    "denseTopK": settings.dense_top_k,
                    "lexicalTopK": settings.lexical_top_k,
                    "candidateK": settings.candidate_k,
                    "evalK": k,
                    "rrfK": settings.rrf_k,
                    "maxAlternativeQueries": settings.max_alternative_queries,
                    "embeddingModel": settings.embedding_model,
                    "rerankerModel": settings.reranker_model,
                    "queryTransformerModel": settings.query_transformer_model,
                },
            },
            "corpus": {
                "source": "data/sample.md",
                "chunkCount": len(chunks),
            },
            "relevantChunks": relevant_chunks,
            "pipelines": pipeline_details,
            "observedComparison": {
                "recallAtKByPipeline": recalls,
                "ndcgAtKByPipeline": ndcgs,
                "firstRelevantRankByPipeline": first_ranks,
                "provenance": "derived-from-measured",
            },
            "steps": [
                {
                    "id": "evaluation-dataset",
                    "detail": {
                        "queryId": case.id,
                        "query": case.query,
                        "datasetSize": len(dataset.queries),
                        "description": dataset.description,
                    },
                },
                {
                    "id": "relevance-judgments",
                    "detail": {
                        "relevance": case.relevance,
                        "rationale": case.rationale,
                        "scheme": dataset.relevance_scheme,
                    },
                },
                {
                    "id": "run-retrieval",
                    "detail": {
                        "pipelines": list(ALL_PIPELINES),
                        "k": k,
                        "note": (
                            "Judgments attach to the original information need; "
                            "query-transform may retrieve with alternatives."
                        ),
                    },
                },
                {
                    "id": "inspect-ranked-results",
                    "detail": {
                        "byPipeline": {
                            p["pipeline"]: p["retrieved"] for p in pipeline_details
                        }
                    },
                },
                {
                    "id": "calculate-recall-at-k",
                    "detail": {
                        "byPipeline": {
                            p["pipeline"]: p["metrics"]["recallAtK"]
                            for p in pipeline_details
                        },
                        "formula": ("Recall@K = |relevant ∩ top-K| / |relevant|"),
                    },
                },
                {
                    "id": "calculate-rr-mrr",
                    "detail": {
                        "reciprocalRankByPipeline": {
                            p["pipeline"]: p["metrics"]["reciprocalRank"]
                            for p in pipeline_details
                        },
                        "firstRelevantRankByPipeline": first_ranks,
                        "formula": "RR = 1/rank(first relevant); MRR = mean RR",
                    },
                },
                {
                    "id": "calculate-ndcg-at-k",
                    "detail": {
                        "byPipeline": {
                            p["pipeline"]: {
                                "dcgAtK": p["metrics"]["dcgAtK"],
                                "idcgAtK": p["metrics"]["idcgAtK"],
                                "ndcgAtK": p["metrics"]["ndcgAtK"],
                            }
                            for p in pipeline_details
                        },
                        "formula": (
                            "DCG@K = Σ (2^rel_i - 1) / log2(i+1); "
                            "nDCG@K = DCG@K / IDCG@K"
                        ),
                    },
                },
                {
                    "id": "compare-pipelines",
                    "detail": {
                        "aggregatesOverFullEvalSet": aggregates,
                        "thisQuery": {
                            "recallAtK": recalls,
                            "ndcgAtK": ndcgs,
                        },
                    },
                },
                {
                    "id": "inspect-failures",
                    "detail": {
                        "byPipeline": {
                            p["pipeline"]: p["failure"] for p in pipeline_details
                        }
                    },
                },
                {
                    "id": "evaluation-summary",
                    "detail": {
                        "teachingClass": spec["teachingClass"],
                        "selectionNote": spec["selectionNote"],
                        "disclaimer": (
                            "Small teaching evaluation set — not a production "
                            "benchmark of retrieval systems."
                        ),
                    },
                },
            ],
            "fullEvaluationAggregates": aggregates,
            "metrics": {
                "chunkingMs": chunk_ms,
                "corpusEmbeddingMs": embed_ms,
                "bm25IndexMs": bm25_ms,
                "rerankerLoadMs": reranker_load_ms,
                "provenance": "measured",
            },
            "relatedEntities": [
                "recall-at-k",
                "mrr",
                "ndcg",
                "bm25",
                "rrf",
                "cross-encoder",
            ],
            "relatedContent": [
                "rag",
                "retrieval-evaluation",
                "hybrid-retrieval",
                "query-transformation",
            ],
            "cookbook": {"path": "examples/rag/05-retrieval-evaluation"},
        }
        traces.append(trace)

    # Also write a machine-readable full report for README regeneration.
    full_report = {
        "recordedAt": recorded_at,
        "k": k,
        "queryCount": len(dataset.queries),
        "pipelines": list(ALL_PIPELINES),
        "aggregates": aggregates,
        "perQuery": {
            name: {
                qid: {
                    "retrievedIds": [h.chunk_id for h in q.retrieved],  # type: ignore[attr-defined]
                    "recallAtK": round(q.recall_at_k, 6),  # type: ignore[attr-defined]
                    "reciprocalRank": round(q.reciprocal_rank, 6),  # type: ignore[attr-defined]
                    "ndcgAtK": round(q.ndcg_at_k, 6),  # type: ignore[attr-defined]
                    "firstRelevantRank": q.first_relevant_rank,  # type: ignore[attr-defined]
                    "failure": q.failure.label,  # type: ignore[attr-defined]
                    "latencyMs": q.latency_ms,  # type: ignore[attr-defined]
                }
                for qid, q in per_pipeline_by_query[name].items()
            }
            for name in ALL_PIPELINES
        },
    }
    report_path = Path(__file__).resolve().parent / "evaluation_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")
    print(f"Wrote full evaluation report to {report_path}")


if __name__ == "__main__":
    main()
