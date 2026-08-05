"""retrieval-evaluation — measure retrieval quality against golden judgments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from config import EXAMPLE_ID, Settings, get_settings
from evaluation.dataset import EvalDataset, load_eval_dataset
from evaluation.evaluator import (
    PipelineEvalResult,
    aggregate_pipeline,
    evaluate_ranking,
)
from evaluation.report import (
    print_comparison,
    print_pipeline_aggregate,
    print_query_result,
)
from pipelines import ALL_PIPELINES, build_pipeline_runners
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, get_client
from rag.loader import load_document
from rag.query_transformer import LLMQueryTransformer
from rag.reranker import CrossEncoderReranker
from rag.store import InMemoryVectorStore


@dataclass
class Indexes:
    store: InMemoryVectorStore
    bm25: BM25Index
    chunk_text_by_id: dict[str, str]


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
    return Indexes(
        store=store,
        bm25=bm25,
        chunk_text_by_id={c.id: c.text for c in chunks},
    )


def run_evaluation(
    settings: Settings,
    dataset: EvalDataset,
    indexes: Indexes,
    *,
    pipelines: tuple[str, ...],
    k: int,
    query_ids: list[str] | None = None,
) -> list[PipelineEvalResult]:
    client = get_client(settings)
    needs_rerank = any(
        name in {"hybrid-reranked", "query-transform"} for name in pipelines
    )
    needs_transform = "query-transform" in pipelines

    reranker = CrossEncoderReranker(settings.reranker_model) if needs_rerank else None
    transform_fn = None
    if needs_transform:
        transformer = LLMQueryTransformer(client, settings.query_transformer_model)
        transform_fn = transformer.transform

    runners = build_pipeline_runners(
        client,
        indexes.store,
        indexes.bm25,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        lexical_top_k=settings.lexical_top_k,
        candidate_k=settings.candidate_k,
        eval_k=k,
        rrf_k=settings.rrf_k,
        max_alternative_queries=settings.max_alternative_queries,
        transform_fn=transform_fn,
        reranker=reranker,
        pipelines=pipelines,
    )

    cases = dataset.queries
    if query_ids:
        wanted = set(query_ids)
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise KeyError(f"Unknown query id(s): {sorted(missing)}")

    results: list[PipelineEvalResult] = []
    for name in pipelines:
        runner = runners[name]
        per_query = []
        for case in cases:
            output = runner(case.query)
            per_query.append(
                evaluate_ranking(
                    case,
                    output.retrieved_ids,
                    pipeline=name,
                    k=k,
                    chunk_text_by_id=indexes.chunk_text_by_id,
                    latency_ms=output.latency_ms,
                )
            )
        results.append(aggregate_pipeline(name, k, per_query))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        choices=list(ALL_PIPELINES),
        help="Pipeline to evaluate (repeatable). Default: all.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Evaluation depth K (default: settings.eval_k / dataset k_default)",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Evaluate a single query id (repeatable)",
    )
    parser.add_argument(
        "--aggregates-only",
        action="store_true",
        help="Print aggregate table only",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.openai_api_key:
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.",
            file=sys.stderr,
        )
        return 1

    dataset = load_eval_dataset(settings.eval_path)
    k = args.k if args.k is not None else settings.eval_k
    pipelines = tuple(args.pipeline) if args.pipeline else ALL_PIPELINES

    print(f"[{EXAMPLE_ID}] teaching evaluation set — not a production benchmark")
    print(
        f"[{EXAMPLE_ID}] loading {settings.eval_path.name} "
        f"({len(dataset.queries)} queries)"
    )
    print(f"[{EXAMPLE_ID}] indexing {settings.data_path.name} …")
    indexes = build_indexes(settings)
    print(
        f"[{EXAMPLE_ID}] indexed {len(indexes.store)} chunks; "
        f"evaluating pipelines={list(pipelines)} at K={k}"
    )

    results = run_evaluation(
        settings,
        dataset,
        indexes,
        pipelines=pipelines,
        k=k,
        query_ids=args.query_ids,
    )

    print("\nRetrieval Evaluation")
    print(f"K: {k}")
    print(f"Corpus: {settings.data_path.name}")
    print(f"Queries: {results[0].query_count if results else 0}")

    if not args.aggregates_only:
        for pipeline_result in results:
            print_pipeline_aggregate(pipeline_result)
            for q in pipeline_result.per_query:
                print_query_result(q)

    print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
