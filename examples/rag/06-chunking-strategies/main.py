"""chunking-strategies — controlled retrieval experiment on chunking."""

from __future__ import annotations

import argparse
import sys

from config import ALL_STRATEGIES, EXAMPLE_ID, get_settings
from evaluation.dataset import load_eval_dataset
from evaluation.report import (
    print_chunk_stats,
    print_comparison,
    print_query_result,
    print_strategy_aggregate,
)
from experiment import run_experiment
from rag.embeddings import get_client
from rag.loader import load_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DataAIHub Cookbook — {EXAMPLE_ID}",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=list(ALL_STRATEGIES),
        help="Strategy to evaluate (repeatable). Default: all.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Evaluation depth K (default: settings.eval_k)",
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

    source_text = load_document(settings.data_path)
    dataset = load_eval_dataset(settings.eval_path, source_text)
    k = args.k if args.k is not None else settings.eval_k
    strategies = tuple(args.strategy) if args.strategy else ALL_STRATEGIES

    print(f"[{EXAMPLE_ID}] evaluation set — not a production benchmark")
    print(
        f"[{EXAMPLE_ID}] corpus={settings.data_path.name}  "
        f"queries={len(dataset.queries)}  evidence={len(dataset.evidence_units)}"
    )
    print(
        f"[{EXAMPLE_ID}] independent variable=chunking strategy; "
        f"held constant: corpus, embedder={settings.embedding_model}, "
        f"dense retrieval, K={k}"
    )

    client = get_client(settings)
    result = run_experiment(
        client,
        source_text,
        dataset,
        settings,
        strategies=strategies,
        k=k,
        query_ids=args.query_ids,
    )

    for name in strategies:
        print_chunk_stats(result.indexes[name].stats)
        timings = result.indexes[name].timings_ms
        print(
            f"  timings: chunking={timings['chunking_ms']}ms  "
            f"embedding={timings['embedding_ms']}ms  "
            f"index={timings['index_ms']}ms"
        )

    print("\nChunking Strategies — Retrieval Evaluation")
    print(f"K: {k}")
    print(f"Corpus: {settings.data_path.name}")
    print(f"Embedding: {settings.embedding_model}")
    print(f"Retrieval: dense cosine top-{k}")
    print(f"Queries: {result.evaluations[0].query_count if result.evaluations else 0}")

    if not args.aggregates_only:
        for strategy_result in result.evaluations:
            print_strategy_aggregate(strategy_result)
            for q in strategy_result.per_query:
                print_query_result(q)

    print_comparison(result.evaluations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
