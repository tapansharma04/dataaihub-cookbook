"""Human-readable evaluation report printing."""

from __future__ import annotations

from evaluation.evaluator import PipelineEvalResult, QueryEvalResult


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def print_query_result(result: QueryEvalResult) -> None:
    print(f"\nQuery: {result.query_id}")
    print(f"  {result.query}")
    print(f"  Pipeline: {result.pipeline}   K={result.k}")
    print(f"  {'Rank':<6}{'Chunk':<14}{'Rel':<6}{'Hit'}")
    for hit in result.retrieved:
        flag = "yes" if hit.is_relevant else "no"
        title = f"  {hit.title}" if hit.title else ""
        print(f"  {hit.rank:<6}{hit.chunk_id:<14}{hit.relevance_grade:<6}{flag}{title}")
    print(
        f"  Recall@{result.k}: {_fmt(result.recall_at_k)}   "
        f"RR: {_fmt(result.reciprocal_rank)}   "
        f"nDCG@{result.k}: {_fmt(result.ndcg_at_k)}"
    )
    print(
        f"  DCG@{result.k}: {_fmt(result.dcg_at_k)}   "
        f"IDCG@{result.k}: {_fmt(result.idcg_at_k)}   "
        f"first_relevant_rank: {result.first_relevant_rank}"
    )
    print(f"  Failure: {result.failure.label} — {result.failure.explanation}")
    if result.latency_ms:
        parts = ", ".join(f"{k}={v}ms" for k, v in result.latency_ms.items())
        print(f"  Latency: {parts}")


def print_pipeline_aggregate(result: PipelineEvalResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Pipeline: {result.pipeline}")
    print(f"Queries:  {result.query_count}   K={result.k}")
    print(f"Recall@{result.k}: {_fmt(result.mean_recall_at_k)}")
    print(f"MRR:        {_fmt(result.mrr)}")
    print(f"nDCG@{result.k}:  {_fmt(result.mean_ndcg_at_k)}")
    if result.total_latency_ms:
        print(f"Total retrieval latency (sum): {result.total_latency_ms}ms")


def print_comparison(results: list[PipelineEvalResult]) -> None:
    if not results:
        return
    k = results[0].k
    print(f"\n{'=' * 60}")
    print("Aggregate comparison")
    print(f"{'Pipeline':<22}{'Recall@' + str(k):<12}{'MRR':<10}{'nDCG@' + str(k):<12}")
    for r in results:
        print(
            f"{r.pipeline:<22}"
            f"{_fmt(r.mean_recall_at_k):<12}"
            f"{_fmt(r.mrr):<10}"
            f"{_fmt(r.mean_ndcg_at_k):<12}"
        )
