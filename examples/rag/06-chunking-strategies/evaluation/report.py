"""Console reporting helpers."""

from __future__ import annotations

from evaluation.evaluator import StrategyEvalResult
from rag.chunking.stats import ChunkStats


def print_chunk_stats(stats: ChunkStats) -> None:
    print(f"\n--- Chunk stats: {stats.strategy} ---")
    print(f"  chunks:              {stats.chunk_count}")
    print(f"  avg length (chars):  {stats.avg_chunk_length:.1f}")
    print(f"  min / max length:    {stats.min_chunk_length} / {stats.max_chunk_length}")
    print(f"  total indexed chars: {stats.total_indexed_chars}")
    print(f"  overlap extra chars: {stats.overlap_extra_chars}")
    print(
        f"  fragmented evidence: {stats.fragmented_evidence_count}/"
        f"{stats.evidence_unit_count} "
        f"({stats.evidence_fragmentation_rate:.1%})"
    )


def print_strategy_aggregate(result: StrategyEvalResult) -> None:
    print(f"\n=== Strategy: {result.strategy} (K={result.k}) ===")
    print(f"  mean Recall@{result.k}: {result.mean_recall_at_k:.4f}")
    print(f"  MRR:                  {result.mrr:.4f}")
    print(f"  mean nDCG@{result.k}:   {result.mean_ndcg_at_k:.4f}")
    print(f"  mean evidence cov.:   {result.mean_evidence_coverage:.4f}")
    print(f"  total retrieval ms:   {result.total_retrieval_ms}")


def print_query_result(q) -> None:
    print(f"\n  [{q.query_id}] {q.query}")
    print(
        f"    Recall@{q.k}={q.recall_at_k:.3f}  RR={q.reciprocal_rank:.3f}  "
        f"nDCG@{q.k}={q.ndcg_at_k:.3f}  evid_cov={q.evidence_coverage:.3f}"
    )
    for hit in q.retrieved:
        flag = "✓" if hit.is_relevant else "·"
        preview = hit.text.replace("\n", " ")[:72]
        print(
            f"    {flag} #{hit.rank} {hit.chunk_id} "
            f"grade={hit.relevance_grade} score={hit.score:.4f}  {preview}"
        )


def print_comparison(results: list[StrategyEvalResult]) -> None:
    print("\n" + "=" * 72)
    print("Aggregate comparison")
    print("=" * 72)
    header = (
        f"{'strategy':<14} {'Recall@K':>10} {'MRR':>8} {'nDCG@K':>8} {'EvidCov':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.strategy:<14} {r.mean_recall_at_k:>10.4f} {r.mrr:>8.4f} "
            f"{r.mean_ndcg_at_k:>8.4f} {r.mean_evidence_coverage:>8.4f}"
        )
