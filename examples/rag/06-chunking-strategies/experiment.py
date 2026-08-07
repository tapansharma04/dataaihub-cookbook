"""Controlled chunking experiment: same corpus → different chunking → same retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from openai import OpenAI

from config import ALL_STRATEGIES, Settings, strategy_config
from evaluation.dataset import EvalDataset
from evaluation.evaluator import (
    StrategyEvalResult,
    aggregate_strategy,
    evaluate_ranking,
)
from evaluation.evidence import attach_evidence
from rag.chunking import chunk_with_strategy, compute_chunk_stats
from rag.chunking.base import Chunk
from rag.chunking.stats import ChunkStats
from rag.embeddings import embed_query, embed_texts
from rag.store import InMemoryVectorStore


def _ms(start: float) -> int:
    return int(round((perf_counter() - start) * 1000))


@dataclass
class StrategyIndex:
    strategy: str
    chunks: list[Chunk]
    store: InMemoryVectorStore
    stats: ChunkStats
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    k: int
    embedding_model: str
    strategy_config: dict
    indexes: dict[str, StrategyIndex]
    evaluations: list[StrategyEvalResult]
    model_init_ms: int = 0


def build_strategy_index(
    client: OpenAI,
    source_text: str,
    dataset: EvalDataset,
    settings: Settings,
    strategy: str,
    *,
    source_name: str = "sample",
) -> StrategyIndex:
    """Chunk → embed → index for one strategy. Timings are strategy-local."""
    t_chunk = perf_counter()
    raw_chunks = chunk_with_strategy(
        source_text,
        strategy,
        source=source_name,
        fixed_chunk_size=settings.fixed_chunk_size,
        fixed_chunk_overlap=settings.fixed_chunk_overlap,
        recursive_target_size=settings.recursive_target_size,
        recursive_chunk_overlap=settings.recursive_chunk_overlap,
    )
    chunks = attach_evidence(raw_chunks, dataset.evidence_units)
    chunk_ms = _ms(t_chunk)

    t_embed = perf_counter()
    vectors = embed_texts(client, [c.text for c in chunks], settings.embedding_model)
    embed_ms = _ms(t_embed)

    t_index = perf_counter()
    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    index_ms = _ms(t_index)

    stats = compute_chunk_stats(
        chunks,
        source_text=source_text,
        evidence_units=dataset.evidence_units,
    )
    return StrategyIndex(
        strategy=strategy,
        chunks=chunks,
        store=store,
        stats=stats,
        timings_ms={
            "chunking_ms": chunk_ms,
            "embedding_ms": embed_ms,
            "index_ms": index_ms,
            # embedding+index together (excludes model HTTP connect separately)
            "embed_index_ms": embed_ms + index_ms,
        },
    )


def retrieve_dense(
    client: OpenAI,
    index: StrategyIndex,
    query: str,
    *,
    embedding_model: str,
    top_k: int,
) -> tuple[list[tuple[Chunk, float]], dict[str, int]]:
    t0 = perf_counter()
    vector = embed_query(client, query, embedding_model)
    ranked = index.store.search(vector, top_k=top_k)
    latency = {"retrieval_ms": _ms(t0), "total_ms": _ms(t0)}
    return [(item.chunk, item.score) for item in ranked], latency


def run_experiment(
    client: OpenAI,
    source_text: str,
    dataset: EvalDataset,
    settings: Settings,
    *,
    strategies: tuple[str, ...] = ALL_STRATEGIES,
    k: int | None = None,
    query_ids: list[str] | None = None,
) -> ExperimentResult:
    """Run the controlled comparison. Only chunking strategy varies."""
    eval_k = k if k is not None else settings.eval_k
    configs = strategy_config(settings)

    indexes: dict[str, StrategyIndex] = {}
    for name in strategies:
        indexes[name] = build_strategy_index(
            client,
            source_text,
            dataset,
            settings,
            name,
            source_name=settings.data_path.stem,
        )

    cases = dataset.queries
    if query_ids:
        wanted = set(query_ids)
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise KeyError(f"Unknown query id(s): {sorted(missing)}")

    units_by_id = dataset.units_by_id()
    evaluations: list[StrategyEvalResult] = []
    for name in strategies:
        index = indexes[name]
        per_query = []
        for case in cases:
            ranked, latency = retrieve_dense(
                client,
                index,
                case.query,
                embedding_model=settings.embedding_model,
                top_k=eval_k,
            )
            per_query.append(
                evaluate_ranking(
                    case,
                    ranked,
                    strategy=name,
                    k=eval_k,
                    all_chunks=index.chunks,
                    units_by_id=units_by_id,
                    binary_threshold=dataset.binary_threshold,
                    latency_ms=latency,
                )
            )
        evaluations.append(aggregate_strategy(name, eval_k, per_query))

    return ExperimentResult(
        k=eval_k,
        embedding_model=settings.embedding_model,
        strategy_config={name: configs[name] for name in strategies},
        indexes=indexes,
        evaluations=evaluations,
    )
