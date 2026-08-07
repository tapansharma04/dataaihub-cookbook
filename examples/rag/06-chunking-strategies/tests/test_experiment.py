"""Experiment invariant tests — stub embeddings, no paid APIs."""

from __future__ import annotations

from pathlib import Path

from config import ALL_STRATEGIES, Settings, strategy_config
from evaluation.dataset import load_eval_dataset
from evaluation.evidence import attach_evidence
from experiment import build_strategy_index, run_experiment
from rag.chunking import chunk_with_strategy
from rag.chunking.stats import compute_chunk_stats
from rag.loader import load_document

ROOT = Path(__file__).resolve().parents[1]


class _StubClient:
    """Deterministic fake embeddings from text hashes — no network."""

    class embeddings:
        @staticmethod
        def create(model, input):  # noqa: A002
            texts = input if isinstance(input, list) else [input]

            class Item:
                def __init__(self, index, embedding):
                    self.index = index
                    self.embedding = embedding

            class Resp:
                pass

            data = []
            for i, text in enumerate(texts):
                # 8-dim pseudo-embedding from character codes — stable.
                vec = [0.0] * 8
                for j, ch in enumerate(text[:64]):
                    vec[j % 8] += (ord(ch) % 31) / 31.0
                # L2-normalize-ish nonzero
                s = sum(v * v for v in vec) ** 0.5 or 1.0
                data.append(Item(i, [v / s for v in vec]))
            resp = Resp()
            resp.data = data
            return resp


def test_same_corpus_queries_k_across_strategies():
    text = load_document(ROOT / "data" / "sample.md")
    dataset = load_eval_dataset(ROOT / "data" / "eval_set.json", text)
    settings = Settings(
        openai_api_key="test",
        data_path=ROOT / "data" / "sample.md",
        eval_path=ROOT / "data" / "eval_set.json",
        eval_k=3,
    )
    result = run_experiment(
        _StubClient(),  # type: ignore[arg-type]
        text,
        dataset,
        settings,
        strategies=ALL_STRATEGIES,
        k=3,
    )
    assert result.k == 3
    assert result.embedding_model == settings.embedding_model
    assert set(result.indexes) == set(ALL_STRATEGIES)

    query_ids = [q.id for q in dataset.queries]
    for strat_result in result.evaluations:
        assert strat_result.k == 3
        assert [q.query_id for q in strat_result.per_query] == query_ids
        assert strat_result.query_count == len(dataset.queries)

    # Only chunking configs differ; embedder name identical.
    configs = strategy_config(settings)
    assert configs["fixed"]["chunk_size"] == settings.fixed_chunk_size
    assert "target_size" in configs["recursive"]
    assert "boundary" in configs["structure"]


def test_chunk_id_namespaces_are_disjoint():
    text = load_document(ROOT / "data" / "sample.md")
    settings = Settings()
    ids = {}
    for name in ALL_STRATEGIES:
        chunks = chunk_with_strategy(
            text,
            name,
            fixed_chunk_size=settings.fixed_chunk_size,
            fixed_chunk_overlap=settings.fixed_chunk_overlap,
            recursive_target_size=settings.recursive_target_size,
            recursive_chunk_overlap=settings.recursive_chunk_overlap,
        )
        ids[name] = {c.id for c in chunks}
        assert all(c.id.startswith(f"{name}-") for c in chunks)
    assert ids["fixed"].isdisjoint(ids["recursive"])
    assert ids["fixed"].isdisjoint(ids["structure"])


def test_build_strategy_index_timings_present():
    text = load_document(ROOT / "data" / "sample.md")
    dataset = load_eval_dataset(ROOT / "data" / "eval_set.json", text)
    settings = Settings(openai_api_key="test")
    index = build_strategy_index(
        _StubClient(),  # type: ignore[arg-type]
        text,
        dataset,
        settings,
        "fixed",
    )
    assert "chunking_ms" in index.timings_ms
    assert "embedding_ms" in index.timings_ms
    assert len(index.chunks) == len(index.store)
    assert index.stats.chunk_count == len(index.chunks)


def test_chunk_stats_fragmentation():
    text = load_document(ROOT / "data" / "sample.md")
    dataset = load_eval_dataset(ROOT / "data" / "eval_set.json", text)
    settings = Settings()
    fixed = attach_evidence(
        chunk_with_strategy(
            text,
            "fixed",
            fixed_chunk_size=settings.fixed_chunk_size,
            fixed_chunk_overlap=settings.fixed_chunk_overlap,
        ),
        dataset.evidence_units,
    )
    structured = attach_evidence(
        chunk_with_strategy(text, "structure"),
        dataset.evidence_units,
    )
    fixed_stats = compute_chunk_stats(
        fixed, source_text=text, evidence_units=dataset.evidence_units
    )
    struct_stats = compute_chunk_stats(
        structured, source_text=text, evidence_units=dataset.evidence_units
    )
    # Structure-aware should fragment fewer evidence units than fixed windows.
    assert (
        struct_stats.fragmented_evidence_count <= fixed_stats.fragmented_evidence_count
    )
