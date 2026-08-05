"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists."""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunker import Chunk
from rag.store import RankedChunk


@dataclass(frozen=True)
class FusedChunk:
    """Hybrid result with provenance from dense and lexical rankings."""

    chunk: Chunk
    rrf_score: float
    rank: int
    dense_rank: int | None
    lexical_rank: int | None
    dense_score: float | None
    lexical_score: float | None


def reciprocal_rank_fusion(
    dense: list[RankedChunk],
    lexical: list[RankedChunk],
    *,
    k: int = 60,
    top_k: int = 3,
) -> list[FusedChunk]:
    """Fuse two rankings with RRF.

    RRF(d) = Σ 1 / (k + rank(d))

    Ranks are 1-based. Candidates missing from a list contribute 0 for that
    list. Raw dense/BM25 scores are intentionally NOT mixed — only ranks.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    dense_by_id = {item.chunk.id: item for item in dense}
    lexical_by_id = {item.chunk.id: item for item in lexical}
    all_ids = list(dict.fromkeys([*dense_by_id.keys(), *lexical_by_id.keys()]))

    fused: list[tuple[str, float]] = []
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in dense_by_id:
            score += 1.0 / (k + dense_by_id[chunk_id].rank)
        if chunk_id in lexical_by_id:
            score += 1.0 / (k + lexical_by_id[chunk_id].rank)
        fused.append((chunk_id, score))

    fused.sort(key=lambda pair: (-pair[1], pair[0]))

    results: list[FusedChunk] = []
    for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1):
        d = dense_by_id.get(chunk_id)
        lex = lexical_by_id.get(chunk_id)
        chunk = d.chunk if d is not None else lex.chunk  # type: ignore[union-attr]
        results.append(
            FusedChunk(
                chunk=chunk,
                rrf_score=score,
                rank=rank,
                dense_rank=d.rank if d else None,
                lexical_rank=lex.rank if lex else None,
                dense_score=d.score if d else None,
                lexical_score=lex.score if lex else None,
            )
        )
    return results
