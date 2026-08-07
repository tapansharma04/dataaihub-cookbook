"""Deterministic structural measurements for a chunk list."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from rag.chunking.base import Chunk


class _Span(Protocol):
    start: int
    end: int


@dataclass(frozen=True)
class ChunkStats:
    strategy: str
    chunk_count: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    total_indexed_chars: int
    source_chars: int
    overlap_extra_chars: int
    """Characters indexed beyond source length due to overlap/duplication."""
    evidence_unit_count: int
    fragmented_evidence_count: int
    """Evidence units whose span overlaps more than one chunk."""
    evidence_fragmentation_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_chunk_stats(
    chunks: list[Chunk],
    *,
    source_text: str,
    evidence_units: list[_Span] | None = None,
) -> ChunkStats:
    strategy = chunks[0].strategy if chunks else "unknown"
    lengths = [c.length for c in chunks]
    total_indexed = sum(lengths)
    source_chars = len(source_text)
    overlap_extra = max(0, total_indexed - source_chars)

    units = evidence_units or []
    fragmented = 0
    for unit in units:
        hits = sum(1 for c in chunks if _overlaps(c.start, c.end, unit.start, unit.end))
        if hits > 1:
            fragmented += 1

    frag_rate = (fragmented / len(units)) if units else 0.0

    return ChunkStats(
        strategy=strategy,
        chunk_count=len(chunks),
        avg_chunk_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        min_chunk_length=min(lengths) if lengths else 0,
        max_chunk_length=max(lengths) if lengths else 0,
        total_indexed_chars=total_indexed,
        source_chars=source_chars,
        overlap_extra_chars=overlap_extra,
        evidence_unit_count=len(units),
        fragmented_evidence_count=fragmented,
        evidence_fragmentation_rate=frag_rate,
    )


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1
