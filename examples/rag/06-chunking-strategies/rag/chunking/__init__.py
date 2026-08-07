"""Chunking strategies package."""

from config import STRATEGY_FIXED, STRATEGY_RECURSIVE, STRATEGY_STRUCTURE
from rag.chunking.base import Chunk
from rag.chunking.fixed import chunk_fixed
from rag.chunking.recursive import chunk_recursive
from rag.chunking.stats import ChunkStats, compute_chunk_stats
from rag.chunking.structure import chunk_structure


def chunk_with_strategy(
    text: str,
    strategy: str,
    *,
    source: str = "sample",
    fixed_chunk_size: int = 400,
    fixed_chunk_overlap: int = 50,
    recursive_target_size: int = 400,
    recursive_chunk_overlap: int = 50,
) -> list[Chunk]:
    """Dispatch to the named strategy. Deterministic for a given config."""
    if strategy == STRATEGY_FIXED:
        return chunk_fixed(
            text,
            chunk_size=fixed_chunk_size,
            chunk_overlap=fixed_chunk_overlap,
            source=source,
        )
    if strategy == STRATEGY_RECURSIVE:
        return chunk_recursive(
            text,
            target_size=recursive_target_size,
            chunk_overlap=recursive_chunk_overlap,
            source=source,
        )
    if strategy == STRATEGY_STRUCTURE:
        return chunk_structure(text, source=source)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "Chunk",
    "ChunkStats",
    "chunk_fixed",
    "chunk_recursive",
    "chunk_structure",
    "chunk_with_strategy",
    "compute_chunk_stats",
]
