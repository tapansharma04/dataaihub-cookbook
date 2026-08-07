"""Strategy A — fixed-size character windows with overlap."""

from __future__ import annotations

from config import STRATEGY_FIXED
from rag.chunking.base import (
    Chunk,
    format_chunk_id,
    link_neighbors,
    section_heading_at,
    strip_span,
)


def chunk_fixed(
    text: str,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    source: str = "sample",
) -> list[Chunk]:
    """Split on fixed character windows.

    Units are Python ``str`` lengths (Unicode code points), **not** tokens.
    Overlap is also measured in characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        piece, abs_start, abs_end = strip_span(text, start, end)
        if piece:
            chunks.append(
                Chunk(
                    id=format_chunk_id(STRATEGY_FIXED, index),
                    text=piece,
                    start=abs_start,
                    end=abs_end,
                    strategy=STRATEGY_FIXED,
                    source=source,
                    section=section_heading_at(text, abs_start),
                    metadata={
                        "window_start": start,
                        "window_end": end,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "size_unit": "characters",
                    },
                )
            )
            index += 1
        if end >= length:
            break
        start = end - chunk_overlap

    return link_neighbors(chunks)
