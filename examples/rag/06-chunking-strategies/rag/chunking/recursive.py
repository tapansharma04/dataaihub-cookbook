"""Strategy B — recursive splitting on progressively smaller separators."""

from __future__ import annotations

from config import STRATEGY_RECURSIVE
from rag.chunking.base import (
    Chunk,
    format_chunk_id,
    link_neighbors,
    section_heading_at,
    strip_span,
)

# Ordered from coarsest document structure to hard character split.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n## ",  # markdown section boundaries
    "\n\n",  # paragraphs
    "\n",  # lines
    ". ",  # sentence-ish
    " ",  # whitespace
    "",  # hard character split fallback
)


def chunk_recursive(
    text: str,
    *,
    target_size: int = 400,
    chunk_overlap: int = 50,
    source: str = "sample",
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Recursively split text until pieces are within ``target_size``.

    Mechanics (deterministic, inspectable):
    1. If the current span length ≤ target_size, emit it as one chunk.
    2. Otherwise try the first separator that appears inside the span.
    3. Split on that separator, then recurse on each piece.
    4. If a separator yields a single oversized piece, try the next separator.
    5. Empty separator ``""`` hard-splits into character windows of target_size.

    Overlap: after producing ordered leaf spans, adjacent chunks optionally
    prepend up to ``chunk_overlap`` characters from the previous span's end
    when the separator path used hard/soft splits (implemented as character
    window overlap only on the hard-split fallback path for adjacent leaves
    that were produced without structural separators). For educational
    clarity, overlap is applied only when two consecutive leaf spans are
    adjacent in the source (end of A == start of B) by extending the later
    chunk's start backward by up to ``chunk_overlap``.

    Size unit: characters (Python ``str`` length), not tokens.
    """
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= target_size:
        raise ValueError("chunk_overlap must be smaller than target_size")

    if not text.strip():
        return []

    spans = _recursive_spans(text, 0, len(text), target_size, separators)
    spans = _apply_adjacent_overlap(spans, chunk_overlap)

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        piece, abs_start, abs_end = strip_span(text, start, end)
        if not piece:
            continue
        chunks.append(
            Chunk(
                id=format_chunk_id(STRATEGY_RECURSIVE, index),
                text=piece,
                start=abs_start,
                end=abs_end,
                strategy=STRATEGY_RECURSIVE,
                source=source,
                section=section_heading_at(text, abs_start),
                metadata={
                    "target_size": target_size,
                    "chunk_overlap": chunk_overlap,
                    "size_unit": "characters",
                    "separators": list(separators),
                },
            )
        )
    return link_neighbors(chunks)


def _recursive_spans(
    text: str,
    start: int,
    end: int,
    target_size: int,
    separators: tuple[str, ...],
) -> list[tuple[int, int]]:
    length = end - start
    if length <= 0:
        return []
    if length <= target_size:
        return [(start, end)]

    for sep in separators:
        if sep == "":
            return _hard_split_spans(start, end, target_size)

        pieces = _split_span(text, start, end, sep)
        if len(pieces) <= 1:
            continue

        out: list[tuple[int, int]] = []
        for p_start, p_end in pieces:
            out.extend(_recursive_spans(text, p_start, p_end, target_size, separators))
        return out

    return _hard_split_spans(start, end, target_size)


def _split_span(
    text: str,
    start: int,
    end: int,
    separator: str,
) -> list[tuple[int, int]]:
    """Split [start, end) on separator, keeping separator attached to the
    following piece when it is a structural marker (starts with newline+##).
    """
    region = text[start:end]
    if separator not in region:
        return [(start, end)]

    # Find split points inside the region (not at absolute index 0 of region
    # when separator would leave an empty first piece unnecessarily handled).
    parts: list[tuple[int, int]] = []
    cursor = 0
    search_from = 0
    while True:
        idx = region.find(separator, search_from)
        if idx == -1:
            break
        # For "\n## ", keep the heading with the following section: split
        # *before* the separator so the next piece starts at the heading.
        if separator.startswith("\n##"):
            split_at = idx  # end of previous piece; separator starts next
            piece_end = start + split_at
            if piece_end > start + cursor:
                parts.append((start + cursor, piece_end))
            cursor = split_at  # next starts at separator (includes \n## )
            search_from = idx + len(separator)
        else:
            # Consume separator into the boundary between pieces.
            piece_end = start + idx
            if piece_end > start + cursor:
                parts.append((start + cursor, piece_end))
            cursor = idx + len(separator)
            search_from = cursor

    if start + cursor < end:
        parts.append((start + cursor, end))
    return [p for p in parts if p[1] > p[0]]


def _hard_split_spans(
    start: int,
    end: int,
    target_size: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        spans.append((cursor, min(cursor + target_size, end)))
        cursor += target_size
    return spans


def _apply_adjacent_overlap(
    spans: list[tuple[int, int]],
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Extend each span (except the first) backward when abutting previous."""
    if chunk_overlap <= 0 or len(spans) <= 1:
        return spans
    out: list[tuple[int, int]] = [spans[0]]
    for i in range(1, len(spans)):
        prev_start, prev_end = out[-1]
        cur_start, cur_end = spans[i]
        if cur_start <= prev_end:
            # Already overlapping or nested — keep as produced.
            out.append((cur_start, cur_end))
            continue
        # Adjacent or gapped: pull start back by overlap into prior content.
        new_start = max(0, cur_start - chunk_overlap)
        # Do not cross into content before previous span start.
        new_start = max(new_start, prev_start)
        out.append((new_start, cur_end))
    return out
