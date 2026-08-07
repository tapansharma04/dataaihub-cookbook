"""Strategy C — structure-aware chunking (markdown ## sections)."""

from __future__ import annotations

from config import STRATEGY_STRUCTURE
from rag.chunking.base import Chunk, format_chunk_id, link_neighbors, strip_span


def chunk_structure(
    text: str,
    *,
    source: str = "sample",
) -> list[Chunk]:
    """One chunk per markdown ``##`` section, heading kept with its body.

    This is **structure-aware** chunking — boundaries follow the handbook's
    headings, not embeddings or semantic similarity detectors.

    Preface material before the first ``##`` becomes its own chunk when
    non-empty. Section lengths are whatever the source document defines;
    there is no fixed character target.
    """
    cleaned = text
    if not cleaned.strip():
        return []

    if "\n## " not in cleaned and not cleaned.lstrip().startswith("## "):
        # No section structure — emit the whole document as one chunk.
        piece, abs_start, abs_end = strip_span(cleaned, 0, len(cleaned))
        if not piece:
            return []
        chunk = Chunk(
            id=format_chunk_id(STRATEGY_STRUCTURE, 0),
            text=piece,
            start=abs_start,
            end=abs_end,
            strategy=STRATEGY_STRUCTURE,
            source=source,
            section=_heading_from_piece(piece),
            metadata={"boundary": "whole-document", "size_unit": "characters"},
        )
        return link_neighbors([chunk])

    lines = cleaned.splitlines(keepends=True)
    headings: list[int] = []
    offset = 0
    for line in lines:
        if line.startswith("## "):
            headings.append(offset)
        offset += len(line)

    if not headings:
        piece, abs_start, abs_end = strip_span(cleaned, 0, len(cleaned))
        if not piece:
            return []
        return link_neighbors(
            [
                Chunk(
                    id=format_chunk_id(STRATEGY_STRUCTURE, 0),
                    text=piece,
                    start=abs_start,
                    end=abs_end,
                    strategy=STRATEGY_STRUCTURE,
                    source=source,
                    section=_heading_from_piece(piece),
                    metadata={"boundary": "whole-document", "size_unit": "characters"},
                )
            ]
        )

    ranges: list[tuple[int, int]] = []
    if headings[0] > 0 and cleaned[: headings[0]].strip():
        ranges.append((0, headings[0]))
    for i, start in enumerate(headings):
        end = headings[i + 1] if i + 1 < len(headings) else len(cleaned)
        ranges.append((start, end))

    chunks: list[Chunk] = []
    index = 0
    for start, end in ranges:
        piece, abs_start, abs_end = strip_span(cleaned, start, end)
        if not piece:
            continue
        chunks.append(
            Chunk(
                id=format_chunk_id(STRATEGY_STRUCTURE, index),
                text=piece,
                start=abs_start,
                end=abs_end,
                strategy=STRATEGY_STRUCTURE,
                source=source,
                section=_heading_from_piece(piece),
                metadata={
                    "boundary": "markdown-h2",
                    "size_unit": "characters",
                },
            )
        )
        index += 1
    return link_neighbors(chunks)


def _heading_from_piece(piece: str) -> str | None:
    for line in piece.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
        return stripped[:80]
    return None
