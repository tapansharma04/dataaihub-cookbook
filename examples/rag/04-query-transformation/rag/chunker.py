"""Split handbook documents into one chunk per markdown section."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 520,
    chunk_overlap: int = 40,
    source: str = "doc",
) -> list[Chunk]:
    """Split on ## headings when present; otherwise fall back to windows.

    Hybrid RAG needs intact topics (error codes vs prose) so dense and lexical
    rankings are interpretable. Character parameters remain in the signature
    for config compatibility and the fallback path.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if "\n## " in cleaned or cleaned.startswith("## "):
        return _chunk_markdown_sections(cleaned, source=source)

    return _chunk_windows(
        cleaned,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=source,
    )


def _chunk_markdown_sections(text: str, *, source: str) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    # Collect [start_offset, end_offset) ranges for each ## section.
    headings: list[int] = []
    offset = 0
    for line in lines:
        if line.startswith("## "):
            headings.append(offset)
        offset += len(line)

    if not headings:
        return _chunk_windows(text, chunk_size=520, chunk_overlap=40, source=source)

    # Preface before the first ## becomes its own chunk when non-empty.
    ranges: list[tuple[int, int]] = []
    if headings[0] > 0 and text[: headings[0]].strip():
        ranges.append((0, headings[0]))
    for i, start in enumerate(headings):
        end = headings[i + 1] if i + 1 < len(headings) else len(text)
        ranges.append((start, end))

    chunks: list[Chunk] = []
    index = 0
    for start, end in ranges:
        piece = text[start:end].strip()
        if not piece:
            continue
        # Recompute exact span of stripped piece within [start, end).
        local = text[start:end]
        lead = len(local) - len(local.lstrip())
        trail = len(local) - len(local.rstrip())
        abs_start = start + lead
        abs_end = end - trail
        chunks.append(
            Chunk(id=f"{source}-{index}", text=piece, start=abs_start, end=abs_end)
        )
        index += 1
    return chunks


def _chunk_windows(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    source: str,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    id=f"{source}-{index}",
                    text=piece,
                    start=start,
                    end=end,
                )
            )
            index += 1
        if end >= length:
            break
        start = end - chunk_overlap

    return chunks
