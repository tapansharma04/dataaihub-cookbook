"""Split documents into overlapping character chunks."""

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
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    source: str = "doc",
) -> list[Chunk]:
    """Split text into overlapping windows.

    Character chunking is intentionally simple so the pipeline stage is
    obvious. Production systems often use token-aware or semantic splitters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        piece = cleaned[start:end].strip()
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
