"""Shared chunk types and helpers for chunking strategies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit produced by one chunking strategy.

    IDs are strategy-scoped and deterministic, e.g. ``fixed-0001``.
    Chunks from different strategies never share IDs or boundaries.
    """

    id: str
    text: str
    start: int
    end: int
    strategy: str
    source: str
    section: str | None = None
    prev_id: str | None = None
    next_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)


def format_chunk_id(strategy: str, index: int) -> str:
    """Stable zero-padded ID: ``fixed-0001``, ``recursive-0001``, …"""
    return f"{strategy}-{index:04d}"


def link_neighbors(chunks: list[Chunk]) -> list[Chunk]:
    """Attach prev/next IDs after the full list is known."""
    if not chunks:
        return []
    linked: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        prev_id = chunks[i - 1].id if i > 0 else None
        next_id = chunks[i + 1].id if i + 1 < len(chunks) else None
        linked.append(
            Chunk(
                id=chunk.id,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                strategy=chunk.strategy,
                source=chunk.source,
                section=chunk.section,
                prev_id=prev_id,
                next_id=next_id,
                evidence_ids=chunk.evidence_ids,
                metadata=dict(chunk.metadata),
            )
        )
    return linked


def strip_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Return stripped piece and absolute [start, end) of non-whitespace content."""
    local = text[start:end]
    if not local.strip():
        return "", start, start
    lead = len(local) - len(local.lstrip())
    trail = len(local) - len(local.rstrip())
    abs_start = start + lead
    abs_end = end - trail
    return text[abs_start:abs_end], abs_start, abs_end


def section_heading_at(text: str, start: int) -> str | None:
    """Best-effort section heading covering ``start`` (nearest preceding ##)."""
    prefix = text[: start + 1]
    last = prefix.rfind("\n## ")
    if last == -1:
        if text.lstrip().startswith("## "):
            line = text.lstrip().splitlines()[0]
            return line.lstrip("# ").strip()
        # Title / preface before first ##
        first_line = text.splitlines()[0].strip() if text.strip() else ""
        if first_line.startswith("# "):
            return first_line.lstrip("# ").strip()
        return None
    line_start = last + 1
    line_end = text.find("\n", line_start)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].lstrip("# ").strip()
