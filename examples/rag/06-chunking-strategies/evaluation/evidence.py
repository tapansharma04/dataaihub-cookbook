"""Evidence units and deterministic chunk↔evidence grading.

Relevance is frozen against *source evidence*, not strategy-specific chunk IDs.
Each evaluation query names required evidence units. Generated chunks inherit
the evidence units whose source spans they cover. Retrieved chunks are graded
from that inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunking.base import Chunk


@dataclass(frozen=True)
class EvidenceUnit:
    """Identifiable span in the source document."""

    id: str
    section: str
    anchor: str
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.anchor


def resolve_evidence_units(
    source_text: str,
    raw_units: list[dict],
) -> list[EvidenceUnit]:
    """Resolve anchor strings to absolute [start, end) offsets.

    Anchors must appear exactly once in the corpus — ambiguity is an error.
    """
    resolved: list[EvidenceUnit] = []
    seen_ids: set[str] = set()
    for item in raw_units:
        eid = str(item["id"])
        if eid in seen_ids:
            raise ValueError(f"Duplicate evidence id: {eid}")
        seen_ids.add(eid)
        anchor = str(item["anchor"])
        count = source_text.count(anchor)
        if count == 0:
            raise ValueError(f"Evidence {eid}: anchor not found in corpus")
        if count > 1:
            raise ValueError(
                f"Evidence {eid}: anchor is ambiguous ({count} occurrences)"
            )
        start = source_text.index(anchor)
        end = start + len(anchor)
        resolved.append(
            EvidenceUnit(
                id=eid,
                section=str(item.get("section", "")),
                anchor=anchor,
                start=start,
                end=end,
            )
        )
    return resolved


def evidence_ids_for_chunk(
    chunk: Chunk,
    units: list[EvidenceUnit],
) -> tuple[str, ...]:
    """Evidence units whose source spans overlap the chunk span."""
    hits = [u.id for u in units if _overlaps(chunk.start, chunk.end, u.start, u.end)]
    return tuple(hits)


def attach_evidence(
    chunks: list[Chunk],
    units: list[EvidenceUnit],
) -> list[Chunk]:
    """Return new chunks with ``evidence_ids`` populated."""
    out: list[Chunk] = []
    for chunk in chunks:
        eids = evidence_ids_for_chunk(chunk, units)
        out.append(
            Chunk(
                id=chunk.id,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                strategy=chunk.strategy,
                source=chunk.source,
                section=chunk.section,
                prev_id=chunk.prev_id,
                next_id=chunk.next_id,
                evidence_ids=eids,
                metadata=dict(chunk.metadata),
            )
        )
    return out


def containment(chunk: Chunk, unit: EvidenceUnit) -> str:
    """Return ``full``, ``partial``, or ``none`` for chunk vs evidence span."""
    if not _overlaps(chunk.start, chunk.end, unit.start, unit.end):
        return "none"
    if chunk.start <= unit.start and chunk.end >= unit.end:
        return "full"
    return "partial"


def grade_chunk_for_query(
    chunk: Chunk,
    *,
    evidence_grades: dict[str, int],
    units_by_id: dict[str, EvidenceUnit],
) -> int:
    """Deterministic graded relevance for one chunk given frozen evidence grades.

    Rules (documented in README):
    - Full containment of evidence E → grade = evidence_grades[E]
    - Partial overlap with evidence E where evidence_grades[E] >= 2 → grade 1
      (partial primary/supporting evidence)
    - Partial overlap with evidence E where evidence_grades[E] == 1 → grade 1
    - No overlap → 0
    - Chunk grade = max over applicable evidence rules
    """
    best = 0
    for eid, eg in evidence_grades.items():
        unit = units_by_id.get(eid)
        if unit is None:
            continue
        status = containment(chunk, unit)
        if status == "full":
            best = max(best, eg)
        elif status == "partial":
            best = max(best, 1)
    return best


def build_chunk_relevance(
    chunks: list[Chunk],
    *,
    evidence_grades: dict[str, int],
    units_by_id: dict[str, EvidenceUnit],
) -> dict[str, int]:
    """Map chunk_id → grade for all chunks with grade > 0."""
    relevance: dict[str, int] = {}
    for chunk in chunks:
        grade = grade_chunk_for_query(
            chunk,
            evidence_grades=evidence_grades,
            units_by_id=units_by_id,
        )
        if grade > 0:
            relevance[chunk.id] = grade
    return relevance


def evidence_coverage(
    retrieved_chunks: list[Chunk],
    *,
    required_evidence_ids: set[str],
    units_by_id: dict[str, EvidenceUnit] | None = None,
) -> dict[str, object]:
    """Fraction of required evidence IDs present in any retrieved chunk.

    Prefers span overlap via ``units_by_id`` when provided; otherwise falls
    back to pre-attached ``chunk.evidence_ids``.
    """
    found: set[str] = set()
    for chunk in retrieved_chunks:
        if units_by_id is not None:
            for eid in required_evidence_ids:
                unit = units_by_id.get(eid)
                if unit is None:
                    continue
                if containment(chunk, unit) != "none":
                    found.add(eid)
        else:
            found.update(
                eid for eid in chunk.evidence_ids if eid in required_evidence_ids
            )
    total = len(required_evidence_ids)
    return {
        "required": sorted(required_evidence_ids),
        "found": sorted(found),
        "missed": sorted(required_evidence_ids - found),
        "coverage": (len(found) / total) if total else 0.0,
    }


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1
