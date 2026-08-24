"""Label-based entity resolution against rdfs:label."""

from __future__ import annotations

import re

from graphrag.graph import RdfGraphStore
from graphrag.model import ResolvedEntity


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_entities(
    question: str, store: RdfGraphStore
) -> tuple[list[ResolvedEntity], int]:
    """Match question text against rdfs:label values.

    Deterministic resolution rule when multiple labels match:
    1. Prefer the longest matching label (greedy, non-overlapping).
    2. Sort remaining matches by IRI ascending.

    This is label-based entity resolution — not semantic entity linking.
    """
    normalized_question = _normalize(question)
    candidates = store.labeled_entities()
    candidate_count = len(candidates)

    # Longest labels first so "Knowledge Platform" wins over partial overlaps.
    sorted_labels = sorted(
        candidates, key=lambda item: (-len(item[0]), item[0].lower())
    )

    matched: list[ResolvedEntity] = []
    consumed_spans: list[tuple[int, int]] = []

    for label, iri in sorted_labels:
        pattern = re.escape(_normalize(label))
        for match in re.finditer(pattern, normalized_question):
            start, end = match.span()
            if any(not (end <= s or start >= e) for s, e in consumed_spans):
                continue
            consumed_spans.append((start, end))
            matched.append(
                ResolvedEntity(
                    iri=iri,
                    label=label,
                    matchSpan=label,
                )
            )

    matched.sort(key=lambda entity: entity.iri)
    return matched, candidate_count
