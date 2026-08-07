"""Load the frozen evaluation set (evidence-based, not chunk-ID judgments)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.evidence import EvidenceUnit, resolve_evidence_units


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation case: information need + required evidence grades."""

    id: str
    query: str
    evidence_grades: dict[str, int]
    rationale: dict[str, str]

    def required_evidence_ids(self, *, min_grade: int = 1) -> set[str]:
        return {eid for eid, g in self.evidence_grades.items() if g >= min_grade}


@dataclass(frozen=True)
class EvalDataset:
    description: str
    corpus: str
    k_default: int
    binary_threshold: int
    relevance_scheme: dict
    evidence_units: list[EvidenceUnit]
    queries: list[EvalQuery]

    def units_by_id(self) -> dict[str, EvidenceUnit]:
        return {u.id: u for u in self.evidence_units}

    def get(self, query_id: str) -> EvalQuery:
        for q in self.queries:
            if q.id == query_id:
                return q
        raise KeyError(f"Unknown evaluation query id: {query_id}")


def load_eval_dataset(path: Path, source_text: str) -> EvalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    units = resolve_evidence_units(source_text, raw["evidence_units"])
    unit_ids = {u.id for u in units}

    queries: list[EvalQuery] = []
    for item in raw["queries"]:
        grades = {str(k): int(v) for k, v in item["evidence_grades"].items()}
        if not grades:
            raise ValueError(f"Query {item['id']} has empty evidence_grades")
        for eid, grade in grades.items():
            if eid not in unit_ids:
                raise ValueError(f"Query {item['id']}: unknown evidence id {eid}")
            if grade < 1:
                raise ValueError(
                    f"Query {item['id']}: store only positive grades; "
                    "unlabeled evidence is grade 0"
                )
        rationale = {str(k): str(v) for k, v in item.get("rationale", {}).items()}
        missing = set(grades) - set(rationale)
        if missing:
            raise ValueError(
                f"Query {item['id']}: missing rationale for {sorted(missing)}"
            )
        queries.append(
            EvalQuery(
                id=item["id"],
                query=item["query"],
                evidence_grades=grades,
                rationale=rationale,
            )
        )

    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation query ids")

    scheme = raw.get("relevance_scheme", {})
    return EvalDataset(
        description=raw.get("description", ""),
        corpus=raw.get("corpus", ""),
        k_default=int(raw.get("k_default", 3)),
        binary_threshold=int(scheme.get("binary_threshold", 1)),
        relevance_scheme=scheme,
        evidence_units=units,
        queries=queries,
    )
