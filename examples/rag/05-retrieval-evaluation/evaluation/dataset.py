"""Load and validate the golden evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation case: information need + graded relevance judgments."""

    id: str
    query: str
    relevance: dict[str, int]
    rationale: dict[str, str]

    def relevant_ids(self, *, min_grade: int = 1) -> set[str]:
        """Chunk IDs with grade >= min_grade (binary relevance for Recall/MRR)."""
        return {cid for cid, grade in self.relevance.items() if grade >= min_grade}

    def grade(self, chunk_id: str) -> int:
        return self.relevance.get(chunk_id, 0)


@dataclass(frozen=True)
class EvalDataset:
    description: str
    corpus: str
    k_default: int
    queries: list[EvalQuery]
    relevance_scheme: dict

    def by_id(self) -> dict[str, EvalQuery]:
        return {q.id: q for q in self.queries}

    def get(self, query_id: str) -> EvalQuery:
        for q in self.queries:
            if q.id == query_id:
                return q
        raise KeyError(f"Unknown evaluation query id: {query_id}")


def load_eval_dataset(path: Path) -> EvalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    for item in raw["queries"]:
        relevance = {str(k): int(v) for k, v in item["relevance"].items()}
        if not relevance:
            raise ValueError(f"Query {item['id']} has empty relevance judgments")
        for grade in relevance.values():
            if grade < 1:
                raise ValueError(
                    f"Query {item['id']}: store only positive grades in "
                    "relevance; unlabeled chunks are grade 0"
                )
        rationale = {str(k): str(v) for k, v in item.get("rationale", {}).items()}
        missing = set(relevance) - set(rationale)
        if missing:
            raise ValueError(
                f"Query {item['id']}: missing rationale for {sorted(missing)}"
            )
        queries.append(
            EvalQuery(
                id=item["id"],
                query=item["query"],
                relevance=relevance,
                rationale=rationale,
            )
        )
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation query ids")
    return EvalDataset(
        description=raw.get("description", ""),
        corpus=raw.get("corpus", ""),
        k_default=int(raw.get("k_default", 3)),
        queries=queries,
        relevance_scheme=raw.get("relevance_scheme", {}),
    )
