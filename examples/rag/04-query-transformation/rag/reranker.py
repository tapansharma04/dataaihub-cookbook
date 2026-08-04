"""Local cross-encoder reranking over hybrid/RRF candidates.

Conceptually:

    pairs = [(query, candidate_1.text), (query, candidate_2.text), ...]
    scores = model.predict(pairs)
    sort candidates by score descending

The model evaluates query and candidate *together* (cross-encoder), unlike
bi-encoder dense retrieval which compares independently computed vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag.chunker import Chunk
from rag.fusion import FusedChunk

ScoreFn = Callable[[str, list[str]], list[float]]


@dataclass(frozen=True)
class RerankedChunk:
    """Final ranking with full retrieval → fusion → rerank provenance."""

    chunk: Chunk
    reranker_score: float
    rank: int
    previous_rank: int
    rrf_score: float
    dense_rank: int | None
    lexical_rank: int | None
    dense_score: float | None
    lexical_score: float | None


def rank_movement(previous_rank: int, new_rank: int) -> str:
    """Human-readable before→after movement label."""
    if previous_rank == new_rank:
        return "—"
    if new_rank < previous_rank:
        return f"↑ {previous_rank} → {new_rank}"
    return f"↓ {previous_rank} → {new_rank}"


def build_pairs(query: str, candidates: list[FusedChunk]) -> list[tuple[str, str]]:
    """Construct (query, document) pairs for the cross-encoder."""
    return [(query, item.chunk.text) for item in candidates]


def order_by_scores(
    candidates: list[FusedChunk],
    scores: list[float],
    *,
    top_k: int,
) -> list[RerankedChunk]:
    """Sort candidates by score desc; preserve RRF provenance on each result."""
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if top_k <= 0:
        return []
    if not candidates:
        return []

    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (-scores[pair[0]], pair[1].chunk.id),
    )

    results: list[RerankedChunk] = []
    for new_rank, (idx, item) in enumerate(indexed[:top_k], start=1):
        results.append(
            RerankedChunk(
                chunk=item.chunk,
                reranker_score=float(scores[idx]),
                rank=new_rank,
                previous_rank=item.rank,
                rrf_score=item.rrf_score,
                dense_rank=item.dense_rank,
                lexical_rank=item.lexical_rank,
                dense_score=item.dense_score,
                lexical_score=item.lexical_score,
            )
        )
    return results


def rerank_candidates(
    query: str,
    candidates: list[FusedChunk],
    *,
    top_k: int,
    score_fn: ScoreFn,
) -> list[RerankedChunk]:
    """Score candidates with ``score_fn`` and return the final top-k order.

    ``score_fn(query, texts) -> scores`` is injectable so tests can supply
    deterministic scores without downloading a model.
    """
    if top_k <= 0 or not candidates:
        return []
    texts = [item.chunk.text for item in candidates]
    scores = score_fn(query, texts)
    return order_by_scores(candidates, scores, top_k=top_k)


class CrossEncoderReranker:
    """Thin wrapper around sentence-transformers CrossEncoder.

    First call may download model weights (~90 MB). Subsequent calls reuse the
    library cache. Runs on CPU by default.
    """

    def __init__(self, model_name: str) -> None:
        # Import lazily so unit tests that inject score_fn never need torch.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        raw = self._model.predict(pairs)
        # predict returns a numpy array for batch input.
        return [float(value) for value in raw]

    def rerank(
        self,
        query: str,
        candidates: list[FusedChunk],
        *,
        top_k: int,
    ) -> list[RerankedChunk]:
        return rerank_candidates(
            query,
            candidates,
            top_k=top_k,
            score_fn=self.score,
        )
