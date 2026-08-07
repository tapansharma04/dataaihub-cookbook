"""In-memory dense vector store with cosine similarity."""

from dataclasses import dataclass

import numpy as np

from rag.chunking.base import Chunk


@dataclass(frozen=True)
class RankedChunk:
    """A chunk with a retrieval score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


class InMemoryVectorStore:
    """Stores chunk texts + embedding vectors in process memory."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self._chunks.extend(chunks)
        new_vectors = np.asarray(embeddings, dtype=np.float64)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[RankedChunk]:
        if self._vectors is None or not self._chunks:
            return []
        if top_k <= 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
        docs = _normalize(self._vectors)
        q = _normalize(query)
        scores = (docs @ q.T).ravel()

        k = min(top_k, len(self._chunks))
        candidate_idx = np.argpartition(-scores, kth=k - 1)[:k]
        ranked = candidate_idx[np.argsort(-scores[candidate_idx])]

        return [
            RankedChunk(
                chunk=self._chunks[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(ranked, start=1)
        ]
