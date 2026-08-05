"""Lightweight Okapi BM25 lexical retriever (no search framework)."""

from __future__ import annotations

import math
import re
from collections import Counter

from rag.chunker import Chunk
from rag.store import RankedChunk

# Keep identifiers like E_CONN_42, v2.3, getUserProfileAsync as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*", re.IGNORECASE)

# Tiny stoplist so BM25 emphasizes content terms / identifiers.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "when",
        "what",
        "should",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; preserve dots and underscores."""
    return [
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    ]


class BM25Index:
    """In-memory Okapi BM25 over chunk texts.

    score(D, Q) = Σ IDF(qi) * f(qi,D)*(k1+1) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

    Defaults (k1=1.5, b=0.75) are conventional — not tuned to this corpus.
    """

    def __init__(
        self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self._chunks = list(chunks)
        self._docs: list[list[str]] = [tokenize(c.text) for c in self._chunks]
        self._doc_len = [len(tokens) for tokens in self._docs]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        self._df: Counter[str] = Counter()
        for tokens in self._docs:
            for term in set(tokens):
                self._df[term] += 1
        self._n = len(self._docs)

    def __len__(self) -> int:
        return len(self._chunks)

    def _idf(self, term: str) -> float:
        # Lucene-style IDF with +1 to keep scores non-negative.
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        if not q_terms or self._n == 0:
            return [0.0] * self._n

        scores = [0.0] * self._n
        for i, tokens in enumerate(self._docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = self._doc_len[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                numer = freq * (self.k1 + 1.0)
                scores[i] += idf * numer / (freq + denom_norm)
        return scores

    def search(self, query: str, top_k: int = 3) -> list[RankedChunk]:
        if top_k <= 0 or self._n == 0:
            return []

        scores = self.score(query)
        # Sort by score desc, then chunk id for stable ties.
        order = sorted(
            range(self._n),
            key=lambda i: (-scores[i], self._chunks[i].id),
        )
        results: list[RankedChunk] = []
        for rank, i in enumerate(order[: min(top_k, self._n)], start=1):
            if scores[i] <= 0:
                break
            results.append(
                RankedChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            )
        return results
