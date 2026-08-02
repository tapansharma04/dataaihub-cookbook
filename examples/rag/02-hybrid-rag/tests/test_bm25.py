"""BM25 lexical retrieval tests — no paid APIs."""

from rag.bm25 import BM25Index, tokenize
from rag.chunker import Chunk


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c-{i}", text=text, start=0, end=len(text))


def test_tokenize_preserves_error_codes_and_versions():
    tokens = tokenize("Fix E_CONN_42 on NebulaAPI v2.3 via getUserProfileAsync")
    assert "e_conn_42" in tokens
    assert "v2.3" in tokens
    assert "getuserprofileasync" in tokens


def test_bm25_ranks_exact_identifier_first():
    chunks = [
        _chunk(0, "Restart the router when connectivity feels intermittent."),
        _chunk(1, "Error code E_CONN_42 means TLS handshake failed on EdgeGateway."),
        _chunk(2, "Optimistic UI helps when profile loading is slow."),
    ]
    index = BM25Index(chunks)
    results = index.search("How do I fix error E_CONN_42?", top_k=3)
    assert results[0].chunk.id == "c-1"
    assert results[0].score > 0


def test_bm25_ignores_zero_score_tails():
    chunks = [
        _chunk(0, "alpha beta"),
        _chunk(1, "completely unrelated text"),
    ]
    index = BM25Index(chunks)
    results = index.search("alpha", top_k=5)
    assert len(results) == 1
    assert results[0].chunk.id == "c-0"
