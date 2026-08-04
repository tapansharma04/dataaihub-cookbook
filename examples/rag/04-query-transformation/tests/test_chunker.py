"""Chunking tests — no network / no paid APIs."""

from pathlib import Path

import pytest

from rag.chunker import chunk_text
from rag.loader import load_document

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.md"


def test_chunk_text_produces_one_chunk_per_section():
    text = load_document(SAMPLE)
    chunks = chunk_text(text, source="sample")
    # Preface + 10 ## sections.
    assert len(chunks) >= 10
    assert chunks[0].id == "sample-0"
    joined = "\n".join(c.text for c in chunks)
    assert "E_CONN_42" in joined
    code_chunks = [
        c
        for c in chunks
        if "E_CONN_42" in c.text and "Remediation for E_CONN_42" in c.text
    ]
    prose_chunks = [
        c for c in chunks if "packets vanish" in c.text and "E_CONN_42" not in c.text
    ]
    assert len(code_chunks) == 1
    assert len(prose_chunks) == 1


def test_sample_contains_ambiguity_anchors():
    text = load_document(SAMPLE)
    assert "E_CONN_42" in text
    assert "getUserProfileAsync" in text
    assert "getUserProfile" in text
    assert "certificate rotation" in text.lower() or "rotate" in text.lower()
    assert "NX-4400-PRO" in text
    assert "relay_handshake_timeout_ms" in text


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
