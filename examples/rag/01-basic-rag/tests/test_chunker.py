"""Chunking tests — no network / no paid APIs."""

import pytest

from rag.chunker import chunk_text


def test_chunk_text_produces_overlapping_chunks():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=80, source="t")
    assert len(chunks) >= 3
    assert chunks[0].id == "t-0"
    assert len(chunks[0].text) <= 400


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
