"""Chunking strategy unit tests — deterministic, no network."""

from __future__ import annotations

from rag.chunking import (
    chunk_fixed,
    chunk_recursive,
    chunk_structure,
    chunk_with_strategy,
)
from rag.chunking.fixed import chunk_fixed as chunk_fixed_direct

SAMPLE = """# Handbook

## Alpha section

Alpha body sentence one. Alpha body sentence two.

## Beta section

Beta body that is a bit longer so recursive and fixed may diverge in places.

## Gamma short

Short.
"""


def test_fixed_deterministic_and_stable_ids():
    a = chunk_fixed(SAMPLE, chunk_size=80, chunk_overlap=10, source="sample")
    b = chunk_fixed(SAMPLE, chunk_size=80, chunk_overlap=10, source="sample")
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.text for c in a] == [c.text for c in b]
    assert a[0].id == "fixed-0000"
    assert a[0].strategy == "fixed"
    assert all(c.length > 0 for c in a)


def test_fixed_overlap_behavior():
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    chunks = chunk_fixed(text, chunk_size=50, chunk_overlap=10, source="doc")
    assert len(chunks) >= 2
    # Windows advance by chunk_size - overlap = 40.
    assert chunks[0].metadata["window_start"] == 0
    assert chunks[1].metadata["window_start"] == 40


def test_fixed_no_empty_chunks():
    text = "hello world " + ("x" * 100)
    chunks = chunk_fixed(text, chunk_size=40, chunk_overlap=5)
    assert all(c.text.strip() for c in chunks)


def test_recursive_respects_section_boundaries_when_small_enough():
    # Each section is under target → expect section-aligned pieces.
    chunks = chunk_recursive(SAMPLE, target_size=200, chunk_overlap=20)
    assert chunks[0].id.startswith("recursive-")
    texts = " ".join(c.text for c in chunks)
    assert "Alpha section" in texts
    assert "Beta section" in texts
    # Deterministic
    again = chunk_recursive(SAMPLE, target_size=200, chunk_overlap=20)
    assert [c.text for c in chunks] == [c.text for c in again]


def test_recursive_hard_split_fallback():
    # No separators inside — hard character split.
    text = "x" * 250
    chunks = chunk_recursive(text, target_size=100, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(c.length <= 110 for c in chunks)  # overlap may extend slightly


def test_structure_preserves_heading_with_body():
    chunks = chunk_structure(SAMPLE, source="sample")
    assert chunks[0].id == "structure-0000"
    # Preface + 3 sections
    assert len(chunks) >= 3
    alpha = next(c for c in chunks if c.section and "Alpha" in c.section)
    assert "Alpha body sentence one" in alpha.text
    assert alpha.text.startswith("## Alpha section") or "Alpha section" in alpha.text


def test_structure_no_empty_chunks():
    chunks = chunk_structure(SAMPLE)
    assert all(c.text.strip() for c in chunks)


def test_neighbors_linked():
    chunks = chunk_fixed(SAMPLE, chunk_size=60, chunk_overlap=10)
    if len(chunks) >= 2:
        assert chunks[0].next_id == chunks[1].id
        assert chunks[1].prev_id == chunks[0].id
        assert chunks[0].prev_id is None
        assert chunks[-1].next_id is None


def test_dispatch_matches_direct():
    via = chunk_with_strategy(
        SAMPLE,
        "fixed",
        fixed_chunk_size=70,
        fixed_chunk_overlap=10,
    )
    direct = chunk_fixed_direct(SAMPLE, chunk_size=70, chunk_overlap=10)
    assert [c.text for c in via] == [c.text for c in direct]


def test_invalid_overlap_raises():
    try:
        chunk_fixed("abc", chunk_size=10, chunk_overlap=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
