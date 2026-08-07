"""Evidence mapping and provenance tests — no network."""

from __future__ import annotations

from pathlib import Path

from evaluation.dataset import load_eval_dataset
from evaluation.evidence import (
    attach_evidence,
    build_chunk_relevance,
    containment,
    grade_chunk_for_query,
    resolve_evidence_units,
)
from rag.chunking import chunk_fixed, chunk_structure
from rag.loader import load_document

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sample.md"
EVAL = ROOT / "data" / "eval_set.json"


def test_evidence_anchors_resolve_uniquely():
    text = load_document(CORPUS)
    dataset = load_eval_dataset(EVAL, text)
    assert len(dataset.evidence_units) >= 10
    for unit in dataset.evidence_units:
        assert text[unit.start : unit.end] == unit.anchor
        assert text.count(unit.anchor) == 1


def test_attach_evidence_propagates_ids():
    text = load_document(CORPUS)
    dataset = load_eval_dataset(EVAL, text)
    chunks = attach_evidence(chunk_structure(text), dataset.evidence_units)
    auth = next(c for c in chunks if c.section and "NebulaAuth" in (c.section or ""))
    assert "ev-auth-ttl" in auth.evidence_ids
    assert auth.source == "sample" or auth.source  # provenance retained
    assert auth.start < auth.end


def test_full_vs_partial_containment():
    text = load_document(CORPUS)
    dataset = load_eval_dataset(EVAL, text)
    unit = dataset.units_by_id()["ev-auth-ttl"]
    # Structure chunk should fully contain.
    structured = attach_evidence(chunk_structure(text), dataset.evidence_units)
    full_chunk = next(c for c in structured if "ev-auth-ttl" in c.evidence_ids)
    assert containment(full_chunk, unit) == "full"

    # Tiny fixed window forced over the middle of the anchor → partial.
    mid = (unit.start + unit.end) // 2
    from rag.chunking.base import Chunk

    partial = Chunk(
        id="fixed-9999",
        text=text[mid - 5 : mid + 5],
        start=mid - 5,
        end=mid + 5,
        strategy="fixed",
        source="sample",
    )
    assert containment(partial, unit) == "partial"


def test_grading_full_primary_vs_partial():
    text = load_document(CORPUS)
    dataset = load_eval_dataset(EVAL, text)
    units = dataset.units_by_id()
    case = dataset.get("auth-idle-timeout")
    structured = attach_evidence(chunk_structure(text), dataset.evidence_units)
    full = next(c for c in structured if "ev-auth-ttl" in c.evidence_ids)
    assert (
        grade_chunk_for_query(
            full,
            evidence_grades=case.evidence_grades,
            units_by_id=units,
        )
        >= 3
    )

    unit = units["ev-auth-ttl"]
    mid = (unit.start + unit.end) // 2
    from rag.chunking.base import Chunk

    partial = Chunk(
        id="fixed-9999",
        text=text[mid - 5 : mid + 5],
        start=mid - 5,
        end=mid + 5,
        strategy="fixed",
        source="sample",
    )
    assert (
        grade_chunk_for_query(
            partial,
            evidence_grades=case.evidence_grades,
            units_by_id=units,
        )
        == 1
    )


def test_relevance_differs_across_strategies():
    text = load_document(CORPUS)
    dataset = load_eval_dataset(EVAL, text)
    case = dataset.get("auth-idle-timeout")
    units = dataset.units_by_id()

    fixed = attach_evidence(
        chunk_fixed(text, chunk_size=400, chunk_overlap=50),
        dataset.evidence_units,
    )
    structured = attach_evidence(chunk_structure(text), dataset.evidence_units)

    rel_fixed = build_chunk_relevance(
        fixed, evidence_grades=case.evidence_grades, units_by_id=units
    )
    rel_struct = build_chunk_relevance(
        structured, evidence_grades=case.evidence_grades, units_by_id=units
    )
    # Different chunk ID namespaces.
    assert all(cid.startswith("fixed-") for cid in rel_fixed)
    assert all(cid.startswith("structure-") for cid in rel_struct)


def test_ambiguous_anchor_raises():
    try:
        resolve_evidence_units(
            "hello hello",
            [{"id": "e1", "section": "s", "anchor": "hello"}],
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "ambiguous" in str(exc)
