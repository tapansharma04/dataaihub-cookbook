"""Prompt and loader smoke tests — no paid APIs."""

from pathlib import Path

from rag.chunker import Chunk
from rag.generator import build_prompt
from rag.loader import load_document
from rag.store import ScoredChunk

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.md"


def test_load_sample_document():
    text = load_document(SAMPLE)
    assert "Retrieval-Augmented Generation" in text
    assert len(text) > 100


def test_build_prompt_includes_context_and_question():
    scored = [
        ScoredChunk(
            chunk=Chunk(
                id="sample-0",
                text="RAG retrieves then generates.",
                start=0,
                end=28,
            ),
            score=0.91,
        )
    ]
    prompt = build_prompt("What is RAG?", scored)
    assert "What is RAG?" in prompt
    assert "RAG retrieves then generates." in prompt
    assert "sample-0" in prompt


def test_build_prompt_handles_empty_retrieval():
    prompt = build_prompt("Anything?", [])
    assert "no documents retrieved" in prompt
