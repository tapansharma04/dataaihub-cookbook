"""Query transformation helpers for multi-query retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable

from openai import OpenAI

TransformFn = Callable[[str, int], list[str]]


def _normalize_queries(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _strip_code_fence(raw: str) -> str:
    """Remove optional markdown fences so JSON parsing stays robust."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_alternative_queries(raw: str, *, max_alternatives: int) -> list[str]:
    """Parse JSON array or newline bullets into a bounded query list."""
    raw = _strip_code_fence(raw)
    parsed: list[str] = []
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            parsed = [str(item) for item in obj]
    except json.JSONDecodeError:
        parsed = []
    if not parsed:
        # Last resort: extract a JSON array substring if the model added prose.
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                if isinstance(obj, list):
                    parsed = [str(item) for item in obj]
            except json.JSONDecodeError:
                parsed = []
    if not parsed:
        lines = raw.splitlines()
        for line in lines:
            item = line.lstrip("-*0123456789. ").strip()
            if item and item not in {"[", "]", "```", "```json"}:
                parsed.append(item)
    return _normalize_queries(parsed)[:max_alternatives]


def build_multi_queries(
    original_query: str,
    *,
    max_alternative_queries: int,
    transform_fn: TransformFn,
) -> list[str]:
    """Build this example's retrieval set: original first, then bounded alts.

    This Cookbook example retrieves with original + transformed queries for
    comparison, provenance, and teaching. That is not a claim that production
    Query Transformation pipelines must always retrieve with the original.
    The original remains available for provenance and is used for final
    reranking/generation.
    """
    alternatives = transform_fn(original_query, max_alternative_queries)
    bounded = _normalize_queries(alternatives)[:max_alternative_queries]
    merged = [original_query, *bounded]
    return _normalize_queries(merged)


class LLMQueryTransformer:
    """Generate retrieval-oriented alternatives for a user question."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def transform(self, original_query: str, max_alternatives: int) -> list[str]:
        if max_alternatives <= 0:
            return []
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a small set of diverse retrieval queries for "
                        "technical documentation search.\n"
                        "Preserve exact product names, error codes, version "
                        "numbers, identifiers, and important constraints.\n"
                        "Prefer handbook terminology (error codes, component "
                        "names, protocol symptoms) over conversational phrasing.\n"
                        "Do not answer the question.\n"
                        "Return retrieval queries only as a JSON array of strings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate up to {max_alternatives} alternative retrieval "
                        f"queries for this user question.\n"
                        f"Question: {original_query}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "[]"
        return parse_alternative_queries(content, max_alternatives=max_alternatives)
