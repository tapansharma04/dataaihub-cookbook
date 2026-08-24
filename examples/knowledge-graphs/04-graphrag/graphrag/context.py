"""Assemble grounded natural-language context from retrieved graph facts."""

from __future__ import annotations

import time

from graphrag.model import GraphFact


def fact_to_statement(fact: GraphFact) -> str:
    return f"{fact.subject.label} {fact.predicate.label} {fact.object.label}."


def assemble_context(facts: list[GraphFact]) -> tuple[list[str], int]:
    """Convert retrieved RDF facts into inspectable context statements."""
    started = time.perf_counter()
    statements = [
        fact_to_statement(fact) for fact in sorted(facts, key=lambda f: f.sort_key())
    ]
    assembly_ms = int((time.perf_counter() - started) * 1000)
    return statements, assembly_ms
