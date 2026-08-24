"""Deterministic answers from retrieved graph facts (GRAPH_GROUNDED mode)."""

from __future__ import annotations

from graphrag.cases import MeasuredCase
from graphrag.model import GraphFact, RetrievalPath

INSUFFICIENT_EVIDENCE = "I don't have enough graph evidence to answer that."


def _join_labels(labels: list[str]) -> str:
    unique = sorted(set(labels), key=str.lower)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def generate_deterministic_answer(
    case: MeasuredCase,
    *,
    facts: list[GraphFact],
    paths: list[RetrievalPath],
    context: list[str],
) -> str:
    del paths  # answer derives from facts/context only
    if not facts:
        return INSUFFICIENT_EVIDENCE

    if case.example_class == "ENTITY_RETRIEVAL":
        people = sorted({fact.subject.label for fact in facts}, key=str.lower)
        project = next((fact.object.label for fact in facts), "the project")
        joined = _join_labels(people)
        return f"{joined} work on {project}."

    if case.example_class == "MULTI_HOP_RETRIEVAL":
        technologies = sorted(
            {fact.object.label for fact in facts if fact.predicate.label == "uses"},
            key=str.lower,
        )
        person = next(
            (
                fact.subject.label
                for fact in facts
                if fact.predicate.label == "works on"
            ),
            "the person",
        )
        joined = _join_labels(technologies)
        return f"{joined} are used by projects {person} works on."

    if case.example_class == "RELATIONSHIP_GROUNDED_ANSWER":
        employer = next(
            (fact.subject.label for fact in facts if fact.predicate.label == "employs"),
            None,
        )
        project = next(
            (fact.object.label for fact in facts if fact.predicate.label == "works on"),
            None,
        )
        person = next(
            (fact.object.label for fact in facts if fact.predicate.label == "employs"),
            next(
                (
                    fact.subject.label
                    for fact in facts
                    if fact.predicate.label == "works on"
                ),
                "the person",
            ),
        )
        parts: list[str] = []
        if employer:
            parts.append(f"{employer} employs {person}.")
        if project:
            parts.append(f"{person} works on {project}.")
        if parts:
            return " ".join(parts)
        return INSUFFICIENT_EVIDENCE

    if case.example_class == "NO_RELEVANT_SUBGRAPH":
        return INSUFFICIENT_EVIDENCE

    # Fallback: join grounded context facts.
    if context:
        return " ".join(context)
    return INSUFFICIENT_EVIDENCE
