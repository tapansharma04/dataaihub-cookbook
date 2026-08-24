"""Bounded graph-native retrieval over RDF structure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rdflib import URIRef

from graphrag.cases import MeasuredCase, TraversalStep
from graphrag.graph import RdfGraphStore
from graphrag.model import GraphFact, PathStep, ResolvedEntity, RetrievalPath
from graphrag.vocab import LOCAL_BY_PREDICATE, PREDICATE_BY_LOCAL


@dataclass
class RetrievalResult:
    facts: list[GraphFact] = field(default_factory=list)
    paths: list[RetrievalPath] = field(default_factory=list)
    hops_used: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    execution_ms: int = 0


def _predicate_uri(step: TraversalStep) -> URIRef:
    return PREDICATE_BY_LOCAL[step.predicate]


def _step_neighbors(
    store: RdfGraphStore,
    current: URIRef,
    step: TraversalStep,
) -> list[tuple[URIRef, GraphFact, PathStep]]:
    pred = _predicate_uri(step)
    results: list[tuple[URIRef, GraphFact, PathStep]] = []
    if step.direction == "forward":
        for obj in store.rdf.objects(current, pred):
            if not isinstance(obj, URIRef):
                continue
            fact = store.make_fact(current, pred, obj)
            path_step = PathStep(
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
            )
            results.append((obj, fact, path_step))
    else:
        for subject in store.rdf.subjects(pred, current):
            if not isinstance(subject, URIRef):
                continue
            fact = store.make_fact(subject, pred, current)
            path_step = PathStep(
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
            )
            results.append((subject, fact, path_step))
    results.sort(key=lambda item: (str(item[1].subject.iri), str(item[1].object.iri)))
    return results


def _dedupe_facts(facts: list[GraphFact]) -> list[GraphFact]:
    seen: set[tuple[str, str, str]] = set()
    out: list[GraphFact] = []
    for fact in sorted(facts, key=lambda f: f.sort_key()):
        key = fact.sort_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out


def _chain_from_seed(
    store: RdfGraphStore,
    seed: URIRef,
    steps: tuple[TraversalStep, ...],
    *,
    max_hops: int,
) -> tuple[list[GraphFact], list[RetrievalPath], list[dict[str, Any]], int]:
    facts: list[GraphFact] = []
    paths: list[RetrievalPath] = []
    trace_steps: list[dict[str, Any]] = []
    max_depth = 0

    frontier: list[tuple[URIRef, list[PathStep], int]] = [(seed, [], 0)]
    step_index = 0

    while frontier and step_index < len(steps):
        step = steps[step_index]
        next_frontier: list[tuple[URIRef, list[PathStep], int]] = []
        for current, path_steps, depth in frontier:
            if depth >= max_hops:
                continue
            neighbors = _step_neighbors(store, current, step)
            trace_steps.append(
                {
                    "hop": depth + 1,
                    "predicate": step.predicate,
                    "direction": step.direction,
                    "from": str(current),
                    "neighborCount": len(neighbors),
                }
            )
            for neighbor, fact, path_step in neighbors:
                facts.append(fact)
                new_steps = [*path_steps, path_step]
                if step_index == len(steps) - 1:
                    paths.append(RetrievalPath(steps=new_steps))
                next_frontier.append((neighbor, new_steps, depth + 1))
                max_depth = max(max_depth, depth + 1)
        frontier = sorted(next_frontier, key=lambda item: str(item[0]))
        step_index += 1

    return _dedupe_facts(facts), paths, trace_steps, max_depth


def _parallel_from_seed(
    store: RdfGraphStore,
    seed: URIRef,
    steps: tuple[TraversalStep, ...],
    *,
    max_hops: int,
    require_direct_only: bool,
) -> tuple[list[GraphFact], list[RetrievalPath], list[dict[str, Any]], int]:
    facts: list[GraphFact] = []
    paths: list[RetrievalPath] = []
    trace_steps: list[dict[str, Any]] = []
    max_depth = 0

    for step in steps:
        if require_direct_only and step.direction != "forward":
            neighbors = _step_neighbors(store, seed, step)
        elif require_direct_only:
            neighbors = _step_neighbors(store, seed, step)
        else:
            neighbors = _step_neighbors(store, seed, step)

        if max_hops < 1:
            continue

        trace_steps.append(
            {
                "hop": 1,
                "predicate": step.predicate,
                "direction": step.direction,
                "from": str(seed),
                "neighborCount": len(neighbors),
            }
        )
        for _neighbor, fact, path_step in neighbors:
            facts.append(fact)
            paths.append(RetrievalPath(steps=[path_step]))
            max_depth = max(max_depth, 1)

    return _dedupe_facts(facts), paths, trace_steps, max_depth


def retrieve_subgraph(
    store: RdfGraphStore,
    resolved_entities: list[ResolvedEntity],
    case: MeasuredCase,
) -> RetrievalResult:
    """Bounded, deterministic graph retrieval from resolved seed entities."""
    started = time.perf_counter()
    if not resolved_entities:
        return RetrievalResult(execution_ms=0)

    all_facts: list[GraphFact] = []
    all_paths: list[RetrievalPath] = []
    all_steps: list[dict[str, Any]] = []
    max_hops = 0

    seeds = sorted({entity.iri for entity in resolved_entities})
    for seed_iri in seeds:
        seed = URIRef(seed_iri)
        if case.example_class == "MULTI_HOP_RETRIEVAL":
            facts, paths, steps, hops = _chain_from_seed(
                store,
                seed,
                case.traversal_steps,
                max_hops=case.max_hops,
            )
        else:
            facts, paths, steps, hops = _parallel_from_seed(
                store,
                seed,
                case.traversal_steps,
                max_hops=case.max_hops,
                require_direct_only=case.require_direct_only,
            )
        all_facts.extend(facts)
        all_paths.extend(paths)
        all_steps.extend(steps)
        max_hops = max(max_hops, hops)

    all_facts = _dedupe_facts(all_facts)
    all_paths.sort(
        key=lambda path: tuple(
            (step.subject.iri, step.predicate.iri, step.object.iri)
            for step in path.steps
        )
    )

    execution_ms = int((time.perf_counter() - started) * 1000)
    return RetrievalResult(
        facts=all_facts,
        paths=all_paths,
        hops_used=max_hops,
        steps=all_steps,
        execution_ms=execution_ms,
    )


def validate_retrieval_config(case: MeasuredCase, *, settings_max_hops: int) -> None:
    """Ensure case traversal stays within application-owned limits."""
    if case.max_hops > settings_max_hops:
        msg = (
            f"case max_hops {case.max_hops} exceeds settings limit {settings_max_hops}"
        )
        raise ValueError(msg)
    for step in case.traversal_steps:
        if step.predicate not in PREDICATE_BY_LOCAL:
            raise ValueError(f"unsupported predicate: {step.predicate}")
        if LOCAL_BY_PREDICATE.get(PREDICATE_BY_LOCAL[step.predicate]) != step.predicate:
            raise ValueError(f"invalid predicate mapping: {step.predicate}")
