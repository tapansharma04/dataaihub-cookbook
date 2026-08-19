"""Deterministic RDF graph traversal over explicit hops.

The application owns identifiers, allowed predicates, direction, and depth.
Requests are structured hop lists — not SPARQL and not executable code.
"""

from __future__ import annotations

import time
from typing import Any

from config import ALLOWED_DIRECTIONS, Settings
from graph.model import (
    Entity,
    EventKind,
    GraphError,
    GraphRunMetrics,
    GraphRunResult,
    Hop,
    MatchedPath,
    SequenceEvent,
    TraversalRequest,
    Triple,
)
from graph.store import GraphStore, labeled_predicate
from graph.vocab import ALLOWED_PREDICATES, predicate_uri


class _Recorder:
    def __init__(self) -> None:
        self.events: list[SequenceEvent] = []

    def emit(
        self,
        kind: EventKind,
        detail: dict[str, Any],
        *,
        latency_ms: int | None = None,
    ) -> None:
        self.events.append(
            SequenceEvent(kind=kind, detail=detail, latency_ms=latency_ms)
        )


def run_case(case: Any, store: GraphStore, settings: Settings) -> GraphRunResult:
    """Execute a measured teaching case against the local RDF graph."""
    request = TraversalRequest(
        start_id=case.start_id,
        hops=list(case.hops),
        max_depth=settings.max_traversal_depth,
    )
    started = time.perf_counter()
    recorder = _Recorder()
    recorder.emit(
        "user_request",
        {
            "question": case.question,
            "startId": case.start_id,
            "hops": [hop.model_dump() for hop in case.hops],
        },
    )
    try:
        paths, answers, start, visited_entities, visited_relationships = traverse(
            store,
            request,
            settings=settings,
            recorder=recorder,
        )
        path_found = bool(paths)
        reason = "completed" if path_found else "no_path"
        errors: list[dict[str, Any]] = []
    except GraphError as exc:
        paths = []
        answers = []
        start = None
        visited_entities = 0
        visited_relationships = 0
        path_found = False
        reason = exc.code
        errors = [{"code": exc.code, "message": exc.message}]
        recorder.emit("result", {"pathFound": False, "answers": [], "error": errors[0]})
        recorder.emit("termination", {"reason": reason})

    execution_ms = max(0, round((time.perf_counter() - started) * 1000))
    if reason in {"completed", "no_path"}:
        recorder.emit(
            "result",
            {
                "pathFound": path_found,
                "answerIds": [entity.id for entity in answers],
                "answers": [entity.public() for entity in answers],
                "pathCount": len(paths),
            },
        )
        recorder.emit("termination", {"reason": reason})

    metrics = GraphRunMetrics(
        entities_visited=visited_entities,
        relationships_visited=visited_relationships,
        traversal_depth=len(request.hops),
        matched_relationships=visited_relationships,
        path_found=path_found,
        execution_ms=execution_ms,
        termination_reason=reason,
        max_depth=settings.max_traversal_depth,
    )
    output = {
        "question": case.question,
        "pathFound": path_found,
        "answers": [entity.public() for entity in answers],
        "paths": [_path_public(path) for path in paths],
        "terminationReason": reason,
    }
    return GraphRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        question=case.question,
        start=start,
        hops=list(case.hops),
        paths=paths,
        answers=answers,
        sequence=recorder.events,
        metrics=metrics,
        output=output,
        errors=errors,
    )


def traverse(
    store: GraphStore,
    request: TraversalRequest,
    *,
    settings: Settings,
    recorder: _Recorder | None = None,
) -> tuple[list[MatchedPath], list[Entity], Entity, int, int]:
    """Follow an explicit hop list over the RDF graph.

    Outgoing hops use Graph.objects(subject, predicate).
    Incoming hops use Graph.subjects(predicate, object).
    Missing triples are reported as no path. The walker does not invent hops.
    """
    recorder = recorder or _Recorder()
    hops = _validate_request(request, settings)

    start = store.lookup(request.start_id)
    recorder.emit(
        "graph_lookup",
        {"entity": start.public(), "found": True},
    )
    recorder.emit(
        "traversal_started",
        {
            "start": start.public(),
            "hopCount": len(hops),
            "maxDepth": settings.max_traversal_depth,
        },
    )

    frontier: list[MatchedPath] = [
        MatchedPath(entities=[start], relationships=[], depth=0)
    ]
    entities_visited = {start.id}
    relationships_visited: set[tuple[str, str, str]] = set()

    for depth, hop in enumerate(hops, start=1):
        pred = labeled_predicate(hop.predicate)
        recorder.emit(
            "traversal_step",
            {
                "depth": depth,
                "predicate": pred,
                "direction": hop.direction,
                "frontierIds": [path.entities[-1].id for path in frontier],
            },
        )
        next_frontier: list[MatchedPath] = []
        for path in frontier:
            current = path.entities[-1]
            neighbors = store.get_neighbors(
                current.id,
                predicate=hop.predicate,
                direction=hop.direction,
            )
            for triple in neighbors:
                neighbor = _neighbor_entity(store, triple, hop)
                recorder.emit(
                    "relationship_match",
                    {
                        "depth": depth,
                        "direction": hop.direction,
                        "triple": _labeled_triple(store, triple),
                        "from": current.public(),
                        "to": neighbor.public(),
                    },
                )
                entities_visited.add(neighbor.id)
                relationships_visited.add(
                    (triple.subject, triple.predicate, triple.object)
                )
                next_frontier.append(
                    MatchedPath(
                        entities=[*path.entities, neighbor],
                        relationships=[*path.relationships, triple],
                        depth=depth,
                    )
                )
        frontier = next_frontier
        if not frontier:
            break

    path_found = bool(frontier) and all(path.depth == len(hops) for path in frontier)
    completed_paths = frontier if path_found else []
    answers = _unique_terminals(completed_paths)
    recorder.emit(
        "traversal_completed",
        {
            "pathFound": path_found,
            "pathCount": len(completed_paths),
            "depthReached": max((path.depth for path in completed_paths), default=0),
        },
    )
    return (
        completed_paths,
        answers,
        start,
        # Distinct RDF entity IRIs touched, including the start entity.
        len(entities_visited),
        # Distinct RDF triples matched during traversal.
        len(relationships_visited),
    )


def _validate_request(request: TraversalRequest, settings: Settings) -> list[Hop]:
    start_id = request.start_id.strip() if isinstance(request.start_id, str) else ""
    if not start_id:
        raise GraphError("invalid_entity", "start_id must be a non-empty identifier")
    hops = list(request.hops)
    if not hops:
        raise GraphError("invalid_relationship", "traversal requires at least one hop")
    ceiling = settings.max_traversal_depth
    requested = request.max_depth if request.max_depth is not None else ceiling
    if requested < 1 or requested > ceiling:
        raise GraphError(
            "depth_limit",
            f"max_depth must be between 1 and {ceiling}",
        )
    if len(hops) > requested:
        raise GraphError(
            "depth_limit",
            f"path length {len(hops)} exceeds max_depth {requested}",
        )
    for hop in hops:
        if hop.predicate not in ALLOWED_PREDICATES:
            raise GraphError(
                "invalid_relationship",
                f"unsupported predicate '{hop.predicate}'",
            )
        if hop.direction not in ALLOWED_DIRECTIONS:
            raise GraphError(
                "invalid_relationship",
                f"unsupported direction '{hop.direction}'",
            )
        predicate_uri(hop.predicate)
    return hops


def _neighbor_entity(store: GraphStore, triple: Triple, hop: Hop) -> Entity:
    neighbor_id = triple.object if hop.direction == "outgoing" else triple.subject
    return store.lookup(neighbor_id)


def _labeled_triple(store: GraphStore, triple: Triple) -> dict[str, Any]:
    return {
        "subject": store.lookup(triple.subject).public(),
        "predicate": labeled_predicate(triple.predicate),
        "object": store.lookup(triple.object).public(),
    }


def _unique_terminals(paths: list[MatchedPath]) -> list[Entity]:
    seen: set[str] = set()
    answers: list[Entity] = []
    for path in paths:
        terminal = path.entities[-1]
        if terminal.id not in seen:
            seen.add(terminal.id)
            answers.append(terminal)
    answers.sort(key=lambda entity: entity.id)
    return answers


def _path_public(path: MatchedPath) -> dict[str, Any]:
    return {
        "depth": path.depth,
        "entities": [entity.public() for entity in path.entities],
        "relationships": [
            _relationship_public(triple) for triple in path.relationships
        ],
    }


def _relationship_public(triple: Triple) -> dict[str, Any]:
    return {
        "subject": triple.subject,
        "predicate": labeled_predicate(triple.predicate),
        "object": triple.object,
    }
