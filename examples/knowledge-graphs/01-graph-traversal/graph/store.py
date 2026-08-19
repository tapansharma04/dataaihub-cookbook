"""RDF graph store. rdflib.Graph is the authoritative state."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef

from config import ALLOWED_DIRECTIONS
from graph.model import Entity, GraphError, Triple
from graph.vocab import (
    ALLOWED_ENTITY_TYPES,
    CLASS_BY_LOCAL,
    EX,
    LOCAL_BY_CLASS,
    PREDICATE_BY_LOCAL,
    RDF,
    RDFS,
    compact,
    entity_uri,
    predicate_local,
    predicate_uri,
)


class GraphStore:
    """Application-owned RDF graph. No SPARQL. No external database."""

    def __init__(self, graph: Graph | None = None) -> None:
        self._graph = graph if graph is not None else Graph()
        self._graph.bind("ex", EX)
        self._graph.bind("rdfs", RDFS)

    @classmethod
    def from_path(cls, path: Path) -> GraphStore:
        graph = Graph()
        graph.parse(path, format="turtle")
        return cls(graph)

    @property
    def rdf(self) -> Graph:
        """Authoritative RDF graph. Do not copy this into a custom triple list."""
        return self._graph

    def add_entity(self, entity_id: str, label: str, entity_type: str) -> Entity:
        uri = entity_uri(entity_id)
        label = label.strip()
        if not label:
            raise GraphError("invalid_entity", "entity label must be non-empty")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            raise GraphError(
                "invalid_entity",
                f"unsupported entity type '{entity_type}'",
            )
        if self._is_entity(uri):
            raise GraphError("invalid_entity", f"entity already exists: {uri}")
        self._graph.add((uri, RDF.type, CLASS_BY_LOCAL[entity_type]))
        self._graph.add((uri, RDFS.label, Literal(label)))
        return self.lookup(str(uri))

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object_id: str,
    ) -> Triple:
        subject_uri = entity_uri(subject)
        object_uri = entity_uri(object_id)
        pred = predicate_uri(predicate)
        if not self._is_entity(subject_uri):
            raise GraphError("invalid_entity", f"unknown subject: {subject}")
        if not self._is_entity(object_uri):
            raise GraphError("invalid_entity", f"unknown object: {object_id}")
        triple = (subject_uri, pred, object_uri)
        if triple in self._graph:
            raise GraphError(
                "invalid_relationship",
                (
                    f"relationship already exists: {subject_uri} "
                    f"-{compact(pred)}-> {object_uri}"
                ),
            )
        self._graph.add(triple)
        return self._to_triple(subject_uri, pred, object_uri)

    def lookup(self, entity_id: str) -> Entity:
        uri = entity_uri(entity_id)
        if not self._is_entity(uri):
            raise GraphError("invalid_entity", f"unknown entity: {entity_id}")
        return self._to_entity(uri)

    def get_neighbors(
        self,
        entity_id: str,
        *,
        predicate: str | None = None,
        direction: str = "outgoing",
    ) -> list[Triple]:
        start = entity_uri(entity_id)
        self.lookup(str(start))
        if direction not in ALLOWED_DIRECTIONS:
            raise GraphError(
                "invalid_relationship",
                f"unsupported direction '{direction}'",
            )
        predicates = (
            [predicate_uri(predicate)]
            if predicate is not None
            else list(PREDICATE_BY_LOCAL.values())
        )
        matched: list[Triple] = []
        for pred in predicates:
            if direction == "outgoing":
                for obj in self._graph.objects(start, pred):
                    if isinstance(obj, URIRef):
                        matched.append(self._to_triple(start, pred, obj))
            else:
                for subj in self._graph.subjects(pred, start):
                    if isinstance(subj, URIRef):
                        matched.append(self._to_triple(subj, pred, start))
        matched.sort(
            key=lambda triple: (triple.predicate, triple.subject, triple.object)
        )
        return matched

    def entities(self) -> list[Entity]:
        found = [self._to_entity(uri) for uri in self._entity_uris()]
        found.sort(key=lambda entity: entity.id)
        return found

    def relationships(self) -> list[Triple]:
        triples = [
            self._to_triple(subject, pred, obj)
            for subject, pred, obj in self._graph
            if pred in PREDICATE_BY_LOCAL.values()
            and isinstance(subject, URIRef)
            and isinstance(obj, URIRef)
        ]
        triples.sort(
            key=lambda triple: (triple.predicate, triple.subject, triple.object)
        )
        return triples

    def snapshot(self) -> dict[str, object]:
        entities = self.entities()
        relationships = self.relationships()
        return {
            "format": "rdf",
            "namespace": str(EX),
            "entityCount": len(entities),
            "relationshipCount": len(relationships),
            "predicates": sorted(PREDICATE_BY_LOCAL),
            "entities": [entity.public() for entity in entities],
            "relationships": [triple.public() for triple in relationships],
        }

    def _is_entity(self, uri: URIRef) -> bool:
        return self._graph.value(uri, RDFS.label) is not None

    def _entity_uris(self) -> list[URIRef]:
        uris: set[URIRef] = set()
        for subject, _, type_uri in self._graph.triples((None, RDF.type, None)):
            if (
                isinstance(subject, URIRef)
                and type_uri in LOCAL_BY_CLASS
                and self._is_entity(subject)
            ):
                uris.add(subject)
        return sorted(uris, key=str)

    def _to_entity(self, uri: URIRef) -> Entity:
        label_node = self._graph.value(uri, RDFS.label)
        if label_node is None:
            raise GraphError("invalid_entity", f"missing rdfs:label for {uri}")
        type_uri = self._graph.value(uri, RDF.type)
        entity_type = LOCAL_BY_CLASS.get(
            type_uri, compact(type_uri) if type_uri else ""
        )
        return Entity(id=str(uri), label=str(label_node), type=entity_type)

    def _to_triple(self, subject: URIRef, predicate: URIRef, obj: URIRef) -> Triple:
        return Triple(
            subject=str(subject),
            predicate=str(predicate),
            object=str(obj),
        )


def labeled_predicate(predicate: str) -> dict[str, str]:
    uri = predicate_uri(predicate)
    return {"id": str(uri), "label": predicate_local(uri)}
