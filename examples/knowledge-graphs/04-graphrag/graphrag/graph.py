"""RDF graph store. rdflib.Graph is the authoritative state."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from graphrag.model import GraphEntityRef, GraphFact, GraphPredicateRef
from graphrag.vocab import (
    EX,
    LOCAL_BY_CLASS,
    LOCAL_BY_PREDICATE,
    PREDICATE_BY_LOCAL,
    RDF,
)


class RdfGraphStore:
    """Application-owned RDF graph loaded from a local Turtle fixture."""

    def __init__(self, graph: Graph | None = None) -> None:
        self._graph = graph if graph is not None else Graph()
        self._graph.bind("ex", EX)
        self._graph.bind("rdfs", RDFS)

    @classmethod
    def from_path(cls, path: Path) -> RdfGraphStore:
        graph = Graph()
        graph.parse(path, format="turtle")
        return cls(graph)

    @property
    def rdf(self) -> Graph:
        return self._graph

    def label_for(self, uri: URIRef | str) -> str | None:
        ref = uri if isinstance(uri, URIRef) else URIRef(str(uri))
        label = self._graph.value(ref, RDFS.label)
        return str(label) if label is not None else None

    def entity_ref(self, uri: URIRef | str) -> GraphEntityRef:
        ref = uri if isinstance(uri, URIRef) else URIRef(str(uri))
        label = self.label_for(ref) or str(ref)
        return GraphEntityRef(iri=str(ref), label=label)

    def predicate_ref(self, pred: URIRef | str) -> GraphPredicateRef:
        ref = pred if isinstance(pred, URIRef) else URIRef(str(pred))
        local = LOCAL_BY_PREDICATE.get(ref, "")
        from graphrag.vocab import predicate_label

        label = predicate_label(local) if local else str(ref)
        return GraphPredicateRef(iri=str(ref), label=label)

    def labeled_entities(self) -> list[tuple[str, str]]:
        """Return (label, iri) pairs sorted deterministically by label then iri."""
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for subject in self._graph.subjects(RDF.type, None):
            if not isinstance(subject, URIRef):
                continue
            type_uri = self._graph.value(subject, RDF.type)
            if type_uri not in LOCAL_BY_CLASS:
                continue
            label = self.label_for(subject)
            if label is None:
                continue
            iri = str(subject)
            if iri in seen:
                continue
            seen.add(iri)
            found.append((label, iri))
        found.sort(key=lambda item: (item[0].lower(), item[1]))
        return found

    def make_fact(self, subject: URIRef, predicate: URIRef, obj: URIRef) -> GraphFact:
        return GraphFact(
            subject=self.entity_ref(subject),
            predicate=self.predicate_ref(predicate),
            object=self.entity_ref(obj),
        )

    def snapshot(self) -> dict[str, object]:
        entities = []
        uris: set[URIRef] = set()
        for subject in self._graph.subjects(RDF.type, None):
            if isinstance(subject, URIRef) and self.label_for(subject):
                uris.add(subject)
        for uri in sorted(uris, key=str):
            type_uri = self._graph.value(uri, RDF.type)
            entities.append(
                {
                    "id": str(uri),
                    "label": self.label_for(uri) or "",
                    "type": LOCAL_BY_CLASS.get(type_uri, ""),
                }
            )
        relationships: list[dict[str, str]] = []
        for subject, pred, obj in self._graph:
            if pred in PREDICATE_BY_LOCAL.values() and isinstance(subject, URIRef):
                if isinstance(obj, URIRef):
                    relationships.append(
                        {
                            "subject": str(subject),
                            "predicate": str(pred),
                            "object": str(obj),
                        }
                    )
        relationships.sort(
            key=lambda triple: (
                triple["predicate"],
                triple["subject"],
                triple["object"],
            )
        )
        return {
            "format": "rdf",
            "namespace": str(EX),
            "entityCount": len(entities),
            "relationshipCount": len(relationships),
            "predicates": sorted(PREDICATE_BY_LOCAL),
            "entities": entities,
            "relationships": relationships,
        }
