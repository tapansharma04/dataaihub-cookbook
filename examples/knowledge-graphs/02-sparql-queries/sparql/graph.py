"""RDF graph store. rdflib.Graph is the authoritative state."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from sparql.model import BindingValue
from sparql.vocab import EX, LOCAL_BY_CLASS, PREDICATE_BY_LOCAL, RDF


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
        """Authoritative RDF graph."""
        return self._graph

    def label_for(self, uri: URIRef | str) -> str | None:
        ref = uri if isinstance(uri, URIRef) else URIRef(str(uri))
        label = self._graph.value(ref, RDFS.label)
        return str(label) if label is not None else None

    def binding_value(self, term: object) -> BindingValue:
        from rdflib import Literal
        from rdflib.term import BNode

        if isinstance(term, URIRef):
            return BindingValue(iri=str(term), label=self.label_for(term))
        if isinstance(term, Literal):
            return BindingValue(
                iri=str(term),
                literal=str(term),
                datatype=str(term.datatype) if term.datatype else None,
            )
        if isinstance(term, BNode):
            return BindingValue(iri=str(term))
        return BindingValue(iri=str(term))

    def entities(self) -> list[dict[str, str]]:
        uris: set[URIRef] = set()
        for subject, _, type_uri in self._graph.triples((None, RDF.type, None)):
            if (
                isinstance(subject, URIRef)
                and type_uri in LOCAL_BY_CLASS
                and self.label_for(subject) is not None
            ):
                uris.add(subject)
        found: list[dict[str, str]] = []
        for uri in sorted(uris, key=str):
            label = self.label_for(uri)
            type_uri = self._graph.value(uri, RDF.type)
            entity_type = LOCAL_BY_CLASS.get(type_uri, "")
            found.append({"id": str(uri), "label": label or "", "type": entity_type})
        return found

    def relationships(self) -> list[dict[str, str]]:
        triples: list[dict[str, str]] = []
        for subject, pred, obj in self._graph:
            if pred in PREDICATE_BY_LOCAL.values() and isinstance(subject, URIRef):
                if isinstance(obj, URIRef):
                    triples.append(
                        {
                            "subject": str(subject),
                            "predicate": str(pred),
                            "object": str(obj),
                        }
                    )
                else:
                    triples.append(
                        {
                            "subject": str(subject),
                            "predicate": str(pred),
                            "object": str(obj),
                        }
                    )
        triples.sort(
            key=lambda triple: (
                triple["predicate"],
                triple["subject"],
                triple["object"],
            )
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
            "entities": entities,
            "relationships": relationships,
        }
