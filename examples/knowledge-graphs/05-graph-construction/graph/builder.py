"""RDF graph store and triple construction.

rdflib.Graph is the authoritative runtime state. The builder COMMITS validated
proposals only — extractors never write here.
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS

from graph.model import ResolvedEntity, TripleRef, ValidatedRelationship
from graph.vocabulary import (
    CLASS_BY_LOCAL,
    EX,
    LOCAL_BY_CLASS,
    PREDICATE_BY_LOCAL,
    RDF,
)


class RdfGraphStore:
    """Application-owned RDF graph."""

    def __init__(self, graph: Graph | None = None) -> None:
        self._graph = graph if graph is not None else Graph()
        self._graph.bind("ex", EX)
        self._graph.bind("rdfs", RDFS)

    @classmethod
    def empty(cls) -> RdfGraphStore:
        return cls(Graph())

    @classmethod
    def from_path(cls, path: Path) -> RdfGraphStore:
        graph = Graph()
        graph.parse(path, format="turtle")
        return cls(graph)

    @classmethod
    def fresh(cls, *, start: str, seed_path: Path) -> RdfGraphStore:
        """Case isolation: empty or seeded copy from Turtle."""
        if start == "seed":
            return cls.from_path(seed_path)
        return cls.empty()

    @property
    def rdf(self) -> Graph:
        return self._graph

    def triple_count(self) -> int:
        return len(self._graph)

    def label_for(self, uri: URIRef | str) -> str | None:
        ref = uri if isinstance(uri, URIRef) else URIRef(str(uri))
        label = self._graph.value(ref, RDFS.label)
        return str(label) if label is not None else None

    def has_triple(self, subject: str, predicate: str, obj: str) -> bool:
        return (
            URIRef(subject),
            URIRef(predicate),
            URIRef(obj),
        ) in self._graph

    def has_predicate_iri(self, predicate_iri: str) -> bool:
        pred = URIRef(predicate_iri)
        return any(True for _ in self._graph.triples((None, pred, None)))

    def commit_entities(self, entities: list[ResolvedEntity]) -> list[TripleRef]:
        created: list[TripleRef] = []
        for entity in entities:
            iri = URIRef(entity.iri)
            type_iri = CLASS_BY_LOCAL[entity.entity_type]
            if (iri, RDF.type, type_iri) not in self._graph:
                self._graph.add((iri, RDF.type, type_iri))
                created.append(
                    TripleRef(
                        subject=entity.iri,
                        predicate=str(RDF.type),
                        object=str(type_iri),
                        kind="type",
                    )
                )
            label_literal = Literal(entity.label)
            if (iri, RDFS.label, label_literal) not in self._graph:
                self._graph.add((iri, RDFS.label, label_literal))
                created.append(
                    TripleRef(
                        subject=entity.iri,
                        predicate=str(RDFS.label),
                        object=entity.label,
                        kind="label",
                    )
                )
        return created

    def commit_relationships(
        self, relationships: list[ValidatedRelationship]
    ) -> list[TripleRef]:
        created: list[TripleRef] = []
        for rel in relationships:
            triple = (
                URIRef(rel.subject_iri),
                URIRef(rel.predicate_iri),
                URIRef(rel.object_iri),
            )
            if triple in self._graph:
                continue
            self._graph.add(triple)
            created.append(
                TripleRef(
                    subject=rel.subject_iri,
                    predicate=rel.predicate_iri,
                    object=rel.object_iri,
                    kind="relationship",
                )
            )
        return created

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
            entity_type = LOCAL_BY_CLASS.get(type_uri, "")  # type: ignore[arg-type]
            found.append({"id": str(uri), "label": label or "", "type": entity_type})
        return found

    def relationships(self) -> list[dict[str, str]]:
        triples: list[dict[str, str]] = []
        for subject, pred, obj in self._graph:
            if pred in PREDICATE_BY_LOCAL.values() and isinstance(subject, URIRef):
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
            "tripleCount": self.triple_count(),
            "predicates": sorted(PREDICATE_BY_LOCAL),
            "entities": entities,
            "relationships": relationships,
        }

    def verify_committed(
        self,
        *,
        entities: list[ResolvedEntity],
        relationships: list[ValidatedRelationship],
        triples: list[TripleRef],
    ) -> dict[str, object]:
        """Observable verification of committed graph state."""
        missing: list[str] = []
        for entity in entities:
            iri = URIRef(entity.iri)
            type_iri = CLASS_BY_LOCAL[entity.entity_type]
            if (iri, RDF.type, type_iri) not in self._graph:
                missing.append(f"type:{entity.iri}")
            if self.label_for(iri) != entity.label:
                missing.append(f"label:{entity.iri}")
        for rel in relationships:
            if not self.has_triple(rel.subject_iri, rel.predicate_iri, rel.object_iri):
                missing.append(
                    f"rel:{rel.subject_iri}|{rel.predicate_local}|{rel.object_iri}"
                )
        # Ensure no non-vocabulary predicates slipped in among relationship commits.
        allowed = set(PREDICATE_BY_LOCAL.values()) | {RDF.type, RDFS.label}
        illicit = [
            str(pred)
            for _, pred, _ in self._graph
            if pred not in allowed and isinstance(pred, URIRef)
        ]
        return {
            "ok": not missing and not illicit,
            "missing": missing,
            "illicitPredicates": sorted(set(illicit)),
            "tripleCount": self.triple_count(),
            "committedTripleCount": len(triples),
        }
