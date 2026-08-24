"""RDF vocabulary for the GraphRAG example."""

from __future__ import annotations

from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from rdflib.term import URIRef

EX = Namespace("https://dataaihub.co/example/kg/")

PREDICATE_BY_LOCAL: dict[str, URIRef] = {
    "employs": EX.employs,
    "worksOn": EX.worksOn,
    "uses": EX.uses,
}
LOCAL_BY_PREDICATE: dict[URIRef, str] = {
    uri: name for name, uri in PREDICATE_BY_LOCAL.items()
}

# Human-readable predicate labels for context assembly.
PREDICATE_LABELS: dict[str, str] = {
    "employs": "employs",
    "worksOn": "works on",
    "uses": "uses",
}

CLASS_BY_LOCAL: dict[str, URIRef] = {
    "Company": EX.Company,
    "Person": EX.Person,
    "Project": EX.Project,
    "Technology": EX.Technology,
}
LOCAL_BY_CLASS: dict[URIRef, str] = {uri: name for name, uri in CLASS_BY_LOCAL.items()}

ALLOWED_PREDICATES: frozenset[str] = frozenset(PREDICATE_BY_LOCAL)


def compact(uri: URIRef | str) -> str:
    text = str(uri)
    prefix = str(EX)
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def predicate_label(local: str) -> str:
    return PREDICATE_LABELS.get(local, local)


__all__ = [
    "ALLOWED_PREDICATES",
    "CLASS_BY_LOCAL",
    "EX",
    "LOCAL_BY_CLASS",
    "LOCAL_BY_PREDICATE",
    "PREDICATE_BY_LOCAL",
    "PREDICATE_LABELS",
    "RDF",
    "RDFS",
    "compact",
    "predicate_label",
]
