"""RDF vocabulary for the graph-traversal example.

The RDF graph is identified by IRIs in this namespace. Compact names such as
`worksOn` are application hop labels that map to these URIRefs.
"""

from __future__ import annotations

from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from rdflib.term import URIRef

from graph.model import GraphError

EX = Namespace("https://dataaihub.co/example/kg/")

PREDICATE_BY_LOCAL: dict[str, URIRef] = {
    "employs": EX.employs,
    "worksOn": EX.worksOn,
    "uses": EX.uses,
}
LOCAL_BY_PREDICATE: dict[URIRef, str] = {
    uri: name for name, uri in PREDICATE_BY_LOCAL.items()
}

CLASS_BY_LOCAL: dict[str, URIRef] = {
    "Company": EX.Company,
    "Person": EX.Person,
    "Project": EX.Project,
    "Technology": EX.Technology,
}
LOCAL_BY_CLASS: dict[URIRef, str] = {uri: name for name, uri in CLASS_BY_LOCAL.items()}

ALLOWED_PREDICATES: frozenset[str] = frozenset(PREDICATE_BY_LOCAL)
ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(CLASS_BY_LOCAL)


def compact(uri: URIRef | str) -> str:
    text = str(uri)
    prefix = str(EX)
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def entity_uri(value: str) -> URIRef:
    text = value.strip()
    if not text:
        raise GraphError("invalid_entity", "entity id must be a non-empty identifier")
    if text.startswith(("http://", "https://")):
        return URIRef(text)
    if text.startswith("ex:"):
        return EX[text.removeprefix("ex:")]
    if "/" in text or ":" in text:
        raise GraphError("invalid_entity", f"unknown entity: {value}")
    return EX[text]


def predicate_uri(value: str) -> URIRef:
    if value in PREDICATE_BY_LOCAL:
        return PREDICATE_BY_LOCAL[value]
    text = value.strip()
    uri = URIRef(text) if text.startswith(("http://", "https://")) else None
    if uri in LOCAL_BY_PREDICATE:
        return uri
    raise GraphError("invalid_relationship", f"unsupported predicate '{value}'")


def predicate_local(uri: URIRef | str) -> str:
    ref = uri if isinstance(uri, URIRef) else URIRef(str(uri))
    local = LOCAL_BY_PREDICATE.get(ref)
    if local is None:
        raise GraphError("invalid_relationship", f"unsupported predicate '{uri}'")
    return local


__all__ = [
    "ALLOWED_ENTITY_TYPES",
    "ALLOWED_PREDICATES",
    "CLASS_BY_LOCAL",
    "EX",
    "LOCAL_BY_CLASS",
    "LOCAL_BY_PREDICATE",
    "PREDICATE_BY_LOCAL",
    "RDF",
    "RDFS",
    "compact",
    "entity_uri",
    "predicate_local",
    "predicate_uri",
]
