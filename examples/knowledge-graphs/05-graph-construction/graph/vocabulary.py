"""RDF vocabulary for graph construction.

Application-owned: extractors may propose semantic predicates; only these IRIs
are ever committed.
"""

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

# Semantic aliases proposed by extractors → application-owned local names.
PREDICATE_ALIASES: dict[str, str] = {
    "employs": "employs",
    "works_on": "worksOn",
    "workson": "worksOn",
    "works on": "worksOn",
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
ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(CLASS_BY_LOCAL)


def compact(uri: URIRef | str) -> str:
    text = str(uri)
    prefix = str(EX)
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def resolve_predicate_local(proposed: str) -> str | None:
    """Map a proposed predicate token to an allowed local name, or None."""
    raw = proposed.strip()
    if not raw:
        return None
    # Reject absolute IRIs supplied by extractors — vocabulary is application-owned.
    if raw.startswith("http://") or raw.startswith("https://"):
        return None
    if raw in PREDICATE_BY_LOCAL:
        return raw
    key = raw.lower().replace("-", "_")
    return PREDICATE_ALIASES.get(key)


__all__ = [
    "ALLOWED_ENTITY_TYPES",
    "ALLOWED_PREDICATES",
    "CLASS_BY_LOCAL",
    "EX",
    "LOCAL_BY_CLASS",
    "LOCAL_BY_PREDICATE",
    "PREDICATE_ALIASES",
    "PREDICATE_BY_LOCAL",
    "RDF",
    "RDFS",
    "compact",
    "resolve_predicate_local",
]
