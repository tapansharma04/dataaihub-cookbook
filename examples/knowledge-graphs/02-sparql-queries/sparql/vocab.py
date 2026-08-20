"""RDF vocabulary for the SPARQL queries example."""

from __future__ import annotations

from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from rdflib.term import URIRef

EX = Namespace("https://dataaihub.co/example/kg/")

PREDICATE_BY_LOCAL: dict[str, URIRef] = {
    "employs": EX.employs,
    "worksOn": EX.worksOn,
    "uses": EX.uses,
    "team": EX.team,
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


def compact(uri: URIRef | str) -> str:
    text = str(uri)
    prefix = str(EX)
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


__all__ = [
    "CLASS_BY_LOCAL",
    "EX",
    "LOCAL_BY_CLASS",
    "LOCAL_BY_PREDICATE",
    "PREDICATE_BY_LOCAL",
    "RDF",
    "RDFS",
    "compact",
]
