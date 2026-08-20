"""Predefined SPARQL UPDATE and verification queries for measured teaching cases.

Each update string is executed by rdflib's SPARQL update engine via Graph.update().
The application selects which update runs; arbitrary user SPARQL is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdflib import URIRef

from config import PROHIBITED_SPARQL_KEYWORDS
from sparql.model import SparqlError, UpdateType
from sparql.vocab import EX

COMMON_PREFIXES = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

INSERT_DATA_UPDATE = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

INSERT DATA {
  ex:billingPortal ex:uses ex:redis .
}
"""

INSERT_DATA_VERIFY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?technology ?technologyLabel
WHERE {
  ex:billingPortal ex:uses ?technology .
  ?technology rdfs:label ?technologyLabel .
}
ORDER BY ?technologyLabel
"""

INSERT_WHERE_UPDATE = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

INSERT {
  ?person ex:uses ?technology .
}
WHERE {
  ?person ex:worksOn ?project .
  ?project ex:uses ?technology .
}
"""

INSERT_WHERE_VERIFY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?person ?personLabel ?technology ?technologyLabel
WHERE {
  ?person a ex:Person .
  ?person ex:uses ?technology .
  ?person rdfs:label ?personLabel .
  ?technology rdfs:label ?technologyLabel .
}
ORDER BY ?personLabel ?technologyLabel
"""

DELETE_DATA_UPDATE = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

DELETE DATA {
  ex:billingPortal ex:uses ex:postgresql .
}
"""

DELETE_DATA_VERIFY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?technology ?technologyLabel
WHERE {
  ex:billingPortal ex:uses ex:postgresql .
  ex:postgresql rdfs:label ?technologyLabel .
  BIND(ex:postgresql AS ?technology)
}
ORDER BY ?technologyLabel
"""

UPDATE_AND_VERIFY_UPDATE = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

DELETE {
  ex:billingPortal ex:uses ex:postgresql .
}
INSERT {
  ex:billingPortal ex:uses ex:redis .
}
WHERE {
  ex:billingPortal ex:uses ex:postgresql .
}
"""

UPDATE_AND_VERIFY_VERIFY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?technology ?technologyLabel
WHERE {
  ex:billingPortal ex:uses ?technology .
  ?technology rdfs:label ?technologyLabel .
}
ORDER BY ?technologyLabel
"""


@dataclass(frozen=True)
class FocusFilter:
    """Which triples to include in before/after focused state."""

    subjects: frozenset[URIRef] | None = None
    predicates: frozenset[URIRef] | None = None
    objects: frozenset[URIRef] | None = None


@dataclass(frozen=True)
class PredefinedUpdate:
    name: UpdateType
    update_query: str
    verification_query: str
    verification_variables: tuple[str, ...]
    focus: FocusFilter
    teaching_point: str


UPDATES: dict[str, PredefinedUpdate] = {
    "INSERT_DATA": PredefinedUpdate(
        name="INSERT_DATA",
        update_query=INSERT_DATA_UPDATE,
        verification_query=INSERT_DATA_VERIFY,
        verification_variables=("technology", "technologyLabel"),
        focus=FocusFilter(
            subjects=frozenset({EX.billingPortal}),
            predicates=frozenset({EX.uses}),
        ),
        teaching_point="INSERT DATA adds explicitly specified RDF triples.",
    ),
    "INSERT_WHERE": PredefinedUpdate(
        name="INSERT_WHERE",
        update_query=INSERT_WHERE_UPDATE,
        verification_query=INSERT_WHERE_VERIFY,
        verification_variables=(
            "person",
            "personLabel",
            "technology",
            "technologyLabel",
        ),
        focus=FocusFilter(
            predicates=frozenset({EX.uses}),
        ),
        teaching_point=(
            "INSERT WHERE uses existing graph patterns to create new triples."
        ),
    ),
    "DELETE_DATA": PredefinedUpdate(
        name="DELETE_DATA",
        update_query=DELETE_DATA_UPDATE,
        verification_query=DELETE_DATA_VERIFY,
        verification_variables=("technology", "technologyLabel"),
        focus=FocusFilter(
            subjects=frozenset({EX.billingPortal}),
            predicates=frozenset({EX.uses}),
        ),
        teaching_point="DELETE DATA removes explicitly specified RDF triples.",
    ),
    "UPDATE_AND_VERIFY": PredefinedUpdate(
        name="UPDATE_AND_VERIFY",
        update_query=UPDATE_AND_VERIFY_UPDATE,
        verification_query=UPDATE_AND_VERIFY_VERIFY,
        verification_variables=("technology", "technologyLabel"),
        focus=FocusFilter(
            subjects=frozenset({EX.billingPortal}),
            predicates=frozenset({EX.uses}),
        ),
        teaching_point=(
            "SPARQL UPDATE can delete and insert within one operation, "
            "then the resulting graph can be queried to verify the new state."
        ),
    ),
}


def get_update(name: str) -> PredefinedUpdate:
    if name not in UPDATES:
        known = ", ".join(sorted(UPDATES))
        raise KeyError(f"Unknown update '{name}'. Known: {known}")
    return UPDATES[name]


def validate_sparql_text(query: str) -> None:
    """Reject prohibited SPARQL operations before execution."""
    upper = query.upper()
    for keyword in PROHIBITED_SPARQL_KEYWORDS:
        if keyword in upper:
            raise SparqlError(
                "update_rejected",
                f"prohibited SPARQL keyword: {keyword}",
            )


def parse_prefixes(query: str) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("PREFIX"):
            continue
        body = stripped.removeprefix("PREFIX").strip()
        if ":" not in body:
            continue
        alias, iri_part = body.split(":", maxsplit=1)
        alias = alias.strip()
        iri = iri_part.strip().strip("<>").strip()
        if alias and iri:
            prefixes[alias] = iri
    return prefixes
