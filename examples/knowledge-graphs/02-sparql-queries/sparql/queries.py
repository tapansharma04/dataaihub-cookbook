"""Predefined SPARQL queries for measured teaching cases.

Each query is a committed string executed by rdflib's SPARQL engine.
The application selects which query runs; arbitrary user SPARQL is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import PROHIBITED_SPARQL_KEYWORDS
from sparql.model import SparqlError

COMMON_PREFIXES = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

BASIC_SELECT = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?person ?personLabel
WHERE {
  ?person ex:worksOn ex:knowledgePlatform .
  ?person rdfs:label ?personLabel .
}
ORDER BY ?personLabel
"""

MULTI_PATTERN_QUERY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?person ?personLabel ?project ?projectLabel ?technology ?technologyLabel
WHERE {
  ?person ex:worksOn ?project .
  ?project rdfs:label ?projectLabel .
  ?project ex:uses ?technology .
  ?technology rdfs:label ?technologyLabel .
  ?person rdfs:label ?personLabel .
  FILTER(?person = ex:alice)
}
ORDER BY ?technologyLabel
"""

FILTER_QUERY = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?person ?personLabel ?project ?projectLabel ?team
WHERE {
  ?person ex:worksOn ?project .
  ?project rdfs:label ?projectLabel .
  ?project ex:team ?team .
  ?person rdfs:label ?personLabel .
  FILTER(?team = "platform")
}
ORDER BY ?personLabel
"""

NO_MATCH = """\
PREFIX ex: <https://dataaihub.co/example/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?person ?personLabel
WHERE {
  ?person ex:worksOn ex:quantumComputingPlatform .
  ?person rdfs:label ?personLabel .
}
ORDER BY ?personLabel
"""


@dataclass(frozen=True)
class PredefinedQuery:
    name: str
    query: str
    triple_patterns: int
    filter_count: int
    variables: tuple[str, ...]
    patterns: tuple[str, ...]


QUERIES: dict[str, PredefinedQuery] = {
    "BASIC_SELECT": PredefinedQuery(
        name="BASIC_SELECT",
        query=BASIC_SELECT,
        triple_patterns=2,
        filter_count=0,
        variables=("person", "personLabel"),
        patterns=(
            "?person ex:worksOn ex:knowledgePlatform .",
            "?person rdfs:label ?personLabel .",
        ),
    ),
    "MULTI_PATTERN_QUERY": PredefinedQuery(
        name="MULTI_PATTERN_QUERY",
        query=MULTI_PATTERN_QUERY,
        triple_patterns=5,
        filter_count=1,
        variables=(
            "person",
            "personLabel",
            "project",
            "projectLabel",
            "technology",
            "technologyLabel",
        ),
        patterns=(
            "?person ex:worksOn ?project .",
            "?project rdfs:label ?projectLabel .",
            "?project ex:uses ?technology .",
            "?technology rdfs:label ?technologyLabel .",
            "?person rdfs:label ?personLabel .",
        ),
    ),
    "FILTER_QUERY": PredefinedQuery(
        name="FILTER_QUERY",
        query=FILTER_QUERY,
        triple_patterns=4,
        filter_count=1,
        variables=("person", "personLabel", "project", "projectLabel", "team"),
        patterns=(
            "?person ex:worksOn ?project .",
            "?project rdfs:label ?projectLabel .",
            "?project ex:team ?team .",
            "?person rdfs:label ?personLabel .",
        ),
    ),
    "NO_MATCH": PredefinedQuery(
        name="NO_MATCH",
        query=NO_MATCH,
        triple_patterns=2,
        filter_count=0,
        variables=("person", "personLabel"),
        patterns=(
            "?person ex:worksOn ex:quantumComputingPlatform .",
            "?person rdfs:label ?personLabel .",
        ),
    ),
}


def get_query(name: str) -> PredefinedQuery:
    if name not in QUERIES:
        known = ", ".join(sorted(QUERIES))
        raise KeyError(f"Unknown query '{name}'. Known: {known}")
    return QUERIES[name]


def validate_query_text(query: str) -> None:
    """Reject prohibited SPARQL operations before execution."""
    upper = query.upper()
    for keyword in PROHIBITED_SPARQL_KEYWORDS:
        if keyword in upper:
            raise SparqlError(
                "query_rejected",
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
