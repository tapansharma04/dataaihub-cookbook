"""Predefined SPARQL query tests."""

from __future__ import annotations

import pytest

from config import PROHIBITED_SPARQL_KEYWORDS
from sparql.model import SparqlError
from sparql.queries import (
    BASIC_SELECT,
    FILTER_QUERY,
    MULTI_PATTERN_QUERY,
    NO_MATCH,
    QUERIES,
    get_query,
    parse_prefixes,
    validate_query_text,
)


def test_all_four_queries_exist():
    names = {"BASIC_SELECT", "MULTI_PATTERN_QUERY", "FILTER_QUERY", "NO_MATCH"}
    assert set(QUERIES) == names


def test_prefix_declarations_present():
    for query in (BASIC_SELECT, MULTI_PATTERN_QUERY, FILTER_QUERY, NO_MATCH):
        prefixes = parse_prefixes(query)
        assert prefixes["ex"] == "https://dataaihub.co/example/kg/"
        assert prefixes["rdfs"] == "http://www.w3.org/2000/01/rdf-schema#"


def test_basic_select_has_fixed_object_pattern():
    query = get_query("BASIC_SELECT")
    assert "?person ex:worksOn ex:knowledgePlatform ." in query.patterns
    assert query.filter_count == 0


def test_multi_pattern_has_shared_variable_join():
    query = get_query("MULTI_PATTERN_QUERY")
    assert "?person ex:worksOn ?project ." in query.patterns
    assert "?project ex:uses ?technology ." in query.patterns
    assert "FILTER(?person = ex:alice)" in query.query


def test_filter_query_contains_sparql_filter():
    query = get_query("FILTER_QUERY")
    assert 'FILTER(?team = "platform")' in query.query
    assert "?project ex:team ?team ." in query.patterns


def test_no_match_queries_missing_project():
    query = get_query("NO_MATCH")
    assert "ex:quantumComputingPlatform" in query.query


@pytest.mark.parametrize("keyword", sorted(PROHIBITED_SPARQL_KEYWORDS))
def test_prohibited_keywords_rejected(keyword: str):
    with pytest.raises(SparqlError, match="prohibited"):
        validate_query_text(
            f"SELECT ?s WHERE {{ ?s ?p ?o . }} {keyword} <http://example.org/>"
        )
