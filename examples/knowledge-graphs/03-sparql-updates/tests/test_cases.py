"""Measured case registry and query alignment tests."""

from __future__ import annotations

import pytest

from config import PROHIBITED_SPARQL_KEYWORDS
from sparql.cases import CASES, get_case
from sparql.model import SparqlError
from sparql.queries import UPDATES, get_update, parse_prefixes, validate_sparql_text


def test_exactly_four_cases():
    assert len(CASES) == 4


def test_case_classes():
    classes = {case.example_class for case in CASES}
    assert classes == {
        "INSERT_DATA",
        "INSERT_WHERE",
        "DELETE_DATA",
        "UPDATE_AND_VERIFY",
    }


def test_get_case_unknown_raises():
    with pytest.raises(KeyError, match="missing-case"):
        get_case("missing-case")


def test_each_case_maps_to_predefined_update():
    for case in CASES:
        assert case.update_name in UPDATES
        assert case.update_name == case.example_class


def test_all_four_updates_exist():
    assert set(UPDATES) == {
        "INSERT_DATA",
        "INSERT_WHERE",
        "DELETE_DATA",
        "UPDATE_AND_VERIFY",
    }


def test_prefix_declarations_present():
    for name in UPDATES:
        update = get_update(name)
        for query in (update.update_query, update.verification_query):
            prefixes = parse_prefixes(query)
            assert prefixes["ex"] == "https://dataaihub.co/example/kg/"
            assert prefixes["rdfs"] == "http://www.w3.org/2000/01/rdf-schema#"


def test_insert_data_query_shape():
    update = get_update("INSERT_DATA")
    assert "INSERT DATA" in update.update_query
    assert "ex:billingPortal ex:uses ex:redis" in update.update_query


def test_insert_where_query_shape():
    update = get_update("INSERT_WHERE")
    assert "INSERT {" in update.update_query
    assert "WHERE {" in update.update_query
    assert "?person ex:worksOn ?project" in update.update_query
    assert "?project ex:uses ?technology" in update.update_query
    assert "?person ex:uses ?technology" in update.update_query


def test_delete_data_query_shape():
    update = get_update("DELETE_DATA")
    assert "DELETE DATA" in update.update_query
    assert "ex:billingPortal ex:uses ex:postgresql" in update.update_query


def test_update_and_verify_query_shape():
    update = get_update("UPDATE_AND_VERIFY")
    assert "DELETE {" in update.update_query
    assert "INSERT {" in update.update_query
    assert "WHERE {" in update.update_query


@pytest.mark.parametrize("keyword", sorted(PROHIBITED_SPARQL_KEYWORDS))
def test_prohibited_keywords_rejected(keyword: str):
    with pytest.raises(SparqlError, match="prohibited"):
        validate_sparql_text(
            f"INSERT DATA {{ <http://a> <http://b> <http://c> }} "
            f"{keyword} <http://example.org/>"
        )
