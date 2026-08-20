"""Measured case registry tests."""

from __future__ import annotations

from sparql.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4


def test_case_classes():
    classes = {case.example_class for case in CASES}
    assert classes == {
        "BASIC_SELECT",
        "MULTI_PATTERN_QUERY",
        "FILTER_QUERY",
        "NO_MATCH",
    }


def test_get_case_unknown_raises():
    try:
        get_case("missing-case")
    except KeyError as exc:
        assert "missing-case" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_each_case_maps_to_predefined_query():
    from sparql.queries import QUERIES

    for case in CASES:
        assert case.query_name in QUERIES
        assert case.query_name == case.example_class
