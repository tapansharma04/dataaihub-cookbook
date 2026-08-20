"""SPARQL runner and binding normalization tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from sparql.cases import CASES, get_case
from sparql.graph import RdfGraphStore
from sparql.runner import assert_query_not_python_filtered, run_case
from sparql.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


@pytest.fixture
def store() -> RdfGraphStore:
    return RdfGraphStore.from_path(GRAPH_PATH)


@pytest.fixture
def settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH)


def test_basic_select_returns_alice_and_bob(store: RdfGraphStore, settings: Settings):
    case = get_case("basic-select-knowledge-platform-people")
    result = run_case(case, store, settings)
    labels = sorted(
        row["person"]["label"] for row in result.matches if row.get("person")
    )
    assert labels == ["Alice", "Bob"]
    assert result.metrics.result_rows == 2
    assert result.metrics.termination_reason == "completed"
    assert result.query in result.output["query"]


def test_basic_select_bindings_have_iri_and_label(
    store: RdfGraphStore, settings: Settings
):
    case = get_case("basic-select-knowledge-platform-people")
    result = run_case(case, store, settings)
    person_iris = {row["person"]["iri"] for row in result.matches}
    assert person_iris == {str(EX.alice), str(EX.bob)}


def test_multi_pattern_join_returns_postgresql(
    store: RdfGraphStore, settings: Settings
):
    case = get_case("multi-pattern-alice-technologies")
    result = run_case(case, store, settings)
    assert len(result.matches) == 1
    row = result.matches[0]
    assert row["person"]["label"] == "Alice"
    assert row["project"]["label"] == "Knowledge Platform"
    assert row["technology"]["label"] == "PostgreSQL"
    assert result.metrics.triple_patterns >= 2


def test_filter_query_excludes_billing_team(store: RdfGraphStore, settings: Settings):
    case = get_case("filter-platform-team-people")
    result = run_case(case, store, settings)
    labels = sorted(row["person"]["label"] for row in result.matches)
    assert labels == ["Alice", "Bob"]
    teams = {row["team"]["literal"] for row in result.matches}
    assert teams == {"platform"}


def test_filter_is_executed_by_sparql_not_python(
    store: RdfGraphStore, settings: Settings
):
    case = get_case("filter-platform-team-people")
    assert assert_query_not_python_filtered(case, store) is True


def test_no_match_returns_zero_rows(store: RdfGraphStore, settings: Settings):
    case = get_case("no-match-quantum-platform")
    result = run_case(case, store, settings)
    assert result.matches == []
    assert result.output["matches"] == []
    assert result.metrics.result_rows == 0
    assert result.metrics.termination_reason == "no_match"
    assert result.errors == []


def test_query_executed_via_graph_query(store: RdfGraphStore, settings: Settings):
    for case in CASES:
        result = run_case(case, store, settings)
        executed = next(
            event for event in result.sequence if event.kind == "query_executed"
        )
        assert executed.detail["engine"] == "rdflib"
        assert executed.detail["success"] is True
        assert executed.detail["query"] == result.query


def test_results_are_normalized_not_rdflib_rows(
    store: RdfGraphStore, settings: Settings
):
    case = get_case("basic-select-knowledge-platform-people")
    result = run_case(case, store, settings)
    for row in result.matches:
        assert isinstance(row, dict)
        for value in row.values():
            assert isinstance(value, dict)
            assert "iri" in value


def test_all_four_cases_align_with_cases_module(
    store: RdfGraphStore, settings: Settings
):
    assert len(CASES) == 4
    for case in CASES:
        result = run_case(case, store, settings)
        assert result.case_id == case.trace_id
        assert result.example_class == case.example_class
        assert result.question == case.question
