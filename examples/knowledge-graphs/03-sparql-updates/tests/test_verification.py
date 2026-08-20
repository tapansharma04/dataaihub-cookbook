"""Verification SELECT query tests after SPARQL UPDATE."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config import Settings
from sparql.cases import CASES, get_case
from sparql.graph import RdfGraphStore
from sparql.runner import run_case
from sparql.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


@pytest.fixture
def settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH)


def test_verification_uses_graph_query(settings: Settings):
    for case in CASES:
        store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
        with patch.object(store.rdf, "query", wraps=store.rdf.query) as query_mock:
            result = run_case(case, settings, store=store)
        query_mock.assert_called()
        assert query_mock.call_args.args[0] == result.verification_query
        verified = next(
            event for event in result.sequence if event.kind == "verification_result"
        )
        assert verified.detail["method"] == "Graph.query"
        assert verified.detail["success"] is True


def test_insert_data_verification_includes_redis(settings: Settings):
    result = run_case(get_case("insert-data-billing-portal-redis"), settings)
    iris = {row.variables["technology"].iri for row in result.verification_bindings}
    assert str(EX.redis) in iris
    assert str(EX.postgresql) in iris


def test_delete_data_verification_empty(settings: Settings):
    result = run_case(get_case("delete-data-billing-portal-postgresql"), settings)
    assert result.verification_bindings == []
    assert result.metrics.verification_rows == 0


def test_update_and_verify_verification_only_redis(settings: Settings):
    result = run_case(get_case("update-and-verify-billing-portal-technology"), settings)
    iris = {row.variables["technology"].iri for row in result.verification_bindings}
    assert iris == {str(EX.redis)}
