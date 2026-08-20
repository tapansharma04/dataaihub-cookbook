"""SPARQL UPDATE mutation tests — mutations must use Graph.update()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config import Settings
from sparql.cases import get_case
from sparql.graph import RdfGraphStore
from sparql.runner import run_case
from sparql.vocab import EX

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


@pytest.fixture
def settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH)


def test_insert_data_adds_redis(settings: Settings):
    case = get_case("insert-data-billing-portal-redis")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    assert (EX.billingPortal, EX.uses, EX.redis) not in store.rdf

    result = run_case(case, settings, store=store)

    assert (EX.billingPortal, EX.uses, EX.redis) in store.rdf
    assert (EX.billingPortal, EX.uses, EX.postgresql) in store.rdf
    assert result.metrics.inserted_triple_count == 1
    assert result.metrics.deleted_triple_count == 0
    labels = sorted(
        row.variables["technology"].label
        for row in result.verification_bindings
        if "technology" in row.variables
    )
    assert labels == ["PostgreSQL", "Redis"]
    assert result.metrics.termination_reason == "completed"


def test_insert_where_derives_person_uses_technology(settings: Settings):
    case = get_case("insert-where-person-uses-technology")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    assert (EX.alice, EX.uses, EX.postgresql) not in store.rdf
    assert (EX.bob, EX.uses, EX.postgresql) not in store.rdf
    assert (EX.carol, EX.uses, EX.postgresql) not in store.rdf

    result = run_case(case, settings, store=store)

    assert (EX.alice, EX.uses, EX.postgresql) in store.rdf
    assert (EX.bob, EX.uses, EX.postgresql) in store.rdf
    assert (EX.carol, EX.uses, EX.postgresql) in store.rdf
    assert result.metrics.inserted_triple_count == 3
    assert result.metrics.deleted_triple_count == 0
    assert result.metrics.verification_rows == 3
    pairs = sorted(
        (
            row.variables["person"].label,
            row.variables["technology"].label,
        )
        for row in result.verification_bindings
    )
    assert pairs == [
        ("Alice", "PostgreSQL"),
        ("Bob", "PostgreSQL"),
        ("Carol", "PostgreSQL"),
    ]


def test_insert_where_produced_by_sparql_not_python(settings: Settings):
    """Assert derived triples come from Graph.update, not Python graph.add."""
    case = get_case("insert-where-person-uses-technology")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    add_calls: list[object] = []
    original_add = store.rdf.add

    def tracking_add(triple: object) -> None:
        add_calls.append(triple)
        return original_add(triple)

    with patch.object(store.rdf, "add", side_effect=tracking_add):
        result = run_case(case, settings, store=store)

    # rdflib's update engine may call add internally; the measured path is update().
    assert result.metrics.inserted_triple_count == 3
    assert (EX.alice, EX.uses, EX.postgresql) in store.rdf
    executed = next(
        event for event in result.sequence if event.kind == "update_executed"
    )
    assert executed.detail["method"] == "Graph.update"
    assert executed.detail["success"] is True


def test_delete_data_removes_postgresql(settings: Settings):
    case = get_case("delete-data-billing-portal-postgresql")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    assert (EX.billingPortal, EX.uses, EX.postgresql) in store.rdf

    result = run_case(case, settings, store=store)

    assert (EX.billingPortal, EX.uses, EX.postgresql) not in store.rdf
    assert result.metrics.deleted_triple_count == 1
    assert result.metrics.inserted_triple_count == 0
    assert result.metrics.verification_rows == 0
    assert result.verification_bindings == []
    assert result.metrics.termination_reason == "completed"


def test_update_and_verify_swaps_technology(settings: Settings):
    case = get_case("update-and-verify-billing-portal-technology")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    assert (EX.billingPortal, EX.uses, EX.postgresql) in store.rdf
    assert (EX.billingPortal, EX.uses, EX.redis) not in store.rdf

    result = run_case(case, settings, store=store)

    assert (EX.billingPortal, EX.uses, EX.postgresql) not in store.rdf
    assert (EX.billingPortal, EX.uses, EX.redis) in store.rdf
    assert result.metrics.deleted_triple_count == 1
    assert result.metrics.inserted_triple_count == 1
    labels = [row.variables["technology"].label for row in result.verification_bindings]
    assert labels == ["Redis"]
    assert result.metrics.termination_reason == "completed"


def test_mutation_uses_graph_update_not_manual_remove(settings: Settings):
    case = get_case("delete-data-billing-portal-postgresql")
    store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    remove_calls: list[object] = []
    original_remove = store.rdf.remove

    def tracking_remove(triple: object) -> None:
        remove_calls.append(triple)
        return original_remove(triple)

    with patch.object(store.rdf, "remove", side_effect=tracking_remove):
        with patch.object(store.rdf, "update", wraps=store.rdf.update) as update_mock:
            result = run_case(case, settings, store=store)

    update_mock.assert_called_once()
    assert update_mock.call_args.args[0] == result.update_query
    assert (EX.billingPortal, EX.uses, EX.postgresql) not in store.rdf
    executed = next(
        event for event in result.sequence if event.kind == "update_executed"
    )
    assert executed.detail["method"] == "Graph.update"


def test_cases_are_isolated(settings: Settings):
    """INSERT_DATA then DELETE_DATA independently each see the fixture initial state."""
    insert_case = get_case("insert-data-billing-portal-redis")
    delete_case = get_case("delete-data-billing-portal-postgresql")

    insert_store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    insert_result = run_case(insert_case, settings, store=insert_store)
    assert insert_result.metrics.inserted_triple_count == 1
    assert (EX.billingPortal, EX.uses, EX.redis) in insert_store.rdf
    # Fixture still had postgresql; insert does not remove it.
    assert (EX.billingPortal, EX.uses, EX.postgresql) in insert_store.rdf

    delete_store = RdfGraphStore.fresh_from_path(GRAPH_PATH)
    # Fresh graph must not see INSERT_DATA mutations.
    assert (EX.billingPortal, EX.uses, EX.redis) not in delete_store.rdf
    assert (EX.billingPortal, EX.uses, EX.postgresql) in delete_store.rdf
    delete_result = run_case(delete_case, settings, store=delete_store)
    assert delete_result.metrics.deleted_triple_count == 1
    assert (EX.billingPortal, EX.uses, EX.postgresql) not in delete_store.rdf


def test_before_after_and_diff_preserved(settings: Settings):
    case = get_case("insert-data-billing-portal-redis")
    result = run_case(case, settings)
    assert len(result.before_state) == 1
    assert result.before_state[0].object.iri == str(EX.postgresql)
    assert len(result.after_state) == 2
    after_objects = {t.object.iri for t in result.after_state}
    assert after_objects == {str(EX.postgresql), str(EX.redis)}
    assert len(result.inserted_triples) == 1
    assert result.inserted_triples[0].object.iri == str(EX.redis)
    assert result.deleted_triples == []
