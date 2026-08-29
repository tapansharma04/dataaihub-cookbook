"""Server-side prompt catalog tests."""

from __future__ import annotations

from server.fixtures import CATALOG, EXPECTED_PROMPT_NAMES


def test_prompt_catalog_lists_three_stable_names():
    assert [entry.name for entry in CATALOG] == list(EXPECTED_PROMPT_NAMES)


def test_catalog_descriptions_are_present():
    for entry in CATALOG:
        assert entry.description
        assert len(entry.description) > 10
