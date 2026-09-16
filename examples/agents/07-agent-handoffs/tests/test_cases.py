"""Measured case catalog tests."""

from __future__ import annotations

from agent.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    assert [case.trace_id for case in CASES] == [
        "invoice-duplicate-basic-handoff",
        "refund-blocked-multi-hop",
        "legal-target-invalid-handoff",
        "unknown-system-handoff-failure",
    ]


def test_example_classes():
    assert [case.example_class for case in CASES] == [
        "BASIC_HANDOFF",
        "MULTI_HOP_HANDOFF",
        "INVALID_HANDOFF",
        "HANDOFF_FAILURE",
    ]


def test_all_cases_start_at_triage():
    assert {case.initial_agent for case in CASES} == {"triage_agent"}


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
