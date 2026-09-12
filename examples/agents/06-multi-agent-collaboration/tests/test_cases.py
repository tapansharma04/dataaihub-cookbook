"""Measured case catalog tests."""

from __future__ import annotations

from agent.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    assert [case.trace_id for case in CASES] == [
        "independent-status-and-docs",
        "sequential-status-then-analysis",
        "docs-agent-failure",
        "operational-short-circuit",
    ]


def test_example_classes():
    assert {case.example_class for case in CASES} == {
        "BASIC_COLLABORATION",
        "SEQUENTIAL_COLLABORATION",
        "AGENT_FAILURE",
        "COLLABORATION_TERMINATION",
    }


def test_failure_case_stops_on_failure():
    assert get_case("docs-agent-failure").stop_on_failure is True
    assert get_case("independent-status-and-docs").stop_on_failure is False


def test_sequential_case_passes_status_into_analysis():
    steps = get_case("sequential-status-then-analysis").steps
    analysis = steps[1]
    assert analysis.agent_id == "analysis_agent"
    assert analysis.input_from_agent == "status_agent"


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
