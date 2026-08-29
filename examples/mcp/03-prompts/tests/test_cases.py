"""Measured case catalog tests."""

from __future__ import annotations

from client.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    ids = [case.trace_id for case in CASES]
    assert ids == [
        "prompt-discovery",
        "single-prompt-get-summarize",
        "prompt-with-arguments-investigate",
        "invalid-prompt-name",
    ]


def test_example_classes():
    classes = {case.example_class for case in CASES}
    assert classes == {
        "PROMPT_DISCOVERY",
        "SINGLE_PROMPT_GET",
        "PROMPT_WITH_ARGUMENTS",
        "INVALID_PROMPT",
    }


def test_discovery_has_no_get_target():
    case = get_case("prompt-discovery")
    assert case.action == "discover_only"
    assert case.prompt_name is None
    assert case.prompt_arguments is None


def test_single_and_multi_argument_cases():
    single = get_case("single-prompt-get-summarize")
    assert single.prompt_name == "summarize-service"
    assert single.prompt_arguments == {"service_name": "knowledge-platform"}
    multi = get_case("prompt-with-arguments-investigate")
    assert multi.prompt_name == "investigate-incident"
    assert multi.prompt_arguments == {
        "service": "billing-api",
        "incident": "INC-2048",
    }


def test_invalid_prompt_case_target():
    case = get_case("invalid-prompt-name")
    assert case.prompt_name == "does-not-exist"
    assert case.prompt_arguments == {}


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
