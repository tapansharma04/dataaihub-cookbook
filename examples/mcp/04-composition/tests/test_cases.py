"""Measured case catalog tests."""

from __future__ import annotations

from client.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    ids = [case.trace_id for case in CASES]
    assert ids == [
        "resource-to-sampling",
        "prompt-to-sampling",
        "tool-resource-prompt-composition",
        "sampling-failure",
    ]


def test_example_classes():
    classes = {case.example_class for case in CASES}
    assert classes == {
        "RESOURCE_TO_SAMPLING",
        "PROMPT_TO_SAMPLING",
        "TOOL_RESOURCE_PROMPT_COMPOSITION",
        "SAMPLING_FAILURE",
    }


def test_sampling_modes():
    assert get_case("resource-to-sampling").sampling_mode == "mock"
    assert get_case("prompt-to-sampling").sampling_mode == "mock"
    assert get_case("tool-resource-prompt-composition").sampling_mode == "mock"
    assert get_case("sampling-failure").sampling_mode == "reject"


def test_resource_case_reads_then_composes():
    kinds = [step.kind for step in get_case("resource-to-sampling").steps]
    assert kinds == ["list_resources", "read_resource", "call_tool"]


def test_prompt_case_gets_then_composes():
    kinds = [step.kind for step in get_case("prompt-to-sampling").steps]
    assert kinds == ["list_prompts", "get_prompt", "call_tool"]


def test_composition_case_uses_all_primitives():
    kinds = [step.kind for step in get_case("tool-resource-prompt-composition").steps]
    assert "call_tool" in kinds
    assert "read_resource" in kinds
    assert "get_prompt" in kinds
    assert kinds.count("call_tool") == 2


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
