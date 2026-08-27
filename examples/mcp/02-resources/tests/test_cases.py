"""Measured case catalog tests."""

from __future__ import annotations

from client.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    ids = [case.trace_id for case in CASES]
    assert ids == [
        "discovery",
        "single-resource-read-knowledge-platform",
        "multi-resource-read-services",
        "invalid-resource-uri",
    ]


def test_example_classes():
    classes = {case.example_class for case in CASES}
    assert classes == {
        "DISCOVERY",
        "SINGLE_RESOURCE_READ",
        "MULTI_RESOURCE_READ",
        "INVALID_RESOURCE",
    }


def test_discovery_has_no_read_uris():
    case = get_case("discovery")
    assert case.action == "discover_only"
    assert case.resource_uris == ()


def test_single_and_multi_uris():
    single = get_case("single-resource-read-knowledge-platform")
    assert single.resource_uris == ("acme://docs/knowledge-platform",)
    multi = get_case("multi-resource-read-services")
    assert multi.resource_uris == (
        "acme://docs/knowledge-platform",
        "acme://status/services",
    )


def test_invalid_uri_case_target():
    case = get_case("invalid-resource-uri")
    assert case.resource_uris == ("acme://docs/does-not-exist",)


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
