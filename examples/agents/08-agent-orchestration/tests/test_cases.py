"""Measured case catalog tests."""

from __future__ import annotations

from agent.cases import CASES, get_case


def test_exactly_four_cases():
    assert len(CASES) == 4
    assert [case.trace_id for case in CASES] == [
        "payments-incident-basic-orchestration",
        "payments-status-dependency-chain",
        "payments-incident-conditional-branch",
        "unknown-service-orchestration-failure",
    ]


def test_example_classes():
    assert [case.example_class for case in CASES] == [
        "BASIC_ORCHESTRATION",
        "DEPENDENCY_CHAIN",
        "CONDITIONAL_BRANCH",
        "ORCHESTRATION_FAILURE",
    ]


def test_basic_has_independent_status_and_docs():
    case = get_case("payments-incident-basic-orchestration")
    by_id = {node.node_id: node for node in case.nodes}
    assert by_id["status"].dependencies == ()
    assert by_id["docs"].dependencies == ()
    assert by_id["analysis"].dependencies == ("status", "docs")
    assert by_id["decision"].dependencies == ("analysis",)


def test_chain_is_strictly_linear():
    case = get_case("payments-status-dependency-chain")
    assert [node.node_id for node in case.nodes] == ["status", "analysis", "decision"]
    by_id = {node.node_id: node for node in case.nodes}
    assert by_id["status"].dependencies == ()
    assert by_id["analysis"].dependencies == ("status",)
    assert by_id["decision"].dependencies == ("analysis",)


def test_conditional_has_mutually_exclusive_branches():
    case = get_case("payments-incident-conditional-branch")
    by_id = {node.node_id: node for node in case.nodes}
    assert by_id["remediation"].condition is not None
    assert by_id["no_action"].condition is not None
    assert by_id["remediation"].condition.equals is True
    assert by_id["no_action"].condition.equals is False
    assert by_id["remediation"].condition.field == "incident"
    assert by_id["no_action"].condition.field == "incident"


def test_failure_case_uses_unknown_service():
    case = get_case("unknown-service-orchestration-failure")
    status = case.nodes[0]
    assert status.assignment["service"] == "does-not-exist"


def test_get_case_unknown():
    try:
        get_case("missing")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "missing" in str(exc)
