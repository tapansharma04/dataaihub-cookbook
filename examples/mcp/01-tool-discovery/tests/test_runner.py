"""MCP protocol runner tests — no network / no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from client.cases import CASES, get_case
from client.runner import run_case_async
from client.trace import build_signature_view, build_trace
from config import EXAMPLE_ID, Settings
from server.app import build_server

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def settings() -> Settings:
    return Settings(data_dir=DATA, mcp_client_mode="legacy")


@pytest.fixture
def server():
    return build_server(DATA)


@pytest.mark.asyncio
async def test_discovery_case_lists_three_tools(settings, server):
    case = get_case("discovery")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.example_class == "DISCOVERY"
    assert result.metrics.tools_discovered == 3
    assert result.metrics.tool_calls == 0
    names = {tool.name for tool in result.discovered_tools}
    assert names == {
        "get_service_status",
        "get_user_profile",
        "search_documentation",
    }
    kinds = [event.kind for event in result.sequence]
    assert "initialize_request" in kinds
    assert "tools_list_response" in kinds
    assert "tool_call_request" not in kinds


@pytest.mark.asyncio
async def test_single_tool_call_returns_structured_result(settings, server):
    case = get_case("single-tool-service-status")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.successful_tool_calls == 1
    inv = result.output["invocation"]
    assert inv["tool"] == "get_service_status"
    assert inv["arguments"] == {"service": "payments"}
    assert inv["isError"] is False
    assert inv["result"]["ok"] is True
    assert inv["result"]["service"]["status"] == "degraded"


@pytest.mark.asyncio
async def test_multi_tool_discovery_invokes_selected_tool(settings, server):
    case = get_case("multi-tool-search-docs")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.tools_discovered == 3
    inv = result.output["invocation"]
    assert inv["tool"] == "search_documentation"
    assert inv["isError"] is False
    assert inv["result"]["hitCount"] >= 1


@pytest.mark.asyncio
async def test_invalid_arguments_rejected_at_protocol_boundary(settings, server):
    case = get_case("invalid-arguments-service-type")
    result = await run_case_async(case, settings=settings, server=server)
    assert result.metrics.failed_tool_calls == 1
    inv = result.output["invocation"]
    assert inv["isError"] is True
    assert inv["arguments"] == {"service": 123}
    kinds = [e.kind for e in result.sequence]
    assert "error" not in kinds
    assert kinds.count("tool_call_response") == 1
    response = next(e for e in result.sequence if e.kind == "tool_call_response")
    assert response.detail["isError"] is True
    assert response.detail["name"] == "get_service_status"
    assert result.errors[0]["stage"] == "tools/call"


def test_signature_view_protocol_error_uses_error_event():
    """Genuine protocol/transport failures remain on the error event."""
    from client.schemas import SequenceEvent

    sequence = [
        SequenceEvent(
            kind="tool_call_request",
            detail={"method": "tools/call", "name": "get_service_status"},
        ),
        SequenceEvent(
            kind="error",
            detail={
                "stage": "transport",
                "message": "connection closed before response",
            },
        ),
    ]
    phases = [v["phase"] for v in build_signature_view(sequence)]
    assert phases == ["INVOKE", "ERROR"]


def test_build_trace_provenance(settings):
    case = get_case("discovery")
    from client.runner import run_case

    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    assert trace["labId"] == EXAMPLE_ID
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["metricsProvenance"] == "measured"
    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]


def test_signature_view_invalid_arguments_flow(settings):
    from client.runner import run_case

    case = get_case("invalid-arguments-service-type")
    result = run_case(case, settings=settings, server=build_server(DATA))
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "INITIALIZE" in phases
    assert "DISCOVER" in phases
    assert "INVOKE" in phases
    assert "REJECTED" in phases
    assert "ERROR" not in phases


def test_invalid_arguments_trace_has_no_redundant_error_event(settings):
    from client.runner import run_case

    case = get_case("invalid-arguments-service-type")
    result = run_case(case, settings=settings, server=build_server(DATA))
    trace = build_trace(case=case, result=result, settings=settings)
    kinds = [event["kind"] for event in trace["sequence"]]
    assert "error" not in kinds
    response = next(
        event for event in trace["sequence"] if event["kind"] == "tool_call_response"
    )
    assert response["detail"]["isError"] is True


def test_committed_lab_traces_schema_if_present():
    path = ROOT / "lab_traces.json"
    if not path.exists():
        return
    traces = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES)
    ids = {t["traceId"] for t in traces}
    assert ids == {c.trace_id for c in CASES}
    for trace in traces:
        assert trace["labId"] == EXAMPLE_ID
        assert trace["provenance"]["model"] == "not_used"
        assert trace["metricsProvenance"] == "measured"
        assert "sequence" in trace and trace["sequence"]
        assert "tools" in trace and len(trace["tools"]) == 3
        for event in trace["sequence"]:
            assert "chainOfThought" not in event.get("detail", {})
            assert "reasoning" not in event.get("detail", {})
    classes = {t["exampleClass"] for t in traces}
    assert classes == {
        "DISCOVERY",
        "SINGLE_TOOL_CALL",
        "MULTI_TOOL_DISCOVERY",
        "INVALID_ARGUMENTS",
    }
