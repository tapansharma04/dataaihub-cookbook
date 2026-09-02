"""MCP composition runner tests — protocol boundary, no paid APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from client.cases import MeasuredCase, ProtocolStep, get_case
from client.runner import run_case_async
from client.sampling import MOCK_MODEL, SAMPLING_REJECT_MESSAGE
from client.trace import build_signature_view
from config import Settings
from server.app import build_server
from server.fixtures import (
    TOOL_COMPOSE_FROM_PROMPT,
    TOOL_COMPOSE_INCIDENT_BRIEF,
    TOOL_COMPOSE_RESOURCE_BRIEF,
    TOOL_GET_SERVICE_STATUS,
    URI_BILLING_PORTAL,
    URI_KNOWLEDGE_PLATFORM,
)
from server.prompts import render_summarize_service

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def settings() -> Settings:
    return Settings(data_dir=DATA, mcp_client_mode="legacy", openai_api_key="")


@pytest.fixture
def server():
    return build_server(DATA)


def test_runner_uses_official_mcp_client_apis():
    source = (ROOT / "client" / "runner.py").read_text(encoding="utf-8")
    assert "from mcp import Client" in source
    assert "sampling_callback=sampling_callback" in source
    assert "await client.list_resources()" in source
    assert "await client.read_resource(" in source
    assert "await client.list_prompts()" in source
    assert "await client.get_prompt(" in source
    assert "await client.list_tools()" in source
    assert "await client.call_tool(" in source
    assert "mode=settings.mcp_client_mode" in source


def test_client_does_not_import_openai_in_mock_path():
    for relative in (
        "client/runner.py",
        "client/cases.py",
        "client/schemas.py",
        "client/trace.py",
        "server/app.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "from openai" not in source
        assert "import openai" not in source


@pytest.mark.asyncio
async def test_mcp_initialization_handshake(settings, server):
    result = await run_case_async(
        get_case("resource-to-sampling"), settings=settings, server=server
    )
    kinds = [event.kind for event in result.sequence]
    assert kinds[0] == "initialize_request"
    assert kinds[1] == "initialize_response"
    assert result.protocol_version is not None
    assert result.server_name == "dataaihub-cookbook-composition"
    init_req = next(e for e in result.sequence if e.kind == "initialize_request")
    assert init_req.detail["samplingCallbackRegistered"] is True


@pytest.mark.asyncio
async def test_resource_to_sampling_grounds_context(settings, server):
    result = await run_case_async(
        get_case("resource-to-sampling"), settings=settings, server=server
    )
    assert result.example_class == "RESOURCE_TO_SAMPLING"
    kinds = [event.kind for event in result.sequence]
    assert kinds.index("resource_read_request") < kinds.index("sampling_request")
    assert kinds.index("sampling_request") < kinds.index("sampling_response")
    assert kinds.index("sampling_response") < kinds.index("tool_call_response")
    assert result.metrics.sampling_requests == 1
    assert result.metrics.successful_samplings == 1
    assert result.metrics.failed_samplings == 0
    assert result.metrics.model_turns == 1
    assert result.metrics.resources_read == 1

    read = result.output["reads"][0]
    resource_text = read["contents"][0]["text"]
    sampling_req = next(e for e in result.sequence if e.kind == "sampling_request")
    params = sampling_req.detail["params"]
    messages = params["messages"]
    assert messages[0]["role"] == "user"
    sampling_text = messages[0]["content"]["text"]
    assert URI_KNOWLEDGE_PLATFORM in sampling_text
    assert "Knowledge Platform" in resource_text
    assert resource_text in sampling_text
    compose_req = next(
        e
        for e in result.sequence
        if e.kind == "tool_call_request"
        and e.detail.get("name") == TOOL_COMPOSE_RESOURCE_BRIEF
    )
    assert compose_req.detail["arguments"]["content"] == resource_text

    sampling_res = next(e for e in result.sequence if e.kind == "sampling_response")
    assert sampling_res.detail["isError"] is False
    assert sampling_res.detail["boundary"] == "mcp-client-sampling-callback"
    assert sampling_res.detail["result"]["model"] == MOCK_MODEL
    assert "Knowledge Platform" in sampling_res.detail["result"]["content"]["text"]

    invocation = result.output["invocations"][0]
    assert invocation["tool"] == TOOL_COMPOSE_RESOURCE_BRIEF
    assert invocation["isError"] is False
    assert invocation["result"]["ok"] is True
    assert invocation["result"]["sampling"]["model"] == MOCK_MODEL


@pytest.mark.asyncio
async def test_prompt_to_sampling_uses_mcp_prompt(settings, server):
    result = await run_case_async(
        get_case("prompt-to-sampling"), settings=settings, server=server
    )
    assert result.metrics.prompts_requested == 1
    assert result.metrics.sampling_requests == 1
    get = result.output["gets"][0]
    assert get["requestedPrompt"] == "summarize-service"
    prompt_text = get["messages"][0]["content"]["text"]
    expected = render_summarize_service("knowledge-platform", "engineering")
    assert prompt_text == expected

    sampling_req = next(e for e in result.sequence if e.kind == "sampling_request")
    sampling_text = sampling_req.detail["params"]["messages"][0]["content"]["text"]
    assert sampling_text == expected
    assert sampling_text == get["messages"][0]["content"]["text"]
    assert (
        get["messages"][0]["role"]
        == sampling_req.detail["params"]["messages"][0]["role"]
    )
    compose_req = next(
        e
        for e in result.sequence
        if e.kind == "tool_call_request"
        and e.detail.get("name") == TOOL_COMPOSE_FROM_PROMPT
    )
    assert compose_req.detail["arguments"]["messages"] == get["messages"]

    invocation = result.output["invocations"][0]
    assert invocation["tool"] == TOOL_COMPOSE_FROM_PROMPT
    assert invocation["result"]["ok"] is True


@pytest.mark.asyncio
async def test_tool_resource_prompt_composition_order(settings, server):
    result = await run_case_async(
        get_case("tool-resource-prompt-composition"),
        settings=settings,
        server=server,
    )
    kinds = [event.kind for event in result.sequence]
    assert kinds.index("tool_call_request") < kinds.index("resource_read_request")
    assert kinds.index("resource_read_request") < kinds.index("prompt_get_request")
    assert kinds.index("prompt_get_request") < kinds.index("sampling_request")
    assert result.metrics.tool_calls == 2
    assert result.metrics.resources_read == 1
    assert result.metrics.prompts_requested == 1
    assert result.metrics.sampling_requests == 1
    assert result.metrics.successful_samplings == 1

    status_inv = result.output["invocations"][0]
    assert status_inv["tool"] == TOOL_GET_SERVICE_STATUS
    assert status_inv["result"]["ok"] is True
    assert status_inv["result"]["service"]["status"] == "degraded"
    assert status_inv["result"]["service"]["incident"] == (
        "BILL-2048: invoice PDF generation delayed"
    )

    sampling_req = next(e for e in result.sequence if e.kind == "sampling_request")
    messages = sampling_req.detail["params"]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert "get_service_status" in messages[0]["content"]["text"]
    assert "BILL-2048" in messages[0]["content"]["text"]
    resource_text = result.output["reads"][0]["contents"][0]["text"]
    prompt_text = result.output["gets"][0]["messages"][0]["content"]["text"]
    assert resource_text in messages[1]["content"]["text"]
    assert prompt_text == messages[2]["content"]["text"]
    compose_req = next(
        e
        for e in result.sequence
        if e.kind == "tool_call_request"
        and e.detail.get("name") == TOOL_COMPOSE_INCIDENT_BRIEF
    )
    assert compose_req.detail["arguments"]["tool_result"] == status_inv["result"]
    assert compose_req.detail["arguments"]["resource_content"] == resource_text
    assert (
        compose_req.detail["arguments"]["prompt_messages"]
        == result.output["gets"][0]["messages"]
    )

    compose = result.output["invocations"][1]
    assert compose["tool"] == TOOL_COMPOSE_INCIDENT_BRIEF
    assert compose["result"]["ok"] is True
    assert compose["result"]["resourceUri"] == URI_BILLING_PORTAL
    assert compose["result"]["promptName"] == "draft-status-update"
    assert compose["result"]["promptArguments"] == result.output["gets"][0]["arguments"]
    assert (
        compose_req.detail["arguments"]["prompt_arguments"]
        == (result.output["gets"][0]["arguments"])
    )


@pytest.mark.asyncio
async def test_sampling_failure_is_protocol_visible(settings, server):
    result = await run_case_async(
        get_case("sampling-failure"), settings=settings, server=server
    )
    assert result.metrics.sampling_requests == 1
    assert result.metrics.successful_samplings == 0
    assert result.metrics.failed_samplings == 1
    assert result.metrics.model_turns == 0
    assert result.metrics.failed_tool_calls == 1
    kinds = [event.kind for event in result.sequence]
    assert "sampling_request" in kinds
    assert "sampling_response" in kinds
    sampling_res = next(e for e in result.sequence if e.kind == "sampling_response")
    assert sampling_res.detail["isError"] is True
    assert sampling_res.detail["error"]["message"] == SAMPLING_REJECT_MESSAGE
    assert "result" not in sampling_res.detail
    sample_out = result.output["sampling"][0]
    assert sample_out["isError"] is True
    assert "result" not in sample_out
    invocation = result.output["invocations"][0]
    assert invocation["isError"] is True
    assert invocation["tool"] == TOOL_COMPOSE_RESOURCE_BRIEF
    assert result.metrics.termination_reason == "sampling_rejected"


@pytest.mark.asyncio
async def test_unknown_tool_is_protocol_error_not_sampling_rejected(settings, server):
    result = await run_case_async(
        MeasuredCase(
            trace_id="unknown-tool-call",
            example_class="RESOURCE_TO_SAMPLING",
            selection_note="Test-only: a missing tool is not a sampling rejection.",
            steps=(
                ProtocolStep(kind="list_tools"),
                ProtocolStep(
                    kind="call_tool",
                    tool_name="does_not_exist",
                    tool_arguments={},
                ),
            ),
            sampling_mode="mock",
        ),
        settings=settings,
        server=server,
    )
    assert result.metrics.sampling_requests == 0
    assert result.metrics.failed_tool_calls == 1
    assert result.metrics.termination_reason == "protocol_error"


@pytest.mark.asyncio
async def test_sampling_failure_does_not_fabricate_llm_output(settings, server):
    result = await run_case_async(
        get_case("sampling-failure"), settings=settings, server=server
    )
    for event in result.sequence:
        if event.kind == "sampling_response":
            assert event.detail.get("isError") is True
            assert event.detail.get("result") in (None, {})
    for sample in result.output["sampling"]:
        assert sample.get("result") in (None, {})


def test_resource_signature_flow(settings):
    from client.runner import run_case

    result = run_case(
        get_case("resource-to-sampling"),
        settings=settings,
        server=build_server(DATA),
    )
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "RESOURCE" in phases
    assert "CONTEXT" in phases
    assert "SAMPLING" in phases
    assert "RESULT" in phases


def test_failure_signature_flow(settings):
    from client.runner import run_case

    result = run_case(
        get_case("sampling-failure"),
        settings=settings,
        server=build_server(DATA),
    )
    phases = [v["phase"] for v in build_signature_view(result.sequence)]
    assert "SAMPLING" in phases
    assert "REJECTED" in phases
    assert phases.count("RESULT") == 0
