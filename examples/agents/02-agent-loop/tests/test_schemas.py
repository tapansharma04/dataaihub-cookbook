"""Schema serialization smoke tests."""

from __future__ import annotations

from agent.schemas import (
    AgentRunMetrics,
    ToolCallRequest,
    ToolDefinition,
    ToolParameterProperty,
    ToolParameters,
)


def test_tool_definition_roundtrip():
    definition = ToolDefinition(
        name="get_service_status",
        description="Status lookup",
        parameters=ToolParameters(
            properties={
                "service": ToolParameterProperty(
                    type="string",
                    description="Service name",
                )
            },
            required=["service"],
        ),
    )
    payload = definition.model_dump()
    assert payload["parameters"]["additionalProperties"] is False
    again = ToolDefinition.model_validate(payload)
    assert again.name == "get_service_status"


def test_tool_call_request_requires_name():
    call = ToolCallRequest(id="1", name="x", arguments={"a": 1})
    assert call.arguments["a"] == 1


def test_metrics_include_termination_and_max_turns():
    metrics = AgentRunMetrics(
        total_ms=1,
        model_ms=1,
        tool_ms=0,
        model_turns=1,
        tool_calls=0,
        successful_tool_calls=0,
        failed_tool_calls=0,
        termination_reason="final_answer",
        max_turns=6,
    )
    assert metrics.provenance == "measured"
    assert metrics.termination_reason == "final_answer"
    assert metrics.max_turns == 6
