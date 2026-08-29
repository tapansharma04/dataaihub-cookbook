"""Run measured MCP prompt protocol cases with observable tracing."""

from __future__ import annotations

import time
from typing import Any

import mcp_types as types
from mcp import Client
from mcp.server.mcpserver import MCPServer

from client.cases import MeasuredCase
from client.schemas import (
    DiscoveredPrompt,
    McpRunMetrics,
    McpRunResult,
    PromptArgumentMeta,
    SequenceEvent,
)
from config import Settings
from server.app import build_server

TRANSPORT_LABEL = (
    "in-process InMemoryTransport with JSON-RPC framing (legacy initialize handshake)"
)


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _prompt_to_discovered(prompt: types.Prompt) -> DiscoveredPrompt:
    arguments = [
        PromptArgumentMeta(
            name=arg.name,
            description=arg.description,
            required=bool(arg.required),
        )
        for arg in (prompt.arguments or [])
    ]
    return DiscoveredPrompt(
        name=prompt.name,
        description=prompt.description,
        arguments=arguments,
    )


def _prompts_payload(prompts: list[DiscoveredPrompt]) -> list[dict[str, Any]]:
    return [
        {
            "name": prompt.name,
            "description": prompt.description,
            "arguments": [
                {
                    "name": arg.name,
                    "description": arg.description,
                    "required": arg.required,
                }
                for arg in prompt.arguments
            ],
        }
        for prompt in prompts
    ]


def _messages_payload(messages: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            payload.append(message.model_dump(mode="json", exclude_none=True))
        else:
            payload.append(dict(message))
    return payload


def _text_bytes(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") or {}
        text = content.get("text")
        if isinstance(text, str):
            total += len(text.encode("utf-8"))
    return total


def _protocol_error_payload(exc: BaseException) -> dict[str, Any] | None:
    """Extract JSON-RPC / MCP error data when the SDK surfaces a protocol error."""
    error = getattr(exc, "error", None)
    if error is None:
        return None
    if hasattr(error, "model_dump"):
        return error.model_dump(mode="json", exclude_none=True)
    if isinstance(error, dict):
        return dict(error)
    return {
        "message": str(error),
        "repr": repr(error),
    }


def _is_mcp_protocol_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"McpError", "MCPError"}:
        return True
    module = type(exc).__module__
    has_payload = _protocol_error_payload(exc) is not None
    return "mcp" in module and "Error" in name and has_payload


async def run_case_async(
    case: MeasuredCase,
    *,
    settings: Settings,
    server: MCPServer | None = None,
) -> McpRunResult:
    """Execute one measured MCP prompt case across the official protocol boundary."""
    mcp_server = server or build_server()
    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    discovered: list[DiscoveredPrompt] = []
    output: dict[str, Any] = {}
    protocol_version: str | None = None
    server_name: str | None = None

    total_start = time.perf_counter()
    initialize_ms = 0
    discovery_ms = 0
    prompt_get_ms = 0
    prompts_requested = 0
    successful_gets = 0
    failed_gets = 0
    message_count = 0
    message_bytes = 0
    termination_reason = "session_closed"
    gets: list[dict[str, Any]] = []

    client_info = types.Implementation(
        name=settings.client_name,
        version=settings.client_version,
    )

    init_start = time.perf_counter()
    sequence.append(
        SequenceEvent(
            kind="initialize_request",
            detail={
                "interactionId": "initialize",
                "method": "initialize",
                "clientInfo": client_info.model_dump(exclude_none=True),
            },
        )
    )

    async with Client(
        mcp_server,
        mode=settings.mcp_client_mode,
        client_info=client_info,
    ) as client:
        init_result = client.session.initialize_result
        initialize_ms = _elapsed_ms(init_start)
        protocol_version = client.protocol_version
        server_name = (
            init_result.server_info.name
            if init_result and init_result.server_info
            else None
        )
        sequence.append(
            SequenceEvent(
                kind="initialize_response",
                detail={
                    "interactionId": "initialize",
                    "method": "initialize",
                    "protocolVersion": protocol_version,
                    "serverInfo": (
                        init_result.server_info.model_dump(exclude_none=True)
                        if init_result and init_result.server_info
                        else None
                    ),
                    "capabilities": (
                        init_result.capabilities.model_dump(exclude_none=True)
                        if init_result and init_result.capabilities
                        else None
                    ),
                },
                latency_ms=initialize_ms,
            )
        )

        list_start = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="prompts_list_request",
                detail={
                    "interactionId": "prompts/list",
                    "method": "prompts/list",
                },
            )
        )
        list_result = await client.list_prompts()
        discovery_ms = _elapsed_ms(list_start)
        discovered = [_prompt_to_discovered(prompt) for prompt in list_result.prompts]
        sequence.append(
            SequenceEvent(
                kind="prompts_list_response",
                detail={
                    "interactionId": "prompts/list",
                    "method": "prompts/list",
                    "prompts": _prompts_payload(discovered),
                    "promptCount": len(discovered),
                },
                latency_ms=discovery_ms,
            )
        )
        output["discoveredPrompts"] = [prompt.name for prompt in discovered]

        if case.action == "get":
            assert case.prompt_name is not None
            prompt_name = case.prompt_name
            arguments = dict(case.prompt_arguments or {})
            discovered_names = {prompt.name for prompt in discovered}
            interaction_id = f"prompts/get:{prompt_name}"
            get_start = time.perf_counter()
            sequence.append(
                SequenceEvent(
                    kind="prompt_get_request",
                    detail={
                        "interactionId": interaction_id,
                        "method": "prompts/get",
                        "name": prompt_name,
                        "arguments": arguments,
                        "discoveredBeforeGet": prompt_name in discovered_names,
                    },
                )
            )
            prompts_requested += 1
            try:
                get_result = await client.get_prompt(prompt_name, arguments)
            except Exception as exc:
                latency = _elapsed_ms(get_start)
                prompt_get_ms += latency
                protocol_error = _protocol_error_payload(exc)
                if _is_mcp_protocol_error(exc) and protocol_error is not None:
                    failed_gets += 1
                    termination_reason = "prompt_get_rejected"
                    response_detail = {
                        "interactionId": interaction_id,
                        "method": "prompts/get",
                        "name": prompt_name,
                        "arguments": arguments,
                        "isError": True,
                        "error": protocol_error,
                        "exceptionType": type(exc).__name__,
                    }
                    sequence.append(
                        SequenceEvent(
                            kind="prompt_get_response",
                            detail=response_detail,
                            latency_ms=latency,
                        )
                    )
                    errors.append(
                        {
                            "stage": "prompts/get",
                            "name": prompt_name,
                            "isError": True,
                            "detail": response_detail,
                        }
                    )
                    gets.append(
                        {
                            "requestedPrompt": prompt_name,
                            "arguments": arguments,
                            "isError": True,
                            "error": protocol_error,
                        }
                    )
                else:
                    termination_reason = "protocol_error"
                    sequence.append(
                        SequenceEvent(
                            kind="error",
                            detail={
                                "interactionId": interaction_id,
                                "stage": "prompts/get",
                                "name": prompt_name,
                                "message": str(exc),
                                "exceptionType": type(exc).__name__,
                            },
                            latency_ms=latency,
                        )
                    )
                    errors.append(
                        {
                            "stage": "prompts/get",
                            "name": prompt_name,
                            "isError": True,
                            "message": str(exc),
                            "exceptionType": type(exc).__name__,
                        }
                    )
                    gets.append(
                        {
                            "requestedPrompt": prompt_name,
                            "arguments": arguments,
                            "isError": True,
                            "message": str(exc),
                        }
                    )
            else:
                latency = _elapsed_ms(get_start)
                prompt_get_ms += latency
                messages = _messages_payload(list(get_result.messages))
                successful_gets += 1
                message_count += len(messages)
                message_bytes += _text_bytes(messages)
                sequence.append(
                    SequenceEvent(
                        kind="prompt_get_response",
                        detail={
                            "interactionId": interaction_id,
                            "method": "prompts/get",
                            "name": prompt_name,
                            "arguments": arguments,
                            "isError": False,
                            "description": get_result.description,
                            "messages": messages,
                        },
                        latency_ms=latency,
                    )
                )
                gets.append(
                    {
                        "requestedPrompt": prompt_name,
                        "arguments": arguments,
                        "isError": False,
                        "description": get_result.description,
                        "messages": messages,
                    }
                )

            output["gets"] = gets

    total_ms = _elapsed_ms(total_start)
    sequence.append(
        SequenceEvent(
            kind="termination",
            detail={"reason": termination_reason},
        )
    )

    return McpRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        transport=TRANSPORT_LABEL,
        protocol_version=protocol_version,
        server_name=server_name,
        discovered_prompts=discovered,
        sequence=sequence,
        metrics=McpRunMetrics(
            total_ms=total_ms,
            initialize_ms=initialize_ms,
            discovery_ms=discovery_ms,
            prompt_get_ms=prompt_get_ms,
            prompts_discovered=len(discovered),
            prompts_requested=prompts_requested,
            successful_gets=successful_gets,
            failed_gets=failed_gets,
            message_count=message_count,
            message_bytes=message_bytes,
            model_turns=0,
            tool_calls=0,
            resources_read=0,
            termination_reason=termination_reason,
        ),
        output=output,
        errors=errors,
    )


def run_case(
    case: MeasuredCase,
    *,
    settings: Settings,
    server: MCPServer | None = None,
) -> McpRunResult:
    import asyncio

    return asyncio.run(run_case_async(case, settings=settings, server=server))
