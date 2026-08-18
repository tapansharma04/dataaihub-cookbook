"""Run measured MCP protocol cases with observable tracing."""

from __future__ import annotations

import json
import time
from typing import Any

import mcp_types as types
from mcp import Client
from mcp.server.mcpserver import MCPServer

from client.cases import MeasuredCase
from client.schemas import DiscoveredTool, McpRunMetrics, McpRunResult, SequenceEvent
from config import Settings
from server.app import build_server

TRANSPORT_LABEL = (
    "in-process InMemoryTransport with JSON-RPC framing (legacy initialize handshake)"
)


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _tool_to_discovered(tool: types.Tool) -> DiscoveredTool:
    return DiscoveredTool(
        name=tool.name,
        description=tool.description,
        input_schema=dict(tool.input_schema or {}),
    )


def _tools_payload(tools: list[DiscoveredTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in tools
    ]


def _parse_tool_result(result: types.CallToolResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "isError": result.is_error,
        "content": [block.model_dump() for block in result.content],
    }
    if result.structured_content is not None:
        payload["structuredContent"] = result.structured_content
        return payload

    texts = [block.text for block in result.content if block.type == "text"]
    if len(texts) == 1:
        try:
            payload["structuredContent"] = json.loads(texts[0])
        except json.JSONDecodeError:
            payload["text"] = texts[0]
    elif texts:
        payload["text"] = "\n".join(texts)
    return payload


def _select_tool_name(case: MeasuredCase, discovered: list[DiscoveredTool]) -> str:
    if case.tool_name is not None:
        names = {tool.name for tool in discovered}
        if case.tool_name not in names:
            raise ValueError(
                f"Case {case.trace_id} expects tool '{case.tool_name}' "
                f"but discovery returned {sorted(names)}"
            )
        return case.tool_name
    if case.select_tool_by is not None:
        for tool in discovered:
            if tool.name == case.select_tool_by:
                return tool.name
    raise ValueError(f"Case {case.trace_id} has no resolvable tool selection")


async def run_case_async(
    case: MeasuredCase,
    *,
    settings: Settings,
    server: MCPServer | None = None,
) -> McpRunResult:
    """Execute one measured MCP case through the official client/server boundary."""
    mcp_server = server or build_server(settings.data_dir)
    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    discovered: list[DiscoveredTool] = []
    output: dict[str, Any] = {}
    protocol_version: str | None = None
    server_name: str | None = None

    total_start = time.perf_counter()
    initialize_ms = 0
    discovery_ms = 0
    tool_call_ms = 0
    tool_calls = 0
    successful_tool_calls = 0
    failed_tool_calls = 0

    client_info = types.Implementation(
        name=settings.client_name,
        version=settings.client_version,
    )

    init_start = time.perf_counter()
    sequence.append(
        SequenceEvent(
            kind="initialize_request",
            detail={
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
                kind="tools_list_request",
                detail={"method": "tools/list"},
            )
        )
        list_result = await client.list_tools()
        discovery_ms = _elapsed_ms(list_start)
        discovered = [_tool_to_discovered(tool) for tool in list_result.tools]
        sequence.append(
            SequenceEvent(
                kind="tools_list_response",
                detail={
                    "method": "tools/list",
                    "tools": _tools_payload(discovered),
                    "toolCount": len(discovered),
                },
                latency_ms=discovery_ms,
            )
        )
        output["discoveredTools"] = [tool.name for tool in discovered]

        if case.action == "invoke":
            tool_name = _select_tool_name(case, discovered)
            arguments = dict(case.tool_arguments or {})

            call_start = time.perf_counter()
            sequence.append(
                SequenceEvent(
                    kind="tool_call_request",
                    detail={
                        "method": "tools/call",
                        "name": tool_name,
                        "arguments": arguments,
                    },
                )
            )
            call_result = await client.call_tool(tool_name, arguments)
            tool_call_ms = _elapsed_ms(call_start)
            tool_calls = 1
            parsed = _parse_tool_result(call_result)
            if call_result.is_error:
                failed_tool_calls = 1
                errors.append(
                    {
                        "stage": "tools/call",
                        "tool": tool_name,
                        "isError": True,
                        "detail": parsed,
                    }
                )
            else:
                successful_tool_calls = 1

            sequence.append(
                SequenceEvent(
                    kind="tool_call_response",
                    detail={
                        "method": "tools/call",
                        "name": tool_name,
                        "isError": call_result.is_error,
                        "result": parsed,
                    },
                    latency_ms=tool_call_ms,
                )
            )
            output["invocation"] = {
                "tool": tool_name,
                "arguments": arguments,
                "isError": call_result.is_error,
                "result": parsed.get("structuredContent") or parsed,
            }

    total_ms = _elapsed_ms(total_start)
    sequence.append(
        SequenceEvent(
            kind="termination",
            detail={"reason": "session_closed"},
        )
    )

    return McpRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        transport=TRANSPORT_LABEL,
        protocol_version=protocol_version,
        server_name=server_name,
        discovered_tools=discovered,
        sequence=sequence,
        metrics=McpRunMetrics(
            total_ms=total_ms,
            initialize_ms=initialize_ms,
            discovery_ms=discovery_ms,
            tool_call_ms=tool_call_ms,
            tool_calls=tool_calls,
            tools_discovered=len(discovered),
            successful_tool_calls=successful_tool_calls,
            failed_tool_calls=failed_tool_calls,
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
