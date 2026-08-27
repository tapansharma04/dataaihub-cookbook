"""Run measured MCP resource protocol cases with observable tracing."""

from __future__ import annotations

import time
from typing import Any

import mcp_types as types
from mcp import Client
from mcp.server.mcpserver import MCPServer

from client.cases import MeasuredCase
from client.schemas import (
    DiscoveredResource,
    McpRunMetrics,
    McpRunResult,
    SequenceEvent,
)
from config import Settings
from server.app import build_server

TRANSPORT_LABEL = (
    "in-process InMemoryTransport with JSON-RPC framing (legacy initialize handshake)"
)


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _uri_str(uri: Any) -> str:
    return str(uri)


def _resource_to_discovered(resource: types.Resource) -> DiscoveredResource:
    return DiscoveredResource(
        uri=_uri_str(resource.uri),
        name=resource.name,
        description=resource.description,
        mime_type=resource.mime_type,
    )


def _resources_payload(resources: list[DiscoveredResource]) -> list[dict[str, Any]]:
    return [
        {
            "uri": resource.uri,
            "name": resource.name,
            "description": resource.description,
            "mimeType": resource.mime_type,
        }
        for resource in resources
    ]


def _content_payload(contents: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for block in contents:
        if hasattr(block, "model_dump"):
            dumped = block.model_dump(mode="json", exclude_none=True)
        else:
            dumped = dict(block)
        if "uri" in dumped:
            dumped["uri"] = _uri_str(dumped["uri"])
        if "mime_type" in dumped and "mimeType" not in dumped:
            dumped["mimeType"] = dumped.pop("mime_type")
        payload.append(dumped)
    return payload


def _text_bytes(contents: list[dict[str, Any]]) -> int:
    total = 0
    for block in contents:
        text = block.get("text")
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
    """Execute one measured MCP resource case across the official protocol boundary."""
    mcp_server = server or build_server(settings.data_dir)
    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    discovered: list[DiscoveredResource] = []
    output: dict[str, Any] = {}
    protocol_version: str | None = None
    server_name: str | None = None

    total_start = time.perf_counter()
    initialize_ms = 0
    discovery_ms = 0
    resource_read_ms = 0
    resources_read = 0
    successful_reads = 0
    failed_reads = 0
    resource_bytes = 0
    termination_reason = "session_closed"
    reads: list[dict[str, Any]] = []

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
                kind="resources_list_request",
                detail={
                    "interactionId": "resources/list",
                    "method": "resources/list",
                },
            )
        )
        list_result = await client.list_resources()
        discovery_ms = _elapsed_ms(list_start)
        discovered = [
            _resource_to_discovered(resource) for resource in list_result.resources
        ]
        sequence.append(
            SequenceEvent(
                kind="resources_list_response",
                detail={
                    "interactionId": "resources/list",
                    "method": "resources/list",
                    "resources": _resources_payload(discovered),
                    "resourceCount": len(discovered),
                },
                latency_ms=discovery_ms,
            )
        )
        output["discoveredResources"] = [resource.uri for resource in discovered]

        if case.action == "read":
            discovered_uris = {resource.uri for resource in discovered}
            for uri in case.resource_uris:
                interaction_id = f"resources/read:{uri}"
                read_start = time.perf_counter()
                sequence.append(
                    SequenceEvent(
                        kind="resource_read_request",
                        detail={
                            "interactionId": interaction_id,
                            "method": "resources/read",
                            "uri": uri,
                            "discoveredBeforeRead": uri in discovered_uris,
                        },
                    )
                )
                resources_read += 1
                try:
                    read_result = await client.read_resource(uri)
                except Exception as exc:
                    latency = _elapsed_ms(read_start)
                    resource_read_ms += latency
                    protocol_error = _protocol_error_payload(exc)
                    if _is_mcp_protocol_error(exc) and protocol_error is not None:
                        failed_reads += 1
                        termination_reason = "resource_read_rejected"
                        response_detail = {
                            "interactionId": interaction_id,
                            "method": "resources/read",
                            "uri": uri,
                            "isError": True,
                            "error": protocol_error,
                            "exceptionType": type(exc).__name__,
                        }
                        sequence.append(
                            SequenceEvent(
                                kind="resource_read_response",
                                detail=response_detail,
                                latency_ms=latency,
                            )
                        )
                        errors.append(
                            {
                                "stage": "resources/read",
                                "uri": uri,
                                "isError": True,
                                "detail": response_detail,
                            }
                        )
                        reads.append(
                            {
                                "requestedUri": uri,
                                "isError": True,
                                "error": protocol_error,
                            }
                        )
                        continue

                    termination_reason = "protocol_error"
                    sequence.append(
                        SequenceEvent(
                            kind="error",
                            detail={
                                "interactionId": interaction_id,
                                "stage": "resources/read",
                                "uri": uri,
                                "message": str(exc),
                                "exceptionType": type(exc).__name__,
                            },
                            latency_ms=latency,
                        )
                    )
                    errors.append(
                        {
                            "stage": "resources/read",
                            "uri": uri,
                            "isError": True,
                            "message": str(exc),
                            "exceptionType": type(exc).__name__,
                        }
                    )
                    reads.append(
                        {
                            "requestedUri": uri,
                            "isError": True,
                            "message": str(exc),
                        }
                    )
                    continue

                latency = _elapsed_ms(read_start)
                resource_read_ms += latency
                contents = _content_payload(list(read_result.contents))
                successful_reads += 1
                resource_bytes += _text_bytes(contents)
                returned_uri = contents[0].get("uri", uri) if contents else uri
                mime_type = contents[0].get("mimeType") if contents else None
                sequence.append(
                    SequenceEvent(
                        kind="resource_read_response",
                        detail={
                            "interactionId": interaction_id,
                            "method": "resources/read",
                            "uri": uri,
                            "isError": False,
                            "contents": contents,
                        },
                        latency_ms=latency,
                    )
                )
                reads.append(
                    {
                        "requestedUri": uri,
                        "returnedUri": returned_uri,
                        "mimeType": mime_type,
                        "isError": False,
                        "contents": contents,
                    }
                )

            output["reads"] = reads

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
        discovered_resources=discovered,
        sequence=sequence,
        metrics=McpRunMetrics(
            total_ms=total_ms,
            initialize_ms=initialize_ms,
            discovery_ms=discovery_ms,
            resource_read_ms=resource_read_ms,
            resources_discovered=len(discovered),
            resources_read=resources_read,
            successful_reads=successful_reads,
            failed_reads=failed_reads,
            resource_bytes=resource_bytes,
            model_turns=0,
            tool_calls=0,
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
