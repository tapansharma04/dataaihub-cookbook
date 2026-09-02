"""Run measured MCP composition cases with observable tracing."""

from __future__ import annotations

import json
import time
from typing import Any

import mcp_types as types
from mcp import Client
from mcp.client import ClientRequestContext
from mcp.server.mcpserver import MCPServer

from client.cases import MeasuredCase, ProtocolStep
from client.sampling import build_sampling_callback
from client.schemas import (
    DiscoveredPrompt,
    DiscoveredResource,
    DiscoveredTool,
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


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return dict(obj)
    return obj


def _uri_str(uri: Any) -> str:
    return str(uri)


def _protocol_error_payload(exc: BaseException) -> dict[str, Any] | None:
    error = getattr(exc, "error", None)
    if error is None:
        return None
    dumped = _dump(error)
    if isinstance(dumped, dict):
        return dumped
    return {"message": str(error), "repr": repr(error)}


def _is_mcp_protocol_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"McpError", "MCPError"}:
        return True
    module = type(exc).__module__
    has_payload = _protocol_error_payload(exc) is not None
    return "mcp" in module and "Error" in name and has_payload


def _resource_to_discovered(resource: types.Resource) -> DiscoveredResource:
    return DiscoveredResource(
        uri=_uri_str(resource.uri),
        name=resource.name,
        description=resource.description,
        mime_type=resource.mime_type,
    )


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


def _tool_to_discovered(tool: types.Tool) -> DiscoveredTool:
    return DiscoveredTool(
        name=tool.name,
        description=tool.description,
        input_schema=dict(tool.input_schema or {}),
    )


def resources_payload(resources: list[DiscoveredResource]) -> list[dict[str, Any]]:
    return [
        {
            "uri": resource.uri,
            "name": resource.name,
            "description": resource.description,
            "mimeType": resource.mime_type,
        }
        for resource in resources
    ]


def prompts_payload(prompts: list[DiscoveredPrompt]) -> list[dict[str, Any]]:
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


def tools_payload(tools: list[DiscoveredTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in tools
    ]


def _content_payload(contents: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for block in contents:
        dumped = _dump(block)
        if isinstance(dumped, dict) and "uri" in dumped:
            dumped["uri"] = _uri_str(dumped["uri"])
        if (
            isinstance(dumped, dict)
            and "mime_type" in dumped
            and "mimeType" not in dumped
        ):
            dumped["mimeType"] = dumped.pop("mime_type")
        payload.append(dumped)
    return payload


def _messages_payload(messages: list[Any]) -> list[dict[str, Any]]:
    return [_dump(message) for message in messages]


def _parse_tool_result(result: types.CallToolResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "isError": result.is_error,
        "content": [_dump(block) for block in result.content],
    }
    if result.structured_content is not None:
        payload["structuredContent"] = result.structured_content
        return payload
    texts = [
        block.text for block in result.content if getattr(block, "type", None) == "text"
    ]
    if len(texts) == 1:
        try:
            payload["structuredContent"] = json.loads(texts[0])
        except json.JSONDecodeError:
            payload["text"] = texts[0]
    elif texts:
        payload["text"] = "\n".join(texts)
    return payload


def _resource_text(read: dict[str, Any]) -> str:
    texts: list[str] = []
    for block in read.get("contents") or []:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts)


def _resource_mime_type(read: dict[str, Any]) -> str:
    for block in read.get("contents") or []:
        if isinstance(block, dict) and block.get("mimeType"):
            return str(block["mimeType"])
    return "text/markdown"


def _matching_read(
    reads: list[dict[str, Any]], uri: str | None
) -> dict[str, Any] | None:
    if uri:
        for read in reversed(reads):
            if read.get("requestedUri") == uri and not read.get("isError"):
                return read
    return None


def _matching_get(
    gets: list[dict[str, Any]], prompt_name: str | None
) -> dict[str, Any] | None:
    if prompt_name:
        for get in reversed(gets):
            if get.get("requestedPrompt") == prompt_name and not get.get("isError"):
                return get
    return None


def _matching_status_result(
    invocations: list[dict[str, Any]], service: str | None
) -> dict[str, Any] | None:
    for invocation in reversed(invocations):
        if invocation.get("tool") != "get_service_status":
            continue
        if invocation.get("isError"):
            continue
        arguments = invocation.get("arguments") or {}
        if service is None or arguments.get("service") == service:
            result = invocation.get("result")
            return result if isinstance(result, dict) else None
    return None


def _successful_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if not item.get("isError")]


def _unique_successful(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = _successful_items(items)
    if len(successful) == 1:
        return successful[0]
    return None


def _had_sampling_rejection(sequence: list[SequenceEvent]) -> bool:
    return any(
        event.kind == "sampling_response" and event.detail.get("isError")
        for event in sequence
    )


def _bind_composition_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    output: dict[str, Any],
    discovered_resources: list[DiscoveredResource],
) -> dict[str, Any]:
    """Attach prior MCP results so composition tools do not reload fixtures."""
    bound = dict(arguments)
    if tool_name == "compose_resource_brief":
        uri = bound.get("uri")
        read = _matching_read(output.get("reads") or [], uri)
        if read is None:
            raise ValueError(
                "compose_resource_brief requires a prior resources/read "
                f"for uri={uri!r}"
            )
        bound["content"] = _resource_text(read)
        bound["mime_type"] = _resource_mime_type(read)
        for resource in discovered_resources:
            if resource.uri == uri and resource.name:
                bound["name"] = resource.name
                break
        return bound
    if tool_name == "compose_from_prompt":
        prompt_name = bound.get("prompt_name")
        get = _matching_get(output.get("gets") or [], prompt_name)
        if get is None:
            raise ValueError(
                "compose_from_prompt requires a prior prompts/get "
                f"for prompt={prompt_name!r}"
            )
        bound["messages"] = list(get.get("messages") or [])
        return bound
    if tool_name == "compose_incident_brief":
        service = bound.get("service")
        status = _matching_status_result(output.get("invocations") or [], service)
        reads = output.get("reads") or []
        gets = output.get("gets") or []
        resource_uri = bound.get("resource_uri")
        prompt_name = bound.get("prompt_name")
        read = (
            _matching_read(reads, resource_uri)
            if resource_uri
            else _unique_successful(reads)
        )
        get = (
            _matching_get(gets, prompt_name)
            if prompt_name
            else _unique_successful(gets)
        )
        if status is None or read is None or get is None:
            raise ValueError(
                "compose_incident_brief requires a matching prior tools/call, "
                "a unique or named resources/read, and a unique or named prompts/get"
            )
        bound["tool_result"] = status
        bound["resource_uri"] = read.get("requestedUri")
        bound["resource_content"] = _resource_text(read)
        bound["resource_mime_type"] = _resource_mime_type(read)
        bound["prompt_name"] = get.get("requestedPrompt")
        bound["prompt_messages"] = list(get.get("messages") or [])
        bound["prompt_arguments"] = dict(get.get("arguments") or {})
        return bound
    return bound


async def _run_step(
    client: Client,
    step: ProtocolStep,
    *,
    index: int,
    sequence: list[SequenceEvent],
    errors: list[dict[str, Any]],
    output: dict[str, Any],
    discovered_resources: list[DiscoveredResource],
    discovered_prompts: list[DiscoveredPrompt],
    discovered_tools: list[DiscoveredTool],
    counters: dict[str, Any],
) -> None:
    if step.kind == "list_resources":
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="resources_list_request",
                detail={
                    "interactionId": "resources/list",
                    "method": "resources/list",
                },
            )
        )
        result = await client.list_resources()
        latency = _elapsed_ms(started)
        counters["discovery_ms"] += latency
        discovered_resources.clear()
        discovered_resources.extend(
            _resource_to_discovered(resource) for resource in result.resources
        )
        sequence.append(
            SequenceEvent(
                kind="resources_list_response",
                detail={
                    "interactionId": "resources/list",
                    "method": "resources/list",
                    "resources": resources_payload(discovered_resources),
                    "resourceCount": len(discovered_resources),
                },
                latency_ms=latency,
            )
        )
        output["discoveredResources"] = [
            resource.uri for resource in discovered_resources
        ]
        return

    if step.kind == "read_resource":
        assert step.uri is not None
        uri = step.uri
        interaction_id = f"resources/read:{uri}"
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="resource_read_request",
                detail={
                    "interactionId": interaction_id,
                    "method": "resources/read",
                    "uri": uri,
                },
            )
        )
        counters["resources_read"] += 1
        result = await client.read_resource(uri)
        latency = _elapsed_ms(started)
        counters["resource_read_ms"] += latency
        contents = _content_payload(list(result.contents))
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
        output["reads"].append(
            {"requestedUri": uri, "isError": False, "contents": contents}
        )
        return

    if step.kind == "list_prompts":
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="prompts_list_request",
                detail={"interactionId": "prompts/list", "method": "prompts/list"},
            )
        )
        result = await client.list_prompts()
        latency = _elapsed_ms(started)
        counters["discovery_ms"] += latency
        discovered_prompts.clear()
        discovered_prompts.extend(
            _prompt_to_discovered(prompt) for prompt in result.prompts
        )
        sequence.append(
            SequenceEvent(
                kind="prompts_list_response",
                detail={
                    "interactionId": "prompts/list",
                    "method": "prompts/list",
                    "prompts": prompts_payload(discovered_prompts),
                    "promptCount": len(discovered_prompts),
                },
                latency_ms=latency,
            )
        )
        output["discoveredPrompts"] = [prompt.name for prompt in discovered_prompts]
        return

    if step.kind == "get_prompt":
        assert step.prompt_name is not None
        prompt_name = step.prompt_name
        arguments = dict(step.prompt_arguments or {})
        interaction_id = f"prompts/get:{prompt_name}"
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="prompt_get_request",
                detail={
                    "interactionId": interaction_id,
                    "method": "prompts/get",
                    "name": prompt_name,
                    "arguments": arguments,
                },
            )
        )
        counters["prompts_requested"] += 1
        result = await client.get_prompt(prompt_name, arguments)
        latency = _elapsed_ms(started)
        counters["prompt_get_ms"] += latency
        messages = _messages_payload(list(result.messages))
        sequence.append(
            SequenceEvent(
                kind="prompt_get_response",
                detail={
                    "interactionId": interaction_id,
                    "method": "prompts/get",
                    "name": prompt_name,
                    "arguments": arguments,
                    "isError": False,
                    "description": result.description,
                    "messages": messages,
                },
                latency_ms=latency,
            )
        )
        output["gets"].append(
            {
                "requestedPrompt": prompt_name,
                "arguments": arguments,
                "isError": False,
                "messages": messages,
            }
        )
        return

    if step.kind == "list_tools":
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="tools_list_request",
                detail={"interactionId": "tools/list", "method": "tools/list"},
            )
        )
        result = await client.list_tools()
        latency = _elapsed_ms(started)
        counters["discovery_ms"] += latency
        discovered_tools.clear()
        discovered_tools.extend(_tool_to_discovered(tool) for tool in result.tools)
        sequence.append(
            SequenceEvent(
                kind="tools_list_response",
                detail={
                    "interactionId": "tools/list",
                    "method": "tools/list",
                    "tools": tools_payload(discovered_tools),
                    "toolCount": len(discovered_tools),
                },
                latency_ms=latency,
            )
        )
        output["discoveredTools"] = [tool.name for tool in discovered_tools]
        return

    if step.kind == "call_tool":
        assert step.tool_name is not None
        tool_name = step.tool_name
        arguments = _bind_composition_arguments(
            tool_name,
            dict(step.tool_arguments or {}),
            output=output,
            discovered_resources=discovered_resources,
        )
        interaction_id = f"tools/call:{tool_name}:{index}"
        started = time.perf_counter()
        sequence.append(
            SequenceEvent(
                kind="tool_call_request",
                detail={
                    "interactionId": interaction_id,
                    "method": "tools/call",
                    "name": tool_name,
                    "arguments": arguments,
                },
            )
        )
        counters["tool_calls"] += 1
        try:
            result = await client.call_tool(tool_name, arguments)
        except Exception as exc:
            latency = _elapsed_ms(started)
            counters["tool_call_ms"] += latency
            counters["failed_tool_calls"] += 1
            protocol_error = _protocol_error_payload(exc)
            is_protocol = _is_mcp_protocol_error(exc) and protocol_error is not None
            if is_protocol:
                counters["termination_reason"] = (
                    "sampling_rejected"
                    if _had_sampling_rejection(sequence)
                    else "protocol_error"
                )
                response_detail = {
                    "interactionId": interaction_id,
                    "method": "tools/call",
                    "name": tool_name,
                    "arguments": arguments,
                    "isError": True,
                    "error": protocol_error,
                    "exceptionType": type(exc).__name__,
                }
                sequence.append(
                    SequenceEvent(
                        kind="tool_call_response",
                        detail=response_detail,
                        latency_ms=latency,
                    )
                )
                errors.append(
                    {
                        "stage": "tools/call",
                        "tool": tool_name,
                        "isError": True,
                        "detail": response_detail,
                    }
                )
                output["invocations"].append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "isError": True,
                        "error": protocol_error,
                    }
                )
            else:
                counters["termination_reason"] = "protocol_error"
                sequence.append(
                    SequenceEvent(
                        kind="error",
                        detail={
                            "interactionId": interaction_id,
                            "stage": "tools/call",
                            "name": tool_name,
                            "message": str(exc),
                            "exceptionType": type(exc).__name__,
                        },
                        latency_ms=latency,
                    )
                )
                errors.append(
                    {
                        "stage": "tools/call",
                        "tool": tool_name,
                        "isError": True,
                        "message": str(exc),
                    }
                )
                output["invocations"].append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "isError": True,
                        "message": str(exc),
                    }
                )
            return
        latency = _elapsed_ms(started)
        counters["tool_call_ms"] += latency
        parsed = _parse_tool_result(result)
        if result.is_error:
            counters["failed_tool_calls"] += 1
            if not _had_sampling_rejection(sequence):
                counters["termination_reason"] = "protocol_error"
        else:
            counters["successful_tool_calls"] += 1
        sequence.append(
            SequenceEvent(
                kind="tool_call_response",
                detail={
                    "interactionId": interaction_id,
                    "method": "tools/call",
                    "name": tool_name,
                    "arguments": arguments,
                    "isError": result.is_error,
                    "result": parsed,
                },
                latency_ms=latency,
            )
        )
        output["invocations"].append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "isError": result.is_error,
                "result": parsed.get("structuredContent") or parsed,
            }
        )
        return

    raise ValueError(f"Unknown step kind '{step.kind}'")


async def run_case_async(
    case: MeasuredCase,
    *,
    settings: Settings,
    server: MCPServer | None = None,
    sampling_mode: str | None = None,
) -> McpRunResult:
    """Execute one composed MCP workflow across the official protocol boundary."""
    mcp_server = server or build_server(settings.data_dir)
    mode = sampling_mode or case.sampling_mode
    sequence: list[SequenceEvent] = []
    errors: list[dict[str, Any]] = []
    discovered_resources: list[DiscoveredResource] = []
    discovered_prompts: list[DiscoveredPrompt] = []
    discovered_tools: list[DiscoveredTool] = []
    output: dict[str, Any] = {
        "reads": [],
        "gets": [],
        "invocations": [],
        "sampling": [],
    }
    protocol_version: str | None = None
    server_name: str | None = None

    total_start = time.perf_counter()
    initialize_ms = 0
    sampling_ms = 0
    sampling_requests = 0
    successful_samplings = 0
    failed_samplings = 0
    counters: dict[str, Any] = {
        "discovery_ms": 0,
        "resource_read_ms": 0,
        "prompt_get_ms": 0,
        "tool_call_ms": 0,
        "resources_read": 0,
        "prompts_requested": 0,
        "tool_calls": 0,
        "successful_tool_calls": 0,
        "failed_tool_calls": 0,
        "termination_reason": "session_closed",
    }

    inner_callback = build_sampling_callback(mode, settings=settings)

    async def sampling_callback(
        context: ClientRequestContext,
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData:
        nonlocal sampling_ms, sampling_requests, successful_samplings, failed_samplings
        sampling_requests += 1
        interaction_id = f"sampling/createMessage:{sampling_requests}"
        request_payload = _dump(params)
        sequence.append(
            SequenceEvent(
                kind="sampling_request",
                detail={
                    "interactionId": interaction_id,
                    "method": "sampling/createMessage",
                    "params": request_payload,
                    "boundary": "mcp-client-sampling-callback",
                },
            )
        )
        started = time.perf_counter()
        result = await inner_callback(context, params)
        latency = _elapsed_ms(started)
        sampling_ms += latency
        if isinstance(result, types.ErrorData):
            failed_samplings += 1
            error_payload = _dump(result)
            sequence.append(
                SequenceEvent(
                    kind="sampling_response",
                    detail={
                        "interactionId": interaction_id,
                        "method": "sampling/createMessage",
                        "isError": True,
                        "error": error_payload,
                        "boundary": "mcp-client-sampling-callback",
                    },
                    latency_ms=latency,
                )
            )
            output["sampling"].append({"isError": True, "error": error_payload})
            errors.append(
                {
                    "stage": "sampling/createMessage",
                    "isError": True,
                    "detail": error_payload,
                }
            )
            return result
        successful_samplings += 1
        result_payload = _dump(result)
        sequence.append(
            SequenceEvent(
                kind="sampling_response",
                detail={
                    "interactionId": interaction_id,
                    "method": "sampling/createMessage",
                    "isError": False,
                    "result": result_payload,
                    "boundary": "mcp-client-sampling-callback",
                },
                latency_ms=latency,
            )
        )
        output["sampling"].append({"isError": False, "result": result_payload})
        return result

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
                "samplingCallbackRegistered": True,
                "samplingMode": mode,
            },
        )
    )

    async with Client(
        mcp_server,
        mode=settings.mcp_client_mode,
        client_info=client_info,
        sampling_callback=sampling_callback,
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

        for index, step in enumerate(case.steps):
            await _run_step(
                client,
                step,
                index=index,
                sequence=sequence,
                errors=errors,
                output=output,
                discovered_resources=discovered_resources,
                discovered_prompts=discovered_prompts,
                discovered_tools=discovered_tools,
                counters=counters,
            )

    if failed_samplings and counters["termination_reason"] == "session_closed":
        counters["termination_reason"] = "sampling_rejected"

    sequence.append(
        SequenceEvent(
            kind="termination",
            detail={"reason": counters["termination_reason"]},
        )
    )

    return McpRunResult(
        case_id=case.trace_id,
        example_class=case.example_class,
        transport=TRANSPORT_LABEL,
        protocol_version=protocol_version,
        server_name=server_name,
        sampling_mode=mode,
        discovered_resources=discovered_resources,
        discovered_prompts=discovered_prompts,
        discovered_tools=discovered_tools,
        sequence=sequence,
        metrics=McpRunMetrics(
            total_ms=_elapsed_ms(total_start),
            initialize_ms=initialize_ms,
            discovery_ms=int(counters["discovery_ms"]),
            resource_read_ms=int(counters["resource_read_ms"]),
            prompt_get_ms=int(counters["prompt_get_ms"]),
            tool_call_ms=int(counters["tool_call_ms"]),
            sampling_ms=sampling_ms,
            resources_discovered=len(discovered_resources),
            resources_read=int(counters["resources_read"]),
            prompts_discovered=len(discovered_prompts),
            prompts_requested=int(counters["prompts_requested"]),
            tools_discovered=len(discovered_tools),
            tool_calls=int(counters["tool_calls"]),
            successful_tool_calls=int(counters["successful_tool_calls"]),
            failed_tool_calls=int(counters["failed_tool_calls"]),
            sampling_requests=sampling_requests,
            successful_samplings=successful_samplings,
            failed_samplings=failed_samplings,
            model_turns=successful_samplings,
            termination_reason=str(counters["termination_reason"]),
        ),
        output=output,
        errors=errors,
    )


def run_case(
    case: MeasuredCase,
    *,
    settings: Settings,
    server: MCPServer | None = None,
    sampling_mode: str | None = None,
) -> McpRunResult:
    import asyncio

    return asyncio.run(
        run_case_async(
            case,
            settings=settings,
            server=server,
            sampling_mode=sampling_mode,
        )
    )
