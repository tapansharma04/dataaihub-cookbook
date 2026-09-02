"""Client-owned sampling handlers.

The MCP server requests generation through sampling/createMessage.
These callbacks are the client boundary: mock completion, controlled
rejection, or an optional live model provider. The server does not import
a model SDK.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import mcp_types as types
from mcp.client import ClientRequestContext
from mcp.types import CreateMessageResult, TextContent

from config import Settings

SamplingCallback = Callable[
    [ClientRequestContext, types.CreateMessageRequestParams],
    Awaitable[CreateMessageResult | types.ErrorData],
]

MOCK_MODEL = "mock"
SAMPLING_REJECT_MESSAGE = "Sampling rejected by client"
INVALID_REQUEST = -32600


def _message_texts(params: types.CreateMessageRequestParams) -> list[str]:
    texts: list[str] = []
    for message in params.messages:
        blocks = message.content_as_list
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return texts


def mock_complete(params: types.CreateMessageRequestParams) -> CreateMessageResult:
    """Deterministic mock completion from the observed sampling request."""
    excerpts = _message_texts(params)
    grounded = "\n---\n".join(excerpts)
    text = (
        "[mock] Client sampling callback completed this request.\n"
        f"Grounded context ({len(excerpts)} message(s)):\n"
        f"{grounded}"
    )
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model=MOCK_MODEL,
    )


def reject_sampling(
    _context: ClientRequestContext,
    _params: types.CreateMessageRequestParams,
) -> types.ErrorData:
    return types.ErrorData(
        code=INVALID_REQUEST,
        message=SAMPLING_REJECT_MESSAGE,
    )


async def mock_sampling_callback(
    _context: ClientRequestContext,
    params: types.CreateMessageRequestParams,
) -> CreateMessageResult:
    return mock_complete(params)


async def reject_sampling_callback(
    context: ClientRequestContext,
    params: types.CreateMessageRequestParams,
) -> types.ErrorData:
    return reject_sampling(context, params)


def _to_provider_messages(
    params: types.CreateMessageRequestParams,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if params.system_prompt:
        messages.append({"role": "system", "content": params.system_prompt})
    for message in params.messages:
        texts = [
            block.text
            for block in message.content_as_list
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        messages.append({"role": message.role, "content": "\n".join(texts)})
    return messages


async def live_sampling_callback(
    _context: ClientRequestContext,
    params: types.CreateMessageRequestParams,
    *,
    settings: Settings,
) -> CreateMessageResult:
    """Call a configured model provider from the client sampling callback."""
    if not settings.openai_api_key:
        raise RuntimeError("Live sampling requires OPENAI_API_KEY")

    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    async with AsyncOpenAI(**kwargs) as client:
        completion = await client.chat.completions.create(
            model=settings.openai_model,
            messages=_to_provider_messages(params),
            max_tokens=params.max_tokens,
        )
    if not completion.choices:
        raise RuntimeError("Live sampling returned no choices")
    choice = completion.choices[0]
    text = choice.message.content or ""
    result_kwargs: dict[str, Any] = {
        "role": "assistant",
        "content": TextContent(type="text", text=text),
        "model": completion.model,
    }
    if choice.finish_reason:
        result_kwargs["stop_reason"] = choice.finish_reason
    return CreateMessageResult(**result_kwargs)


def build_sampling_callback(
    mode: str,
    *,
    settings: Settings,
) -> SamplingCallback:
    if mode == "reject":
        return reject_sampling_callback
    if mode == "live":

        async def _live(
            context: ClientRequestContext,
            params: types.CreateMessageRequestParams,
        ) -> CreateMessageResult:
            return await live_sampling_callback(context, params, settings=settings)

        return _live
    if mode == "mock":
        return mock_sampling_callback
    raise ValueError(f"Unknown sampling mode '{mode}'")
