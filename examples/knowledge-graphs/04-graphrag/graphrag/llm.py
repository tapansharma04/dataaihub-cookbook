"""LLM clients for GraphRAG answer synthesis.

The LLM receives only assembled graph context — never the raw RDF graph.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from openai import OpenAI

from config import Settings

SYSTEM_PROMPT = """You answer questions using only the supplied graph context.
Do not use outside knowledge.
Do not infer relationships that are not present in the context.
If the context does not support the answer, say there is insufficient graph evidence.
Provide a concise answer."""


class LLMClient(Protocol):
    model_name: str
    provider: str

    def complete(self, *, question: str, context: list[str]) -> tuple[str, int]: ...


def build_user_prompt(question: str, context: list[str]) -> str:
    if context:
        context_block = "\n".join(f"- {line}" for line in context)
    else:
        context_block = "(no graph context retrieved)"
    return (
        f"Graph context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the graph context above."
    )


def get_openai_client(settings: Settings) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


class OpenAILLMClient:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model_name = model
        self.provider = "openai"

    def complete(self, *, question: str, context: list[str]) -> tuple[str, int]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or ""
        return content.strip(), latency_ms


class MockLLMClient:
    """Deterministic mock for tests — never masquerades as a live provider."""

    def __init__(self) -> None:
        self.model_name = "mock"
        self.provider = "mock"
        self.last_context: list[str] = []
        self.last_question: str = ""

    def complete(self, *, question: str, context: list[str]) -> tuple[str, int]:
        started = time.perf_counter()
        self.last_question = question
        self.last_context = list(context)
        if not context:
            answer = "There is insufficient graph evidence to answer that question."
        else:
            answer = f"[mock grounded answer from {len(context)} fact(s)] " + " ".join(
                context
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return answer, latency_ms


def build_llm_client(settings: Settings, *, use_mock: bool = False) -> LLMClient | None:
    if use_mock:
        return MockLLMClient()
    if not settings.openai_api_key:
        return None
    return OpenAILLMClient(get_openai_client(settings), settings.openai_model)
