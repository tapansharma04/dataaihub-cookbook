"""Build a grounded prompt and call the chat model."""

from __future__ import annotations

from openai import OpenAI

from rag.fusion import FusedChunk
from rag.reranker import RerankedChunk
from rag.store import RankedChunk

SYSTEM_PROMPT = """You are a helpful assistant that answers using only the
provided context. If the context does not contain enough information,
say you do not know based on the available documents. Be concise."""


def build_prompt(
    question: str,
    chunks: list[RankedChunk] | list[FusedChunk] | list[RerankedChunk],
) -> str:
    if not chunks:
        context = "(no documents retrieved)"
    else:
        parts = []
        for i, item in enumerate(chunks, start=1):
            if isinstance(item, RerankedChunk):
                meta = (
                    f"rerank={item.reranker_score:.4f}, id={item.chunk.id}, "
                    f"prev_rank={item.previous_rank}, rrf={item.rrf_score:.6f}, "
                    f"dense_rank={item.dense_rank}, lexical_rank={item.lexical_rank}"
                )
            elif isinstance(item, FusedChunk):
                meta = (
                    f"rrf={item.rrf_score:.6f}, id={item.chunk.id}, "
                    f"dense_rank={item.dense_rank}, lexical_rank={item.lexical_rank}"
                )
            else:
                meta = f"score={item.score:.3f}, id={item.chunk.id}"
            parts.append(f"[{i}] ({meta})\n{item.chunk.text}")
        context = "\n\n".join(parts)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def generate_answer(
    client: OpenAI,
    question: str,
    chunks: list[RankedChunk] | list[FusedChunk] | list[RerankedChunk],
    *,
    model: str,
) -> str:
    user_prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content or ""
