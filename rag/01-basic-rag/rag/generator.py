"""Build a grounded prompt and call the chat model."""

from openai import OpenAI

from rag.store import ScoredChunk

SYSTEM_PROMPT = """You are a helpful assistant that answers using only the
provided context. If the context does not contain enough information,
say you do not know based on the available documents. Be concise."""


def build_prompt(question: str, scored_chunks: list[ScoredChunk]) -> str:
    if not scored_chunks:
        context = "(no documents retrieved)"
    else:
        parts = []
        for i, item in enumerate(scored_chunks, start=1):
            parts.append(
                f"[{i}] (score={item.score:.3f}, id={item.chunk.id})\n{item.chunk.text}"
            )
        context = "\n\n".join(parts)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def generate_answer(
    client: OpenAI,
    question: str,
    scored_chunks: list[ScoredChunk],
    *,
    model: str,
) -> str:
    user_prompt = build_prompt(question, scored_chunks)
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
