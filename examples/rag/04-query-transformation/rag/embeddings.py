"""Embedding client (OpenAI / OpenAI-compatible)."""

from openai import OpenAI

from config import Settings


def get_client(settings: Settings) -> OpenAI:
    kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input string."""
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


def embed_query(client: OpenAI, query: str, model: str) -> list[float]:
    return embed_texts(client, [query], model)[0]
