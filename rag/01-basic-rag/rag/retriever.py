"""Retrieve top-k chunks for a query."""

from openai import OpenAI

from rag.embeddings import embed_query
from rag.store import InMemoryVectorStore, ScoredChunk


def retrieve(
    client: OpenAI,
    store: InMemoryVectorStore,
    query: str,
    *,
    embedding_model: str,
    top_k: int,
) -> list[ScoredChunk]:
    query_vector = embed_query(client, query, embedding_model)
    return store.search(query_vector, top_k=top_k)
