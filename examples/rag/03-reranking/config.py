"""Configuration for the reranking example.

Stable example ID: reranking
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "reranking"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "sample.md"

# Lightweight MS MARCO cross-encoder — see README engineering decisions.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL",
    )

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # Section chunking is primary; these remain for fallback windows / docs.
    chunk_size: int = 520
    chunk_overlap: int = 40

    # Retrieve broadly so RRF can build a candidate pool for reranking.
    dense_top_k: int = 8
    lexical_top_k: int = 8
    candidate_k: int = 8

    # After cross-encoder scoring, keep only the final context window.
    final_context_k: int = 3

    # Conventional RRF constant; not tuned to this corpus.
    rrf_k: int = 60

    reranker_model: str = DEFAULT_RERANKER_MODEL

    data_path: Path = DEFAULT_DATA_PATH


def get_settings() -> Settings:
    return Settings()
