"""Configuration for the hybrid-rag example.

Stable example ID: hybrid-rag
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "hybrid-rag"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "sample.md"


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

    dense_top_k: int = 3
    lexical_top_k: int = 3
    hybrid_top_k: int = 3

    # Conventional RRF constant; not tuned to this corpus.
    rrf_k: int = 60

    data_path: Path = DEFAULT_DATA_PATH


def get_settings() -> Settings:
    return Settings()
