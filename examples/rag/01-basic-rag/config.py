"""Configuration for the basic-rag example.

Stable example ID: basic-rag
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "basic-rag"

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

    # Character-based chunking keeps the demo free of tokenizer deps.
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = 3

    data_path: Path = DEFAULT_DATA_PATH


def get_settings() -> Settings:
    return Settings()
