"""Configuration for the GraphRAG example.

Stable example ID: graphrag
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "graphrag"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph.ttl"

# Application-owned retrieval boundary.
DEFAULT_MAX_HOPS = 2

Mode = str  # "graph_grounded" | "graphrag_llm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    graph_path: Path = DEFAULT_GRAPH_PATH
    max_hops: int = Field(
        default=DEFAULT_MAX_HOPS,
        ge=1,
        le=5,
        description="Maximum graph hops during bounded retrieval.",
    )
    openai_api_key: str = Field(default="", description="OpenAI API key for LLM mode.")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model for LLM mode. Not hard-coded at call sites.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL.",
    )


def get_settings() -> Settings:
    return Settings()
