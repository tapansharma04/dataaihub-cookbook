"""Configuration for the graph-construction example.

Stable example ID: graph-construction
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "graph-construction"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph.ttl"
DEFAULT_SOURCES_PATH = DEFAULT_DATA_DIR / "sources.json"

# Fixed timestamp for deterministic structured-mode traces.
FIXED_RECORDED_AT = "2026-01-01T00:00:00Z"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    graph_path: Path = DEFAULT_GRAPH_PATH
    sources_path: Path = DEFAULT_SOURCES_PATH
    openai_api_key: str = Field(default="", description="OpenAI API key for LLM mode.")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model for LLM-assisted extraction.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL.",
    )


def get_settings() -> Settings:
    return Settings()
