"""Configuration for the agent handoffs example.

Stable example ID: agent-handoffs
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "agent-handoffs"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"


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
    chat_model: str = "gpt-4o-mini"
    max_handoffs: int = 6
    data_dir: Path = DEFAULT_DATA_DIR


def get_settings() -> Settings:
    return Settings()
