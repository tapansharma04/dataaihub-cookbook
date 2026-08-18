"""Configuration for the agent-memory example.

Stable example ID: agent-memory
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "agent-memory"

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

    # Runtime bounds (application-enforced, not model-enforced).
    max_turns: int = 6
    tool_timeout_ms: int = 2000

    data_dir: Path = DEFAULT_DATA_DIR


def get_settings() -> Settings:
    return Settings()
