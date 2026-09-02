"""Configuration for the MCP composition example.

Stable example ID: mcp-composition
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "mcp-composition"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    mcp_client_mode: str = Field(
        default="legacy",
        description=(
            "MCP Client connection mode. 'legacy' uses in-process InMemoryTransport "
            "with JSON-RPC framing and an explicit initialize handshake. Sampling "
            "requires this handshake-era back-channel."
        ),
    )
    client_name: str = "dataaihub-cookbook-mcp-client"
    client_version: str = "0.1.0"
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for optional live sampling export. Unused by CI.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model for live sampling. Not used by the MCP server.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL for live sampling.",
    )


def get_settings() -> Settings:
    return Settings()
