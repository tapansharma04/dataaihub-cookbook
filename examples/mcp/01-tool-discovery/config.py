"""Configuration for the MCP tool-discovery example.

Stable example ID: mcp-tool-discovery
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "mcp-tool-discovery"

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
            "with JSON-RPC framing and an explicit initialize handshake."
        ),
    )
    client_name: str = "dataaihub-cookbook-mcp-client"
    client_version: str = "0.1.0"


def get_settings() -> Settings:
    return Settings()
