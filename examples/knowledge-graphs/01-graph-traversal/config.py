"""Configuration for the graph-traversal example.

Stable example ID: graph-traversal
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "graph-traversal"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph.ttl"

# Application-owned ceiling. A request cannot raise this from a query string.
DEFAULT_MAX_TRAVERSAL_DEPTH = 8

ALLOWED_DIRECTIONS: frozenset[str] = frozenset({"outgoing", "incoming"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    graph_path: Path = DEFAULT_GRAPH_PATH
    max_traversal_depth: int = Field(
        default=DEFAULT_MAX_TRAVERSAL_DEPTH,
        ge=1,
        le=32,
        description="Hard ceiling on traversal hops. Requests cannot exceed this.",
    )


def get_settings() -> Settings:
    return Settings()
