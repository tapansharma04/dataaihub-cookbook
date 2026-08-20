"""Configuration for the SPARQL updates example.

Stable example ID: sparql-updates
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "sparql-updates"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "graph.ttl"

# Application-owned ceiling on verification result rows.
DEFAULT_MAX_RESULT_ROWS = 256

# Prohibited SPARQL keywords for predefined measured updates/queries.
PROHIBITED_SPARQL_KEYWORDS: frozenset[str] = frozenset(
    {
        "SERVICE",
        "LOAD",
        "FROM NAMED",
        "FROM <",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR
    graph_path: Path = DEFAULT_GRAPH_PATH
    max_result_rows: int = Field(
        default=DEFAULT_MAX_RESULT_ROWS,
        ge=1,
        le=10_000,
        description=(
            "Hard ceiling on verification SELECT result rows. "
            "Requests cannot exceed this."
        ),
    )


def get_settings() -> Settings:
    return Settings()
