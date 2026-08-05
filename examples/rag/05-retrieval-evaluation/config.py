"""Configuration for the retrieval-evaluation example."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "retrieval-evaluation"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "sample.md"
DEFAULT_EVAL_PATH = ROOT / "data" / "eval_queries.json"

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


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

    chunk_size: int = 520
    chunk_overlap: int = 40

    # Match examples 03/04 candidate breadth so comparisons stay meaningful.
    dense_top_k: int = 8
    lexical_top_k: int = 8
    candidate_k: int = 5
    max_alternative_queries: int = 2
    query_transformer_model: str = "gpt-4o-mini"

    # Evaluation / final ranking depth (same K for all compared pipelines).
    eval_k: int = 3
    rrf_k: int = 60

    reranker_model: str = DEFAULT_RERANKER_MODEL

    data_path: Path = DEFAULT_DATA_PATH
    eval_path: Path = DEFAULT_EVAL_PATH


def get_settings() -> Settings:
    return Settings()
