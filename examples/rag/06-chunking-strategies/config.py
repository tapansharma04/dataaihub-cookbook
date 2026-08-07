"""Configuration for the chunking-strategies experiment.

Independent variable: chunking strategy.
Held constant: corpus, embedding model, dense retrieval, evaluation set, K.

Chunk size / overlap units are **characters** (UTF-8 code points as Python
``str`` length), not tokens.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EXAMPLE_ID = "chunking-strategies"

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "sample.md"
DEFAULT_EVAL_PATH = ROOT / "data" / "eval_set.json"

STRATEGY_FIXED = "fixed"
STRATEGY_RECURSIVE = "recursive"
STRATEGY_STRUCTURE = "structure"

ALL_STRATEGIES = (STRATEGY_FIXED, STRATEGY_RECURSIVE, STRATEGY_STRUCTURE)

# Size unit for all chunk_size / overlap settings below.
CHUNK_SIZE_UNIT = "characters"


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

    # Fixed-size strategy (characters). Chosen so typical handbook sections
    # (~220–600 chars) sit near the operating range of structure-aware chunks
    # without forcing identical lengths.
    fixed_chunk_size: int = 400
    fixed_chunk_overlap: int = 50

    # Recursive strategy target size (characters) — comparable operating range.
    recursive_target_size: int = 400
    recursive_chunk_overlap: int = 50

    # Evaluation / retrieval depth (same K for all strategies).
    eval_k: int = 3

    data_path: Path = DEFAULT_DATA_PATH
    eval_path: Path = DEFAULT_EVAL_PATH


def get_settings() -> Settings:
    return Settings()


def strategy_config(settings: Settings) -> dict[str, dict]:
    """Frozen strategy configuration recorded before scoring."""
    return {
        STRATEGY_FIXED: {
            "name": STRATEGY_FIXED,
            "label": "Fixed-size",
            "chunk_size": settings.fixed_chunk_size,
            "chunk_overlap": settings.fixed_chunk_overlap,
            "size_unit": CHUNK_SIZE_UNIT,
        },
        STRATEGY_RECURSIVE: {
            "name": STRATEGY_RECURSIVE,
            "label": "Recursive",
            "target_size": settings.recursive_target_size,
            "chunk_overlap": settings.recursive_chunk_overlap,
            "size_unit": CHUNK_SIZE_UNIT,
            "separators": [
                "\n## ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        },
        STRATEGY_STRUCTURE: {
            "name": STRATEGY_STRUCTURE,
            "label": "Structure-aware",
            "boundary": "markdown ## headings (section = chunk)",
            "size_unit": CHUNK_SIZE_UNIT,
            "note": (
                "No fixed length target; preserves heading + section body. "
                "Section lengths vary with the source handbook."
            ),
        },
    }
