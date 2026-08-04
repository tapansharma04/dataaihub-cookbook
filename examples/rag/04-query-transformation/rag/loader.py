"""Load documents from disk."""

from pathlib import Path


def load_document(path: Path) -> str:
    """Read a single UTF-8 text/markdown file."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    return path.read_text(encoding="utf-8")
