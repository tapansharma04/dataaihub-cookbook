"""Deterministic catalogs owned by specialist agents.

Status receives only the status store. Docs receives only the docs store.
Downstream agents cannot reach another specialist's store.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class StatusStore:
    """Service status records. No documentation access."""

    def __init__(self, services: dict[str, dict[str, Any]]) -> None:
        self._services = services

    def get_service(self, service: str) -> dict[str, Any] | None:
        record = self._services.get(service)
        return copy.deepcopy(record) if record is not None else None

    @property
    def ids(self) -> list[str]:
        return sorted(self._services)


class DocsStore:
    """Documentation records. No status access."""

    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self._docs = docs

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        record = self._docs.get(doc_id)
        return copy.deepcopy(record) if record is not None else None

    @property
    def ids(self) -> list[str]:
        return sorted(self._docs)


class Catalog:
    def __init__(self, data_dir: Path) -> None:
        self.services: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "services.json").read_text(encoding="utf-8")
        )
        docs = json.loads((data_dir / "docs.json").read_text(encoding="utf-8"))
        self.docs: dict[str, dict[str, Any]] = {item["id"]: item for item in docs}

    def status_store(self) -> StatusStore:
        return StatusStore(self.services)

    def docs_store(self) -> DocsStore:
        return DocsStore(self.docs)

    def get_service(self, service: str) -> dict[str, Any] | None:
        return self.status_store().get_service(service)

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        return self.docs_store().get_doc(doc_id)
