"""Deterministic catalogs owned by specialist agents.

Status and docs agents read these fixtures. The analysis agent does not.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class Catalog:
    def __init__(self, data_dir: Path) -> None:
        self.services: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "services.json").read_text(encoding="utf-8")
        )
        docs = json.loads((data_dir / "docs.json").read_text(encoding="utf-8"))
        self.docs: dict[str, dict[str, Any]] = {item["id"]: item for item in docs}

    def get_service(self, service: str) -> dict[str, Any] | None:
        record = self.services.get(service)
        return copy.deepcopy(record) if record is not None else None

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        record = self.docs.get(doc_id)
        return copy.deepcopy(record) if record is not None else None
