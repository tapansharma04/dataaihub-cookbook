"""Current authoritative preference source — distinct from memory.

Memory retains information associated with a prior interaction.
This store is the current source of record for notification preferences.
It is not a vector index and is not updated by memory writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AuthoritativeStore:
    """Deterministic local fixture of current notification preferences."""

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> AuthoritativeStore:
        payload = json.loads(
            (data_dir / "preferences.json").read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("preferences.json must be an object keyed by user id")
        return cls(payload)

    def get(self, scope: str, key: str) -> dict[str, Any] | None:
        scoped = self._records.get(scope)
        if not isinstance(scoped, dict):
            return None
        record = scoped.get(key)
        if not isinstance(record, dict):
            return None
        channel = record.get("channel")
        version = record.get("version")
        if not isinstance(channel, str) or not isinstance(version, int):
            return None
        return {
            "scope": scope,
            "key": key,
            "channel": channel,
            "version": version,
            "updatedAt": record.get("updatedAt"),
            "source": record.get("source", "system"),
        }
