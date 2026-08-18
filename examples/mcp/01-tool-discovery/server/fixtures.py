"""Deterministic local fixtures for MCP server tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


class FixtureStore:
    """Read-only fixture access for MCP tool handlers."""

    def __init__(self, data_dir: Path) -> None:
        self.services: dict[str, dict[str, Any]] = load_json(data_dir / "services.json")
        self.users: dict[str, dict[str, Any]] = load_json(data_dir / "users.json")
        self.docs: list[dict[str, Any]] = load_json(data_dir / "docs.json")

    def get_service_status(self, service: str) -> dict[str, Any]:
        record = self.services.get(service)
        if record is None:
            return {
                "ok": False,
                "error": {
                    "code": "unknown_service",
                    "message": f"Unknown service '{service}'",
                    "validServices": sorted(self.services),
                },
            }
        return {"ok": True, "service": record}

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        record = self.users.get(user_id)
        if record is None:
            return {
                "ok": False,
                "error": {
                    "code": "unknown_user",
                    "message": f"Unknown user_id '{user_id}'",
                    "hint": "User ids look like u-1001",
                },
            }
        return {"ok": True, "profile": record}

    def search_documentation(self, query: str) -> dict[str, Any]:
        query_tokens = tokenize(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.docs:
            hay = tokenize(f"{doc['title']} {doc['body']}")
            score = len(query_tokens & hay)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        hits = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "snippet": doc["body"],
                "score": score,
            }
            for score, doc in scored[:3]
        ]
        return {
            "ok": True,
            "query": query,
            "hitCount": len(hits),
            "hits": hits,
        }
