"""Deterministic local tools — schemas + handlers.

Tool execution never calls paid external APIs. Results are derived from
local JSON fixtures under data/.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.schemas import ToolDefinition, ToolParameterProperty, ToolParameters

Handler = Callable[[BaseModel], dict[str, Any]]


class GetServiceStatusArgs(BaseModel):
    service: str = Field(description="Canonical service name: billing, payments, auth")

    @field_validator("service")
    @classmethod
    def normalize(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("service must be a non-empty string")
        return cleaned


class GetUserProfileArgs(BaseModel):
    user_id: str = Field(description="User id, e.g. u-1001")

    @field_validator("user_id")
    @classmethod
    def normalize(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("user_id must be a non-empty string")
        return cleaned


class SearchDocumentationArgs(BaseModel):
    query: str = Field(description="Short keyword query over internal docs")

    @field_validator("query")
    @classmethod
    def normalize(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("query must be at least 2 characters")
        return cleaned


class ToolSpec:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Handler,
        definition: ToolDefinition,
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.handler = handler
        self.definition = definition


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


class ToolRegistry:
    """Allowlisted tools with typed args and deterministic handlers."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.services: dict[str, dict[str, Any]] = _load_json(
            data_dir / "services.json"
        )
        self.users: dict[str, dict[str, Any]] = _load_json(data_dir / "users.json")
        self.docs: list[dict[str, Any]] = _load_json(data_dir / "docs.json")
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            name="get_service_status",
            description=(
                "Return the current status for a canonical service name "
                "(billing, payments, or auth)."
            ),
            args_model=GetServiceStatusArgs,
            handler=self._get_service_status,
            properties={
                "service": ToolParameterProperty(
                    type="string",
                    description="Canonical service name: billing, payments, auth",
                ),
            },
            required=["service"],
        )
        self.register(
            name="get_user_profile",
            description="Return a user profile by user id (e.g. u-1001).",
            args_model=GetUserProfileArgs,
            handler=self._get_user_profile,
            properties={
                "user_id": ToolParameterProperty(
                    type="string",
                    description="User id such as u-1001",
                ),
            },
            required=["user_id"],
        )
        self.register(
            name="search_documentation",
            description="Search internal documentation snippets by keyword query.",
            args_model=SearchDocumentationArgs,
            handler=self._search_documentation,
            properties={
                "query": ToolParameterProperty(
                    type="string",
                    description="Keyword query over internal docs",
                ),
            },
            required=["query"],
        )

    def register(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Handler,
        properties: dict[str, ToolParameterProperty],
        required: list[str],
    ) -> None:
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=ToolParameters(
                properties=properties,
                required=required,
            ),
        )
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            args_model=args_model,
            handler=handler,
            definition=definition,
        )

    def names(self) -> set[str]:
        return set(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [spec.definition for spec in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI Chat Completions tool schema list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters.model_dump(by_alias=True),
                },
            }
            for d in self.definitions()
        ]

    def parse_arguments(self, name: str, raw: dict[str, Any]) -> BaseModel:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(name)
        try:
            return spec.args_model.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    def _get_service_status(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, GetServiceStatusArgs)
        record = self.services.get(args.service)
        if record is None:
            return {
                "ok": False,
                "error": {
                    "code": "unknown_service",
                    "message": f"Unknown service '{args.service}'",
                    "validServices": sorted(self.services),
                },
            }
        return {"ok": True, "service": record}

    def _get_user_profile(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, GetUserProfileArgs)
        record = self.users.get(args.user_id)
        if record is None:
            return {
                "ok": False,
                "error": {
                    "code": "unknown_user",
                    "message": f"Unknown user_id '{args.user_id}'",
                    "hint": "User ids look like u-1001",
                },
            }
        return {"ok": True, "profile": record}

    def _search_documentation(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, SearchDocumentationArgs)
        query_tokens = _tokenize(args.query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.docs:
            hay = _tokenize(f"{doc['title']} {doc['body']}")
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
            "query": args.query,
            "hitCount": len(hits),
            "hits": hits,
        }


def build_registry(data_dir: Path) -> ToolRegistry:
    return ToolRegistry(data_dir)
