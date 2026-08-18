"""Application-owned memory records, store, and freshness comparison.

The model/case harness may propose a store or retrieve. This module is the
only place that writes, reads, or scopes memory. Lookup is by explicit
scope and key — not embeddings or similarity search.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from agent.schemas import FreshnessResolution, MemorySource

ALLOWED_KEYS: frozenset[str] = frozenset({"notification_channel"})
ALLOWED_CHANNELS: frozenset[str] = frozenset({"email", "sms", "slack"})
ALLOWED_SOURCES: frozenset[str] = frozenset({"user", "system", "tool", "application"})
TRUSTED_WRITE_SOURCES: frozenset[str] = frozenset({"user", "system", "application"})


class MemoryValidationError(ValueError):
    """Proposed memory operation failed application validation."""


class Clock(Protocol):
    def now(self) -> str: ...


class FixedClock:
    """Deterministic clock so measured memory timestamps stay stable."""

    def __init__(self, timestamp: str = "2026-08-01T12:00:00Z") -> None:
        self.timestamp = timestamp

    def now(self) -> str:
        return self.timestamp


class MemoryRecord(BaseModel):
    """One application-owned memory record."""

    id: str
    scope: str
    key: str
    value: dict[str, Any]
    created_at: str
    updated_at: str
    source: MemorySource
    version: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "key": self.key,
            "value": dict(self.value),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "source": self.source,
            "version": self.version,
        }


class FreshnessAssessment(BaseModel):
    """Observable comparison between stored memory and the current source."""

    stale: bool
    resolution: FreshnessResolution
    memory_version: int | None = None
    current_source_version: int | None = None
    memory_value: dict[str, Any] | None = None
    current_value: dict[str, Any] | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "stale": self.stale,
            "resolution": self.resolution,
            "memoryVersion": self.memory_version,
            "currentSourceVersion": self.current_source_version,
            "memoryValue": self.memory_value,
            "currentValue": self.current_value,
        }


class MemoryStore:
    """In-process deterministic memory store, scoped by owner identity.

    Records are keyed by (scope, key). A record stored for one scope is
    never returned for another. Missing keys return None — the store does
    not fabricate records.
    """

    def __init__(
        self,
        *,
        known_scopes: set[str],
        clock: Clock | None = None,
        allowed_keys: frozenset[str] = ALLOWED_KEYS,
    ) -> None:
        self.known_scopes = set(known_scopes)
        self.clock = clock or FixedClock()
        self.allowed_keys = allowed_keys
        self._records: dict[tuple[str, str], MemoryRecord] = {}

    def store(
        self,
        *,
        scope: str,
        key: str,
        value: dict[str, Any],
        source: str,
        session_scope: str | None = None,
    ) -> MemoryRecord:
        self._validate_scope(scope, session_scope=session_scope)
        self._validate_key(key)
        validated_value = self._validate_value(value)
        validated_source = self._validate_write_source(source)
        existing = self._records.get((scope, key))
        now = self.clock.now()
        version = 1 if existing is None else existing.version + 1
        record = MemoryRecord(
            id=f"mem-{scope}-{key}-v{version}",
            scope=scope,
            key=key,
            value=validated_value,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            source=validated_source,
            version=version,
        )
        self._records[(scope, key)] = record
        return record

    def retrieve(
        self,
        *,
        scope: str,
        key: str,
        session_scope: str | None = None,
    ) -> MemoryRecord | None:
        self._validate_scope(scope, session_scope=session_scope)
        self._validate_key(key)
        return self._records.get((scope, key))

    def update(
        self,
        *,
        scope: str,
        key: str,
        value: dict[str, Any],
        source: str,
        session_scope: str | None = None,
    ) -> MemoryRecord:
        existing = self.retrieve(scope=scope, key=key, session_scope=session_scope)
        if existing is None:
            raise MemoryValidationError(
                f"Cannot update missing memory for scope={scope!r} key={key!r}"
            )
        return self.store(
            scope=scope,
            key=key,
            value=value,
            source=source,
            session_scope=session_scope,
        )

    def exported_records(self) -> list[dict[str, Any]]:
        return [
            record.to_public_dict()
            for _, record in sorted(self._records.items(), key=lambda item: item[0])
        ]

    def _validate_scope(self, scope: str, *, session_scope: str | None) -> None:
        cleaned = scope.strip()
        if not cleaned:
            raise MemoryValidationError("scope must be a non-empty string")
        if cleaned not in self.known_scopes:
            raise MemoryValidationError(f"unknown scope '{cleaned}'")
        if session_scope is not None and cleaned != session_scope:
            raise MemoryValidationError(
                f"scope '{cleaned}' is outside the session scope '{session_scope}'"
            )

    def _validate_key(self, key: str) -> None:
        cleaned = key.strip()
        if cleaned not in self.allowed_keys:
            raise MemoryValidationError(
                f"key '{key}' is not allowlisted (allowed: {sorted(self.allowed_keys)})"
            )

    def _validate_value(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise MemoryValidationError("value must be an object")
        channel = value.get("channel")
        if not isinstance(channel, str) or channel.strip() not in ALLOWED_CHANNELS:
            raise MemoryValidationError(
                f"value.channel must be one of {sorted(ALLOWED_CHANNELS)}"
            )
        return {"channel": channel.strip()}

    def _validate_write_source(self, source: str) -> MemorySource:
        cleaned = source.strip()
        if cleaned not in ALLOWED_SOURCES:
            raise MemoryValidationError(
                f"source '{source}' is not allowed (allowed: {sorted(ALLOWED_SOURCES)})"
            )
        if cleaned not in TRUSTED_WRITE_SOURCES:
            raise MemoryValidationError(
                f"source '{cleaned}' is not a trusted write provenance. "
                "Model-inferred statements are not stored as memory."
            )
        return cleaned  # type: ignore[return-value]


def assess_freshness(
    record: MemoryRecord,
    current: dict[str, Any] | None,
) -> FreshnessAssessment:
    """Compare stored memory with the current authoritative source.

    Stale memory is not a store failure. It means a current source exists
    and differs from the stored record.
    """
    if current is None:
        return FreshnessAssessment(
            stale=False,
            resolution="memory_used",
            memory_version=record.version,
            memory_value=dict(record.value),
        )
    current_value = {"channel": current["channel"]}
    current_version = int(current["version"])
    values_differ = current_value["channel"] != record.value.get("channel")
    version_ahead = current_version > record.version
    stale = values_differ or version_ahead
    return FreshnessAssessment(
        stale=stale,
        resolution="current_source_preferred" if stale else "memory_matches_current",
        memory_version=record.version,
        current_source_version=current_version,
        memory_value=dict(record.value),
        current_value=current_value,
    )


def parse_memory_write(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MemoryValidationError("memory write proposal must be an object")
    return {
        "scope": raw.get("scope"),
        "key": raw.get("key"),
        "value": raw.get("value") if isinstance(raw.get("value"), dict) else {},
        "source": raw.get("source"),
    }


def parse_memory_read(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MemoryValidationError("memory read proposal must be an object")
    return {
        "scope": raw.get("scope"),
        "key": raw.get("key"),
    }
