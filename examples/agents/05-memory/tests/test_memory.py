"""Memory store, retrieve, scope, provenance, and freshness tests."""

from __future__ import annotations

import pytest

from agent.memory import (
    FixedClock,
    MemoryStore,
    MemoryValidationError,
    assess_freshness,
)
from agent.source import AuthoritativeStore

KNOWN = {"u-1001", "u-1002", "u-1003"}


def _store() -> MemoryStore:
    return MemoryStore(known_scopes=KNOWN, clock=FixedClock())


def test_store_valid_record():
    store = _store()
    record = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    assert record.id == "mem-u-1001-notification_channel-v1"
    assert record.scope == "u-1001"
    assert record.key == "notification_channel"
    assert record.value == {"channel": "email"}
    assert record.source == "user"
    assert record.version == 1
    assert record.created_at == "2026-08-01T12:00:00Z"
    assert record.updated_at == "2026-08-01T12:00:00Z"


def test_store_preserves_scope_provenance_and_version():
    store = _store()
    record = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    payload = record.to_public_dict()
    assert payload["scope"] == "u-1001"
    assert payload["source"] == "user"
    assert payload["version"] == 1
    assert payload["createdAt"]
    assert payload["updatedAt"]


def test_retrieve_by_scope_and_key():
    store = _store()
    stored = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    found = store.retrieve(scope="u-1001", key="notification_channel")
    assert found is not None
    assert found.id == stored.id
    assert found.value == {"channel": "email"}


def test_missing_memory_is_explicit_none():
    store = _store()
    found = store.retrieve(scope="u-1002", key="notification_channel")
    assert found is None


def test_scope_isolation():
    store = _store()
    store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    other = store.retrieve(scope="u-1002", key="notification_channel")
    assert other is None
    own = store.retrieve(scope="u-1001", key="notification_channel")
    assert own is not None
    assert own.scope == "u-1001"


def test_session_scope_rejects_cross_user_write():
    store = _store()
    with pytest.raises(MemoryValidationError, match="outside the session scope"):
        store.store(
            scope="u-1002",
            key="notification_channel",
            value={"channel": "sms"},
            source="user",
            session_scope="u-1001",
        )


def test_session_scope_rejects_cross_user_read():
    store = _store()
    store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
        session_scope="u-1001",
    )
    with pytest.raises(MemoryValidationError, match="outside the session scope"):
        store.retrieve(
            scope="u-1001",
            key="notification_channel",
            session_scope="u-1002",
        )


def test_untrusted_source_is_rejected():
    store = _store()
    with pytest.raises(MemoryValidationError, match="trusted write provenance"):
        store.store(
            scope="u-1001",
            key="notification_channel",
            value={"channel": "email"},
            source="tool",
        )


def test_unknown_key_is_rejected():
    store = _store()
    with pytest.raises(MemoryValidationError, match="not allowlisted"):
        store.store(
            scope="u-1001",
            key="favorite_color",
            value={"channel": "email"},
            source="user",
        )


def test_update_increments_version():
    store = _store()
    first = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    updated = store.update(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "sms"},
        source="user",
    )
    assert first.version == 1
    assert updated.version == 2
    assert updated.created_at == first.created_at
    assert updated.value == {"channel": "sms"}


def test_stale_when_current_source_differs():
    store = _store()
    record = store.store(
        scope="u-1003",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    current = {
        "channel": "sms",
        "version": 2,
        "updatedAt": "2026-08-18T09:00:00Z",
        "source": "system",
    }
    assessment = assess_freshness(record, current)
    assert assessment.stale is True
    assert assessment.resolution == "current_source_preferred"
    assert assessment.memory_version == 1
    assert assessment.current_source_version == 2
    assert assessment.memory_value == {"channel": "email"}
    assert assessment.current_value == {"channel": "sms"}


def test_freshness_without_current_source_uses_memory():
    store = _store()
    record = store.store(
        scope="u-1001",
        key="notification_channel",
        value={"channel": "email"},
        source="user",
    )
    assessment = assess_freshness(record, None)
    assert assessment.stale is False
    assert assessment.resolution == "memory_used"


def test_authoritative_store_is_scoped():
    authoritative = AuthoritativeStore(
        {
            "u-1003": {
                "notification_channel": {
                    "channel": "sms",
                    "version": 2,
                    "updatedAt": "2026-08-18T09:00:00Z",
                    "source": "system",
                }
            }
        }
    )
    found = authoritative.get("u-1003", "notification_channel")
    assert found is not None
    assert found["channel"] == "sms"
    assert found["version"] == 2
    assert authoritative.get("u-1001", "notification_channel") is None
