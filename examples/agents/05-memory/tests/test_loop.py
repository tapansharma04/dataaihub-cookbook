"""Memory loop tests — scripted model, real store, no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path

from agent.cases import get_case, scripted_client_for
from agent.loop import classify_decision, run_memory_loop
from agent.memory import FixedClock, MemoryStore
from agent.model import ScriptedModelClient, ScriptedTurn
from agent.schemas import ModelTurn
from agent.source import AuthoritativeStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _known_scopes() -> set[str]:
    users = json.loads((DATA / "users.json").read_text(encoding="utf-8"))
    return set(users)


def _run(case_id: str):
    case = get_case(case_id)
    max_turns = case.max_turns if case.max_turns is not None else 6
    return run_memory_loop(
        interactions=[item.request for item in case.interactions],
        scope=case.scope,
        model=scripted_client_for(case),
        memory_store=MemoryStore(known_scopes=_known_scopes(), clock=FixedClock()),
        authoritative=AuthoritativeStore.from_data_dir(DATA),
        max_turns=max_turns,
    )


def test_no_memory_is_a_miss_not_an_error():
    result = _run("no-memory-notification-preference")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.memory_reads == 1
    assert result.metrics.memory_hits == 0
    assert result.metrics.memory_misses == 1
    assert result.metrics.memory_writes == 0
    kinds = [e.kind for e in result.sequence]
    assert "memory_not_found" in kinds
    assert "memory_store_error" not in kinds
    assert "error" not in kinds
    assert kinds[-2:] == ["final_answer", "termination"]
    answer = result.answer.lower()
    assert "not invent" in answer or "no stored" in answer
    assert "email" not in answer
    assert "sms" not in answer


def test_store_persists_user_provided_record():
    result = _run("store-email-notification-preference")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.memory_writes == 1
    assert result.metrics.memory_scope == "u-1001"
    assert result.metrics.memory_version == 1
    kinds = [e.kind for e in result.sequence]
    assert kinds[0] == "user_request"
    write_req = kinds.index("memory_write_requested")
    stored = kinds.index("memory_stored")
    assert write_req < stored
    event = next(e for e in result.sequence if e.kind == "memory_stored")
    assert event.detail["scope"] == "u-1001"
    assert event.detail["key"] == "notification_channel"
    assert event.detail["value"]["channel"] == "email"
    assert event.detail["source"] == "user"
    assert event.detail["version"] == 1
    records = result.state["memoryRecords"]
    assert len(records) == 1
    assert records[0]["source"] == "user"


def test_recall_uses_information_from_interaction_1():
    result = _run("recall-email-notification-preference")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.memory_writes == 1
    assert result.metrics.memory_reads == 1
    assert result.metrics.memory_hits == 1
    assert result.metrics.memory_misses == 0
    assert result.metrics.stale_memory_detected is False

    interactions = result.state["interactions"]
    assert [item["id"] for item in interactions] == [
        "interaction-1",
        "interaction-2",
    ]
    second_request = interactions[1]["request"]
    assert "email" not in second_request.lower()
    assert "sms" not in second_request.lower()

    stored = next(e for e in result.sequence if e.kind == "memory_stored")
    retrieved = next(e for e in result.sequence if e.kind == "memory_retrieved")
    assert stored.interaction_id == "interaction-1"
    assert retrieved.interaction_id == "interaction-2"
    assert retrieved.detail["record"]["value"]["channel"] == "email"
    assert retrieved.detail["record"]["id"] == stored.detail["id"]
    assert "email" in result.answer.lower()

    user_requests = [e for e in result.sequence if e.kind == "user_request"]
    assert len(user_requests) == 2
    assert user_requests[0].interaction_id == "interaction-1"
    assert user_requests[1].interaction_id == "interaction-2"


def test_stale_memory_prefers_current_source():
    result = _run("stale-memory-notification-preference")
    assert result.metrics.termination_reason == "final_answer"
    assert result.metrics.stale_memory_detected is True
    assert result.metrics.memory_hits == 1
    kinds = [e.kind for e in result.sequence]
    retrieved = kinds.index("memory_retrieved")
    observed = kinds.index("observation")
    assert retrieved < observed
    observation = next(e for e in result.sequence if e.kind == "observation")
    assert observation.detail["kind"] == "current_source"
    assert observation.detail["staleMemoryDetected"] is True
    assert observation.detail["resolution"] == "current_source_preferred"
    assert observation.detail["current"]["channel"] == "sms"
    assert observation.detail["current"]["version"] == 2
    retrieved_event = next(e for e in result.sequence if e.kind == "memory_retrieved")
    assert retrieved_event.detail["record"]["value"]["channel"] == "email"
    assert retrieved_event.detail["record"]["version"] == 1
    answer = result.answer.lower()
    assert "sms" in answer
    assert "version 2" in answer or "version 1" in answer
    assert result.metrics.termination_reason != "error"


def test_cross_scope_proposal_is_rejected():
    model = ScriptedModelClient(
        [
            ScriptedTurn(
                content="store for another user",
                decision="store_memory",
                memory_write={
                    "scope": "u-1002",
                    "key": "notification_channel",
                    "value": {"channel": "email"},
                    "source": "user",
                },
            )
        ]
    )
    result = run_memory_loop(
        interactions=["I prefer email notifications."],
        scope="u-1001",
        model=model,
        memory_store=MemoryStore(known_scopes=_known_scopes(), clock=FixedClock()),
        authoritative=AuthoritativeStore.from_data_dir(DATA),
        max_turns=6,
    )
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.memory_writes == 0
    assert result.state["memoryRecords"] == []


def test_unrecognized_decision_is_invalid_action():
    model = ScriptedModelClient(
        [ScriptedTurn(content="keep going", decision="continue")]
    )
    result = run_memory_loop(
        interactions=["How should I be notified?"],
        scope="u-1002",
        model=model,
        memory_store=MemoryStore(known_scopes=_known_scopes(), clock=FixedClock()),
        authoritative=AuthoritativeStore.from_data_dir(DATA),
        max_turns=6,
    )
    assert result.metrics.termination_reason == "invalid_action"
    assert result.metrics.memory_writes == 0


def test_metrics_are_measured_non_negative():
    result = _run("store-email-notification-preference")
    m = result.metrics
    assert m.provenance == "measured"
    assert m.total_ms >= 0
    assert m.model_ms >= 0
    assert m.tool_ms >= 0
    assert "memoryQuality" not in m.model_dump()


def test_classify_decision_rejects_unknown():
    turn = ModelTurn(content="keep going", decision="invalid_action")
    assert classify_decision(turn) == "invalid_action"


def test_repeated_execution_is_semantically_stable():
    first = _run("recall-email-notification-preference")
    second = _run("recall-email-notification-preference")
    left = [
        (
            e.kind,
            e.interaction_id,
            e.detail.get("key"),
            (e.detail.get("record") or {}).get("value"),
            e.detail.get("answer"),
        )
        for e in first.sequence
    ]
    right = [
        (
            e.kind,
            e.interaction_id,
            e.detail.get("key"),
            (e.detail.get("record") or {}).get("value"),
            e.detail.get("answer"),
        )
        for e in second.sequence
    ]
    assert left == right
    assert first.metrics.termination_reason == second.metrics.termination_reason
    assert first.answer == second.answer
    assert first.state["memoryRecords"] == second.state["memoryRecords"]
