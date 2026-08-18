"""Schema serialization smoke tests."""

from __future__ import annotations

from agent.schemas import AgentRunMetrics, MemoryOperationDefinition, SequenceEvent


def test_memory_operation_definition_roundtrip():
    definition = MemoryOperationDefinition(
        name="store_memory",
        description="Store a record",
        parameters={"type": "object"},
    )
    payload = definition.model_dump()
    again = MemoryOperationDefinition.model_validate(payload)
    assert again.name == "store_memory"


def test_sequence_event_carries_interaction_id():
    event = SequenceEvent(
        kind="memory_stored",
        turn=1,
        interaction_id="interaction-1",
        detail={"key": "notification_channel"},
    )
    assert event.interaction_id == "interaction-1"
    assert event.kind == "memory_stored"


def test_metrics_include_memory_fields():
    metrics = AgentRunMetrics(
        total_ms=1,
        model_ms=1,
        tool_ms=0,
        model_turns=1,
        tool_calls=0,
        successful_tool_calls=0,
        failed_tool_calls=0,
        termination_reason="final_answer",
        max_turns=6,
        memory_writes=1,
        memory_reads=1,
        memory_hits=1,
        memory_misses=0,
        memory_scope="u-1001",
        memory_version=1,
        stale_memory_detected=False,
    )
    assert metrics.provenance == "measured"
    assert metrics.memory_scope == "u-1001"
    assert metrics.stale_memory_detected is False
