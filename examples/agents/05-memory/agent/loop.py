"""Memory runtime — application-managed store / retrieve across interactions.

Interaction
  → Agent
  → Memory Store  (STORE / RETRIEVE)
  → Later Interaction
  → Agent uses recalled context

This is not an agent loop that repeatedly selects data tools, and it is
not a planner. The model proposes a memory operation or a final answer.
The application owns scope, validation, persistence, retrieval, and
freshness comparison with the current authoritative source.
"""

from __future__ import annotations

import json
import time

from agent.memory import (
    MemoryStore,
    MemoryValidationError,
    assess_freshness,
    parse_memory_read,
    parse_memory_write,
)
from agent.model import ModelClient, memory_operation_definitions, memory_tools
from agent.schemas import (
    AgentRunMetrics,
    AgentRunResult,
    DecisionKind,
    ModelTurn,
    SequenceEvent,
)
from agent.source import AuthoritativeStore
from agent.state import RecordedDecision, initial_state

SYSTEM_PROMPT = """\
You are a support assistant. The application owns memory.

When the user explicitly provides a preference that should persist, propose
store_memory with source=user. When a later request needs a previously
stored fact, propose retrieve_memory.

Do not invent a preference that was not stored. Do not treat stored memory
as automatically newer than the current source of record. After a retrieve,
the application may attach a current-source observation.

Produce a final answer from stored memory and current-source observations
only. Do not claim a missing memory exists.
"""


def classify_decision(turn: ModelTurn) -> DecisionKind:
    """Map an observable model turn to a runtime memory decision.

    Unknown tools and unrecognized decisions become invalid_action.
    """
    if turn.decision == "invalid_action":
        return "invalid_action"
    data_tool_names = {
        tc.name
        for tc in turn.tool_calls
        if tc.name not in {"store_memory", "retrieve_memory"}
    }
    if data_tool_names:
        return "invalid_action"
    if turn.decision in {"store_memory", "retrieve_memory", "final_answer"}:
        return turn.decision
    if turn.memory_write:
        return "store_memory"
    if turn.memory_read:
        return "retrieve_memory"
    if turn.content and not turn.tool_calls:
        return "final_answer"
    return "invalid_action"


def _note(state, text: str) -> None:
    state.messages.append({"role": "user", "content": text})


def run_memory_loop(
    *,
    interactions: list[str],
    scope: str,
    model: ModelClient,
    memory_store: MemoryStore,
    authoritative: AuthoritativeStore,
    max_turns: int = 6,
    system_prompt: str | None = None,
) -> AgentRunResult:
    started = time.perf_counter()
    operations = memory_operation_definitions()
    prompt = (system_prompt or SYSTEM_PROMPT) + f"\n\nSession scope (user id): {scope}."

    state = initial_state(scope=scope, max_turns=max_turns)
    sequence: list[SequenceEvent] = []

    model_ms = 0
    model_turns = 0
    memory_writes = 0
    memory_reads = 0
    memory_hits = 0
    memory_misses = 0
    stale_memory_detected = False
    last_memory_version: int | None = None

    def interaction_id() -> str:
        return state.current_interaction_id or "interaction-1"

    def record_model_turn(turn: ModelTurn, decision: DecisionKind) -> int:
        nonlocal model_ms, model_turns
        turn_number = state.begin_turn()
        model_turns += 1
        model_ms += turn.latency_ms
        state.record_decision(
            RecordedDecision(
                turn=turn_number,
                interaction_id=interaction_id(),
                decision=decision,
                content=turn.content,
                memory_write=turn.memory_write,
                memory_read=turn.memory_read,
                finish_reason=turn.finish_reason,
                latency_ms=turn.latency_ms,
            )
        )
        sequence.append(
            SequenceEvent(
                kind="model_decision",
                turn=turn_number,
                interaction_id=interaction_id(),
                latency_ms=turn.latency_ms,
                detail={
                    "decision": decision,
                    "content": turn.content,
                    "finishReason": turn.finish_reason,
                    "memoryWrite": turn.memory_write,
                    "memoryRead": turn.memory_read,
                    "promptTokens": turn.prompt_tokens,
                    "completionTokens": turn.completion_tokens,
                    "interactionId": interaction_id(),
                },
            )
        )
        return turn_number

    def stop_invalid(turn_number: int | None, message: str) -> None:
        state.terminate(
            "invalid_action",
            answer=message,
            error={"code": "invalid_action", "message": message},
        )
        sequence.append(
            SequenceEvent(
                kind="error",
                turn=turn_number,
                interaction_id=interaction_id(),
                detail={
                    "code": "invalid_action",
                    "message": message,
                    "interactionId": interaction_id(),
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                turn=turn_number if turn_number is not None else state.current_turn,
                interaction_id=interaction_id(),
                detail={
                    "reason": "invalid_action",
                    "interactionId": interaction_id(),
                },
            )
        )

    def stop_max_turns() -> None:
        answer = (
            "Stopped: reached max_turns without completing the memory session. "
            "The runtime enforces a hard turn limit to prevent runaway loops."
        )
        state.terminate(
            "max_turns",
            answer=answer,
            error={"code": "max_turns", "message": answer},
        )
        sequence.append(
            SequenceEvent(
                kind="error",
                interaction_id=interaction_id(),
                detail={
                    "code": "max_turns",
                    "message": answer,
                    "interactionId": interaction_id(),
                },
            )
        )
        sequence.append(
            SequenceEvent(
                kind="termination",
                turn=state.current_turn,
                interaction_id=interaction_id(),
                detail={
                    "reason": "max_turns",
                    "maxTurns": state.max_turns,
                    "currentTurn": state.current_turn,
                    "interactionId": interaction_id(),
                },
            )
        )

    def consult() -> tuple[int, ModelTurn, DecisionKind] | None:
        if state.current_turn >= state.max_turns:
            stop_max_turns()
            return None
        turn = model.complete(state.messages, memory_tools())
        decision = classify_decision(turn)
        turn_number = record_model_turn(turn, decision)
        if decision == "invalid_action":
            stop_invalid(
                turn_number,
                "Stopped: model produced an unrecognized action. "
                "The memory runtime only accepts store_memory, retrieve_memory, "
                "or final_answer.",
            )
            return None
        return turn_number, turn, decision

    def handle_store(turn: ModelTurn, turn_number: int) -> bool:
        nonlocal memory_writes, last_memory_version
        try:
            proposal = parse_memory_write(turn.memory_write)
        except MemoryValidationError as exc:
            stop_invalid(turn_number, f"Stopped: invalid memory write ({exc}).")
            return False
        proposed_scope = proposal["scope"] or scope
        sequence.append(
            SequenceEvent(
                kind="memory_write_requested",
                turn=turn_number,
                interaction_id=interaction_id(),
                detail={
                    "scope": proposed_scope,
                    "key": proposal["key"],
                    "value": proposal["value"],
                    "source": proposal["source"],
                    "sessionScope": scope,
                    "interactionId": interaction_id(),
                },
            )
        )
        try:
            record = memory_store.store(
                scope=proposed_scope,
                key=str(proposal["key"] or ""),
                value=proposal["value"],
                source=str(proposal["source"] or ""),
                session_scope=scope,
            )
        except MemoryValidationError as exc:
            stop_invalid(turn_number, f"Stopped: invalid memory write ({exc}).")
            return False
        memory_writes += 1
        last_memory_version = record.version
        state.record_write(record)
        sequence.append(
            SequenceEvent(
                kind="memory_stored",
                turn=turn_number,
                interaction_id=interaction_id(),
                detail={
                    **record.to_public_dict(),
                    "interactionId": interaction_id(),
                },
            )
        )
        _note(
            state,
            "The application stored memory: " + json.dumps(record.to_public_dict()),
        )
        return True

    def handle_retrieve(turn: ModelTurn, turn_number: int) -> bool:
        nonlocal memory_reads, memory_hits, memory_misses
        nonlocal stale_memory_detected, last_memory_version
        try:
            proposal = parse_memory_read(turn.memory_read)
        except MemoryValidationError as exc:
            stop_invalid(turn_number, f"Stopped: invalid memory read ({exc}).")
            return False
        proposed_scope = proposal["scope"] or scope
        key = str(proposal["key"] or "")
        sequence.append(
            SequenceEvent(
                kind="memory_retrieval_requested",
                turn=turn_number,
                interaction_id=interaction_id(),
                detail={
                    "scope": proposed_scope,
                    "key": key,
                    "sessionScope": scope,
                    "interactionId": interaction_id(),
                },
            )
        )
        try:
            record = memory_store.retrieve(
                scope=proposed_scope,
                key=key,
                session_scope=scope,
            )
        except MemoryValidationError as exc:
            stop_invalid(turn_number, f"Stopped: invalid memory read ({exc}).")
            return False
        memory_reads += 1
        if record is None:
            memory_misses += 1
            read_payload = {
                "scope": proposed_scope,
                "key": key,
                "found": False,
                "record": None,
                "interactionId": interaction_id(),
            }
            state.record_read(read_payload)
            sequence.append(
                SequenceEvent(
                    kind="memory_not_found",
                    turn=turn_number,
                    interaction_id=interaction_id(),
                    detail=read_payload,
                )
            )
            _note(
                state,
                (
                    "Memory lookup returned no record for "
                    f"scope={proposed_scope} key={key}. "
                    "Do not invent a preference."
                ),
            )
            return True

        memory_hits += 1
        last_memory_version = record.version
        read_payload = {
            "scope": record.scope,
            "key": record.key,
            "found": True,
            "record": record.to_public_dict(),
            "interactionId": interaction_id(),
        }
        state.record_read(read_payload)
        sequence.append(
            SequenceEvent(
                kind="memory_retrieved",
                turn=turn_number,
                interaction_id=interaction_id(),
                detail=read_payload,
            )
        )
        current = authoritative.get(record.scope, record.key)
        assessment = assess_freshness(record, current)
        freshness_payload = assessment.to_public_dict()
        freshness_payload["interactionId"] = interaction_id()
        state.record_freshness(freshness_payload)
        if current is not None:
            if assessment.stale:
                stale_memory_detected = True
            sequence.append(
                SequenceEvent(
                    kind="observation",
                    turn=turn_number,
                    interaction_id=interaction_id(),
                    detail={
                        "kind": "current_source",
                        "scope": record.scope,
                        "key": record.key,
                        "current": current,
                        "freshness": freshness_payload,
                        "staleMemoryDetected": assessment.stale,
                        "resolution": assessment.resolution,
                        "interactionId": interaction_id(),
                    },
                )
            )
            _note(
                state,
                "Current authoritative source: "
                + json.dumps(
                    {
                        "current": current,
                        "freshness": freshness_payload,
                    }
                ),
            )
        else:
            _note(
                state,
                "The application retrieved memory: "
                + json.dumps(record.to_public_dict())
                + ". No current authoritative source exists for this key.",
            )
        return True

    for index, request in enumerate(interactions, start=1):
        if state.terminated:
            break
        iid = f"interaction-{index}"
        is_last = index == len(interactions)
        state.begin_interaction(iid, request, prompt)
        sequence.append(
            SequenceEvent(
                kind="user_request",
                interaction_id=iid,
                detail={
                    "request": request,
                    "scope": scope,
                    "interactionId": iid,
                },
            )
        )

        interaction_complete = False
        while not state.terminated and not interaction_complete:
            consulted = consult()
            if consulted is None:
                break
            turn_number, turn, decision = consulted
            if decision == "store_memory":
                if not handle_store(turn, turn_number):
                    break
                continue
            if decision == "retrieve_memory":
                if not handle_retrieve(turn, turn_number):
                    break
                continue
            if decision != "final_answer":
                stop_invalid(
                    turn_number,
                    "Stopped: expected store_memory, retrieve_memory, or final_answer.",
                )
                break
            answer = (turn.content or "").strip()
            state.set_interaction_answer(answer)
            sequence.append(
                SequenceEvent(
                    kind="final_answer",
                    turn=turn_number,
                    interaction_id=iid,
                    detail={
                        "answer": answer,
                        "interactionId": iid,
                    },
                )
            )
            if is_last:
                state.terminate("final_answer", answer=answer)
                sequence.append(
                    SequenceEvent(
                        kind="termination",
                        turn=turn_number,
                        interaction_id=iid,
                        detail={
                            "reason": "final_answer",
                            "interactionId": iid,
                            "scope": scope,
                        },
                    )
                )
            interaction_complete = True

    if not state.terminated:
        stop_invalid(
            state.current_turn,
            "Stopped: memory session ended without a final answer.",
        )

    total_ms = int(round((time.perf_counter() - started) * 1000))
    primary_request = interactions[-1] if interactions else ""
    return AgentRunResult(
        request=primary_request,
        answer=state.final_answer or "",
        model=model.model_name,
        model_driver=model.driver,
        memory_operations=operations,
        sequence=sequence,
        metrics=AgentRunMetrics(
            total_ms=total_ms,
            model_ms=model_ms,
            tool_ms=0,
            model_turns=model_turns,
            tool_calls=0,
            successful_tool_calls=0,
            failed_tool_calls=0,
            termination_reason=state.termination_reason or "error",
            max_turns=max_turns,
            memory_writes=memory_writes,
            memory_reads=memory_reads,
            memory_hits=memory_hits,
            memory_misses=memory_misses,
            memory_scope=scope,
            memory_version=last_memory_version,
            stale_memory_detected=stale_memory_detected,
        ),
        state={
            **state.to_public_dict(),
            "memoryRecords": memory_store.exported_records(),
        },
        errors=list(state.errors),
    )
