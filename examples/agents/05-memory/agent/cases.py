"""Measured cases for the agent-memory example.

Case harness turns are predetermined so sequences are reproducible. Memory
writes and reads are real against an application-owned in-process store.
The harness proposes store/retrieve; the runtime owns validation, scope,
persistence, and freshness comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.model import ScriptedModelClient, ScriptedTurn

STORE_EMAIL = {
    "scope": "u-1001",
    "key": "notification_channel",
    "value": {"channel": "email"},
    "source": "user",
}

STORE_EMAIL_U1003 = {
    "scope": "u-1003",
    "key": "notification_channel",
    "value": {"channel": "email"},
    "source": "user",
}

RETRIEVE_U1001 = {
    "scope": "u-1001",
    "key": "notification_channel",
}

RETRIEVE_U1002 = {
    "scope": "u-1002",
    "key": "notification_channel",
}

RETRIEVE_U1003 = {
    "scope": "u-1003",
    "key": "notification_channel",
}


@dataclass(frozen=True)
class MeasuredInteraction:
    request: str
    turns: tuple[ScriptedTurn, ...]


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: str
    scope: str
    selection_note: str
    interactions: tuple[MeasuredInteraction, ...]
    max_turns: int | None = None


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="no-memory-notification-preference",
        example_class="NO_MEMORY",
        scope="u-1002",
        selection_note=(
            "Measured case: a later request asks for a notification "
            "preference that was never stored. Memory lookup returns no "
            "record. The agent reports the miss and does not invent a "
            "channel. A miss is a normal observable state, not a memory "
            "system failure."
        ),
        interactions=(
            MeasuredInteraction(
                request="How should I be notified about the next incident?",
                turns=(
                    ScriptedTurn(
                        content="retrieve notification preference",
                        decision="retrieve_memory",
                        memory_read=RETRIEVE_U1002,
                    ),
                    ScriptedTurn(
                        content=(
                            "No stored notification preference exists for this "
                            "user. I will not invent a channel. Ask the user "
                            "which channel to use, or store a preference first."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="store-email-notification-preference",
        example_class="STORE",
        scope="u-1001",
        selection_note=(
            "Measured case: the user explicitly provides a notification "
            "preference. The application validates the write and stores a "
            "scoped memory record. Information does not persist just "
            "because the model saw it; persistence begins with an explicit "
            "application-owned write."
        ),
        interactions=(
            MeasuredInteraction(
                request="I prefer email notifications for service incidents.",
                turns=(
                    ScriptedTurn(
                        content="store email notification preference",
                        decision="store_memory",
                        memory_write=STORE_EMAIL,
                    ),
                    ScriptedTurn(
                        content=(
                            "Stored email as the notification channel for this "
                            "user. The record is scoped to u-1001 with "
                            "provenance user and version 1."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="recall-email-notification-preference",
        example_class="RECALL",
        scope="u-1001",
        selection_note=(
            "Measured case: interaction 1 stores a user-provided email "
            "preference. Interaction 2 asks how the user should be notified "
            "and does not repeat the channel. The runtime retrieves the "
            "stored record and the answer uses that recalled information."
        ),
        interactions=(
            MeasuredInteraction(
                request="I prefer email notifications for service incidents.",
                turns=(
                    ScriptedTurn(
                        content="store email notification preference",
                        decision="store_memory",
                        memory_write=STORE_EMAIL,
                    ),
                    ScriptedTurn(
                        content=(
                            "Stored email as the notification channel for this user."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
            MeasuredInteraction(
                request="How should I be notified about the next incident?",
                turns=(
                    ScriptedTurn(
                        content="retrieve notification preference",
                        decision="retrieve_memory",
                        memory_read=RETRIEVE_U1001,
                    ),
                    ScriptedTurn(
                        content=(
                            "Notify this user by email, using the stored "
                            "notification preference for u-1001."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
        ),
    ),
    MeasuredCase(
        trace_id="stale-memory-notification-preference",
        example_class="STALE_MEMORY",
        scope="u-1003",
        selection_note=(
            "Measured case: interaction 1 stores an email preference. "
            "Interaction 2 retrieves that record, then observes the current "
            "authoritative source (SMS, version 2). The stored record is "
            "stale. The final answer uses the current source. Stale memory "
            "is not a store failure."
        ),
        interactions=(
            MeasuredInteraction(
                request="I prefer email notifications for service incidents.",
                turns=(
                    ScriptedTurn(
                        content="store email notification preference",
                        decision="store_memory",
                        memory_write=STORE_EMAIL_U1003,
                    ),
                    ScriptedTurn(
                        content=(
                            "Stored email as the notification channel for this user."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
            MeasuredInteraction(
                request="How should I be notified about the next incident?",
                turns=(
                    ScriptedTurn(
                        content="retrieve notification preference",
                        decision="retrieve_memory",
                        memory_read=RETRIEVE_U1003,
                    ),
                    ScriptedTurn(
                        content=(
                            "Stored memory is email at version 1, but the "
                            "current source of record is SMS at version 2. "
                            "Notify this user by SMS. Memory is context, not "
                            "the latest source of truth."
                        ),
                        decision="final_answer",
                    ),
                ),
            ),
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(c.trace_id for c in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")


def scripted_client_for(case: MeasuredCase) -> ScriptedModelClient:
    turns: list[ScriptedTurn] = []
    for interaction in case.interactions:
        turns.extend(interaction.turns)
    return ScriptedModelClient(turns, model_name="case-harness")
