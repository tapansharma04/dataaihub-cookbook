"""Deterministic prompt catalog metadata for the Acme AI domain.

Prompt names and argument contracts are owned by the server. Clients discover
them through prompts/list and retrieve rendered messages through prompts/get.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_SUMMARIZE_SERVICE = "summarize-service"
PROMPT_INVESTIGATE_INCIDENT = "investigate-incident"
PROMPT_DRAFT_STATUS_UPDATE = "draft-status-update"

EXPECTED_PROMPT_NAMES = (
    PROMPT_SUMMARIZE_SERVICE,
    PROMPT_INVESTIGATE_INCIDENT,
    PROMPT_DRAFT_STATUS_UPDATE,
)


@dataclass(frozen=True)
class PromptCatalogEntry:
    name: str
    description: str


CATALOG: tuple[PromptCatalogEntry, ...] = (
    PromptCatalogEntry(
        name=PROMPT_SUMMARIZE_SERVICE,
        description=(
            "Produce a concise service summary tailored to a target audience."
        ),
    ),
    PromptCatalogEntry(
        name=PROMPT_INVESTIGATE_INCIDENT,
        description=(
            "Start an incident investigation workflow for an Acme AI service."
        ),
    ),
    PromptCatalogEntry(
        name=PROMPT_DRAFT_STATUS_UPDATE,
        description="Draft a stakeholder status update for a service.",
    ),
)
