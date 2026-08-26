"""Extraction: structured fixtures and LLM-assisted structured proposals.

Extractors PROPOSE entities and relationships. They never receive or mutate
an RDF graph.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

from openai import OpenAI

from config import Settings
from graph.model import (
    EntityProposal,
    ExtractionProposal,
    RelationshipProposal,
)
from graph.vocabulary import ALLOWED_ENTITY_TYPES, ALLOWED_PREDICATES

SYSTEM_PROMPT = """You extract entities and relationships for an RDF knowledge graph.
Return ONLY structured data matching the schema.
Use only these entity types: {entity_types}.
Allowed relationship predicates for successful commits: employs, works_on, uses.
Preserve the relationship verb expressed in the source text as the predicate token.
Do NOT rewrite a source verb into a different allowed predicate.
If the source says "supervises", propose predicate "supervises"
(not employs or works_on).
The application validates predicates; unsupported verbs must still be
proposed as written.
Choose entity_type by the entity's nature in the text:
- Person for people
- Project for platforms, products, and projects being built or used
- Technology for databases, libraries, and tools
- Company for organizations
Do not invent predicates. Do not return Turtle or SPARQL.
Do not include reasoning, thought, or chain-of-thought fields.
Entity labels should be the surface forms from the source text."""


def build_extraction_user_prompt(source_text: str) -> str:
    """Observable extraction contract sent to the LLM (no hidden reasoning)."""
    allowed = ", ".join(sorted(ALLOWED_PREDICATES))
    return (
        f"Allowed commit predicates: {allowed} "
        f"(propose works_on for the worksOn relationship).\n"
        "Predicate rule: copy the source relationship verb into the predicate "
        "field. Do not substitute an allowed predicate for a different source "
        'verb. Example: source "supervises" → predicate "supervises".\n'
        "Entity-type rule: platforms/products are Project, not Company; "
        "people are Person; databases/tools are Technology; organizations "
        "are Company.\n\n"
        f"Source text:\n{source_text}\n\n"
        "Extract entities and relationships from the source text above."
    )


# Deterministic fixture proposals keyed by source/case id.
STRUCTURED_PROPOSALS: dict[str, ExtractionProposal] = {
    "entity-extraction-alice": ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ],
        relationships=[],
    ),
    "relationship-extraction-alice-platform": ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="works_on",
                object="Knowledge Platform",
            )
        ],
    ),
    "entity-linking-known-entities": ExtractionProposal(
        entities=[
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
            EntityProposal(label="PostgreSQL", entity_type="Technology"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Knowledge Platform",
                predicate="uses",
                object="PostgreSQL",
            )
        ],
    ),
    "invalid-fact-unsupported-predicate": ExtractionProposal(
        entities=[
            EntityProposal(label="Alice", entity_type="Person"),
            EntityProposal(label="Knowledge Platform", entity_type="Project"),
        ],
        relationships=[
            RelationshipProposal(
                subject="Alice",
                predicate="supervises",
                object="Knowledge Platform",
            )
        ],
    ),
}


class Extractor(Protocol):
    mode: str
    model_name: str | None
    provider: str | None

    def extract(
        self, *, case_id: str, source_text: str
    ) -> tuple[ExtractionProposal, int]:
        """Return (proposal, model_latency_ms). Latency is 0 when no model ran."""
        ...


class StructuredExtractor:
    """Deterministic fixture extractor — no API key required."""

    mode = "structured"
    model_name: str | None = None
    provider: str | None = None

    def extract(
        self, *, case_id: str, source_text: str
    ) -> tuple[ExtractionProposal, int]:
        del source_text  # fixture keyed by case id
        if case_id not in STRUCTURED_PROPOSALS:
            raise KeyError(f"No structured proposal fixture for case '{case_id}'")
        return STRUCTURED_PROPOSALS[case_id].model_copy(deep=True), 0


class OpenAIExtractor:
    """LLM-assisted extractor returning a structured ExtractionProposal."""

    mode = "llm_assisted"

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model_name = model
        self.provider = "openai"

    def extract(
        self, *, case_id: str, source_text: str
    ) -> tuple[ExtractionProposal, int]:
        del case_id
        started = time.perf_counter()
        system = SYSTEM_PROMPT.format(
            entity_types=", ".join(sorted(ALLOWED_ENTITY_TYPES))
        )
        user = build_extraction_user_prompt(source_text)
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=ExtractionProposal,
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Fallback: attempt JSON content parse
            content = response.choices[0].message.content or "{}"
            parsed = ExtractionProposal.model_validate(json.loads(content))
        return parsed, latency_ms


class MockLLMExtractor:
    """Deterministic mock for tests — never masquerades as a live provider."""

    mode = "llm_assisted"
    model_name = "mock"
    provider = "mock"

    def __init__(
        self,
        proposals: dict[str, ExtractionProposal] | None = None,
    ) -> None:
        self._proposals = proposals or STRUCTURED_PROPOSALS
        self.last_source_text: str = ""
        self.last_case_id: str = ""

    def extract(
        self, *, case_id: str, source_text: str
    ) -> tuple[ExtractionProposal, int]:
        started = time.perf_counter()
        self.last_case_id = case_id
        self.last_source_text = source_text
        if case_id not in self._proposals:
            raise KeyError(f"No mock proposal for case '{case_id}'")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._proposals[case_id].model_copy(deep=True), latency_ms


def get_openai_client(settings: Settings) -> OpenAI:
    kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def build_extractor(
    settings: Settings,
    *,
    mode: str,
    use_mock: bool = False,
) -> Extractor | None:
    if mode == "structured":
        return StructuredExtractor()
    if use_mock:
        return MockLLMExtractor()
    if not settings.openai_api_key:
        return None
    return OpenAIExtractor(get_openai_client(settings), settings.openai_model)
