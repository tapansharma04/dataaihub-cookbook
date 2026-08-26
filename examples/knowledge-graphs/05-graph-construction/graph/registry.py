"""Deterministic local entity registry.

Extracted labels are not RDF identifiers. The application maps known labels to
stable IRIs. Unknown labels are not invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdflib.term import URIRef

from graph.vocabulary import CLASS_BY_LOCAL, EX


@dataclass(frozen=True)
class RegistryEntry:
    label: str
    local_name: str
    entity_type: str

    @property
    def iri(self) -> URIRef:
        return EX[self.local_name]


# Application-owned Acme AI domain registry (same entities as KG #1–#4).
ENTITY_REGISTRY: dict[str, RegistryEntry] = {
    "Acme AI": RegistryEntry("Acme AI", "acmeAI", "Company"),
    "Alice": RegistryEntry("Alice", "alice", "Person"),
    "Bob": RegistryEntry("Bob", "bob", "Person"),
    "Carol": RegistryEntry("Carol", "carol", "Person"),
    "Knowledge Platform": RegistryEntry(
        "Knowledge Platform", "knowledgePlatform", "Project"
    ),
    "Billing Portal": RegistryEntry("Billing Portal", "billingPortal", "Project"),
    "PostgreSQL": RegistryEntry("PostgreSQL", "postgresql", "Technology"),
    "Redis": RegistryEntry("Redis", "redis", "Technology"),
}


def normalize_label(label: str) -> str:
    return " ".join(label.strip().split())


def resolve_label(label: str) -> RegistryEntry | None:
    key = normalize_label(label)
    return ENTITY_REGISTRY.get(key)


def type_iri_for(entity_type: str) -> URIRef | None:
    return CLASS_BY_LOCAL.get(entity_type)


def known_labels() -> list[str]:
    return sorted(ENTITY_REGISTRY)
