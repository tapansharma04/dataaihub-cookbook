"""Application-owned validation boundary for extraction proposals.

Extractors propose; this module validates. RDF commits happen only after
validation succeeds for the intended triples.
"""

from __future__ import annotations

from graph.model import (
    ExtractionProposal,
    RejectedRelationship,
    ResolvedEntity,
    ValidatedRelationship,
    ValidationResult,
)
from graph.registry import resolve_label, type_iri_for
from graph.vocabulary import (
    ALLOWED_ENTITY_TYPES,
    PREDICATE_BY_LOCAL,
    resolve_predicate_local,
)


def validate_proposal(proposal: ExtractionProposal) -> ValidationResult:
    """Validate entities and relationships against application vocabulary."""
    resolved: list[ResolvedEntity] = []
    unresolved: list[str] = []
    type_errors: list[str] = []
    seen_labels: set[str] = set()
    # Labels proposed as entities that did not successfully resolve.
    # Relationship endpoints must not bypass this via registry lookup.
    failed_entity_labels: set[str] = set()

    for entity in proposal.entities:
        label = entity.label.strip()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)

        if entity.entity_type not in ALLOWED_ENTITY_TYPES:
            type_errors.append(f"unsupported_entity_type:{entity.entity_type}:{label}")
            failed_entity_labels.add(label)
            continue
        if type_iri_for(entity.entity_type) is None:
            type_errors.append(f"unsupported_entity_type:{entity.entity_type}:{label}")
            failed_entity_labels.add(label)
            continue

        entry = resolve_label(label)
        if entry is None:
            unresolved.append(label)
            failed_entity_labels.add(label)
            continue
        # Registry owns identity and type; do not silently normalize mismatches.
        if entity.entity_type != entry.entity_type:
            type_errors.append(
                f"entity_type_mismatch:{label}:{entity.entity_type}:{entry.entity_type}"
            )
            failed_entity_labels.add(label)
            continue
        resolved.append(
            ResolvedEntity(
                label=entry.label,
                iri=str(entry.iri),
                entity_type=entry.entity_type,
            )
        )

    # Also resolve relationship endpoints that were not listed as entities.
    for rel in proposal.relationships:
        for endpoint in (rel.subject, rel.object):
            label = endpoint.strip()
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            entry = resolve_label(label)
            if entry is None:
                unresolved.append(label)
            else:
                resolved.append(
                    ResolvedEntity(
                        label=entry.label,
                        iri=str(entry.iri),
                        entity_type=entry.entity_type,
                    )
                )

    by_label = {entity.label: entity for entity in resolved}
    accepted: list[ValidatedRelationship] = []
    rejected: list[RejectedRelationship] = []

    for rel in proposal.relationships:
        subject_label = rel.subject.strip()
        object_label = rel.object.strip()
        predicate_raw = rel.predicate.strip()

        local = resolve_predicate_local(predicate_raw)
        if local is None:
            rejected.append(
                RejectedRelationship(
                    subject=subject_label,
                    predicate=predicate_raw,
                    object=object_label,
                    reason="unsupported_predicate",
                )
            )
            continue

        subject = _endpoint_for_relationship(
            subject_label,
            by_label=by_label,
            failed_entity_labels=failed_entity_labels,
        )
        if subject == "entity_validation_failed":
            rejected.append(
                RejectedRelationship(
                    subject=subject_label,
                    predicate=predicate_raw,
                    object=object_label,
                    reason="entity_validation_failed",
                )
            )
            continue
        obj = _endpoint_for_relationship(
            object_label,
            by_label=by_label,
            failed_entity_labels=failed_entity_labels,
        )
        if obj == "entity_validation_failed":
            rejected.append(
                RejectedRelationship(
                    subject=subject_label,
                    predicate=predicate_raw,
                    object=object_label,
                    reason="entity_validation_failed",
                )
            )
            continue

        if subject is None:
            rejected.append(
                RejectedRelationship(
                    subject=subject_label,
                    predicate=predicate_raw,
                    object=object_label,
                    reason="unresolved_subject",
                )
            )
            continue
        if obj is None:
            rejected.append(
                RejectedRelationship(
                    subject=subject_label,
                    predicate=predicate_raw,
                    object=object_label,
                    reason="unresolved_object",
                )
            )
            continue

        predicate_iri = PREDICATE_BY_LOCAL[local]
        accepted.append(
            ValidatedRelationship(
                subject_iri=subject.iri,
                predicate_local=local,
                predicate_iri=str(predicate_iri),
                object_iri=obj.iri,
                subject_label=subject.label,
                object_label=obj.label,
            )
        )

    return ValidationResult(
        resolved_entities=resolved,
        unresolved_labels=unresolved,
        accepted_relationships=accepted,
        rejected_relationships=rejected,
        entity_type_errors=type_errors,
    )


def _endpoint_for_relationship(
    label: str,
    *,
    by_label: dict[str, ResolvedEntity],
    failed_entity_labels: set[str],
) -> ResolvedEntity | None | str:
    """Resolve a relationship endpoint.

    Returns ResolvedEntity, None (unknown), or \"entity_validation_failed\" when
    the label was proposed as an entity and failed validation — do not accept
    relationships through a registry bypass in that case.
    """
    if label in failed_entity_labels:
        return "entity_validation_failed"
    if label in by_label:
        return by_label[label]
    return _resolve_endpoint(label)


def _resolve_endpoint(label: str) -> ResolvedEntity | None:
    entry = resolve_label(label)
    if entry is None:
        return None
    return ResolvedEntity(
        label=entry.label,
        iri=str(entry.iri),
        entity_type=entry.entity_type,
    )
