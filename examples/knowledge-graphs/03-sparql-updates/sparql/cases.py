"""Four measured SPARQL UPDATE teaching cases."""

from __future__ import annotations

from dataclasses import dataclass

from sparql.model import UpdateType


@dataclass(frozen=True)
class MeasuredCase:
    trace_id: str
    example_class: UpdateType
    question: str
    update_name: UpdateType
    selection_note: str


CASES: tuple[MeasuredCase, ...] = (
    MeasuredCase(
        trace_id="insert-data-billing-portal-redis",
        example_class="INSERT_DATA",
        question="Add Redis as a technology used by the Billing Portal.",
        update_name="INSERT_DATA",
        selection_note=(
            "Measured case: INSERT DATA adds the explicit triple "
            "ex:billingPortal ex:uses ex:redis . "
            "Billing Portal already uses PostgreSQL in the fixture."
        ),
    ),
    MeasuredCase(
        trace_id="insert-where-person-uses-technology",
        example_class="INSERT_WHERE",
        question=(
            "Derive person→uses→technology triples from "
            "person→worksOn→project→uses→technology."
        ),
        update_name="INSERT_WHERE",
        selection_note=(
            "Measured case: INSERT WHERE matches worksOn + uses patterns and "
            "inserts derived ex:uses triples for Alice, Bob, and Carol."
        ),
    ),
    MeasuredCase(
        trace_id="delete-data-billing-portal-postgresql",
        example_class="DELETE_DATA",
        question="Remove PostgreSQL as a technology used by the Billing Portal.",
        update_name="DELETE_DATA",
        selection_note=(
            "Measured case: DELETE DATA removes the explicit triple "
            "ex:billingPortal ex:uses ex:postgresql ."
        ),
    ),
    MeasuredCase(
        trace_id="update-and-verify-billing-portal-technology",
        example_class="UPDATE_AND_VERIFY",
        question=(
            "Change the Billing Portal technology from PostgreSQL to Redis, "
            "then verify the new graph state."
        ),
        update_name="UPDATE_AND_VERIFY",
        selection_note=(
            "Measured case: one SPARQL UPDATE deletes "
            "ex:billingPortal ex:uses ex:postgresql and inserts "
            "ex:billingPortal ex:uses ex:redis when the WHERE pattern matches."
        ),
    ),
)


def get_case(trace_id: str) -> MeasuredCase:
    for case in CASES:
        if case.trace_id == trace_id:
            return case
    known = ", ".join(case.trace_id for case in CASES)
    raise KeyError(f"Unknown case '{trace_id}'. Known: {known}")
