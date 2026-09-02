"""Server-owned prompt templates used by prompts/get handlers.

Prompt names, argument contracts, and rendered message text live on the
server. Composition tools consume the prompts/get payloads rather than
re-rendering these templates.
"""

from __future__ import annotations


def render_summarize_service(
    service_name: str,
    audience: str = "engineering",
) -> str:
    return (
        f"Provide a concise summary of the {service_name} service for a "
        f"{audience} audience. Cover purpose, upstream/downstream "
        f"dependencies, and current operational posture."
    )


def render_draft_status_update(service: str, status: str) -> str:
    return (
        f"Draft a brief stakeholder status update for {service}. "
        f"Current status: {status}. Include impact, mitigation steps taken, "
        f"and the next checkpoint time."
    )
