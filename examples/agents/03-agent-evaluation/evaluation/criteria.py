"""Explicit per-case evaluation criteria.

Criteria describe observable constraints, not a unique gold path and not
hidden model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArgumentExpectation:
    """Observable argument constraint for a tool.

    `arguments` must match exactly on the listed keys.
    `argument_contains` requires a case-insensitive substring on a string arg.
    A trajectory satisfies the expectation when at least one *successful*
    call of `tool` meets the constraints. Extra calls of other tools do not
    fail this check.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    argument_contains: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseCriteria:
    """Case-level rubric.

    Hard trajectory constraints (used for `trajectory_success`):
    required tools, required arguments, max useful tool calls, and whether
    recovery is expected. Successful execution of required tools is also
    required by the evaluator.

    Outcome fields: `required_answer_facts` / `forbidden_answer_facts` feed
    final-answer correctness and task success.

    Result interpretation is computed from observations vs the answer and is
    reported separately; it is not a hard trajectory constraint.
    """

    required_tools: tuple[str, ...]
    required_arguments: tuple[ArgumentExpectation, ...]
    required_answer_facts: tuple[str, ...]
    max_useful_tool_calls: int | None = None
    recovery_expected: bool = False
    forbidden_answer_facts: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "requiredTools": list(self.required_tools),
            "requiredArguments": [
                {
                    "tool": item.tool,
                    "arguments": item.arguments,
                    "argumentContains": item.argument_contains,
                }
                for item in self.required_arguments
            ],
            "requiredAnswerFacts": list(self.required_answer_facts),
            "forbiddenAnswerFacts": list(self.forbidden_answer_facts),
            "maxUsefulToolCalls": self.max_useful_tool_calls,
            "recoveryExpected": self.recovery_expected,
        }


# Shared support-investigation task used by the four measured cases.
PAYMENTS_INVESTIGATION_CRITERIA = CaseCriteria(
    required_tools=("get_service_status", "search_documentation"),
    required_arguments=(
        ArgumentExpectation(
            tool="get_service_status",
            arguments={"service": "payments"},
        ),
        ArgumentExpectation(
            tool="search_documentation",
            argument_contains={"query": "payment"},
        ),
    ),
    required_answer_facts=("degraded", "PAY-2041", "checkout"),
    max_useful_tool_calls=2,
    recovery_expected=False,
)

RECOVERY_CRITERIA = CaseCriteria(
    required_tools=("get_service_status", "search_documentation"),
    required_arguments=(
        ArgumentExpectation(
            tool="get_service_status",
            arguments={"service": "payments"},
        ),
        ArgumentExpectation(
            tool="search_documentation",
            argument_contains={"query": "payment"},
        ),
    ),
    required_answer_facts=("degraded", "PAY-2041", "checkout"),
    # Failed canonical-name lookup + corrected status + docs.
    max_useful_tool_calls=3,
    recovery_expected=True,
)
