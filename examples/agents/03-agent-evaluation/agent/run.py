"""Shared measured-case runner used by CLI, export, and tests."""

from __future__ import annotations

from pathlib import Path

from agent.cases import MeasuredCase, scripted_client_for
from agent.executor import ToolExecutor
from agent.loop import run_agent_loop
from agent.schemas import AgentRunResult
from agent.tools import build_registry
from config import Settings, get_settings


def run_measured_case(
    case: MeasuredCase,
    *,
    settings: Settings | None = None,
    data_dir: Path | None = None,
) -> AgentRunResult:
    cfg = settings or get_settings()
    registry = build_registry(data_dir or cfg.data_dir)
    executor = ToolExecutor(registry, timeout_ms=cfg.tool_timeout_ms)
    max_turns = case.max_turns if case.max_turns is not None else cfg.max_turns
    return run_agent_loop(
        request=case.request,
        model=scripted_client_for(case),
        registry=registry,
        executor=executor,
        max_turns=max_turns,
        max_tool_calls_per_turn=cfg.max_tool_calls_per_turn,
    )
