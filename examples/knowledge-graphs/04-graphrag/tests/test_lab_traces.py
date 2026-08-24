"""Trace schema, provenance, CoT, and export tests — no network / no paid APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EXAMPLE_ID, Settings
from graphrag.cases import CASES, get_case
from graphrag.graph import RdfGraphStore
from graphrag.runner import run_case
from graphrag.trace import build_signature_view, build_trace

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"
LAB_TRACES_PATH = ROOT / "lab_traces.json"
LAB_TRACES_LLM_PATH = ROOT / "lab_traces_llm.json"

EXPECTED_EXAMPLE_CLASSES = frozenset(
    {
        "ENTITY_RETRIEVAL",
        "MULTI_HOP_RETRIEVAL",
        "RELATIONSHIP_GROUNDED_ANSWER",
        "NO_RELEVANT_SUBGRAPH",
    }
)
EXPECTED_TRACE_IDS = frozenset(
    {
        "entity-retrieval-knowledge-platform",
        "multi-hop-alice-technologies",
        "relationship-grounded-alice-employer-project",
        "no-relevant-subgraph-alice-direct-uses",
    }
)
SUPPORTED_TERMINATION_REASONS = frozenset(
    {
        "completed",
        "no_entity_match",
        "no_relevant_subgraph",
        "model_unavailable",
        "model_failed",
    }
)
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "groundingScore",
        "graphragScore",
        "answerQualityScore",
        "confidenceScore",
        "benchmarkScore",
        "grounding_score",
        "graphrag_score",
        "answer_quality_score",
        "confidence_score",
        "benchmark_score",
    }
)
COT_FIELD_NAMES = frozenset(
    {
        "chainOfThought",
        "chain_of_thought",
        "reasoning",
        "hiddenReasoning",
        "hidden_reasoning",
        "internalReasoning",
        "internal_reasoning",
        "thoughtProcess",
        "thought_process",
        "thought",
        "thoughts",
        "privateReasoning",
        "private_reasoning",
    }
)
VOLATILE_KEYS = frozenset(
    {
        "recordedAt",
        "latencyMs",
        "latency_ms",
        "executionMs",
        "execution_ms",
        "retrievalExecutionMs",
        "retrieval_execution_ms",
        "contextAssemblyMs",
        "context_assembly_ms",
        "answerGenerationMs",
        "answer_generation_ms",
        "totalMs",
        "total_ms",
        "modelLatencyMs",
        "model_latency_ms",
    }
)


def _collect_cot_violations(obj: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in COT_FIELD_NAMES:
                violations.append(child_path)
            violations.extend(_collect_cot_violations(value, child_path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            violations.extend(_collect_cot_violations(item, f"{path}[{index}]"))
    return violations


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_volatile(value)
            for key, value in obj.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(item) for item in obj]
    return obj


def _build(trace_id: str, *, mode: str = "graph_grounded") -> dict[str, Any]:
    settings = Settings(graph_path=GRAPH_PATH, openai_api_key="")
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = get_case(trace_id)
    result = run_case(case, settings, mode=mode, store=store)  # type: ignore[arg-type]
    return build_trace(case=case, result=result, settings=settings, store=store)


def _assert_grounded_trace_contract(trace: dict[str, Any]) -> None:
    assert trace["labId"] == EXAMPLE_ID
    assert trace["executionMode"] == "graph_grounded"
    assert isinstance(trace["traceId"], str) and trace["traceId"]
    assert trace["metricsProvenance"] == "measured"
    assert trace["provenance"] == {
        "model": "not_used",
        "tools": "measured",
        "metrics": "measured",
    }
    assert trace["exampleClass"] in EXPECTED_EXAMPLE_CLASSES
    assert "graph" in trace
    assert "sequence" in trace and len(trace["sequence"]) >= 2
    assert "steps" in trace and len(trace["steps"]) >= 2
    assert "metrics" in trace
    assert set(trace["metrics"]) >= {
        "entityCandidates",
        "resolvedEntityCount",
        "retrievalHops",
        "entitiesRetrieved",
        "relationshipsRetrieved",
        "subgraphTripleCount",
        "contextFactCount",
        "retrievalExecutionMs",
        "contextAssemblyMs",
        "answerGenerationMs",
        "totalMs",
        "modelTurns",
        "terminationReason",
    }
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(trace["metrics"])
    assert trace["metrics"]["modelTurns"] == 0
    termination_reason = trace["metrics"]["terminationReason"]
    assert termination_reason in SUPPORTED_TERMINATION_REASONS
    assert trace["termination"] == termination_reason

    assert "presentation" in trace
    assert "signatureView" in trace["presentation"]
    assert "signatureFlow" in trace["presentation"]
    assert trace["architecture"]["layout"] == "graphrag"
    assert trace["architecture"]["graphModel"] == "rdf"
    assert trace["architecture"]["executionEngine"] == "rdflib"

    kinds = [event["kind"] for event in trace["sequence"]]
    assert "user_request" in kinds
    assert "entity_resolution" in kinds
    assert "termination" in kinds
    assert "model_request" not in kinds
    assert "model_response" not in kinds

    cot_violations = _collect_cot_violations(trace)
    assert cot_violations == []


def test_all_four_cases_have_graph_grounded_traces():
    traces = [_build(case.trace_id) for case in CASES]
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    assert {trace["exampleClass"] for trace in traces} == EXPECTED_EXAMPLE_CLASSES
    for trace in traces:
        _assert_grounded_trace_contract(trace)


def test_entity_retrieval_trace_preserves_subgraph():
    trace = _build("entity-retrieval-knowledge-platform")
    subjects = {fact["subject"]["label"] for fact in trace["subgraph"]}
    assert subjects == {"Alice", "Bob"}
    assert trace["answer"] == "Alice and Bob work on Knowledge Platform."


def test_multi_hop_trace_preserves_paths():
    trace = _build("multi-hop-alice-technologies")
    techs = {
        fact["object"]["label"]
        for fact in trace["subgraph"]
        if fact["predicate"]["label"] == "uses"
    }
    assert techs == {"PostgreSQL", "Redis"}
    assert len(trace["paths"]) == 2


def test_no_relevant_subgraph_trace_shape():
    trace = _build("no-relevant-subgraph-alice-direct-uses")
    assert trace["subgraph"] == []
    assert trace["context"] == []
    assert trace["termination"] == "no_relevant_subgraph"
    phases = [item["phase"] for item in trace["presentation"]["signatureView"]]
    assert "NO_RELEVANT_SUBGRAPH" in phases


def test_signature_view_ends_with_termination():
    settings = Settings(graph_path=GRAPH_PATH, openai_api_key="")
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = get_case("entity-retrieval-knowledge-platform")
    result = run_case(case, settings, mode="graph_grounded", store=store)
    view = build_signature_view(result, example_class=case.example_class)
    assert view[-1]["phase"] == "TERMINATION"


def test_committed_lab_traces_schema():
    if not LAB_TRACES_PATH.exists():
        return
    traces = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    assert isinstance(traces, list)
    assert len(traces) == len(CASES) == 4
    assert {trace["traceId"] for trace in traces} == EXPECTED_TRACE_IDS
    for trace in traces:
        _assert_grounded_trace_contract(trace)


def test_committed_grounded_traces_match_regenerated_semantically():
    if not LAB_TRACES_PATH.exists():
        return
    committed = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
    regenerated = [_build(case.trace_id) for case in CASES]
    committed_by_id = {trace["traceId"]: trace for trace in committed}
    for trace in regenerated:
        left = _strip_volatile(committed_by_id[trace["traceId"]])
        right = _strip_volatile(trace)
        assert left == right


def test_committed_llm_traces_are_separate_file():
    if not LAB_TRACES_LLM_PATH.exists():
        return
    if LAB_TRACES_PATH.exists():
        grounded = json.loads(LAB_TRACES_PATH.read_text(encoding="utf-8"))
        assert all(trace.get("executionMode") == "graph_grounded" for trace in grounded)
    llm_traces = json.loads(LAB_TRACES_LLM_PATH.read_text(encoding="utf-8"))
    assert isinstance(llm_traces, list)
    assert len(llm_traces) == len(CASES) == 4
    assert {trace["traceId"] for trace in llm_traces} == EXPECTED_TRACE_IDS
    assert all(trace.get("executionMode") == "graphrag_llm" for trace in llm_traces)


def test_no_chain_of_thought_in_lab_traces():
    for path in (LAB_TRACES_PATH, LAB_TRACES_LLM_PATH):
        if not path.exists():
            continue
        traces = json.loads(path.read_text(encoding="utf-8"))
        for trace in traces:
            violations = _collect_cot_violations(trace)
            assert violations == []


def test_no_hidden_reasoning_in_fresh_traces():
    for case in CASES:
        trace = _build(case.trace_id)
        assert _collect_cot_violations(trace) == []
