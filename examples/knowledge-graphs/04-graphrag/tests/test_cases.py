"""Measured case execution tests."""

from pathlib import Path

from config import Settings
from graphrag.cases import CASES
from graphrag.graph import RdfGraphStore
from graphrag.runner import run_case

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH, openai_api_key="")


def test_four_cases_defined():
    assert len(CASES) == 4
    classes = {case.example_class for case in CASES}
    assert classes == {
        "ENTITY_RETRIEVAL",
        "MULTI_HOP_RETRIEVAL",
        "RELATIONSHIP_GROUNDED_ANSWER",
        "NO_RELEVANT_SUBGRAPH",
    }


def test_entity_retrieval_deterministic_answer():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "ENTITY_RETRIEVAL")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    assert result.termination == "completed"
    assert "Alice" in result.answer and "Bob" in result.answer
    assert result.metrics.model_turns == 0
    assert result.provenance["model"] == "not_used"


def test_multi_hop_answer_lists_technologies():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "MULTI_HOP_RETRIEVAL")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    assert "PostgreSQL" in result.answer
    assert "Redis" in result.answer
    assert result.metrics.retrieval_hops == 2


def test_relationship_grounded_answer():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "RELATIONSHIP_GROUNDED_ANSWER")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    assert "Acme AI" in result.answer
    assert "Knowledge Platform" in result.answer


def test_no_relevant_subgraph_terminates_without_context():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "NO_RELEVANT_SUBGRAPH")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    assert result.termination == "no_relevant_subgraph"
    assert result.context == []
    assert "enough graph evidence" in result.answer
    assert result.metrics.model_turns == 0
    kinds = [event.kind for event in result.sequence]
    assert "context_assembled" not in kinds
    assert "model_request" not in kinds
