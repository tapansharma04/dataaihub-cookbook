"""LLM grounding and cross-mode retrieval parity tests."""

from pathlib import Path

from config import Settings
from graphrag.cases import CASES
from graphrag.graph import RdfGraphStore
from graphrag.llm import MockLLMClient
from graphrag.runner import run_case

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"

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


def _collect_cot_violations(obj, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else key
            if key in COT_FIELD_NAMES:
                violations.append(child)
            violations.extend(_collect_cot_violations(value, child))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            violations.extend(_collect_cot_violations(item, f"{path}[{index}]"))
    return violations


def _settings() -> Settings:
    return Settings(graph_path=GRAPH_PATH, openai_api_key="")


def _retrieval_signature(result) -> dict:
    return {
        "resolved": [entity.model_dump() for entity in result.resolved_entities],
        "subgraph": [fact.sort_key() for fact in result.subgraph],
        "paths": [
            [
                (step.subject.iri, step.predicate.iri, step.object.iri)
                for step in path.steps
            ]
            for path in result.paths
        ],
        "context": result.context,
    }


def test_same_retrieval_across_modes_for_shared_cases():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    mock = MockLLMClient()
    for case in CASES:
        if case.example_class == "NO_RELEVANT_SUBGRAPH":
            continue
        grounded = run_case(case, _settings(), mode="graph_grounded", store=store)
        llm = run_case(
            case,
            _settings(),
            mode="graphrag_llm",
            store=store,
            llm_client=mock,
        )
        assert _retrieval_signature(grounded) == _retrieval_signature(llm)


def test_llm_receives_only_assembled_context():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    mock = MockLLMClient()
    case = next(c for c in CASES if c.example_class == "MULTI_HOP_RETRIEVAL")
    result = run_case(
        case,
        _settings(),
        mode="graphrag_llm",
        store=store,
        llm_client=mock,
    )
    assert mock.last_context == result.context
    assert all(
        "PostgreSQL" in line or "Redis" in line or "works on" in line
        for line in mock.last_context
    )
    assert not any("Carol" in line for line in mock.last_context)


def test_unrelated_graph_triples_excluded_from_context():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "ENTITY_RETRIEVAL")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    joined = " ".join(result.context)
    assert "Carol" not in joined
    assert "Billing Portal" not in joined


def test_multi_hop_paths_preserved():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    case = next(c for c in CASES if c.example_class == "MULTI_HOP_RETRIEVAL")
    result = run_case(case, _settings(), mode="graph_grounded", store=store)
    assert len(result.paths) == 2
    for path in result.paths:
        assert len(path.steps) == 2


def test_no_relevant_subgraph_does_not_fabricate_context_or_call_llm():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    mock = MockLLMClient()
    case = next(c for c in CASES if c.example_class == "NO_RELEVANT_SUBGRAPH")
    result = run_case(
        case,
        _settings(),
        mode="graphrag_llm",
        store=store,
        llm_client=mock,
    )
    assert result.context == []
    assert result.metrics.model_turns == 0
    assert mock.last_context == []
    kinds = [event.kind for event in result.sequence]
    assert kinds.index("subgraph_retrieved") < kinds.index("final_answer")
    assert "model_request" not in kinds


def test_llm_mode_occurs_after_context_assembly():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    mock = MockLLMClient()
    case = next(c for c in CASES if c.example_class == "ENTITY_RETRIEVAL")
    result = run_case(
        case,
        _settings(),
        mode="graphrag_llm",
        store=store,
        llm_client=mock,
    )
    kinds = [event.kind for event in result.sequence]
    assert kinds.index("context_assembled") < kinds.index("model_request")
    assert kinds.index("model_request") < kinds.index("model_response")
    assert kinds.index("model_response") < kinds.index("final_answer")


def test_no_hidden_reasoning_fields_in_result():
    store = RdfGraphStore.from_path(GRAPH_PATH)
    mock = MockLLMClient()
    case = next(c for c in CASES if c.example_class == "RELATIONSHIP_GROUNDED_ANSWER")
    result = run_case(
        case,
        _settings(),
        mode="graphrag_llm",
        store=store,
        llm_client=mock,
    )
    payload = result.model_dump()
    assert _collect_cot_violations(payload) == []
