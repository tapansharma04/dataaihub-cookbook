"""Pydantic schema smoke tests."""

from graphrag.model import GraphEntityRef, GraphFact, GraphPredicateRef, GraphRunMetrics


def test_graph_fact_public_shape():
    fact = GraphFact(
        subject=GraphEntityRef(iri="https://example/alice", label="Alice"),
        predicate=GraphPredicateRef(iri="https://example/worksOn", label="works on"),
        object=GraphEntityRef(iri="https://example/kp", label="Knowledge Platform"),
    )
    public = fact.public()
    assert public["subject"]["label"] == "Alice"
    assert public["predicate"]["label"] == "works on"
    assert public["object"]["label"] == "Knowledge Platform"


def test_metrics_model_aliases():
    metrics = GraphRunMetrics(
        entity_candidates=8,
        resolved_entity_count=1,
        retrieval_hops=2,
        entities_retrieved=3,
        relationships_retrieved=2,
        subgraph_triple_count=2,
        context_fact_count=2,
        retrieval_execution_ms=1,
        context_assembly_ms=0,
        answer_generation_ms=0,
        total_ms=2,
        model_turns=0,
        termination_reason="completed",
    )
    dumped = metrics.model_dump(by_alias=True)
    assert dumped["entity_candidates"] == 8
    assert dumped["termination_reason"] == "completed"
