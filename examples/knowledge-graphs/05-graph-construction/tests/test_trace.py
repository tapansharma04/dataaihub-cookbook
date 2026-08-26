"""Trace sequence and presentation tests."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from graph.builder import RdfGraphStore
from graph.cases import get_case
from graph.extractor import StructuredExtractor
from graph.runner import run_case
from graph.trace import build_signature_view, build_trace

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "graph.ttl"


def _build(trace_id: str):
    settings = Settings(graph_path=GRAPH_PATH, openai_api_key="")
    case = get_case(trace_id)
    store = RdfGraphStore.fresh(start=case.start_graph, seed_path=GRAPH_PATH)
    result = run_case(
        case,
        settings,
        mode="structured",
        extractor=StructuredExtractor(),
        store=store,
    )
    trace = build_trace(case=case, result=result, settings=settings, store=store)
    return result, trace


def test_entity_linking_shows_label_to_iri():
    result, trace = _build("entity-linking-known-entities")
    view = build_signature_view(result, example_class="ENTITY_LINKING")
    phases = [item["phase"] for item in view]
    assert "ENTITY_LINKING" in phases
    linking = next(item for item in view if item["phase"] == "ENTITY_LINKING")
    links = {item["label"]: item["iri"] for item in linking["links"]}
    assert "Knowledge Platform" in links
    assert links["Knowledge Platform"].endswith("knowledgePlatform")
    assert "PostgreSQL" in links
    assert links["PostgreSQL"].endswith("postgresql")
    assert trace["validation"]["resolvedEntities"]


def test_invalid_fact_signature_shows_rejection():
    result, _ = _build("invalid-fact-unsupported-predicate")
    view = build_signature_view(result, example_class="INVALID_FACT")
    phases = [item["phase"] for item in view]
    assert "VALIDATION_REJECTED" in phases
    assert "GRAPH_UNCHANGED" in phases
    assert view[-1]["phase"] == "TERMINATION"


def test_trace_metrics_keys():
    _, trace = _build("entity-extraction-alice")
    assert set(trace["metrics"]) >= {
        "sourceCharacters",
        "entitiesProposed",
        "entitiesResolved",
        "relationshipsProposed",
        "relationshipsAccepted",
        "relationshipsRejected",
        "triplesCreated",
        "triplesRejected",
        "graphTripleCount",
        "validationErrors",
        "modelTurns",
        "totalMs",
        "modelMs",
        "terminationReason",
    }
    forbidden = {
        "extractionQuality",
        "graphQuality",
        "accuracy",
        "confidence",
        "intelligence",
        "benchmarkScore",
    }
    assert forbidden.isdisjoint(trace["metrics"])


def test_steps_mirror_sequence():
    _, trace = _build("relationship-extraction-alice-platform")
    assert len(trace["steps"]) == len(trace["sequence"])
    assert trace["steps"][0]["type"] == trace["sequence"][0]["kind"]
