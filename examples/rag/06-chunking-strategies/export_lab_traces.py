"""Export measured chunking-strategy traces for Interactive Lab #6."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import ALL_STRATEGIES, EXAMPLE_ID, get_settings, strategy_config
from evaluation.dataset import load_eval_dataset
from evaluation.evidence import build_chunk_relevance
from experiment import run_experiment
from rag.embeddings import get_client
from rag.loader import load_document

# Selected AFTER the measured run. Labels describe observed behavior.
# Do not invent classifications before seeing results — populate via
# classify_example_cases() from measured metrics, then lock notes here.
EXAMPLE_CASE_SPECS: list[dict] = []


def ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def _query_payload(result) -> dict:
    return {
        "queryId": result.query_id,
        "query": result.query,
        "strategy": result.strategy,
        "k": result.k,
        "retrieved": [
            {
                "chunkId": h.chunk_id,
                "rank": h.rank,
                "relevanceGrade": h.relevance_grade,
                "isRelevant": h.is_relevant,
                "score": round(h.score, 6),
                "section": h.section,
                "evidenceIds": list(h.evidence_ids),
                "start": h.start,
                "end": h.end,
                "text": h.text,
            }
            for h in result.retrieved
        ],
        "metrics": {
            "recallAtK": round(result.recall_at_k, 6),
            "reciprocalRank": round(result.reciprocal_rank, 6),
            "firstRelevantRank": result.first_relevant_rank,
            "dcgAtK": round(result.dcg_at_k, 6),
            "idcgAtK": round(result.idcg_at_k, 6),
            "ndcgAtK": round(result.ndcg_at_k, 6),
            "evidenceCoverage": round(result.evidence_coverage, 6),
            "evidenceFound": result.evidence_found,
            "evidenceMissed": result.evidence_missed,
            "provenance": "computed",
        },
        "chunkRelevance": result.chunk_relevance,
        "latencyMs": result.latency_ms,
    }


def _aggregate_payload(result) -> dict:
    return {
        "strategy": result.strategy,
        "k": result.k,
        "queryCount": result.query_count,
        "meanRecallAtK": round(result.mean_recall_at_k, 6),
        "mrr": round(result.mrr, 6),
        "meanNdcgAtK": round(result.mean_ndcg_at_k, 6),
        "meanEvidenceCoverage": round(result.mean_evidence_coverage, 6),
        "totalRetrievalMs": result.total_retrieval_ms,
        "provenance": "computed",
    }


def _chunk_payload(chunk) -> dict:
    return {
        "chunkId": chunk.id,
        "strategy": chunk.strategy,
        "text": chunk.text,
        "start": chunk.start,
        "end": chunk.end,
        "length": chunk.length,
        "section": chunk.section,
        "source": chunk.source,
        "prevId": chunk.prev_id,
        "nextId": chunk.next_id,
        "evidenceIds": list(chunk.evidence_ids),
        "metadata": chunk.metadata,
    }


def build_presentation(trace: dict) -> dict:
    """Frontend-friendly projection of fields already on the trace.

    Does not invent measurements — only reorganizes existing corpus, boundary,
    and evidence data.
    """
    required = set(trace["input"]["evidenceGrades"])
    evidence_for_query = [u for u in trace["evidenceUnits"] if u["id"] in required]
    bc = trace["boundaryComparison"]
    chunks_containing_evidence = {}
    for strategy, chunks in bc["chunksByStrategy"].items():
        containing = []
        for c in chunks:
            hit = [eid for eid in c.get("evidenceIds", []) if eid in required]
            if hit:
                containing.append(
                    {
                        "chunkId": c["chunkId"],
                        "start": c["start"],
                        "end": c["end"],
                        "section": c.get("section"),
                        "evidenceIds": hit,
                        "text": c["text"],
                    }
                )
        chunks_containing_evidence[strategy] = containing
    return {
        "purpose": (
            "Frontend-friendly projection of existing corpus, boundary, and "
            "evidence fields. Not a new measurement."
        ),
        "originalDocument": {
            "source": trace["corpus"]["source"],
            "charCount": trace["corpus"]["charCount"],
            "text": trace["corpus"]["text"],
        },
        "evidenceSpans": [
            {
                "id": u["id"],
                "section": u["section"],
                "start": u["start"],
                "end": u["end"],
                "text": u["anchor"],
                "grade": trace["input"]["evidenceGrades"][u["id"]],
            }
            for u in evidence_for_query
        ],
        "regionOfInterest": {
            "start": bc["regionStart"],
            "end": bc["regionEnd"],
            "text": bc["regionText"],
        },
        "chunkBoundariesByStrategy": {
            strategy: [
                {
                    "chunkId": c["chunkId"],
                    "start": c["start"],
                    "end": c["end"],
                    "length": c.get("length", c["end"] - c["start"]),
                    "section": c.get("section"),
                    "evidenceIds": list(c.get("evidenceIds", [])),
                    "containsRequiredEvidence": any(
                        eid in required for eid in c.get("evidenceIds", [])
                    ),
                }
                for c in chunks
            ]
            for strategy, chunks in bc["chunksByStrategy"].items()
        },
        "chunksContainingEvidenceByStrategy": chunks_containing_evidence,
    }


def classify_example_cases(
    per_strategy_by_query: dict[str, dict[str, object]],
    dataset,
) -> list[dict]:
    """Derive ~3 example cases from measured per-query metrics.

    Labels are observational, not winners.
    """
    candidates: list[dict] = []

    for case in dataset.queries:
        metrics = {
            name: per_strategy_by_query[name][case.id]  # type: ignore[index]
            for name in ALL_STRATEGIES
        }
        recalls = {n: q.recall_at_k for n, q in metrics.items()}  # type: ignore[attr-defined]
        ndcgs = {n: q.ndcg_at_k for n, q in metrics.items()}  # type: ignore[attr-defined]
        rrs = {n: q.reciprocal_rank for n, q in metrics.items()}  # type: ignore[attr-defined]
        covs = {n: q.evidence_coverage for n, q in metrics.items()}  # type: ignore[attr-defined]

        recall_vals = list(recalls.values())
        ndcg_vals = list(ndcgs.values())

        # BOUNDARY / FRAGMENTATION: structure or recursive covers evidence
        # better than fixed (coverage or recall gap).
        if (
            covs.get("structure", 0) > covs.get("fixed", 0) + 0.01
            or recalls.get("structure", 0) > recalls.get("fixed", 0) + 0.01
        ):
            candidates.append(
                {
                    "traceId": f"boundary-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "BOUNDARY",
                    "selectionNote": (
                        "Measured: structure evidence_coverage="
                        f"{covs['structure']:.3f} "
                        f"Recall@K={recalls['structure']:.3f} vs fixed "
                        f"coverage={covs['fixed']:.3f} "
                        f"Recall@K={recalls['fixed']:.3f}. "
                        "Chunk boundaries changed which evidence "
                        "units stayed intact."
                    ),
                    "score": abs(covs["structure"] - covs["fixed"])
                    + abs(recalls["structure"] - recalls["fixed"]),
                }
            )

        if (
            covs.get("fixed", 0) > covs.get("structure", 0) + 0.01
            or recalls.get("fixed", 0) > recalls.get("structure", 0) + 0.01
        ):
            candidates.append(
                {
                    "traceId": f"regression-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "REGRESSION",
                    "selectionNote": (
                        "Measured: fixed outscored structure on this query "
                        f"(Recall {recalls['fixed']:.3f} vs "
                        f"{recalls['structure']:.3f}; "
                        f"coverage {covs['fixed']:.3f} vs "
                        f"{covs['structure']:.3f}). "
                        "A more structure-aware strategy is not "
                        "automatically better."
                    ),
                    "score": abs(recalls["fixed"] - recalls["structure"])
                    + abs(covs["fixed"] - covs["structure"]),
                }
            )

        # EQUIVALENT: all strategies within a tight band.
        if (
            max(recall_vals) - min(recall_vals) < 1e-9
            and max(ndcg_vals) - min(ndcg_vals) < 0.05
        ):
            candidates.append(
                {
                    "traceId": f"equivalent-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "EQUIVALENT",
                    "selectionNote": (
                        f"Measured: Recall@K identical ({recall_vals[0]:.3f}) and "
                        f"nDCG@K within 0.05 across strategies "
                        f"({ {n: round(v, 3) for n, v in ndcgs.items()} }). "
                        "Different chunk texts, equivalent retrieval quality."
                    ),
                    "score": 0.5 - (max(ndcg_vals) - min(ndcg_vals)),
                }
            )

        # RANKING_SENSITIVITY: same recall, different nDCG.
        if (
            max(recall_vals) - min(recall_vals) < 1e-9
            and max(ndcg_vals) - min(ndcg_vals) > 0.05
        ):
            candidates.append(
                {
                    "traceId": f"ranking-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "RANKING_SENSITIVITY",
                    "selectionNote": (
                        f"Measured: Recall@K tied at {recall_vals[0]:.3f} while "
                        f"nDCG@K differs "
                        f"({ {n: round(v, 3) for n, v in ndcgs.items()} }). "
                        "Same hit presence, different ranking quality."
                    ),
                    "score": max(ndcg_vals) - min(ndcg_vals),
                }
            )

        # CONTEXT_PRESERVED: structure has RR=1 and others lower.
        if rrs.get("structure", 0) == 1.0 and any(
            rrs[n] < 1.0 for n in ("fixed", "recursive")
        ):
            candidates.append(
                {
                    "traceId": f"context-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "CONTEXT_PRESERVED",
                    "selectionNote": (
                        "Measured: structure RR=1.0 while "
                        f"fixed RR={rrs['fixed']:.3f}, "
                        f"recursive RR={rrs['recursive']:.3f}. "
                        "Keeping heading+body together ranked "
                        "primary evidence first."
                    ),
                    "score": 1.0
                    - min(rrs["fixed"], rrs["recursive"])
                    + (covs["structure"] - min(covs["fixed"], covs["recursive"])),
                }
            )

        # FRAGMENTATION: fixed coverage lower because evidence split.
        if covs.get("fixed", 1) + 0.01 < covs.get("structure", 0):
            candidates.append(
                {
                    "traceId": f"fragmentation-{case.id}",
                    "queryId": case.id,
                    "exampleClass": "FRAGMENTATION",
                    "selectionNote": (
                        f"Measured: fixed evidence_coverage={covs['fixed']:.3f} "
                        f"< structure={covs['structure']:.3f}. "
                        "Fixed windows likely split required evidence across chunks."
                    ),
                    "score": covs["structure"] - covs["fixed"],
                }
            )

    # Deduplicate by queryId+exampleClass, keep highest score, pick diverse classes.
    by_key: dict[tuple[str, str], dict] = {}
    for c in candidates:
        key = (c["queryId"], c["exampleClass"])
        if key not in by_key or c["score"] > by_key[key]["score"]:
            by_key[key] = c

    ranked = sorted(by_key.values(), key=lambda c: -c["score"])
    selected: list[dict] = []
    used_queries: set[str] = set()
    used_classes: set[str] = set()
    for c in ranked:
        if c["queryId"] in used_queries:
            continue
        if c["exampleClass"] in used_classes and len(selected) < 3:
            # Prefer diversity of classes first.
            continue
        selected.append(c)
        used_queries.add(c["queryId"])
        used_classes.add(c["exampleClass"])
        if len(selected) >= 3:
            break

    # If diversity filter was too strict, fill up.
    if len(selected) < 3:
        for c in ranked:
            if c["queryId"] in used_queries:
                continue
            selected.append(c)
            used_queries.add(c["queryId"])
            if len(selected) >= 3:
                break

    # Strip internal score field.
    out = []
    for c in selected:
        out.append(
            {
                "traceId": c["traceId"],
                "queryId": c["queryId"],
                "exampleClass": c["exampleClass"],
                "selectionNote": c["selectionNote"],
            }
        )
    return out


def _chunks_overlapping_region(
    chunks: list,
    start: int,
    end: int,
) -> list[dict]:
    return [_chunk_payload(c) for c in chunks if c.start < end and start < c.end]


def main() -> None:
    global EXAMPLE_CASE_SPECS

    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY")

    source_text = load_document(settings.data_path)
    dataset = load_eval_dataset(settings.eval_path, source_text)
    k = settings.eval_k
    configs = strategy_config(settings)

    t_init = time.perf_counter()
    client = get_client(settings)
    # Touch embeddings API with empty skip — model init is client construction.
    model_init_ms = ms(t_init)

    result = run_experiment(
        client,
        source_text,
        dataset,
        settings,
        strategies=ALL_STRATEGIES,
        k=k,
    )

    aggregates = [_aggregate_payload(r) for r in result.evaluations]
    per_strategy_by_query: dict[str, dict[str, object]] = {
        name: {} for name in ALL_STRATEGIES
    }
    for strat_result in result.evaluations:
        for q in strat_result.per_query:
            per_strategy_by_query[strat_result.strategy][q.query_id] = q

    EXAMPLE_CASE_SPECS = classify_example_cases(per_strategy_by_query, dataset)

    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    units_by_id = dataset.units_by_id()

    # Full strategy payloads for lab stages 2–4.
    strategies_payload = {}
    for name in ALL_STRATEGIES:
        index = result.indexes[name]
        strategies_payload[name] = {
            "config": configs[name],
            "stats": index.stats.to_dict(),
            "timingsMs": index.timings_ms,
            "chunks": [_chunk_payload(c) for c in index.chunks],
        }

    evidence_payload = [
        {
            "id": u.id,
            "section": u.section,
            "anchor": u.anchor,
            "start": u.start,
            "end": u.end,
        }
        for u in dataset.evidence_units
    ]

    traces = []
    for spec in EXAMPLE_CASE_SPECS:
        case = dataset.get(spec["queryId"])
        strategy_details = [
            _query_payload(per_strategy_by_query[name][case.id])  # type: ignore[index]
            for name in ALL_STRATEGIES
        ]
        recalls = {p["strategy"]: p["metrics"]["recallAtK"] for p in strategy_details}
        ndcgs = {p["strategy"]: p["metrics"]["ndcgAtK"] for p in strategy_details}
        rrs = {p["strategy"]: p["metrics"]["reciprocalRank"] for p in strategy_details}
        covs = {
            p["strategy"]: p["metrics"]["evidenceCoverage"] for p in strategy_details
        }

        # Region covering all evidence for this query — for boundary comparison.
        evid_spans = [
            units_by_id[eid] for eid in case.evidence_grades if eid in units_by_id
        ]
        region_start = min(u.start for u in evid_spans) if evid_spans else 0
        region_end = max(u.end for u in evid_spans) if evid_spans else 0

        boundary_compare = {
            name: _chunks_overlapping_region(
                result.indexes[name].chunks, region_start, region_end
            )
            for name in ALL_STRATEGIES
        }

        # Per-strategy derived relevance for this query (for lab inspection).
        relevance_by_strategy = {
            name: build_chunk_relevance(
                result.indexes[name].chunks,
                evidence_grades=case.evidence_grades,
                units_by_id=units_by_id,
            )
            for name in ALL_STRATEGIES
        }

        trace = {
            "labId": EXAMPLE_ID,
            "traceId": spec["traceId"],
            "executionMode": "guided",
            "recordedAt": recorded_at,
            "metricsProvenance": "measured",
            "exampleClass": spec["exampleClass"],
            "selectionNote": spec["selectionNote"],
            "architecture": {
                "layout": "fan-out-compare",
                "stages": [
                    "source",
                    "fixed-chunking",
                    "recursive-chunking",
                    "structure-chunking",
                    "compare-boundaries",
                    "embed-index",
                    "retrieve",
                    "compare-ranked",
                    "measure",
                    "inspect-example",
                    "compare-aggregates",
                    "takeaway",
                ],
            },
            "input": {
                "queryId": case.id,
                "query": case.query,
                "k": k,
                "evidenceGrades": case.evidence_grades,
                "rationale": case.rationale,
                "config": {
                    "embeddingModel": settings.embedding_model,
                    "retrieval": "dense-cosine",
                    "evalK": k,
                    "strategies": configs,
                    "chunkSizeUnit": "characters",
                },
            },
            "corpus": {
                "source": "data/sample.md",
                "charCount": len(source_text),
                "text": source_text,
            },
            "evidenceUnits": evidence_payload,
            "strategies": strategies_payload,
            "boundaryComparison": {
                "regionStart": region_start,
                "regionEnd": region_end,
                "regionText": source_text[region_start:region_end],
                "chunksByStrategy": boundary_compare,
            },
            "relevanceByStrategy": relevance_by_strategy,
            "retrievalByStrategy": strategy_details,
            "observedComparison": {
                "recallAtKByStrategy": recalls,
                "ndcgAtKByStrategy": ndcgs,
                "reciprocalRankByStrategy": rrs,
                "evidenceCoverageByStrategy": covs,
                "provenance": "derived-from-measured",
            },
            "steps": [
                {
                    "id": "source-document",
                    "detail": {
                        "source": "data/sample.md",
                        "charCount": len(source_text),
                    },
                },
                {
                    "id": "fixed-chunking",
                    "detail": {
                        "config": configs["fixed"],
                        "stats": result.indexes["fixed"].stats.to_dict(),
                        "chunkCount": len(result.indexes["fixed"].chunks),
                    },
                },
                {
                    "id": "recursive-chunking",
                    "detail": {
                        "config": configs["recursive"],
                        "stats": result.indexes["recursive"].stats.to_dict(),
                        "chunkCount": len(result.indexes["recursive"].chunks),
                    },
                },
                {
                    "id": "structure-chunking",
                    "detail": {
                        "config": configs["structure"],
                        "stats": result.indexes["structure"].stats.to_dict(),
                        "chunkCount": len(result.indexes["structure"].chunks),
                    },
                },
                {
                    "id": "compare-boundaries",
                    "detail": {
                        "regionStart": region_start,
                        "regionEnd": region_end,
                        "chunksByStrategy": {
                            n: [c["chunkId"] for c in boundary_compare[n]]
                            for n in ALL_STRATEGIES
                        },
                    },
                },
                {
                    "id": "embed-index",
                    "detail": {
                        "embeddingModel": settings.embedding_model,
                        "timingsMsByStrategy": {
                            n: result.indexes[n].timings_ms for n in ALL_STRATEGIES
                        },
                        "note": (
                            "Each strategy embedded independently with the same model. "
                            "No cross-strategy embedding cache in this run."
                        ),
                    },
                },
                {
                    "id": "retrieve",
                    "detail": {
                        "query": case.query,
                        "method": "dense-cosine",
                        "k": k,
                    },
                },
                {
                    "id": "compare-ranked",
                    "detail": {
                        "byStrategy": {
                            p["strategy"]: p["retrieved"] for p in strategy_details
                        }
                    },
                },
                {
                    "id": "measure",
                    "detail": {
                        "recallAtK": recalls,
                        "mrrComponentRR": rrs,
                        "ndcgAtK": ndcgs,
                        "evidenceCoverage": covs,
                    },
                },
                {
                    "id": "inspect-example",
                    "detail": {
                        "exampleClass": spec["exampleClass"],
                        "selectionNote": spec["selectionNote"],
                    },
                },
                {
                    "id": "compare-aggregates",
                    "detail": {"aggregates": aggregates},
                },
                {
                    "id": "takeaway",
                    "detail": {
                        "message": (
                            "There is no universally best chunking strategy. "
                            "Chunking changes the retrieval units; measure the "
                            "effect on your information needs."
                        )
                    },
                },
            ],
            "fullEvaluationAggregates": aggregates,
            "metrics": {
                "modelInitMs": model_init_ms,
                "timingsMsByStrategy": {
                    n: result.indexes[n].timings_ms for n in ALL_STRATEGIES
                },
                "provenance": "measured",
            },
            "relatedEntities": [
                "chunking",
                "recall-at-k",
                "mrr",
                "ndcg",
                "dense-retrieval",
            ],
            "relatedContent": [
                "rag",
                "chunking-strategies",
                "retrieval-evaluation",
            ],
            "cookbook": {"path": "examples/rag/06-chunking-strategies"},
        }
        trace["presentation"] = build_presentation(trace)
        traces.append(trace)

    # Full evaluation report for README regeneration.
    full_report = {
        "recordedAt": recorded_at,
        "k": k,
        "queryCount": len(dataset.queries),
        "embeddingModel": settings.embedding_model,
        "retrieval": "dense-cosine",
        "chunkSizeUnit": "characters",
        "strategyConfig": configs,
        "strategies": list(ALL_STRATEGIES),
        "chunkStats": {n: result.indexes[n].stats.to_dict() for n in ALL_STRATEGIES},
        "timingsMs": {n: result.indexes[n].timings_ms for n in ALL_STRATEGIES},
        "aggregates": aggregates,
        "exampleCases": EXAMPLE_CASE_SPECS,
        "perQuery": {
            name: {
                qid: {
                    "retrievedIds": [h.chunk_id for h in q.retrieved],  # type: ignore[attr-defined]
                    "grades": [h.relevance_grade for h in q.retrieved],  # type: ignore[attr-defined]
                    "scores": [round(h.score, 6) for h in q.retrieved],  # type: ignore[attr-defined]
                    "recallAtK": round(q.recall_at_k, 6),  # type: ignore[attr-defined]
                    "reciprocalRank": round(q.reciprocal_rank, 6),  # type: ignore[attr-defined]
                    "ndcgAtK": round(q.ndcg_at_k, 6),  # type: ignore[attr-defined]
                    "evidenceCoverage": round(q.evidence_coverage, 6),  # type: ignore[attr-defined]
                    "latencyMs": q.latency_ms,  # type: ignore[attr-defined]
                }
                for qid, q in per_strategy_by_query[name].items()
            }
            for name in ALL_STRATEGIES
        },
    }
    report_path = Path(__file__).resolve().parent / "evaluation_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    out = Path(__file__).resolve().parent / "lab_traces.json"
    out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Wrote {len(traces)} traces to {out}")
    print(f"Wrote full evaluation report to {report_path}")
    print("Example cases:")
    for spec in EXAMPLE_CASE_SPECS:
        print(f"  {spec['exampleClass']}: {spec['queryId']}")


if __name__ == "__main__":
    main()
