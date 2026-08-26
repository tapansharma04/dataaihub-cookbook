# Graph Construction: Source → Proposal → Validation → RDF

**Example ID:** `graph-construction`

Follows: [04 — GraphRAG](../04-graphrag/)

## Why graph construction matters

Building an RDF knowledge graph from source text is not “ask an LLM for
Turtle.” Extraction may propose entities and relationships; the **application**
owns the RDF vocabulary, stable identifiers, validation, provenance, and graph
writes.

```text
SOURCE → EXTRACTION PROPOSAL → VALIDATION → RDF TRIPLES → RDF GRAPH
```

**The LLM proposes structured facts; it does not own the RDF graph.**

**Entity labels extracted from source text are not trusted as RDF identifiers.**

## Pipeline

```text
Source document
  → Extractor (proposes only)
  → Structured ExtractionProposal
  → Application validation
  → Identifier resolution (entity registry)
  → RDF mapping
  → rdflib.Graph (commits)
  → Measured trace
```

The extractor never receives a graph object and never mutates RDF state.

## Two modes

### Structured (`structured`)

Deterministic fixture proposals. No API key. Stable reference implementation of
the construction pipeline.

Provenance: `model=not_used`, `tools=measured`, `metrics=measured`.

### LLM-assisted (`llm_assisted`)

OpenAI structured output returns an `ExtractionProposal` (entities +
relationships). The model receives source text and the allowed vocabulary. It
does **not** return arbitrary Turtle, and it does not write to the graph.

CI does not require an API key. Tests use a deterministic mock extractor.

## Application-owned RDF vocabulary

Allowed predicates (same Acme AI domain as KG #1–#4):

| Local name | IRI |
|------------|-----|
| `employs` | `https://dataaihub.co/example/kg/employs` |
| `worksOn` | `https://dataaihub.co/example/kg/worksOn` |
| `uses` | `https://dataaihub.co/example/kg/uses` |

Extractors may propose semantic tokens such as `works_on`. The application maps
those tokens onto vocabulary IRIs. Arbitrary predicate IRIs from a model
(for example `http://evil.example/supervises`) are rejected.

## Stable IRIs vs labels

| Label (source) | Stable IRI |
|----------------|------------|
| Alice | `https://dataaihub.co/example/kg/alice` |
| Knowledge Platform | `https://dataaihub.co/example/kg/knowledgePlatform` |
| PostgreSQL | `https://dataaihub.co/example/kg/postgresql` |

Human-readable labels are stored with `rdfs:label`. Identifiers come from a
deterministic local entity registry. Unknown labels are not invented as IRIs.

## Entity extraction vs entity linking

- **Entity extraction** converts proposed labels into stable RDF entities
  (`rdf:type` + `rdfs:label`).
- **Entity linking** resolves extracted labels to **existing** registry IRIs,
  then commits validated relationships. The label is not the identifier.

## Relationship validation

The runtime validates:

1. Predicate is in the allowed vocabulary (no arbitrary IRIs).
2. Subject and object resolve via the entity registry.
3. Direction is preserved as proposed.
4. Unsupported predicates (for example `supervises`) are rejected — not rewritten.

## Invalid / unsupported facts

When a proposal includes an unsupported predicate, the application rejects it.
No unsupported triple is committed. The graph remains unchanged. The trace
records:

```text
relationship proposed → validation rejected → graph unchanged → termination
```

Reason code: `unsupported_predicate`.

## Graph construction vs GraphRAG

| | KG #5 (this example) | KG #4 (GraphRAG) |
|--|----------------------|------------------|
| Direction | Source → construct graph | Question → retrieve graph → answer |
| Focus | Validation boundary before RDF commits | Retrieval + optional LLM answering |
| Artifact | Built / mutated RDF graph | Retrieved subgraph + answer |

This example does not perform GraphRAG retrieval or answer generation.

## Provenance

| Mode | `model` | `tools` | `metrics` |
|------|---------|---------|-----------|
| `structured` | `not_used` | `measured` | `measured` |
| `llm_assisted` (model invoked) | configured model id | `measured` | `measured` |
| `llm_assisted` (no model turn) | `not_used` | `measured` | `measured` |

## Traces

- `lab_traces.json` — structured mode (deterministic; safe to regenerate)
- `lab_traces_llm.json` — LLM-assisted mode (generated only when an API key is
  available; not required for CI)

Observable events include: `source_loaded`, `extraction_started`,
`entity_proposed`, `relationship_proposed`, `validation_started`,
`validation_passed`, `validation_rejected`, `entity_resolved`,
`triple_created`, `graph_committed`, `graph_verified`, `result`,
`termination`.

Chain-of-thought, hidden reasoning, and quality/benchmark scores are **not**
stored.

A future Lab can visualize SOURCE → EXTRACTION → VALIDATION → RDF GRAPH
(and LLM EXTRACTION for LLM mode), including accepted/rejected triples and
before/after graph state. This example does not implement a Lab UI.

## Metrics (observable only)

`sourceCharacters`, `entitiesProposed`, `entitiesResolved`,
`relationshipsProposed`, `relationshipsAccepted`, `relationshipsRejected`,
`triplesCreated`, `triplesRejected`, `graphTripleCount`, `validationErrors`,
`modelTurns`, `totalMs`, `modelMs`, `terminationReason`.

This is not a benchmark — no extraction-quality or accuracy scores.

## Security / runtime boundary

- Vocabulary and predicate aliases are application-owned
- Entity registry owns identifier resolution
- Validator rejects unsupported / unresolved facts before commit
- Builder is the only component that mutates `rdflib.Graph`
- Extractors (including LLMs) never receive the graph object

## Quick start

```bash
cd examples/knowledge-graphs/05-graph-construction
uv sync
uv run python main.py --case entity-extraction-alice
uv run python main.py --case relationship-extraction-alice-platform
uv run python main.py --case entity-linking-known-entities
uv run python main.py --case invalid-fact-unsupported-predicate
uv run pytest -q
```

Export structured traces:

```bash
uv run python export_lab_traces.py --mode structured --force
```

LLM-assisted mode (requires `OPENAI_API_KEY`):

```bash
cp .env.example .env   # set OPENAI_API_KEY
uv run python main.py --case entity-extraction-alice --mode llm_assisted
uv run python export_lab_traces.py --mode llm_assisted
```

## Measured cases

| Case | Source | Outcome |
|------|--------|---------|
| `ENTITY_EXTRACTION` | Alice works on the Knowledge Platform. | Resolve Alice + Knowledge Platform; commit type/label triples |
| `RELATIONSHIP_EXTRACTION` | Alice works on the Knowledge Platform. | Commit `ex:alice ex:worksOn ex:knowledgePlatform` |
| `ENTITY_LINKING` | Knowledge Platform uses PostgreSQL. | Link labels to stable IRIs; commit `ex:uses` |
| `INVALID_FACT` | Alice supervises the Knowledge Platform. | Reject `supervises`; graph unchanged |

## Limitations

- Small local Acme AI domain and registry
- Deterministic label→IRI registry — not fuzzy entity resolution
- Three application predicates only (`employs`, `worksOn`, `uses`)
- No Neo4j, property graphs, embeddings, vector DBs, or GraphRAG
- No SPARQL as the primary focus
- No LangChain / LangGraph / agent frameworks
- Structured mode is fully deterministic; LLM mode is model/provider-specific
- Not a construction benchmark

## Path

```text
01 — RDF & Graph Traversal
02 — SPARQL & Graph Queries
03 — SPARQL Updates
04 — GraphRAG
05 — Graph Construction (this example)
```
