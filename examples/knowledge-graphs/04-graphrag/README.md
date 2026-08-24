# GraphRAG: Graph-Grounded Retrieval & LLM Answering

**Example ID:** `graphrag`

Follows: [03 — SPARQL Updates](../03-sparql-updates/)

## What this example teaches

GraphRAG separates **graph retrieval** from **answer generation**.

```text
GraphRAG uses graph structure to retrieve connected, relevant context;
an LLM is an optional answer-generation layer.
```

This is not simply “RAG + a graph database.” The teaching artifact is the
**graph retrieval stage** — entity resolution, bounded traversal, subgraph
construction, and context assembly happen **before** any LLM call.

## Two modes

### Graph-grounded (`graph_grounded`)

```text
Question → entity resolution → graph retrieval → subgraph → grounded facts
→ deterministic answer
```

No LLM. Works with `OPENAI_API_KEY=""`. Provenance: `model=not_used`.

### GraphRAG + LLM (`graphrag_llm`)

```text
Question → entity resolution → graph retrieval → subgraph → context assembly
→ LLM → answer
```

The LLM receives **only** assembled graph context. It does not perform graph
retrieval, does not receive the raw RDF graph, and does not execute SPARQL.

For the same case, retrieval output is identical across both modes.

## Why the graph matters

Relationships connect facts. Multi-hop structure lets retrieval follow paths
such as:

```text
Alice → worksOn → Knowledge Platform → uses → PostgreSQL
```

That connected subgraph is richer than isolated text chunks with similar keywords.

## Retrieval pipeline

```text
Question
  → label-based entity resolution (rdfs:label)
  → bounded graph traversal (max hops)
  → relevant subgraph
  → context assembly
  → [optional LLM]
  → answer
```

Graph-native retrieval uses rdflib `Graph.objects()` / `Graph.subjects()` and
real RDF triples. No embeddings. No vector search.

## Graph-grounded mode

Deterministic answers are generated from retrieved facts — not from a model.
This mode is fully reproducible and requires no network access.

## LLM mode

The LLM is downstream of graph retrieval. The prompt instructs the model to:

- answer using supplied graph context only
- not invent unsupported facts
- state when context is insufficient
- avoid external knowledge
- provide a concise answer

Chain-of-thought is **not** requested or stored.

## Grounding boundary

The LLM receives only retrieved context — never the full graph, never arbitrary
SPARQL, and never unrelated triples.

## No relevant subgraph

When retrieval finds no supporting relationships (for example, asking for a
direct `uses` edge that does not exist), the system terminates with
`no_relevant_subgraph`. The LLM is **not** called in this case.

## Provenance

| Mode | `model` | `tools` | `metrics` |
|------|---------|---------|-----------|
| `graph_grounded` | `not_used` | `measured` | `measured` |
| `graphrag_llm` | configured model id | `measured` | `measured` |

LLM traces record the model used during export. They are teaching artifacts,
not performance benchmarks.

## Traces

Grounded traces live in `lab_traces.json`. LLM traces live in `lab_traces_llm.json`.
The two files are independent — LLM export does not merge into the grounded file.

Observable events:

```text
user_request → entity_resolution → retrieval_started → retrieval_step
→ subgraph_retrieved → context_assembled → [model_request → model_response]
→ final_answer → termination
```

Deliberately **not** stored: chain-of-thought, hidden reasoning, internal
deliberation, grounding scores, or answer-quality scores.

## Security

Application-owned boundaries:

- graph source (`data/graph.ttl`)
- label-based entity resolution
- max hops and allowed predicates
- subgraph construction and context assembly
- termination rules

The LLM does not control traversal or graph access.

## Trace export

```bash
# Deterministic — writes lab_traces.json (safe to regenerate with --force)
uv run python export_lab_traces.py --mode graph_grounded --force

# Requires OPENAI_API_KEY — writes lab_traces_llm.json (separate file)
uv run python export_lab_traces.py --mode graphrag_llm
```

CI runs deterministic validation only. Tests use `model=mock` for LLM grounding
checks — mock provenance is never stored as production measured traces.

## Quick start

```bash
cd examples/knowledge-graphs/04-graphrag
cp .env.example .env   # optional — only for LLM mode
uv sync
uv run python main.py --case entity-retrieval-knowledge-platform
uv run python main.py --case multi-hop-alice-technologies --show-sequence
```

## Measured cases

| Case | Question |
|------|----------|
| `ENTITY_RETRIEVAL` | Who works on the Knowledge Platform? |
| `MULTI_HOP_RETRIEVAL` | Which technologies are used by projects Alice works on? |
| `RELATIONSHIP_GROUNDED_ANSWER` | Which company employs Alice and what project does she work on? |
| `NO_RELEVANT_SUBGRAPH` | What technology does Alice directly use? |

## Limitations

- Small local RDF graph (Turtle + rdflib)
- Label-based entity resolution — not semantic entity linking
- Bounded retrieval (`max_hops`)
- No embeddings or vector search
- No graph database (Neo4j, etc.)
- No LangChain / LlamaIndex GraphRAG abstractions
- Deterministic graph-grounded mode; LLM mode is model/provider-specific
- Not a GraphRAG benchmark — no answer quality score
- No claim that graph retrieval is universally superior to conventional RAG

## Learning path

```text
01 — RDF & Graph Traversal
02 — SPARQL & Graph Queries
03 — SPARQL Updates
04 — GraphRAG (this example)
```
