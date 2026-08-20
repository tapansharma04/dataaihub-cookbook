# SPARQL Updates: INSERT, DELETE & Graph Mutation

**Example ID:** `sparql-updates`

Follows: [02 — SPARQL & Graph Queries](../02-sparql-queries/)

## What this example teaches

SPARQL UPDATE modifies an RDF graph. `SELECT` reads the graph; SPARQL UPDATE
**changes** it. This example demonstrates INSERT, DELETE, and combined
delete+insert operations executed by rdflib against a local Turtle fixture.

```text
SPARQL UPDATE → changed RDF graph → SELECT verification → result
```

No LLM, embeddings, vector database, GraphRAG, or external graph database is
used.

## From querying to mutation

**KG #2** asks:

> How do I declaratively describe the pattern I want to read?

**KG #3** asks:

> How do I declaratively change the graph, then verify the new state?

```text
KG #2:  RDF graph → SPARQL query → bindings → result
KG #3:  RDF graph → SPARQL UPDATE → changed graph → verification
```

Core teaching statement:

> **SELECT reads the graph. SPARQL UPDATE changes the graph.**

## INSERT DATA

`INSERT DATA` adds explicitly specified RDF triples:

```sparql
INSERT DATA {
  ex:billingPortal ex:uses ex:redis .
}
```

Before: Billing Portal uses PostgreSQL.  
After: Billing Portal uses PostgreSQL **and** Redis.

## INSERT WHERE

`INSERT WHERE` derives new triples from existing graph patterns:

```sparql
INSERT {
  ?person ex:uses ?technology .
}
WHERE {
  ?person ex:worksOn ?project .
  ?project ex:uses ?technology .
}
```

Alice, Bob, and Carol each gain an `ex:uses` link to the technology used by
their project. The SPARQL engine performs the derivation — not Python.

## DELETE DATA

`DELETE DATA` removes explicitly specified RDF triples:

```sparql
DELETE DATA {
  ex:billingPortal ex:uses ex:postgresql .
}
```

Before: the triple exists. After: it does not. Verification returns zero rows.

## DELETE + INSERT

One SPARQL UPDATE can delete and insert together:

```sparql
DELETE {
  ex:billingPortal ex:uses ex:postgresql .
}
INSERT {
  ex:billingPortal ex:uses ex:redis .
}
WHERE {
  ex:billingPortal ex:uses ex:postgresql .
}
```

This example demonstrates graph mutation and subsequent verification. It does
**not** claim database transaction or atomicity guarantees beyond what the
local rdflib update operation provides.

## Before and after

Each case records a focused before/after triple state relevant to the
mutation. Inserted and deleted triples are derived from that measured
difference so they cannot disagree with the graph state.

## Verification

After every UPDATE, the example runs a real `SELECT` via `Graph.query()` and
inspects the bindings. That reinforces the KG #2 → KG #3 relationship:
mutation changes the graph; query reads the new state.

## Fresh graph per case

Every measured case starts from a clean load of `data/graph.ttl`. Cases never
share a mutated graph. Running INSERT_DATA does not affect a later independent
DELETE_DATA run.

## SPARQL UPDATE vs application graph mutation

This example executes real SPARQL UPDATE through `rdflib.Graph.update(...)`.
It does **not** implement the measured mutation with `graph.add(...)` /
`graph.remove(...)`. Python may inspect and normalize the graph afterward.

## Security

Arbitrary external SPARQL UPDATE execution is out of scope. The example uses:

- a local Turtle fixture
- predefined measured update queries
- a local in-memory rdflib graph

Prohibited operations include `SERVICE`, `LOAD`, and external `FROM` URLs.
There is no public update endpoint.

## Provenance

| Field | Value |
|-------|-------|
| model | `not_used` |
| tools | `measured` |
| metrics | `measured` |

No LLM generates or interprets the updates.

## Limitations

- Local RDF graph only (no Neo4j, Neptune, Stardog, etc.)
- Deterministic fixtures
- No external SPARQL endpoint
- No LLM / embeddings / GraphRAG
- No production transaction semantics
- Not an update performance benchmark

## Measured cases

| Case | Trace ID | Idea |
|------|----------|------|
| INSERT_DATA | `insert-data-billing-portal-redis` | Explicit triple insert |
| INSERT_WHERE | `insert-where-person-uses-technology` | Pattern-derived insert |
| DELETE_DATA | `delete-data-billing-portal-postgresql` | Explicit triple delete |
| UPDATE_AND_VERIFY | `update-and-verify-billing-portal-technology` | Delete + insert + verify |

## Layout

```text
examples/knowledge-graphs/03-sparql-updates/
├── README.md
├── pyproject.toml
├── .env.example
├── config.py
├── main.py
├── export_lab_traces.py
├── lab_traces.json
├── data/
│   └── graph.ttl
├── sparql/
│   ├── graph.py
│   ├── queries.py
│   ├── runner.py
│   ├── cases.py
│   ├── trace.py
│   ├── model.py
│   └── vocab.py
└── tests/
```

## Run

```bash
cd examples/knowledge-graphs/03-sparql-updates
uv sync --extra dev
uv run python main.py --list-cases
uv run python main.py --case insert-data-billing-portal-redis --show-sequence
uv run pytest -q
uv run python export_lab_traces.py
```

No API key. No network.
