# Query Transformation

**Example ID:** `query-transformation`

This extends [`03-reranking`](../03-reranking) with an explicit **multi-query
retrieval** stage. Multi-query retrieval is **one** Query Transformation
technique — not HyDE, not decomposition, not step-back prompting.

The teaching point:

> The user's question is not always the best retrieval query.

## Scope

- Reuses chunking, embeddings, dense retrieval, BM25, RRF, local cross-encoder
  reranking, and grounded generation from Hybrid RAG / Reranking.
- Adds bounded LLM query transformation only in this example; `01`–`03` are
  unchanged.
- Caps alternatives with `max_alternative_queries` (example default: `2`).
  That is an example configuration, not a universal optimum.

## Original vs transformed queries

This example retrieves with:

```text
original query + transformed alternatives
```

for comparison, provenance, and teaching.

That is **not** a claim that production Query Transformation pipelines must
always retrieve using the original query. The original question is always kept
for provenance/debugging and is the relevance target for **reranking** and
**generation**.

## Architecture

```text
original user question
        ↓
query transformation  (bounded alternatives)
        ↓
Q0 original + Q1..Qn alternatives
        ↓
[for each Qi] dense + BM25 + RRF   (independent Hybrid retrieval)
        ↓
merge + dedup by chunk id  (with query→candidate provenance)
        ↓
cross-query RRF aggregation
        ↓
cross-encoder rerank  (ORIGINAL question only)
        ↓
final_context_k
        ↓
generation  (ORIGINAL question + final reranked context only)
```

Per-query retrieval is sequential in this implementation (not parallelized).
Recorded timings are measured wall-clock values for this run — not benchmarks.

## Configuration

| Setting | Default | Role |
|---------|---------|------|
| `max_alternative_queries` | `2` | Cap on generated alternatives |
| `dense_top_k` / `lexical_top_k` | `8` | Per-query retrieval breadth |
| `candidate_k` | `5` | Per-query fused pool / merged top-k |
| `final_context_k` | `3` | Chunks sent to the prompt after rerank |
| `rrf_k` | `60` | Conventional RRF constant |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local CE |
| `query_transformer_model` | `gpt-4o-mini` | Alternative-query generation |

`candidate_k=5` is intentional for this larger handbook corpus so vocabulary
mismatch can leave a relevant section outside the original-only pool.

## Run

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and `OPENAI_API_KEY`
for embeddings, query transformation, and generation. Reranking is local.

```bash
cd examples/rag/04-query-transformation
cp .env.example .env
# set OPENAI_API_KEY

uv sync
uv run python main.py
```

API key is required for: transformation, embeddings, generation.
No API key is required for: BM25, RRF, cross-encoder reranking, unit tests.

## Measured results

Traces regenerated via `uv run python export_lab_traces.py` after corpus/query
iteration. Classifications follow **measured** original-only vs multi-query
behavior.

| Case | Class | Original-only behavior | Multi-query behavior | Final context impact | Lesson |
|------|-------|------------------------|----------------------|----------------------|--------|
| Idle re-auth | `TRANSFORM_HELPS` | Relevant `AUTH_TOKEN_EXPIRED` chunk (`sample-12`) **absent** from `candidate_k` | A transformed query discovers `sample-12`; merge provenance records which Qi found it | `sample-12` enters `final_context_k` after rerank against the original question | Transformation can surface evidence original retrieval never made available |
| `E_CONN_42` | `REDUNDANT` | Already retrieves the remediation chunk | Alternatives retrieve the **same** candidate set (`discovered=[]`) | Final context unchanged | When the original query already matches well, extra queries mostly add duplicate work |
| Vague platform complaint | `ADDS_NOISE` | Broad but coherent pool | Alternatives introduce extra candidates (e.g. caching / logs / metrics) | At least one weaker candidate can survive into final context | Improving recall can also increase candidate noise |

### Signature Case A (detail)

Original question:

> Why does everyone have to unlock the app again after being idle half a day?

Handbook answer uses NebulaAuth jargon (`AUTH_TOKEN_EXPIRED`, token TTL, silent
credential refresh) rather than “unlock the app”.

Measured pattern:

1. Original-only Hybrid pool does **not** include `sample-12`.
2. Multi-query transformation emits retrieval-oriented alternatives.
3. At least one alternative discovers `sample-12` (see `foundBy` / provenance).
4. Cross-encoder reranks against the **original** question.
5. `sample-12` enters final context — evidence the original-only path could not
   evaluate.

### Case B deduplication

Identifier-rich queries such as `E_CONN_42` produce high duplicate rates across
Q0/Q1/Q2 (same chunks retrieved repeatedly). Dedup metrics
(`beforeDedup` / `afterDedup` / `duplicates`) make that cost visible.

## Tradeoffs

- Transformation can improve recall when user language ≠ handbook language.
- Extra queries multiply retrieval work (sequential here).
- Broader formulations can inject noise; reranking helps but is not perfect.

## Limitations

This example does **not** include:

- HyDE
- query decomposition
- step-back prompting
- agentic query planning
- adaptive / self-RAG
- evaluation frameworks
- Interactive Lab UI (traces prepare a future Lab)

## Export traces

```bash
uv run python export_lab_traces.py
```

Writes measured `lab_traces.json` with original-only baseline **and**
multi-query path for every teaching case.

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```

Tests inject transformers/reranker scores — no paid APIs, no model downloads.
