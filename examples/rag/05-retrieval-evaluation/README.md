# Retrieval Evaluation

**Example ID:** `retrieval-evaluation`

This example does **not** introduce another retrieval technique.

It answers:

> How do we know whether the retrieval improvements from examples 01–04
> actually improve retrieval quality?

The teaching point:

> Better-looking retrieval is not enough. Retrieval quality should be measured
> against explicit relevance judgments.

**This is a small teaching evaluation set, not a benchmark of retrieval systems.**

## Where it fits

```text
01 Basic RAG
02 Hybrid RAG
03 Reranking
04 Query Transformation
05 Retrieval Evaluation   ← you are here
```

Examples 01–04 add retrieval machinery. Example 05 measures that machinery.

### Not a guaranteed quality ladder

The Cookbook progression is a sequence of *capabilities*, not a promise that
each stage is better than the last:

| Stage | What it changes |
|-------|-----------------|
| Dense retrieval | Semantic candidate discovery |
| Hybrid retrieval | Combines semantic and lexical signals |
| Reranking | Reorders candidates with deeper query–document scoring |
| Query Transformation | Changes the search formulations used for discovery |
| Evaluation | Determines whether those changes helped for *this* workload |

Do **not** read:

```text
Basic → Hybrid → Reranking → Query Transformation
```

as:

```text
worse → better → even better → best
```

Engineering principle:

> More sophisticated retrieval is not automatically better retrieval. Each
> technique changes system behavior, and its value depends on the queries,
> corpus, configuration, relevance judgments, and metric being optimized.

## Retrieval evaluation vs generation evaluation

| | Retrieval evaluation | Generation evaluation |
|---|---|---|
| Question | Did we retrieve the evidence? | Did the model produce a good answer from that evidence? |
| Inputs | ranked chunk IDs + judgments | answer text + references / rubrics |
| Metrics here | Recall@K, MRR, nDCG@K | *(not this example)* |

Higher retrieval metrics do **not** automatically guarantee better generated
answers. This Cookbook example stops at retrieval.

It deliberately does **not** include LLM-as-judge, RAGAS, faithfulness,
hallucination scoring, or answer correctness.

## Golden evaluation set

Path: [`data/eval_queries.json`](data/eval_queries.json)

Each case has:

- stable query ID
- user query (the information need)
- graded relevance judgments (`chunk_id → grade`)
- rationale explaining why each chunk is relevant

### Graded relevance (0–3)

| Grade | Meaning |
|------:|---------|
| 3 | Directly answers the information need |
| 2 | Strongly related supporting evidence |
| 1 | Tangentially related / easy to confuse |
| 0 | Not relevant (implicit for unlabeled chunks) |

### How metrics use those grades

| Metric family | How grades are used |
|---------------|---------------------|
| Recall@K, RR / MRR | Binary: **grade ≥ 1** counts as relevant |
| nDCG@K | Graded: preserves 0–3 so stronger evidence near the top scores higher |

Recall and MRR collapse the graded judgments into a binary relevant / not
relevant decision using `grade ≥ 1`. nDCG preserves the relevance grades, which
makes it sensitive to whether the *strongest* evidence appears near the top.

Judgments were written from the handbook content **before** measuring pipeline
outputs, then frozen for scoring.

Caveats of a tiny golden set:

- Incomplete judgments: unlabeled relevant chunks become false negatives.
- Subjective relevance: different judges may disagree on grade 1 vs 2.
- Representativeness: 8 queries cannot stand in for production traffic.
- Overfitting: do not tune systems solely against this teaching set.

## Metrics

### Recall@K

\[
\mathrm{Recall@K} = \frac{|\{\text{relevant}\} \cap \mathrm{top\text{-}K}|}{|\{\text{relevant}\}|}
\]

Relevant = chunks with grade ≥ 1. Multiple relevant chunks are supported.

Asks: *how much relevant evidence appeared in top-K?*

### Reciprocal Rank / MRR

Per query:

\[
\mathrm{RR} = \begin{cases}
1 / \mathrm{rank}(\text{first relevant}) & \text{if found within evaluated ranking} \\
0 & \text{otherwise}
\end{cases}
\]

\[
\mathrm{MRR} = \mathrm{mean}(\mathrm{RR})
\]

RR is per-query; MRR is the dataset aggregate. “Relevant” again means grade ≥ 1.

Asks: *how early does the first relevant result appear?*

### nDCG@K (graded)

Standard Järvelin & Kekäläinen formulation:

\[
\mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)}
\]

\[
\mathrm{IDCG@K} = \mathrm{DCG@K}\text{ of the ideal ranking of known grades}
\]

\[
\mathrm{nDCG@K} = \begin{cases}
\mathrm{DCG@K} / \mathrm{IDCG@K} & \mathrm{IDCG@K} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

Asks: *was the more highly relevant evidence placed near the top?*

Two systems can share the same Recall@K and still differ in nDCG@K when the
ordering (or graded strength) of retrieved hits changes. A single metric does
not fully describe retrieval quality — evaluate per query as well as in
aggregate.

## Architecture

```text
                    Golden evaluation set
                           │
                           ▼
                     Evaluation runner
                           │
     ┌─────────────┬───────┼───────────┬────────────────┐
     ▼             ▼       ▼           ▼                │
  Dense         Hybrid  Hybrid+     Query Transform     │
 (01)           (02)    Rerank (03) + Hybrid+Rerank (04)│
     │             │       │           │                │
     ▼             ▼       ▼           ▼                │
 ranked chunks  ranked   ranked     ranked chunks       │
     │             │       │           │                │
     └─────────────┴───────┼───────────┘                │
                           ▼                            │
                 Relevance comparison                   │
                           │                            │
                           ▼                            │
               Recall@K / RR·MRR / nDCG@K               │
                           │                            │
                           ▼                            │
                    Evaluation report ◄─────────────────┘
```

Conceptual separation:

```text
retriever → ranked results
evaluator + golden judgments → metrics
```

Metric functions do not import retrieval implementations.

## Pipelines compared

| Pipeline | Corresponds to | Ranking evaluated |
|----------|----------------|-------------------|
| `dense` | 01 Basic RAG | Dense top-K |
| `hybrid` | 02 Hybrid RAG | Dense + BM25 → RRF top-K |
| `hybrid-reranked` | 03 Reranking | Hybrid `candidate_k` → cross-encoder top-K |
| `query-transform` | 04 Query Transformation | Multi-query hybrid → aggregate → rerank top-K |

All use:

- the same corpus (`data/sample.md`)
- the same evaluation queries
- the same relevance judgments
- the same K (default `eval_k=3`)

Documented differences (intentional, matching prior examples):

- `dense` / `hybrid` produce a top-K list directly.
- `hybrid-reranked` / `query-transform` first retrieve `candidate_k=5`, then
  rerank down to K.
- `query-transform` may retrieve with transformed alternatives, but **reranking
  and relevance judgments stay on the original information need**.

## Query transformation fairness

For `query-transform`:

1. The evaluation case query is the original information need.
2. Alternatives may be generated as in example 04.
3. Retrieval may use original + alternatives.
4. Reranking uses the original query.
5. Metrics compare retrieved chunk IDs against judgments for that original need.

Transformed strings are never treated as independent evaluation questions.

## Candidate discovery is not the same as final top-K quality

Readers coming from [`04-query-transformation`](../04-query-transformation)
may notice that example 04 and this example look at the idle-auth case
differently. They are **not** contradictory — they measure different points in
the pipeline.

| Example | Question it asks |
|---------|------------------|
| **04** | Can query transformation *discover* evidence the original query did not make adequately available? |
| **05** | Does that evidence *survive* the complete pipeline into the evaluated final top-K? |

Example 04 showed that, for the idle-auth information need, multi-query
retrieval can surface `sample-12` (`AUTH_TOKEN_EXPIRED`) into a broader
candidate pool so downstream stages can consider it.

Example 05 evaluates **final** ranked lists at **K = 3**. In the measured run
here:

- Dense retrieves `sample-12` at rank 3.
- Hybrid, hybrid+rerank, and query-transform do **not** retain `sample-12` in
  the evaluated final top-3.

So:

> Query Transformation can make a chunk available to downstream ranking without
> guaranteeing that the chunk survives into the final top-K.

Candidate recall and final top-K retrieval quality are different evaluation
cuts. Measure the cut that matches the decision you care about.

## Failure analysis

Per query, relevant chunks (grade ≥ 1) are labeled:

| Label | Meaning |
|-------|---------|
| GOOD | Relevant evidence near the top (rank 1 by default) |
| LATE | Retrieved in top-K but ranked lower |
| MISS | Relevant chunk absent from top-K |
| MIXED | Some relevant chunks hit, others missed |

These labels explain outcomes; they do not replace metrics.

## Run

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and `OPENAI_API_KEY`
for embeddings (and query transformation when that pipeline is selected).
Reranking is local.

```bash
cd examples/rag/05-retrieval-evaluation
cp .env.example .env
# set OPENAI_API_KEY

uv sync
uv run python main.py
```

Useful options:

```bash
# One pipeline
uv run python main.py --pipeline hybrid-reranked

# Inspect one query
uv run python main.py --query-id auth-idle-timeout

# Aggregates only
uv run python main.py --aggregates-only

# Different K
uv run python main.py --k 5
```

API key is required for: embeddings, query transformation.
No API key is required for: BM25, RRF, metric computation, unit tests.
Reranker weights download on first use of reranked pipelines.

## Measured results

Regenerate with:

```bash
uv run python export_lab_traces.py
```

Results below are from the committed measured run in `evaluation_report.json`
(`recordedAt: 2026-08-05`). They are **not** hard-coded teaching fiction — if a
supposedly advanced pipeline ties or loses, that is the lesson.

**This dataset is a teaching set, not a benchmark.**

### Aggregate (K=3, 8 queries)

| Pipeline | Recall@3 | MRR | nDCG@3 | Σ retrieval latency |
|----------|---------:|----:|-------:|--------------------:|
| `dense` | 0.9167 | 0.9167 | 0.9211 | 4121 ms |
| `hybrid` | 0.7917 | 0.8750 | 0.8586 | 2866 ms |
| `hybrid-reranked` | 0.7500 | 0.8750 | 0.8324 | 3957 ms |
| `query-transform` | 0.7500 | 0.8750 | 0.8457 | 19176 ms |

On this small 8-query teaching set at K=3, Dense retrieval produced the highest
aggregate Recall, MRR, and nDCG. That does **not** establish Dense as a
universally superior retrieval strategy. It demonstrates why retrieval changes
should be evaluated against the application's own queries and relevance
judgments.

In this recorded run, Query Transformation had substantially higher latency
and lower aggregate retrieval metrics on this teaching set than Dense. That is
a measured observation for this workload and cut — not a general claim that
Query Transformation is “worse.”

### Aggregate metrics hide query-level behavior

Different queries can respond differently to the same pipeline change. Evaluate
**per query** as well as in aggregate. In principle:

- lexical identifiers / error codes may benefit from lexical signals
- semantic wording may already work extremely well with Dense alone
- query expansion may improve candidate coverage but introduce noise
- reranking may change ordering without improving Recall@K
- an added stage may help some queries and regress others

Only the teaching cases below make specific claims about *this* evaluation set,
and only where the measured numbers support them.

### Teaching cases from measured runs

| Trace | Class | What the numbers show |
|-------|-------|------------------------|
| `discover-auth-idle-timeout` | DISCOVER | Dense retrieves `sample-12` at rank 3; hybrid / reranked / query-transform **miss** it in final top-3 |
| `ranking-sensitivity-profile-async` | RANKING_SENSITIVITY | Recall@3 tied at 0.667; nDCG@3 moves (~0.947 → ~0.798) when secondary graded evidence is reordered — not a ranking “win” |
| `regression-econn-42-remediation` | REGRESSION | Dense/hybrid Recall@3=0.667; reranked/transform drop to 0.333 and lower nDCG while RR stays 1.0 |

#### DISCOVER — `auth-idle-timeout`

Dense finds the relevant `sample-12` inside final top-3 (late, rank 3). The
hybrid-based pipelines miss it at this cut. Lesson: you cannot rerank evidence
that never enters the evaluated ranking. Together with the Example 04 → 05
section above, this also shows candidate discovery ≠ final top-K retention.

#### RANKING_SENSITIVITY — `profile-async`

Recall asks how much relevant evidence appeared in top-K. nDCG additionally
asks whether more highly relevant evidence was placed near the top.

On this query, Recall@3 stays tied at 0.667 across pipelines (primary
`sample-4` at rank 1), while nDCG@3 changes when secondary graded evidence is
swapped (`sample-5` grade 2 vs `sample-3` grade 1). In the measured run the
hybrid-reranked nDCG is *lower* (~0.798) than dense/hybrid (~0.947) — this is
ranking *sensitivity*, not a claim that the advanced pipeline improved
ordering.

#### REGRESSION — `econn-42-remediation`

Dense/hybrid Recall@3=0.667; hybrid-reranked and query-transform drop to 0.333
and lower nDCG while RR stays 1.0. An advanced retrieval pipeline can regress a
specific information need even when the added technique is useful elsewhere —
scoped to this measured case, not a universal rule.

### Engineering takeaway

Do not choose a retrieval architecture by complexity. Start with a baseline,
define a representative evaluation set, measure it, and add retrieval stages
only when they improve the quality or behavior your application needs:

```text
baseline → measure → change → measure again
```

That is not an argument that everyone should use Dense retrieval. It is an
argument for selecting architecture through evaluation rather than assuming
quality from sophistication.

## Latency / cost

Where pipelines record timings, the report includes measured:

- retrieval latency
- transformation latency (query-transform)
- reranking latency
- total per-query retrieval latency

These timings are recorded executions for this environment and model stack —
**not benchmarks**, and not portable cost estimates.

Evaluation should consider both retrieval quality and operational cost. A more
complex pipeline may add latency or model calls without improving the metric
that matters for a particular workload. In this recorded run, that trade-off is
visible in the aggregate table above.

## Export traces

```bash
uv run python export_lab_traces.py
```

Writes measured `lab_traces.json` for a future Interactive Lab at
`/labs/rag/retrieval-evaluation`, plus `evaluation_report.json` for README
inspection.

Trace stages (approx.):

1. Evaluation dataset
2. Relevance judgments
3. Run retrieval
4. Inspect ranked results
5. Calculate Recall@K
6. Calculate RR / MRR
7. Calculate nDCG@K
8. Compare pipelines
9. Inspect failure cases
10. Evaluation summary

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```

Tests cover metric edge cases, evaluator aggregation, failure labels, pipeline
injection (no paid APIs / no model downloads), and lab-trace schema checks.

## Limitations

- Tiny teaching set (8 queries) — not production-representative.
- Judgments are author-written and may be incomplete.
- Dense/hybrid vs reranked pipelines differ in candidate breadth by design.
- Query transformation depends on an LLM; alternatives vary across providers.
- Does not evaluate generation quality.
- Does not claim statistical significance.
- Aggregate winners on this set do not generalize to other corpora or K values.

## Relationship to previous examples

| Example | Contribution | Evaluated here as |
|---------|--------------|-------------------|
| 01 | Dense retrieval | `dense` |
| 02 | Dense + BM25 + RRF | `hybrid` |
| 03 | Cross-encoder reranking | `hybrid-reranked` |
| 04 | Multi-query transformation | `query-transform` |
| 05 | Measure the above | metrics + report |

Examples 01–04 are left behaviorally unchanged.
