# Chunking Strategies

**Example ID:** `chunking-strategies`

This example applies the evaluation discipline from Example 05 to an early RAG
design decision:

> How does chunking strategy affect what a RAG system can retrieve?

It is a **controlled retrieval experiment**, not a gallery of chunking demos.

**This is a small evaluation set, not a benchmark of chunking systems.**

## What this example demonstrates

Same corpus → different chunking strategy → same embedding model → same dense
retrieval → same evaluation queries → compare measured retrieval quality.

Do **not** assume a more sophisticated chunker must win. Measured regressions,
ties, and mixed aggregates are part of the lesson.

## Why chunking matters

Chunking decides the retrieval units. The same handbook sentence can sit intact
inside one section chunk, or be split across fixed windows. Embedding and ranking
only see those units — so boundary choices change Recall, MRR, and nDCG even when
everything downstream is fixed.

## Where it fits

```text
01 Basic RAG
02 Hybrid RAG
03 Reranking
04 Query Transformation
05 Retrieval Evaluation   ← learn how to measure retrieval
06 Chunking Strategies    ← use evaluation to test a design decision  ← you are here
```

### Connection to Example 05

> **Retrieval Evaluation (Example 05)** taught us how to measure retrieval
> quality.
>
> **Chunking Strategies (Example 06)** uses those same measurements to evaluate
> one specific RAG design decision while keeping the rest of the retrieval
> pipeline constant.

## Controlled experiment

**Only the chunking strategy changes.**

| Held constant | Value |
|---------------|-------|
| Corpus | `data/sample.md` (Acme Edge handbook; same family as Example 05) |
| Embedding model | `text-embedding-3-small` |
| Retrieval | Dense cosine similarity |
| Evaluation set | `data/eval_set.json` (10 queries, 16 evidence units) |
| K | `3` |
| Relevance method | Evidence-span grading (frozen before scoring) |

Independent variable: **chunking strategy**.

**Why dense-only?** Examples 02–04 already vary hybrid / rerank / query
transform on a fixed chunker. This lab isolates chunking, so advanced retrieval
machinery is omitted on purpose.

```text
                    SOURCE
                      │
         ┌────────────┼────────────┐
         ↓            ↓            ↓
      FIXED       RECURSIVE    STRUCTURE
         ↓            ↓            ↓
       chunks       chunks       chunks
         ↓            ↓            ↓
      embedding    embedding    embedding
         ↓            ↓            ↓
      retrieval    retrieval    retrieval
         └────────────┼────────────┘
                      ↓
                   COMPARE
                      ↓
                 EVALUATION
```

Generation is out of scope. Better retrieval is upstream of generation; this
experiment measures retrieval only.

## Strategies

Configuration was chosen **before** the final measured run so operating ranges
are comparable without forcing identical boundaries.

### Fixed-size (baseline)

| Setting | Value |
|---------|-------|
| `chunk_size` | 400 |
| `chunk_overlap` | 50 |
| **Unit** | **characters** (Python `str` length — not tokens) |

Straightforward sliding windows. Explicitly the baseline.

### Recursive

| Setting | Value |
|---------|-------|
| `target_size` | 400 characters |
| `chunk_overlap` | 50 characters |
| Separators (in order) | `\n## ` → `\n\n` → `\n` → `. ` → ` ` → hard split |

Deterministic, inspectable recursion — not an opaque framework splitter.

On this handbook, recursion often continues past section splits into
paragraph/sentence pieces, so measured average length ends up finer than the
400-character target (see Results). That length distribution is a fairness
confound to read alongside the metrics for this corpus.

### Structure-aware

One chunk per markdown `##` section. The heading stays with its body.

This is **structure-aware** chunking — not “semantic chunking.” Boundaries
follow document headings, not embeddings or similarity detectors.

No fixed length target; section lengths vary with the source.

## Corpus

Reuses the Acme Edge Platform support handbook from Example 05.

**Why reuse (documented before scoring):** sections already mix heading +
explanation, procedures, caveats, and multi-paragraph topics with lengths from
~220–600 characters — enough for fixed windows to cross boundaries while
structure-aware keeps sections intact. No corpus edits were made after seeing
retrieval results.

## Evaluation methodology

Relevance cannot be copied as strategy-specific chunk IDs — each strategy
produces different boundaries.

Frozen model in `data/eval_set.json`:

1. **Evidence units** — unique source anchors resolved to `[start, end)`.
2. **Evaluation queries** — each lists required evidence with grades 1–3.
3. **Inheritance** — chunks inherit evidence IDs whose spans they overlap.
4. **Chunk grade (deterministic):**
   - full containment of evidence E → grade = `evidence_grades[E]`
   - partial overlap with labeled evidence → grade at least 1
   - chunk grade = max over applicable rules
5. **Recall@K / RR / MRR** — binary relevance with threshold **grade ≥ 1**
6. **nDCG@K** — graded relevance from the derived per-strategy map
7. **Evidence coverage** — fraction of required evidence IDs present in top-K

No LLM judge. Judgments frozen before pipeline scoring.

## Metrics

| Metric | Definition |
|--------|------------|
| Recall@K | \|relevant ∩ top-K\| / \|relevant\| |
| MRR | mean of per-query reciprocal ranks |
| nDCG@K | DCG@K / IDCG@K (`2^rel − 1` gain) |
| Evidence coverage | required evidence IDs found in top-K / required |

Chunking-specific measurements: chunk count, length min/avg/max, overlap extra
characters, evidence fragmentation rate, plus separate timings for chunking,
embedding, indexing, and retrieval.

## Run

```bash
cd examples/rag/06-chunking-strategies
cp .env.example .env   # add OPENAI_API_KEY
uv sync
uv run python main.py
uv run python export_lab_traces.py
```

Tests (no API keys / no model downloads):

```bash
uv sync --extra dev
uv run pytest -q
```

## Why boundaries matter (illustration)

Educational only — not a measured claim. Adapted from the handbook's NebulaAuth
section to show how the *same* source text becomes different retrieval units.

```text
SOURCE DOCUMENT
------------------------------------------------
## NebulaAuth token TTL and silent credential refresh

NebulaAuth access tokens expire
after eight hours of inactivity.
When the TTL elapses, Edge clients
receive HTTP 401 …
------------------------------------------------

Fixed-size (character windows)

  fixed-0013
  -----------------------------
  … ## NebulaAuth token TTL …
  NebulaAuth access tokens expire
  after eight hours of inactivity.
  When the TTL elapses …

  fixed-0014
  -----------------------------
  … Enable silent credential refresh
  with `nebula auth refresh --background`
  so tokens renew …
------------------------------------------------

Structure-aware (## section = one chunk)

  structure-0012
  -----------------------------
  ## NebulaAuth token TTL and silent
  credential refresh

  NebulaAuth access tokens expire
  after eight hours of inactivity.
  When the TTL elapses, Edge clients
  receive HTTP 401 …
  Enable silent credential refresh …
------------------------------------------------
```

Different chunking strategies expose different retrieval units to the search
system.

## Results

Measured run recorded in `evaluation_report.json` / `lab_traces.json`
(`recordedAt`: 2026-08-06T05:31:55Z).

### Chunk statistics

| Strategy | Chunks | Avg len | Min | Max | Overlap extra chars | Fragmented evidence |
|----------|-------:|--------:|----:|----:|--------------------:|--------------------:|
| Fixed | 24 | 398.6 | 371 | 400 | 1145 | 6 / 16 (37.5%) |
| Recursive | 47 | 205.1 | 43 | 440 | 1219 | 3 / 16 (18.8%) |
| Structure-aware | 24 | 349.0 | 221 | 593 | 0 | 0 / 16 (0%) |

**Fairness note:** On this corpus, recursive landed finer-grained than the
400-character target (avg ~205). Structure-aware allows longer sections
(max 593). Length distribution is part of interpreting these measured results —
not a reason to treat any strategy as universally preferable.

### Aggregate retrieval metrics (K=3)

| Strategy | Recall@3 | MRR | nDCG@3 | Evidence coverage |
|----------|---------:|----:|-------:|------------------:|
| Fixed | 0.775 | **1.000** | 0.814 | 0.817 |
| Recursive | 0.753 | 0.900 | 0.778 | 0.867 |
| Structure-aware | **0.933** | 0.933 | **0.937** | **0.942** |

Different chunking strategies optimize different retrieval behaviors on this
evaluation set:

- Structure-aware produced the highest Recall@3 and nDCG@3 measured here.
- Fixed-size produced the highest MRR.
- Recursive did not lead any aggregate metric in this experiment.

That pattern illustrates an engineering trade-off, not a universally best
strategy.

### Index / retrieval timings (measured, no cross-strategy cache)

Each strategy embedded independently with the same model.

| Strategy | Chunking | Embedding | Index | Total retrieval (10 queries) |
|----------|---------:|----------:|------:|-----------------------------:|
| Fixed | 0 ms | 1494 ms | 1 ms | 3648 ms |
| Recursive | 1 ms | 498 ms | 1 ms | 3584 ms |
| Structure-aware | 0 ms | 760 ms | 1 ms | 3424 ms |

Embedding latency varies with batch size and API conditions — do not treat
embedding-time gaps as a property of the chunking algorithm itself.

## Example cases

Selected from measured behavior on this evaluation set (not pre-scripted
winners).

### BOUNDARY — `auth-idle-timeout`

> Why does everyone have to unlock the app again after being idle half a day?

On this query, NebulaAuth TTL evidence sits in one structure-aware section
chunk, while fixed windows create multiple graded-relevant pieces (intact
primary + partial overlap). Measured in this experiment: structure
Recall@3=1.0 and evidence coverage=1.0; fixed Recall@3=0.5 and coverage=0.5
(top-K hit a partial fragment, missed the primary intact window).

### CONTEXT_PRESERVED — `auth-vs-sso`

> Is AUTH_TOKEN_EXPIRED the same problem as an IdP outage during SSO sign-in?

Needs two evidence units from different sections. In this evaluation,
structure-aware retrieved both primary sections at ranks 1–2 (nDCG@3=1.0).
Recursive covered the required evidence but ranked it later (RR=0.5). Fixed
retrieved more partial grades and a lower nDCG on this query.

### EQUIVALENT — `dns-relay`

> Which DNS resolvers should edge sites use so relay hostnames resolve?

All three strategies: Recall@3=1.0, RR=1.0, nDCG@3=1.0. Different chunk texts;
equivalent retrieval quality for this information need on this evaluation set.

## What the results mean

- There is **no universally best chunking strategy**.
- Chunking changes the retrieval units presented to search.
- A strategy can help one information need and hurt another.
- On this evaluation set, aggregate leaders disagree across metrics (structure
  on Recall/nDCG; fixed on MRR).
- Finer recursive pieces and longer structure sections are confounds — always
  read length stats with the scores.

## Limitations

- Small evaluation set (10 queries)
- One corpus / domain
- One embedding model
- One K
- Chunk-size configuration sensitivity
- Recursive effective granularity finer than its nominal target on this corpus
- Structure-aware max chunk length higher than fixed
- Results are not a universal benchmark
- Generation quality not evaluated

## Engineering takeaway

Chunking is a retrieval design decision rather than just a preprocessing step.

Before changing chunk sizes or chunking algorithms, measure the effect on
Recall@K, MRR, and nDCG using a representative evaluation set.

Prefer:

baseline → change one variable → measure → inspect failures → decide
