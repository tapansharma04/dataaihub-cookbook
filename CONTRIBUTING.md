# Contributing

Thank you for helping improve the DataAIHub Cookbook.

## What belongs here

This repo is for **small, concept-focused reference implementations**.

It is not for:

- production applications
- large demo projects
- framework wrappers that hide the idea
- vendor-specific forks of the same concept

## Before you open a PR

Ask:

1. Does this teach **one** primary concept?
2. Is the code smaller than it needs to be?
3. Can a developer understand the architecture in two minutes?
4. Can they run it in about five minutes?
5. Does the README explain **why** key decisions were made?
6. Are paid API calls kept out of CI?

If the answer to any of these is no, revise before submitting.

## Example checklist

Every example should include:

- stable identifier (e.g. `basic-rag`) matching DataAIHub mapping
- `README.md` following the cookbook README standard
- `.env.example` (never commit secrets)
- `pyproject.toml` managed with `uv`
- smoke tests that do **not** require paid APIs
- Python 3.12+

Default layout:

```text
01-example-name/
├── README.md
├── main.py
├── config.py
├── .env.example
├── pyproject.toml
└── tests/
```

Add folders only when they make the pipeline clearer.

## Adding a new example

1. Place it in the correct learning path under `examples/` (`examples/rag/`, `examples/agents/`, etc.).
2. Number it in sequence (`02-…`, `03-…`) only when it extends that path.
3. Do not create empty category directories.
4. Prefer direct SDK usage for foundational examples.
5. Prefer a simple provider abstraction over separate vendor examples.

## Shared code

Be conservative with `shared/`.

Educational clarity beats DRY. Extract code only when repetition is clearly harmful.

## Tests and CI

CI verifies:

- formatting
- linting
- imports / syntax
- tests that do not call paid APIs

Do not add tests that require live LLM or embedding calls in CI.

## Local checks

From an example directory:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Pull requests

- Keep PRs focused on one example or one shared concern
- Link related DataAIHub Guide / Lab / Architecture pages when they exist
- Include a short note on what concept is taught and what was deliberately left out
