# DataAIHub Cookbook

Public reference implementations for [DataAIHub](https://www.dataaihub.co).

DataAIHub explains the concepts.  
This repository shows how they are implemented.

```text
Guide  →  "What is it?"
Lab    →  "How does it work?"
Cookbook →  "How is it implemented?"
```

## Principles

- **Concept first** — each example teaches one primary engineering idea
- **Minimal code** — smallest correct implementation, not a production app
- **Progressive complexity** — examples form learning sequences
- **Provider-agnostic where useful** — concepts over vendors
- **Independent examples** — readable without navigating half the repo

## Examples

| ID | Path | Concept |
|----|------|---------|
| `basic-rag` | [`examples/rag/01-basic-rag`](examples/rag/01-basic-rag) | Chunk → embed → retrieve → generate |
| `hybrid-rag` | [`examples/rag/02-hybrid-rag`](examples/rag/02-hybrid-rag) | Dense + BM25 → RRF → generate |

More examples will be added as the RAG path progresses.

## Quick start

```bash
cd examples/rag/01-basic-rag
cp .env.example .env   # add your API key
uv sync
uv run python main.py "What is retrieval-augmented generation?"
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Repository layout

```text
dataaihub-cookbook/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── examples/
│   └── rag/
│       ├── 01-basic-rag/
│       └── 02-hybrid-rag/
├── shared/
├── docs/
└── scripts/
```

Example categories under `examples/` appear when an implementation exists.

## Relationship to DataAIHub

| Surface | Role |
|---------|------|
| Guide | Conceptual explanation |
| Architecture / Lab | Interactive learning |
| Cookbook (this repo) | Runnable reference code |

Cookbook code and Labs correspond conceptually but stay operationally independent.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
