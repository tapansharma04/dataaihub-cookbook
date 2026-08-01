# DataAIHub Cookbook

Public reference implementations for [DataAIHub](https://dataaihub.com).

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
| `basic-rag` | [`rag/01-basic-rag`](rag/01-basic-rag) | Chunk → embed → retrieve → generate |

More examples will be added after this first pattern is solid.

## Quick start

```bash
cd rag/01-basic-rag
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
└── rag/
    └── 01-basic-rag/
```

Categories appear only when an implementation exists.

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
