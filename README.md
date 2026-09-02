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

- **Concept first** — each example demonstrates one primary engineering idea
- **Minimal code** — smallest correct implementation, not a production app
- **Progressive complexity** — examples form learning sequences
- **Provider-agnostic where useful** — concepts over vendors
- **Independent examples** — readable without navigating half the repo

## Examples

### RAG

| ID | Path | Concept |
|----|------|---------|
| `basic-rag` | [`examples/rag/01-basic-rag`](examples/rag/01-basic-rag) | Chunk → embed → retrieve → generate |
| `hybrid-rag` | [`examples/rag/02-hybrid-rag`](examples/rag/02-hybrid-rag) | Dense + BM25 → RRF → generate |
| `reranking` | [`examples/rag/03-reranking`](examples/rag/03-reranking) | Hybrid candidates → cross-encoder → generate |
| `query-transformation` | [`examples/rag/04-query-transformation`](examples/rag/04-query-transformation) | Multi-query retrieval aggregation → rerank → generate |
| `retrieval-evaluation` | [`examples/rag/05-retrieval-evaluation`](examples/rag/05-retrieval-evaluation) | Measure retrieval with Recall@K, MRR, and nDCG@K against golden judgments |
| `chunking-strategies` | [`examples/rag/06-chunking-strategies`](examples/rag/06-chunking-strategies) | Compare fixed-size, recursive, and structure-aware chunking under the same retrieval evaluation setup |

### AI Agents

| ID | Path | Concept |
|----|------|---------|
| `tool-calling` | [`examples/agents/01-tool-calling`](examples/agents/01-tool-calling) | Model ↔ tool interaction loop (select → execute → observe → answer) |
| `agent-loop` | [`examples/agents/02-agent-loop`](examples/agents/02-agent-loop) | Application-owned runtime loop (state → decide → act → observe → terminate) |
| `agent-evaluation` | [`examples/agents/03-agent-evaluation`](examples/agents/03-agent-evaluation) | Outcome + trajectory evaluation of a measured agent run under explicit constraints |
| `planning` | [`examples/agents/04-planning`](examples/agents/04-planning) | Explicit multi-step plan as a runtime artifact (create → execute → observe → revise) |
| `agent-memory` | [`examples/agents/05-memory`](examples/agents/05-memory) | Application-owned memory across interactions (store → retrieve → use, including miss and stale) |

### MCP

| ID | Path | Concept |
|----|------|---------|
| `mcp-tool-discovery` | [`examples/mcp/01-tool-discovery`](examples/mcp/01-tool-discovery) | MCP client/server protocol lifecycle (initialize → tools/list → tools/call) |
| `mcp-resources` | [`examples/mcp/02-resources`](examples/mcp/02-resources) | MCP resource discovery and reading (initialize → resources/list → resources/read) |
| `mcp-prompts` | [`examples/mcp/03-prompts`](examples/mcp/03-prompts) | MCP prompt discovery and retrieval (initialize → prompts/list → prompts/get) |
| `mcp-composition` | [`examples/mcp/04-composition`](examples/mcp/04-composition) | Composed MCP workflow: resources, prompts, and tools with Sampling through the client |

### Knowledge Graphs

| ID | Path | Concept |
|----|------|---------|
| `graph-traversal` | [`examples/knowledge-graphs/01-graph-traversal`](examples/knowledge-graphs/01-graph-traversal) | RDF triples and graph traversal (entity → relationship → evidence) |
| `sparql-queries` | [`examples/knowledge-graphs/02-sparql-queries`](examples/knowledge-graphs/02-sparql-queries) | SPARQL query patterns over RDF (pattern → bindings → result rows) |
| `sparql-updates` | [`examples/knowledge-graphs/03-sparql-updates`](examples/knowledge-graphs/03-sparql-updates) | SPARQL Updates: INSERT, DELETE & Graph Mutation |
| `graphrag` | [`examples/knowledge-graphs/04-graphrag`](examples/knowledge-graphs/04-graphrag) | GraphRAG: Graph-Grounded Retrieval & LLM Answering |
| `graph-construction` | [`examples/knowledge-graphs/05-graph-construction`](examples/knowledge-graphs/05-graph-construction) | Graph Construction: Source → Proposal → Validation → RDF |

More examples will be added as each learning path progresses.

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
├── LICENSE
├── examples/
│   ├── rag/
│   │   ├── 01-basic-rag/
│   │   ├── 02-hybrid-rag/
│   │   ├── 03-reranking/
│   │   ├── 04-query-transformation/
│   │   ├── 05-retrieval-evaluation/
│   │   └── 06-chunking-strategies/
│   ├── agents/
│   │   ├── 01-tool-calling/
│   │   ├── 02-agent-loop/
│   │   ├── 03-agent-evaluation/
│   │   ├── 04-planning/
│   │   └── 05-memory/
│   ├── mcp/
│   │   ├── 01-tool-discovery/
│   │   ├── 02-resources/
│   │   ├── 03-prompts/
│   │   └── 04-composition/
│   └── knowledge-graphs/
│       ├── 01-graph-traversal/
│       ├── 02-sparql-queries/
│       ├── 03-sparql-updates/
│       ├── 04-graphrag/
│       └── 05-graph-construction/
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

## License

[MIT](LICENSE)
