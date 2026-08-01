# Sample knowledge base for basic RAG
# Tiny on purpose — enough to retrieve against, easy to inspect.

## What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with
large language models. Instead of answering only from model parameters,
the system first retrieves relevant documents, then conditions the LLM
on that retrieved context.

RAG reduces hallucinations for knowledge-intensive questions and lets
answers stay current without retraining the model.

## Chunking

Documents are split into smaller pieces called chunks before embedding.
Chunk size is a tradeoff: too small and context is lost; too large and
retrieval becomes noisy. Overlap between adjacent chunks helps preserve
meaning that would otherwise be cut at boundaries.

## Embeddings

An embedding model maps text to a dense vector. Semantically similar
texts land near each other in vector space. RAG embeds both document
chunks (offline) and the user question (at query time).

## Vector search

At query time, the system embeds the question and finds the nearest
chunk vectors by similarity (often cosine similarity). The top-k chunks
become the context passed to the language model.

## Prompt construction

Retrieved chunks are inserted into a prompt that instructs the model to
answer using only the provided context. If the context is insufficient,
a good prompt asks the model to say so rather than invent details.

## Limitations of basic RAG

Basic RAG uses dense retrieval only. It may miss exact keyword matches,
does not rerank results, and has no agentic tool use. Production systems
often add hybrid search, reranking, evaluation, and monitoring.
