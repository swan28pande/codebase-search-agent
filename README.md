# Codebase Search Agent

A minimal multi-agent code search system built on [LangGraph](https://github.com/langchain-ai/langgraph).

```
question -> plan (query rewrites) -> N parallel search agents -> consensus -> ranked results
                                       │ hybrid BM25 + FAISS
                                       └ self-verification
```

- **Hybrid retrieval**: BM25 (`rank-bm25`) and dense vectors (FAISS) fused with reciprocal-rank fusion. Identifiers are split on snake_case/camelCase.
- **Parallel agents**: each query rewrite runs as its own LangGraph branch (`Send`), with async verification inside each branch.
- **Self-verification**: each agent asks the LLM whether every hit is relevant (or uses a term-overlap check offline).
- **Consensus**: a chunk is returned only if enough agents independently verified it.
- **vLLM**: talks to any vLLM server through its OpenAI-compatible API.

## Quickstart

```bash
pip install -e .
csa /path/to/repo "where are auth tokens refreshed?"
```

With a vLLM server for query planning and verification:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct
VLLM_BASE_URL=http://localhost:8000/v1 csa /path/to/repo "where are auth tokens refreshed?" --vllm
```

The default embedding is a dependency-free hashing embedding. To use a real model, pass `CodeIndex(root, embed=fn)`, where `fn(list[str]) -> np.ndarray` returns L2-normalized float32 vectors.

## Layout

```
csa/
  index.py  # chunking, BM25 + FAISS hybrid search
  graph.py  # LangGraph: plan, search_agent, consensus
  llm.py    # async vLLM client
  cli.py    # `csa` command
```
