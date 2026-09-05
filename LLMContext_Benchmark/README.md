# ISONGraph LLM Context Window Benchmark

> This benchmark measures **ISON graph serialization**, not embeddings. It
> was extracted alongside `isongraph-vector` and kept because it still
> runs, but it exercises `ison-graph`; nothing here touches `SemanticGraph`
> or `EmbeddingStore`.

A benchmark focused on **token efficiency** and **LLM context window capacity** for graph serialization formats.

## Overview

This benchmark answers a critical question for LLM developers:

**Which graph format fits the most data into an LLM's context window?**

### Key Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Token Count** | Tokens used to serialize graph | Lower = more room for other content |
| **Nodes/1K Tokens** | Graph density per 1000 tokens | Higher = more efficient format |
| **Context Capacity** | Max nodes in 128K context | Practical limit for GPT-4o |
| **Serialization Size** | Bytes on disk | Storage and transfer cost |

## Formats Compared

| Store | Format | Description |
|-------|--------|-------------|
| **ISONGraph** | ISON (Tabular) | Token-efficient tabular format |
| **NetworkX** | GML | Standard graph exchange format |
| **igraph** | GraphML (XML) | XML-based graph format |
| **SQLite** | SQL Export | Relational SQL statements |
| **DictGraph** | JSON | Common but verbose format |

## Quick Start

### Prerequisites

```bash
# Required
pip install tiktoken networkx

# Optional (for additional comparisons)
pip install igraph
```

### Running the Benchmark

```bash
# Quick benchmark (1K nodes)
python benchmark_llm_context.py

# Full benchmark (10K nodes)
python benchmark_llm_context.py --full

# Custom scale (5K nodes)
python benchmark_llm_context.py --scale 5000

# Specific stores only
python benchmark_llm_context.py --stores isongraph networkx
```

## Expected Results

From the committed run in
[`llm_context_benchmark_20251230_173619.md`](llm_context_benchmark_20251230_173619.md)
(1,000 nodes + 4,981 edges), reproduced since.

### Token Efficiency (1K nodes + 5K edges)

| Store | Tokens | Nodes/1K Tokens | Relative |
| --- | ---: | ---: | :---: |
| **ISONGraph** | 66,787 | 15.0 | Baseline |
| SQLite | 142,575 | 7.0 | 2.1x worse |
| NetworkX | 178,457 | 5.6 | 2.7x worse |
| igraph | 287,392 | 3.5 | 4.3x worse |
| DictGraph | 352,800 | 2.8 | 5.3x worse |

An earlier version of this table listed SQLite last at ~364K tokens. That
was stale: the committed report and every re-run since put SQLite second,
and DictGraph last at the ~353K the old table attributed to SQLite.

### Context Window Capacity

Nodes that fit, quoted verbatim from the same committed report:

| Store | 4K | 8K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: |
| **ISONGraph** | 61 | 122 | 490 | 1,962 |
| SQLite | 28 | 57 | 229 | 919 |
| NetworkX | 22 | 45 | 183 | 734 |
| igraph | 14 | 28 | 114 | 456 |
| DictGraph | 11 | 23 | 92 | 371 |

## Output Files

The benchmark generates three output files:

1. **`llm_context_benchmark_TIMESTAMP.json`** - Machine-readable results
2. **`llm_context_benchmark_TIMESTAMP.log`** - Human-readable log
3. **`llm_context_benchmark_TIMESTAMP.md`** - Markdown report

## Why ISONGraph Wins

### 1. Tabular Format (No Key Repetition)
```text
# ISONGraph: Headers appear once
nodes.person
id name age
1 Alice 28
2 Bob 34

# JSON: Keys repeated for every node
{"nodes": [{"id": 1, "name": "Alice", "age": 28}, ...]}
```

### 2. Compact Node References
```text
# ISONGraph: 11 characters
:person:1

# JSON: 35 characters
{"type": "person", "id": 1}
```

### 3. Minimal Punctuation
```text
# ISONGraph: Whitespace delimited
1 Alice 28 true 1500

# JSON: Heavy punctuation
{"id": 1, "name": "Alice", "age": 28, "verified": true}
```

## Use Cases

### When to Use ISONGraph

| Use Case | ISONGraph Advantage |
|----------|---------------------|
| **LLM Context Injection** | 2.7-5x more nodes fit |
| **RAG Knowledge Graphs** | Token-efficient retrieval |
| **Agent Memory** | More history in context |
| **Multi-hop Reasoning** | Larger traversable graphs |

### When Other Formats Excel

| Format | Best For |
|--------|----------|
| NetworkX/GML | Algorithm research (not LLM contexts) |
| GraphML | XML tooling integration |
| JSON | Human debugging, REST APIs |
| SQL | Persistent storage, complex queries |

## Tokenizer

This benchmark uses `tiktoken` with the `o200k_base` encoding (GPT-4o/GPT-5).

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
tokens = len(enc.encode(content))
```

## Reproducibility

All tests use:
- Random seed: 42
- Node count: 1,000 (default) or 10,000 (--full)
- Edge multiplier: 5x nodes
- Same data across all formats

## License

MIT License - See LICENSE file for details.

## Related Benchmarks

`KnowledgeGraph_Benchmark` and `DataTraversal_Benchmark` (LLM accuracy with
different formats) live in the `ison-graph` repository, not here.

## Author

Mahesh Vaikri

## Links

- [ISONGraph Repository](https://github.com/maheshvaikri-code/ison)
- ISON Format Specification (`docs/ISON_FORMAT.md` in the `ison` repository)
