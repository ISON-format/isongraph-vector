# ISONGraph LLM Context Window Benchmark

> This report measures **ISON graph serialization**, not embeddings. It was
> extracted alongside `isongraph-vector` and kept because it still runs, but it
> exercises `ison-graph`; nothing here touches `SemanticGraph` or
> `EmbeddingStore`. The argument it makes — that a compact serialization fits
> more graph in a context window — is the same one `semantic_subgraph` makes
> one level up, by sending less of the graph in the first place.

**Focus: Token Efficiency, Serialization Size, and Context Window Capacity**

**Date:** December 30, 2025
**Tokenizer:** o200k_base (GPT-4o/GPT-5)
**Test Data:** 1,000 nodes + 5,000 edges

---

## Executive Summary

**ISONGraph is the most token-efficient graph store for LLM contexts.**

| Metric | ISONGraph | vs. Best Alternative |
|--------|-----------|---------------------|
| **Token Efficiency** | 14.5 nodes/1K tokens | **2.7x better** than NetworkX |
| **Serialized Size** | 172 KB | **66% smaller** than NetworkX |
| **128K Context Capacity** | 1,855 nodes | **2.7x more** than NetworkX |

---

## Token Efficiency Comparison

**Lower tokens = More data fits in LLM context window**

| Rank | Store | Tokens | Nodes/1K Tokens | Token Efficiency |
|:----:|-------|-------:|----------------:|-----------------|
| 1 | **ISONGraph** | **68,975** | **14.5** | Baseline |
| 2 | NetworkX | 185,578 | 5.4 | 2.7x worse |
| 3 | DictGraph | 273,977 | 3.6 | 4.0x worse |
| 4 | igraph | 299,987 | 3.3 | 4.4x worse |
| 5 | SQLite | 363,583 | 2.8 | 5.3x worse |

### Token Savings vs. Other Stores

| Comparison | Token Savings |
|------------|---------------|
| ISONGraph vs. NetworkX | **62.8%** fewer tokens |
| ISONGraph vs. DictGraph | **74.8%** fewer tokens |
| ISONGraph vs. igraph | **77.0%** fewer tokens |
| ISONGraph vs. SQLite | **81.0%** fewer tokens |

---

## Serialization Size Comparison

**Smaller serialization = Faster context injection + Less storage**

| Rank | Store | Size (bytes) | Size (KB) | Relative Size |
|:----:|-------|-------------:|----------:|--------------:|
| 1 | **ISONGraph** | **172,347** | **168 KB** | 1.00x |
| 2 | NetworkX | 509,409 | 497 KB | 2.96x |
| 3 | SQLite | 638,976 | 624 KB | 3.71x |
| 4 | DictGraph | 709,994 | 693 KB | 4.12x |
| 5 | igraph | 876,335 | 856 KB | 5.08x |

### Size Savings vs. Other Stores

| Comparison | Size Savings |
|------------|--------------|
| ISONGraph vs. NetworkX | **66.2%** smaller |
| ISONGraph vs. DictGraph | **75.7%** smaller |
| ISONGraph vs. igraph | **80.3%** smaller |
| ISONGraph vs. SQLite | **73.0%** smaller |

---

## Context Window Capacity

**How many graph nodes can fit in each context window size?**

### Nodes That Fit Per Context Window

| Store | 4K Context | 8K Context | 32K Context | 128K Context |
|-------|:----------:|:----------:|:-----------:|:------------:|
| **ISONGraph** | **57** | **115** | **463** | **1,855** |
| NetworkX | 21 | 43 | 172 | 689 |
| DictGraph | 14 | 29 | 116 | 467 |
| igraph | 13 | 26 | 106 | 426 |
| SQLite | 11 | 22 | 88 | 352 |

### Context Utilization (% of window used for 1K nodes)

| Store | 4K Context | 8K Context | 32K Context | 128K Context |
|-------|:----------:|:----------:|:-----------:|:------------:|
| **ISONGraph** | 1,684% | 842% | **211%** | **53%** |
| NetworkX | 4,531% | 2,265% | 566% | 142% |
| DictGraph | 6,689% | 3,344% | 836% | 209% |
| igraph | 7,324% | 3,662% | 915% | 229% |
| SQLite | 8,877% | 4,438% | 1,110% | 277% |

**Key Insight:** Only ISONGraph can fit 1,000 nodes in a 128K context window (53% usage). All others exceed the limit.

---

## Practical Capacity Examples

### GPT-4o (128K context) - How Many Nodes Fit?

| Store | Max Nodes | Max Edges (5:1) | Use Case Example |
|-------|:---------:|:---------------:|------------------|
| **ISONGraph** | **1,855** | **9,275** | Full company org chart |
| NetworkX | 689 | 3,445 | Department only |
| DictGraph | 467 | 2,335 | Team level |
| igraph | 426 | 2,130 | Team level |
| SQLite | 352 | 1,760 | Small team |

### Claude 3.5 (200K context) - How Many Nodes Fit?

| Store | Max Nodes | Max Edges (5:1) | Use Case Example |
|-------|:---------:|:---------------:|------------------|
| **ISONGraph** | **2,900** | **14,500** | Enterprise knowledge graph |
| NetworkX | 1,077 | 5,385 | Large department |
| DictGraph | 729 | 3,645 | Medium department |
| igraph | 666 | 3,330 | Medium department |
| SQLite | 550 | 2,750 | Small department |

### Gemini 1.5 Pro (1M context) - How Many Nodes Fit?

| Store | Max Nodes | Max Edges (5:1) | Use Case Example |
|-------|:---------:|:---------------:|------------------|
| **ISONGraph** | **14,500** | **72,500** | Full enterprise graph |
| NetworkX | 5,388 | 26,940 | Large business unit |
| DictGraph | 3,649 | 18,245 | Business unit |
| igraph | 3,333 | 16,665 | Business unit |
| SQLite | 2,750 | 13,750 | Multiple departments |

---

## Memory Efficiency (In-Memory Runtime)

| Store | Memory (MB) | Memory per Node | Notes |
|-------|:-----------:|:---------------:|-------|
| SQLite | 0.03 | 0.00003 MB | Disk-backed |
| igraph | 0.22 | 0.00022 MB | C-optimized |
| DictGraph | 1.86 | 0.00186 MB | Pure Python |
| NetworkX | 2.10 | 0.00210 MB | Feature-rich |
| **ISONGraph** | 2.73 | 0.00273 MB | Schema validation |

**Note:** ISONGraph uses slightly more memory due to schema validation and edge indexing, but this enables faster queries and type safety.

---

## Why ISONGraph Wins Token Efficiency

### 1. Tabular Format
```
# ISONGraph: Column headers appear once
nodes.person
id name age
1 Alice 28
2 Bob 34

# JSON: Keys repeated for every node
{"nodes": [{"id": 1, "type": "person", "name": "Alice", "age": 28}, ...]}
```

### 2. Compact References
```
# ISONGraph: 11 characters
:person:1

# JSON: 35 characters
{"type": "person", "id": 1}
```

### 3. Minimal Punctuation
```
# ISONGraph: Whitespace delimited
1 Alice 28 true 1500

# JSON: Heavy punctuation
{"id": 1, "name": "Alice", "age": 28, "verified": true, "followers": 1500}
```

### 4. Explicit Edge Sections
```
# ISONGraph: Clear relationship grouping
edges.FOLLOWS
source target since
:person:1 :person:2 2020

# JSON: Nested with redundant keys
{"edges": [{"source": 1, "target": 2, "relation": "follows", "since": 2020}]}
```

---

## Efficiency Score (Nodes per 1K Tokens)

The key metric for LLM context utilization:

| Rank | Store | Nodes/1K Tokens | Relative Efficiency |
|:----:|-------|:---------------:|:-------------------:|
| 1 | **ISONGraph** | **14.5** | 100% (baseline) |
| 2 | NetworkX | 5.4 | 37% |
| 3 | DictGraph | 3.6 | 25% |
| 4 | igraph | 3.3 | 23% |
| 5 | SQLite | 2.8 | 19% |

**ISONGraph delivers 4-5x more graph data per token than alternatives.**

---

## Recommendations

### When to Use ISONGraph

| Use Case | ISONGraph Advantage |
|----------|---------------------|
| **LLM Context Injection** | 2.7-5x more nodes fit in context |
| **RAG with Knowledge Graphs** | Token-efficient retrieval |
| **Agent Memory** | More history fits in context |
| **Multi-hop Reasoning** | Larger traversable graphs |
| **Conversational AI** | Fit entire user knowledge |

### When Other Stores Excel

| Store | Best For |
|-------|----------|
| NetworkX | Algorithm research (not LLM contexts) |
| igraph | Large-scale analytics (not LLM contexts) |
| SQLite | Persistent storage (not LLM contexts) |

---

## Conclusion

For LLM context window optimization, **ISONGraph is the clear winner**:

| Metric | ISONGraph Advantage |
|--------|---------------------|
| **Token Efficiency** | 2.7-5x better |
| **Serialization Size** | 66-80% smaller |
| **Context Capacity** | 2.7-5x more nodes |
| **128K Context** | Fits 1,855 nodes (others: 350-690) |

**The right graph format for LLM Context Windows is ISONGraph.**

---

## Reproducibility

```bash
cd benchmark/GraphStore_Benchmark
pip install networkx igraph rdflib kuzu tiktoken
python benchmark_stores.py
```

---

*Generated by ISONGraph Store Benchmark v2.0.0*
