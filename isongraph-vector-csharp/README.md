# IsonGraph.Vector

[ISONGraph](https://github.com/ISON-format/isongraph) can walk a graph. It can
tell you who Alice knows, and who they know in turn. What it cannot tell you is
which node in the graph is *about* the thing you just asked for.

`IsonGraph.Vector` closes that gap. It embeds nodes as vectors, finds the ones
closest in meaning to a query, and then traverses the graph from there — so a
question like *"who works on machine learning?"* starts at the nodes that
actually mean that, rather than at a node you had to know the id of.

This is the .NET port. See the
[repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the cross-port picture.

## Installation

```bash
dotnet add package IsonGraph.Vector          # core
dotnet add package IsonGraph.Vector.Sqlite   # + SQLite storage and sqlite-vec
```

Targets `net8.0` and builds on
[`IsonGraph`](https://www.nuget.org/packages/IsonGraph) 1.4.0.

## The idea, in one example

```csharp
using IsonGraph;
using IsonGraph.Vector;

var graph = new SemanticGraph("social", new MockEncoder(384));

graph.AddNode("person", "1", new() { ["name"] = "Alice", ["description"] = "software engineer" });
graph.AddNode("person", "2", new() { ["name"] = "Bob", ["description"] = "data scientist" });
graph.AddNode("person", "3", new() { ["name"] = "Carol", ["description"] = "product manager" });
graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
graph.Graph.AddEdge("KNOWS", new NodeRef("person", "2"), new NodeRef("person", "3"));

var results = graph.SemanticMultiHop("Alice software engineer", new MultiHopOptions
{
    RelType = "KNOWS",
    MaxHops = 2,
    Threshold = -1.0f,
});

foreach (var r in results)
    Console.WriteLine($"{r.NodeRef.Type}:{r.NodeRef.Id}  score={r.Score:F4}  hops={r.HopCount}");
```

```text
person:1  score=1.0000  hops=0
person:2  score=0.8000  hops=1
person:3  score=0.6400  hops=2
```

Alice matches the query directly, so she seeds the search at hop 0. Bob and
Carol are reached by traversal, and their relevance decays by `Decay ** hops` —
0.8 and 0.64 with the default decay of 0.8. That decay is the whole point: a
node three hops from anything relevant should not outrank a direct match.

That example passes `Threshold = -1.0f`, which deserves an explanation.
`MockEncoder` is deterministic and dependency-free, but its vectors carry no
meaning — they are hash-derived noise for testing plumbing. Because they are
near-orthogonal and cosine similarity is signed, mock scores straddle zero, and
the default threshold of 0.3 would reject every seed. Supply your own
`IEmbeddingEncoder` over a real model and the default is sensible again.

`ISONGraph` is `sealed`, so `SemanticGraph` wraps it by composition rather than
inheriting from it. The underlying graph stays fully available through
`.Graph`, which is why the edges above are added there.

## Beyond a single search

**Blending in a node's own relevance.** By default a node's score comes purely
from its seed and its distance. Sometimes a node two hops out is itself highly
relevant and deserves to outrank a closer but unrelated neighbour. `Blend`
mixes the two:

```csharp
var results = graph.SemanticMultiHop(query, new MultiHopOptions { Blend = 0.5f });
// score = 0.5 * (seedScore * decay^hops) + 0.5 * ownSimilarity
```

At `Blend = 0` (the default) scoring is pure seed-and-decay. Seeds score the
same either way, so raising it only ever reorders the nodes you reached.

**Pulling out the slice that matters.** `SemanticSubgraph` runs the same search
and hands back a real `SemanticGraph` — the matching nodes, every edge induced
between them, and their embeddings. This is what makes an ISON graph useful as
LLM context: rather than serializing a whole knowledge graph into a prompt, you
serialize the part the question is actually about.

**More like this.** `SimilarToNode(new NodeRef("person", "1"))` searches with a
node's own stored vector — no query text, no encoder call — and drops the node
itself from the results.

Alongside those: `SemanticPath` to reach a specific target from a
semantically-relevant start, `EmbedAllNodes` to backfill an existing graph in
one batched encoder call, and `EmbedNodes` for batching your own pairs.

## Storing the vectors

The core package keeps embeddings in memory. Add `IsonGraph.Vector.Sqlite` and
they persist:

```csharp
using IsonGraph.Vector.Sqlite;

using var store = new SqliteEmbeddingStore("embeddings.db", new MockEncoder(384));
store.Add(new NodeRef("person", "1"), "Alice is a software engineer");
store.AddBatch(new[] {                    // one encoder call, one transaction
    (new NodeRef("person", "2"), "Bob analyses ML models"),
    (new NodeRef("person", "3"), "Carol plans roadmaps"),
});
```

Pass `sqliteVec: true` and top-k searches are answered by a sqlite-vec `vec0`
virtual table instead of a scan:

```csharp
using var store = new SqliteEmbeddingStore("embeddings.db", encoder, sqliteVec: true);
```

It does not change what you get back. `vec0` runs an exact brute-force KNN in
C, not an approximate index — measured identical ordering and scores within
2e-07 of the scan. Measured over 20,000 vectors of 384 dimensions, top-10,
median of five runs: **37 ms** for the SQLite scan, **7.1 ms** through the
index, and **5.9 ms** for the in-memory store. Recall of the index against the
exact scan is **1.000** — the same answers, faster. Below a few thousand
vectors the scan wins outright, which is why the index is opt-in.

Note which pair that speedup is between: the index replaces the SQLite scan,
not the in-memory store. `IsonGraph.Vector`'s own store is faster still at
this size, because it never leaves managed memory. Reach for the index when
the vectors have to be on disk anyway.

Top-k results come out of a bounded insertion rather than sorting all of them,
which is where a third of the query time used to go. Filtering by node type
stays exact too: the type is a `vec0` metadata column, so it narrows the
search rather than filtering its results.

`Dispose` closes the connection and clears the ADO.NET pool, so the database
file is genuinely released and can be moved or deleted.

## One format, six languages

The SQLite schema and the portable JSON payload are shared by every port —
Python, C#, TypeScript, JavaScript, Rust and C++ — so a database or payload
written by one can be read by another. `MockEncoder` is byte-identical across
all six, which is what makes that guarantee testable rather than aspirational;
this suite proves it by importing a payload captured from the Python port and
checking the vectors match what C# encodes for the same text.

```csharp
var payload = store.ExportEmbeddings();
var json = store.ToJson();
store.FromJson(json);
```

## Things that will bite you

- **Search is brute force**, sqlite-vec included: linear in the number of
  nodes. This is built for an application's own knowledge graph, not as a
  vector-database replacement.
- **One embedding per node**, from one text field or a concatenation. No
  multi-vector nodes, no edge embeddings.
- **`MockEncoder` has no semantic structure.** Use it to test plumbing, never
  to judge relevance quality. This port ships no real encoder — implement
  `IEmbeddingEncoder` over the model of your choice.
- **Auto-embed renders typed properties.** Since ISONGraph 1.4.0 a property can
  be a number, a bool or null. Nulls are skipped and booleans render as ISON
  spells them (`true`/`false`, not .NET's `True`), so the same graph produces
  the same text in every port. Floats use the invariant culture — `1.5` never
  `1,5`.
- **The vec0 natives come from a redistribution.** asg017's own `sqlite-vec`
  package has never left prerelease on NuGet, and a stable package cannot
  depend on a prerelease one, so this port takes its natives from
  `HiraokaHyperTools.sqlite-vec` 0.1.9 - the same upstream C, at the same
  version the Rust, Node and C++ ports use. The core `IsonGraph.Vector`
  package has no such dependency.

## Tests

```bash
dotnet test        # 61 tests
```

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
