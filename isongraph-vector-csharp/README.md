# IsonGraph.Vector

`isongraph-vector` extends [ISONGraph](https://github.com/ISON-format/isongraph) with embedding-based
similarity search and semantic multi-hop traversal: find the nodes closest in
meaning to a query, then walk the graph from there with the relevance score
decaying per hop.

This is the C# port. See the [repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the full API, the cross-port feature matrix, and the design notes.

## Installation

```bash
dotnet add package IsonGraph.Vector
```

Targets `net8.0` and builds on the [`IsonGraph`](https://www.nuget.org/packages/IsonGraph) package.

## Usage

```csharp
using IsonGraph;
using IsonGraph.Vector;

var graph = new SemanticGraph("social", new MockEncoder(384));

graph.AddNode("person", "1", new() { ["name"] = "Alice", ["description"] = "software engineer" });
graph.AddNode("person", "2", new() { ["name"] = "Bob", ["description"] = "data scientist" });
graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));

var results = graph.SemanticMultiHop("engineering", new MultiHopOptions
{
    RelType = "KNOWS",
    MaxHops = 2,
});

foreach (var result in results)
    Console.WriteLine($"{result.NodeRef} score={result.Score:F4} hops={result.HopCount}");
```

## What you get

- `EmbeddingStore` - in-memory vector storage, cosine similarity search,
  dimension-checked, and safe to share across threads
- `SemanticGraph` - wraps `ISONGraph` (which is sealed) and exposes it through
  `Graph`; `AddNode` auto-embeds from node properties
- `SemanticMultiHop` - similarity seeds plus traversal, with optional `Blend`
  to mix in each node's own relevance
- `SemanticPath`, `SimilarToNode`, `EmbedAllNodes`
- `SemanticSubgraph` - the on-topic slice of a graph as a new graph, ready to
  serialize into an LLM context
- A portable embedding payload every language port reads and writes, so a store
  built by one can be searched by another

## Tests

```bash
dotnet test
```

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
