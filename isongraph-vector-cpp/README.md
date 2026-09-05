# isongraph-vector-cpp

`isongraph-vector` extends [ISONGraph](https://github.com/ISON-format/isongraph) with embedding-based
similarity search and semantic multi-hop traversal: find the nodes closest in
meaning to a query, then walk the graph from there with the relevance score
decaying per hop.

This is the C++ port. See the [repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the full API, the cross-port feature matrix, and the design notes.

## Installation

Header-only, C++17. Add both include directories to your path:

```cmake
add_subdirectory(isongraph-vector-cpp)
target_link_libraries(your_target PRIVATE isongraph_vector)
```

`ison_graph.hpp` is vendored under `vendor/ison-graph-cpp/`; there is no C++
package registry for it. Verify that copy is current before relying on it.

## Usage

```cpp
#include "isongraph_vector.hpp"

using namespace isongraph_vector;

auto encoder = std::make_shared<MockEncoder>(384);
SemanticGraph graph("social", encoder);

graph.addNode("person", "1", {{"name", "Alice"}, {"description", "software engineer"}});
graph.addNode("person", "2", {{"name", "Bob"}, {"description", "data scientist"}});
graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));

MultiHopOptions options;
options.relType = "KNOWS";
options.maxHops = 2;
auto results = graph.semanticMultiHop("engineering", options);
```

## What you get

- `EmbeddingStore` - in-memory vector storage, cosine similarity search, dimension-checked
- `SemanticGraph` - wraps `ISONGraph` (reachable via `graph()`); `addNode` auto-embeds from node properties
- `semanticMultiHop` - similarity seeds plus traversal, with optional `blend`
  to mix in each node's own relevance
- `semanticPath`, `similarToNode`, `embedAllNodes`
- `semanticSubgraph` - the on-topic slice of a graph as a new graph, ready to
  serialize into an LLM context
- A portable embedding payload every language port reads and writes, so a store
  built by one can be searched by another

## Tests

```bash
cmake -S . -B build && cmake --build build
./build/test_isongraph_vector
```

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
