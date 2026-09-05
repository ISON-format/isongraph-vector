# isongraph-vector-cpp

[ISONGraph](https://github.com/ISON-format/isongraph) can walk a graph. It can
tell you who Alice knows, and who they know in turn. What it cannot tell you is
which node in the graph is *about* the thing you just asked for.

`isongraph-vector-cpp` closes that gap. It embeds nodes as vectors, finds the
ones closest in meaning to a query, and then traverses the graph from there —
so a question like *"who works on machine learning?"* starts at the nodes that
actually mean that, rather than at a node you had to know the id of.

This is the C++ port. See the
[repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the cross-port picture.

## Installation

Header-only, C++17. The core needs nothing but the headers:

```cmake
add_subdirectory(isongraph-vector-cpp)
target_link_libraries(your_target PRIVATE isongraph_vector)
```

`ison_graph.hpp` (ISONGraph 1.4.0) and `ison_parser.hpp` (ison-cpp 1.2.0) are
vendored under `vendor/ison-graph-cpp/`. There is no package registry for C++,
so unlike the other ports these are copies rather than a dependency
declaration; they are taken from the canonical checkouts.

## The idea, in one example

```cpp
#include "isongraph_vector.hpp"

using namespace isongraph_vector;

SemanticGraph graph("social", std::make_shared<MockEncoder>(384));

graph.addNode("person", "1", {{"name", "Alice"}, {"description", "software engineer"}});
graph.addNode("person", "2", {{"name", "Bob"}, {"description", "data scientist"}});
graph.addNode("person", "3", {{"name", "Carol"}, {"description", "product manager"}});
graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
graph.graph().addEdge("KNOWS", NodeRef("person", "2"), NodeRef("person", "3"));

MultiHopOptions options;
options.relType = "KNOWS";
options.maxHops = 2;
options.threshold = -1.0f;

for (const auto& r : graph.semanticMultiHop("Alice software engineer", options)) {
    printf("%s:%s  score=%.4f  hops=%zu\n",
           r.node_ref.type.c_str(), r.node_ref.id.c_str(), r.score, r.hop_count);
}
```

```text
person:1  score=1.0000  hops=0
person:2  score=0.8000  hops=1
person:3  score=0.6400  hops=2
```

Alice matches the query directly, so she seeds the search at hop 0. Bob and
Carol are reached by traversal, and their relevance decays by `decay ^ hops` —
0.8 and 0.64 with the default decay of 0.8. That decay is the whole point: a
node three hops from anything relevant should not outrank a direct match.

That example sets `threshold = -1.0f`, which deserves an explanation.
`MockEncoder` is deterministic and dependency-free, but its vectors carry no
meaning — they are hash-derived noise for testing plumbing. Because they are
near-orthogonal and cosine similarity is signed, mock scores straddle zero, and
the default threshold of 0.3 would reject every seed. Subclass
`EmbeddingEncoder` over a real model and the default is sensible again.

`SemanticGraph` wraps `ISONGraph` by composition, so the underlying graph stays
available through `graph()` — which is where the edges above are added.
Property values are `ison::Value`, typed since ISONGraph 1.4.0, but a string
literal converts implicitly, so the brace-initialised map above reads the same
as it always did.

## Beyond a single search

**Blending in a node's own relevance.** By default a node's score comes purely
from its seed and its distance. Sometimes a node two hops out is itself highly
relevant and deserves to outrank a closer but unrelated neighbour. `blend`
mixes the two:

```cpp
MultiHopOptions options;
options.blend = 0.5f;
// score = 0.5 * (seedScore * decay^hops) + 0.5 * ownSimilarity
```

At `blend = 0` (the default) scoring is pure seed-and-decay. Seeds score the
same either way, so raising it only ever reorders the nodes you reached.

**Pulling out the slice that matters.** `semanticSubgraph` runs the same search
and hands back a real `SemanticGraph` — the matching nodes, every edge induced
between them, and their embeddings. This is what makes an ISON graph useful as
LLM context: rather than serializing a whole knowledge graph into a prompt, you
serialize the part the question is actually about.

**More like this.** `similarToNode(NodeRef("person", "1"))` searches with a
node's own stored vector — no query text, no encoder call — and drops the node
itself from the results.

Alongside those: `semanticPath` to reach a specific target from a
semantically-relevant start, and `embedAllNodes` to backfill an existing graph
in one batched encoder call.

## Storing the vectors

The default store is in memory, and the library stays header-only and
dependency-free that way. Turn the option on and you get a SQLite-backed store:

```bash
cmake -S . -B build -DISONGRAPH_VECTOR_SQLITE=ON
cmake --build build
```

That fetches the SQLite and sqlite-vec amalgamations at configure time — both
pinned by SHA256 — and compiles them in, so no system SQLite is required.

```cpp
#include "isongraph_vector_sqlite.hpp"

SqliteEmbeddingStore store("embeddings.db", std::make_shared<MockEncoder>(384));
store.add(NodeRef("person", "1"), "Alice is a software engineer");
store.addBatch({                          // one encoder call, one transaction
    {NodeRef("person", "2"), "Bob analyses ML models"},
    {NodeRef("person", "3"), "Carol plans roadmaps"},
});
```

Pass `true` as the third constructor argument and top-k searches are answered
by a sqlite-vec `vec0` virtual table instead of a scan. It does not change what
you get back: `vec0` runs an exact brute-force KNN in C, not an approximate
index — measured identical ordering and scores within 2e-07 of the scan. It is
worth roughly 4x over 20,000 vectors and is *slower* below a few thousand,
which is why it is opt-in. Filtering by node type stays exact too: the type is
a `vec0` metadata column, so it narrows the search rather than filtering its
results.

The fetched SQLite targets are build-tree only and are not installed. A
consumer who wants the store either builds with the option on, or links their
own SQLite and includes `isongraph_vector_sqlite.hpp` directly.

## One format, six languages

The SQLite schema and the portable JSON payload are shared by every port —
Python, C#, TypeScript, JavaScript, Rust and C++ — so a database or payload
written by one can be read by another. `MockEncoder` is byte-identical across
all six, and the `id` column is an MD5 prefix computed the same way everywhere;
the suite checks that MD5 against known vectors so a row written here matches a
row written by Python byte for byte.

## Things that will bite you

- **Search is brute force**, sqlite-vec included: linear in the number of
  nodes. This is built for an application's own knowledge graph, not as a
  vector-database replacement.
- **One embedding per node**, from one text field or a concatenation. No
  multi-vector nodes, no edge embeddings.
- **`MockEncoder` has no semantic structure.** Use it to test plumbing, never
  to judge relevance quality. This port ships no real encoder — subclass
  `EmbeddingEncoder` over the model of your choice.
- **The base library is vendored, not versioned.** Nothing tells you when
  ISONGraph moves on; re-copy the headers when it does.
- **Auto-embed renders typed properties.** Nulls and references contribute
  nothing and booleans render as `true`/`false`, so the same graph produces the
  same text in every port.

## Tests

```bash
cmake -S . -B build && cmake --build build
./build/test_isongraph_vector                 # 51 tests

cmake -S . -B build -DISONGRAPH_VECTOR_SQLITE=ON && cmake --build build
./build/test_isongraph_vector_sqlite          # + 16 SQLite tests
```

On a multi-config generator such as Visual Studio the binaries land in
`build/Debug/`.

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
