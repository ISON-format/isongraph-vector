# Vendored headers

There is no package registry for C++, so ISONGraph and the ISON parser are
copied in here rather than declared as dependencies. That is a real cost: a
copy has no version resolution and no upgrade notification, and this directory
once sat three minor versions behind without anything signalling it.

This file is the signal. `test_isongraph_vector` asserts the versions below, so
a re-copy that forgets to update this file, or a stale copy that never gets
re-copied, fails the suite rather than passing quietly.

| Header | Upstream | Version | Lines | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `ison_graph.hpp` | `ison-graph-cpp` | 1.1.0 | 1894 | `a80e8e55f32647cf39dba5c261754d84dadfcf67b7329bbb79eca6347d9f3694` |
| `ison_parser.hpp` | `ison-cpp` | 1.2.0 | 2005 | `6d2330a3a5a51f08fbb81a2af43caef500e8eb9b8c94650bff06b395d85d1f73` |

ISONGraph refuses to compile against an ison-cpp older than 1.2.0 — earlier
releases lose floats in transit — which is why the parser is vendored alongside
it rather than left to the consumer.

## A note on that 1.1.0

`ison_graph.hpp` declares `VERSION = "1.1.0"`, but it was copied from the
`ison-graph-cpp` checkout whose `CMakeLists.txt` says `VERSION 1.4.0`. The
header constant lags the package upstream; this is the 1.4.0 source. The table
records what the header itself declares, because that is the only thing a test
can read.

## Re-copying

```bash
cp <ison-graph-cpp>/include/ison_graph.hpp   vendor/ison-graph-cpp/
cp <ison-cpp>/include/ison_parser.hpp        vendor/ison-graph-cpp/
```

Then update the versions in this table and in
`tests/test_isongraph_vector.cpp`, and run the suite.
