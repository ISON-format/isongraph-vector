# Search engine design (reference implementation)

This is the specification the Rust crate implements and the C++, C#, Python and
JS ports should follow. Code: `src/table.rs` (data), `src/ann.rs` (scores,
search, index), `src/pool.rs` (threads), `src/shared.rs` (reader/writer handle).

## 1. Data layout: one copy of every vector

A **table** holds the whole store. A record *is a slot number*; nothing else
owns the floats.

| part | shape | notes |
|---|---|---|
| matrix | chunks of `CHUNK_ROWS = 1024` rows, `dim` f32 each, row-major | chunked so growth never copies stored data and a scan block never straddles chunks. Chunk 0 grows like a vector; later chunks are allocated whole. |
| `norms[slot]` | f32 | Euclidean norm; **NaN = tombstone** (a NaN score is dropped by the collector, so scans need no liveness branch) |
| `codes[slot]` | u32 | node type as an index into a type-name table (never shrinks) |
| `meta[slot]` | optional {key `"type:id"`, text, model} | `None` = tombstone. Model names are interned. |
| `index` | hash map key -> slot | the only lookup structure; the key string is shared with `meta` |

Slots are in arrival order. An update overwrites its slot in place; an insert
takes `n`; a delete tombstones (`norm = NaN`, `meta = None`, key removed).
When more than 25% of slots are dead, **compaction** moves live rows down,
preserving order, and the index follows (section 5). Equal scores are ordered by
slot, so answers are reproducible for any thread count.
Measured: 1M x 384 = 1.12x the raw vector bytes including the IVF index.

## 2. One definition of a score

`score = dot(q, row) / (norm(row) * norm(q))` (0 if the product is 0). `dot` has
**one fixed summation order**: one 8-lane accumulator walked in ascending
dimension (`acc[l] += q[i+l]*row[i+l]`, fused multiply-add where the CPU has it),
reduced as `t[l] = acc[l] + acc[l+4]`, `s = (t0 + t2) + (t1 + t3)`, then the
`dim % 8` tail added one element at a time. Every kernel (1 query x 4 rows, 4
queries x 2 rows, k-means 2 points x 4 centroids, a lone dot) computes each
(query, row) pair in this order; they differ only in how many pairs they hold in
registers. Consequences: **a batch answer is bit-identical to the one-query
answer**, and does not depend on tiling or thread count. A port may pick another
order, but must use it everywhere.

## 3. Search paths

* **Exact, one query**: blocks of 1024 rows claimed from an atomic counter by up
  to `min(pool, 8, n*dim / 2^21)` participants; each keeps a bounded heap
  (`TopK`: min-heap, root = worst kept, `bar` = its score so most rows are
  rejected by one compare; ordering `(score desc, slot asc)`); heaps merge.
* **Exact, batch**: tasks = (block of rows) x (tile of `BATCH_TILE = 16`
  queries), block-major so threads share a block in cache. Inner kernel: 4
  queries x 2 rows (8 accumulators, 6 loads per 8 FMAs). Per-thread heaps per
  query, merged at the end. Big batches run in passes of `BATCH_CHUNK = 4096`.
* **IVF, one query**: rank `nlist` centroids by dot, take the `nprobe` best
  (cell ids ascending), score those cells' rows and the pending rows with the
  1 x 4 gather kernel. Fewer than `k` survivors (filters) -> widen `nprobe` x4,
  ending in the exact scan.
* **IVF, batch (cell-major)**: choose cells per query, invert to "queries that
  probe cell c", score each probed cell **once** against all its queries with the
  4 x 2 kernel. Exactly the rows the single path scores, hence the same answer;
  short queries are re-run on the single path.
* A type filter is a compare of `codes[slot]`; a threshold is the initial `bar`.

## 4. Concurrency model

* **Reads are lock-free.** A search borrows the table immutably and touches no
  shared mutable state: counters are relaxed atomics, scratch is per call. Any
  number run at once. The only lock on the read path is a shared read lock on the
  lazily built index (taken for write only by a build or retrain).
* **Writes are exclusive** (`&mut self` in Rust). A search therefore sees the
  store before a write or after it, never between: *no torn reads*, by
  construction, not by timing.
* **Mixed readers/writers**: `SharedEmbeddingStore` = reader/writer lock.
  Searches hold the read lock for their scan (ms); a write holds the write lock
  for one row copy, a few index updates and (every 256th) one bounded fold
  (~1-3 ms at 1M x 384). Encoding happens before the lock is taken.
  *Rejected*: copy-on-write snapshots. Per-write copying of GBs is impossible;
  chunk-level sharing saves readers nothing the lock does not, and is much
  harder to port.
* **Pool**: one persistent set of workers (`SearchPool`, default = logical CPUs,
  override `ISONGRAPH_VECTOR_THREADS` or `set_search_pool`). `run(n, f)` runs `f`
  on the caller plus up to `n-1` idle workers; `f` must be work-stealing (claim
  units from a shared counter), so a call completes even if no worker is free.
  Callers share the pool without deadlock. `set_scan_threads(t)` caps one call
  (0 = auto, 1 = never touch the pool).

## 5. Write path

* **put(key, type, vector)**: copy into slot (existing key -> overwrite; else
  append), set norm/code/meta/index; if an index exists: detach the slot from its
  cell or pending list (O(1), see below), push it on `pending`; **if
  `pending.len() == FOLD_BATCH (256)`: fold**.
* **fold**: gather the pending rows, find each one's nearest centroid
  (`nlist x dim` MACs per row, parallel in blocks of 16-64 points), append the
  slot to that cell's list. Cost ~6-8 us/row at nlist 4000, dim 384 on 32 threads;
  a pause of ~1.5 ms every 256 writes. Until folded a row is scored exactly by
  every query, so a search never waits for, and is never wrong because of, a fold.
* **remove**: tombstone; detach from its cell/pending list (swap-remove using
  `pos[slot]`, fixing the moved element's `pos`); `churn++`. Drop the index if
  live rows fall under `min_size`.
* **Index state**: `centroids[nlist][dim]`, `assign[slot]` (cell, `PENDING`, or
  none), `pos[slot]` (position in its list), `lists[cell]`, `pending`. Lists hold
  exactly the live folded rows: no stale entries.
* **Compaction** (dead > 25% of slots, or explicit): O(n) row moves, order
  preserving; the index remaps `assign`, rebuilds lists, remaps `pending`.
  This is the one unbounded write section.
* **Retrain**: `churn > 0.5 * trained_n` -> the next search that needs the index
  rebuilds it under the write lock (k-means on `min(40*nlist, n)` live rows, 8
  iterations, seeded; `nlist = 4*sqrt(n)` in [16, 65536]). Build is also
  `build_ann_index()`; call it off the hot path. Never persisted.

## 6. Snapshot semantics

A search observes the state after the last write that completed before it
started (and, with `SharedEmbeddingStore`, possibly later writes that complete
while it waits for the lock, but none partially). A batch observes **one** state
for all its queries. Results never contain a deleted row, a row twice, or a row
whose score is not the cosine of one stored version of it.

## 7. Invariants (checked by `verify_invariants()` and the tests)

1. `norms`, `codes`, `meta` have `n` entries; matrix has `n*dim` floats, all
   chunks but the last full.
2. `dead` = count of `meta == None`; those and only those have NaN norm.
3. `index` has one entry per live slot, and `meta[index[k]].key == k`.
4. Every key is `type + ":" + id` for the slot's type code.
5. Per-model counts equal the rows carrying each model.
6. Index: `assign.len() == n`; a live slot is in exactly one list (cell or
   pending) at `pos[slot]`; dead slots are in none; `pending.len() < 256` between
   writes.
7. Batch == single, bit for bit (same ids, scores, order).

## 8. Knobs and defaults

| knob | default | meaning |
|---|---|---|
| `CHUNK_ROWS` | 1024 | rows per matrix chunk (= scan block) |
| `FOLD_BATCH` | 256 | pending rows that trigger a fold |
| retrain threshold | 0.5 x trained size | churn before the next search retrains |
| compaction threshold | 25% dead | |
| `AnnConfig.nlist / nprobe / min_size / seed` | auto (4 sqrt n) / 5% of nlist / 50,000 / 0 | |
| `set_scan_threads` | 0 | per-call cap; auto = 8 for one scan, whole pool for batch/build |
| pool size | logical CPUs | `ISONGRAPH_VECTOR_THREADS`, `SearchPool::new(n)` |
| `BATCH_TILE` / `BATCH_CHUNK` | 16 / 4096 queries | |

## 9. Portable vs language-specific

*Portable (copy exactly)*: chunked matrix + parallel arrays, tombstone-by-NaN,
slot order and tie-break, the score definition, the pending/fold policy with
`FOLD_BATCH`, `pos`-based O(1) list removal, compaction, retrain rule,
cell-major batch, the invariants.
*Language-specific*: SIMD (Rust: AVX2+FMA intrinsics picked at run time, portable
fallback with the same order; C++: intrinsics/`#pragma omp simd`; C#:
`System.Numerics.Vector<T>` or `Avx2`; Python: numpy/BLAS with the batch as a
matrix product, bit-identity then holds per kernel, not across them; JS:
`Float32Array` loops or WASM SIMD); the lock (Rust `&mut`/`&` + `RwLock`; C++
`shared_mutex`; C# `ReaderWriterLockSlim`; Python a lock around writes with
searches in numpy outside the GIL; JS single-threaded, workers need
`SharedArrayBuffer`); the pool (Rust hand-written, one `unsafe` lifetime erasure;
others use their standard pool); hash map and heap (any).

## 10. Int8 scalar-quantised scan with exact rerank (opt-in)

Code: `src/quant.rs` (child module of `ann`), hooks in `ann.rs`, storage in
`table.rs`. Off by default; `ScanPrecision::Int8 { rerank_factor }` via
`EmbeddingStore::set_scan_precision` (default factor `DEFAULT_RERANK_FACTOR = 4`).
With it off nothing below exists (no memory, no code path).

### 10.1 The copy

Beside every f32 row the table keeps `dim` int8 codes and one f32 scale, slot for
slot, in chunks that mirror the f32 chunks (`CHUNK_ROWS` rows). It is **derived
data**: never persisted, never in a sidecar or digest, rebuilt in one pass by
`set_scan_precision`. Cost: `dim + 4` bytes per row = 0.25x the f32 matrix
(+4 bytes/row).

Quantisation (the same function for rows and queries, so a port must use one
definition everywhere):

```
m     = max over i of |x_i|          (a NaN component is ignored)
if m == 0 or m is not finite:  scale = 0, every code = 0
else:  inv = 127 / m   (f32);   code_i = clamp(round_half_away(x_i * inv), -127, 127)
       scale = m / 127 (f32)
```
Symmetric, no zero point, **-128 is never produced** (the AVX2 kernel relies on
it). A zero scale makes the row's approximate dot 0, like its exact one.

Maintenance (all inside the table, so it cannot go stale): a put overwrites or
appends the codes with the f32 row; a delete leaves them (the NaN norm already
drops the row); compaction moves codes and scales with the rows; `clear` empties
them. Invariant 8 (checked by `verify_invariants`): for **every** slot, tombstones
included, codes and scale equal `quantize(f32 row)` bit for bit.

### 10.2 The search

1. Quantise the query. A query with no direction (zero or non-finite norm) is
   answered by the f32 scan.
2. First pass over exactly the rows the f32 path would score (all rows of the
   blocks, or the probed cells plus the pending rows), with the type filter:
   `d = sum_i q_i * r_i` as an **integer** (exact in i32 for any realistic `dim`:
   384 * 127 * 127 < 2^23), then
   `approx = cosine(float(d) * row_scale * query_scale, row_norm, query_norm)`
   where `cosine` is the section-2 function (zero denominator -> 0; a tombstone's
   NaN norm -> NaN -> dropped by the collector). Keep the best `kc = k *
   rerank_factor` in the section-3 heap (order: score desc, slot asc), with the
   bar starting at `threshold - 0.05` (so quantisation noise cannot lose a row
   that passes the exact threshold).
3. **Exact rerank**: score those `kc` slots with the exact f32 kernel (section 2,
   same summation order as the f32 scan), apply the exact threshold, keep the best
   `k`. Returned scores are therefore exactly the cosines the f32 scan returns;
   only *which* rows reach the rerank is approximate.
4. If `kc >= rows the first pass would score` (live rows for a full scan; the
   probed rows for a cell scan), int8 cannot skip anything: answer in f32. The
   graph's unbounded filtered search (`k = usize::MAX`) therefore stays f32.

Because step 2 is integer arithmetic (associative: lane count, tiling, kernel,
thread count do not matter) and selection is by the total order (score, slot), the
candidate set of a query is a function of that query and the store only. So **batch
== single bit for bit still holds**, and the answer does not depend on the thread
count. The exact batch tiles (block of rows) x (16 queries) and runs the same
per-row first pass; the IVF batch is cell-major as in section 3 (each probed
cell's int8 rows are scored once against all queries that probe it, into
per-query `kc` heaps); a query that comes up short or whose probe holds `<= kc`
rows is re-run on the single-query path.

Kernel (Rust): AVX2 `sign(r, q)` -> `maddubs(|q|, .)` -> `madd(., 1)` -> add: 32
products per 4 instructions; a pair sum is at most 2*127*127 < 2^15 so the 16-bit
step cannot saturate. The `unsafe` block sits behind an `assert!` that every
operand has `dim` bytes; a portable loop is the fallback and the specification
(they are tested equal on every dimension 1..1000 sampled, including +-127
extremes). A port may use any integer kernel (NEON `sdot`, VNNI): the result is
identical by construction.

### 10.3 Reporting recall

`int8_recall_report(queries, k, node_type, threshold, rerank_factors)` runs the
full scan in f32 and, per factor, in int8 + rerank, and returns the mean and worst
per-query overlap of the two top-k sets, whether every returned score is
bit-identical to the exact cosine of its row, whether int8 applied (`k * factor <
live`), and median times. It measures full scans; an IVF probe adds its own
approximation on top (measure it with `recall_check`; the int8 error is the same
mechanism on a smaller candidate set).

### 10.4 Measured (indicative; this machine was shared with another agent's work)

Clustered synthetic data, 384-d, k = 10, rerank factor 4. Timings are the fastest
of 180 searches (5 batches): medians swung 3x run to run on this machine, minimums
were stable to ~5%.

| 1M x 384 | f32 | int8 | |
|---|---|---|---|
| exact scan, 1 thread | 155 ms | 43 ms | 3.6x |
| exact scan, auto threads | 31.9 ms | 11.2 ms | 2.9x |
| IVF (nlist 4000, nprobe 200), 1 thread | 10.2 ms | 2.9 ms | 3.5x |
| IVF, auto threads | 2.87 ms | 1.75 ms | 1.6x |
| exact batch of 128, auto / 1 thread | 309 / 3202 ms | 240 / 2947 ms | 1.3x / 1.1x |
| IVF batch of 128, auto / 1 thread | 75 / 502 ms | 69 / 431 ms | 1.1x / 1.2x |

100k x 384: exact 1 thread 5.7 -> 1.85 ms (3.1x), auto threads 3.3 -> 0.58 ms
(5.7x), batch 1.0-1.1x. Recall@10 against the f32 scan (128 queries): factor 1
0.970 (1M) / 0.976 (100k), factor >= 2 **1.000**, worst single query 1.00 from
factor 2. Memory (working set, 1M x 384): 1658 -> 2033 MiB, **+375 MiB (+23%)**;
the copy itself is 370 MiB against a 1466 MiB matrix.

Honest reading: int8 buys **single-query latency** (2-4x on exact scans, 1.6-3.5x on
IVF cells) because a scan is memory-bandwidth bound and the copy is 4x smaller. It
buys **nothing for batch throughput** (1.0-1.3x): the f32 batch kernel (4 queries x
2 rows in registers) is compute-bound, and an AVX2 int8 dot does the same number of
MACs per instruction as an f32 FMA; only VNNI (`vpdpbusd`, not assumed here) would
change that. It costs +23% RSS at 384-d. Recommendation: keep it as an opt-in for
latency-sensitive single-query serving of large stores; do not enable it for
batch/offline workloads; the recall figure is for synthetic clustered data and
must be re-measured with `int8_recall_report` on real embeddings before relying on
factor 4 (use a larger factor if it shows < 0.99).

Ports: implement 10.1 and 10.2 exactly (including the quantisation rounding rule
and the `0.05` slack) and the invariant; kernels and threading are free.

## 11. Binary snapshot (optional, beside the JSON sidecar)

Code: `src/snapshot.rs`; glue in `SemanticGraph::{save_with_snapshot, load_with_report}`.

Evidence for it: at 1M x 384 the JSON sidecar is 4.5 GB and takes 30-50 s to write
and 20-30 s to read here; a snapshot of the same store is 1.5 GB and
saves in ~3.5 s / loads in ~4.3 s. The JSON sidecar is **unchanged and remains the
portable, authoritative format**; the snapshot is a cache of it. Graph files, the
sidecar bytes and the graph digest are byte-identical with or without a snapshot.

### 11.1 Files and protocol

* Path: `<graph file>.embeddings.snap` (`.embeddings.json` is the sidecar).
* **Save with snapshot**: write the graph file, then the JSON sidecar (both atomic,
  as before), then the snapshot: temp file in the same directory, flush + fsync,
  rename over the old one. A failure at any point leaves the previous file
  untouched and removes the temp file. If the snapshot cannot be written the pair
  is still saved, a stale `.snap` is removed, and the error is returned. A save
  **without** snapshot deletes any existing `.snap` first (the sidecar is about to
  change).
* **Load**: read the graph; if a `.snap` exists, take the cheap identity checks
  (below), and only then stream the body into a *staging* store, verify the
  trailer checksum, and install the staging store; **any** failure (missing file,
  bad magic/version/flags, identity mismatch, truncation, malformed body,
  duplicate key, checksum) leaves the store untouched and falls back to the JSON
  sidecar, exactly as without a snapshot (its own digest check and
  `GraphMismatch` policy apply). The load reports which path ran
  (`LoadReport { source: Snapshot | Json | NoEmbeddings, snapshot_skipped:
  Option<reason>, embeddings, seconds }`).
* **Identity** (all required): `graph_digest` in the header equals the digest of the
  graph file read now (same `GRAPH_DIGEST_BASIS`; no Python-view leniency), and
  the header's sidecar identity equals the identity of the sidecar file on disk.
  A snapshot is never used when the sidecar is missing.
* Sidecar identity = `len` (u64) and `fingerprint` = SHA-256 over: `len` as u64 LE,
  the first 65,536 bytes, the last 65,536 bytes, and for i in 1..=64 the 4,096
  bytes at offset `floor(len * i / 65)` (each clipped to the file end; for a file
  shorter than that the windows overlap; every window is still hashed, in order, even
  where it overlaps another, and a window that starts at or past the end is empty). Any
  serializer rewriting the sidecar changes it; **limit**: an in-place edit that
  keeps the length and touches no sampled byte is not detected. Tools that rewrite
  the sidecar must delete the `.snap` (this crate's `save` does).

### 11.2 Byte layout, version 1 (all integers little endian)

```
header
  0   8   magic        "ISONVSNP"
  8   4   version      u32 = 1            (reader rejects any other)
  12  4   flags        u32                bit 0: sidecar identity present; other bits must be 0
  16  4   dim          u32                must be > 0 when count > 0
  20  4   block_rows   u32 (1..=2^20)     writer uses 1024; reader honours the file's
  24  8   count        u64                live rows (<= 2^32-1)
  32  .   graph_digest        opt string
          graph_digest_basis  opt string  ("nodes-edges-json-v1" when a digest is given)
          encoder_model       opt string
          encoder_version     opt string
          [if flags bit 0]  sidecar_len u64, sidecar_fingerprint 32 bytes
          n_models  u32, then n_models x string   (model names, order of first use)
block * ceil(count / block_rows)       R = min(block_rows, rows left)
  R x row metadata:   type string, id string, text string, model_index u32
  R x dim x f32       row-major, in the same row order
trailer  32 bytes  SHA-256 of every byte before it (from offset 0)
```
`string` = `u32 length` + that many UTF-8 bytes. `opt string` = the same, with length
`0xFFFFFFFF` meaning absent. Rows are in **the writing port's own sidecar order**,
live rows only, so that loading a snapshot yields the same slots, tombstone-free,
as loading that port's JSON (equal score ties resolve identically *within a port*).
The Rust crate sorts by node type then id (byte order); Python, TypeScript and
JavaScript write arrival order, C# its own comparer. A reader must **not** check or
rely on row order, and tie order after a load can differ between ports.
`key = type ":" id`; two rows whose keys are equal as strings (type `a:b` with id
`c`, and type `a` with id `b:c`) are a repeated key: a reader rejects the file and
falls back to the JSON, which keeps them apart in ports with tuple keys.
A reader must reject: a length past the end of the file, a string that is not
valid UTF-8, a model index >= n_models, a repeated key, `count * dim * 4` larger
than the file, bytes between the last block and the trailer (bytes "after" the
trailer cannot be told apart from these: the trailer is the last 32 bytes), a
count above what the reader can hold (the format allows 2^32-1; a port whose arrays
are limited, such as .NET, rejects more and falls back), a trailer mismatch.
**Ids are text.** The format carries no id kind, so a snapshot cannot say that
`("person", 1)` was an integer id. Ports whose ids can be integers (Python,
TypeScript, JavaScript) restore the kind from the graph (a string id that names a
node stays a string; a canonical integer string that names an integer node becomes
the integer), and refuse to write a snapshot for an embedding that would not restore
identically (they save the JSON pair, remove any stale `.snap` and report why).
A version 2 should add an id-kind bit per row.
The reader sets `dimension` from the first row (as the JSON import does), and the
import metadata (encoder model/version, digest, basis) from the header.

Smallest example (1 row: type `t`, id `a`, text `x`, model `provided`, vector
`[1.0, -2.0]`, no digest, no sidecar), 123 bytes:
```
49 53 4f 4e 56 53 4e 50  01 00 00 00  00 00 00 00      magic, version 1, flags 0
02 00 00 00  00 04 00 00  01 00 00 00 00 00 00 00      dim 2, block_rows 1024, count 1
ff ff ff ff x4                                          four absent strings
01 00 00 00  08 00 00 00 "provided"                     1 model
01 00 00 00 "t"  01 00 00 00 "a"  01 00 00 00 "x"  00 00 00 00      row metadata, model 0
00 00 80 3f  00 00 00 c0                                1.0, -2.0
<32-byte SHA-256 of the 91 bytes above>
```
(A port that has no float byte order issue reads/writes f32 as IEEE-754 LE.)

### 11.3 What is deliberately not in it

* **IVF state.** The index is rebuilt (lazily, or by `build_ann_index`). A rebuild is
  deterministic for a given store and seed, but it is *not* the saved store's index:
  the saved index depends on the slot order and fold history of the live store,
  while a canonical-order file loads into a different slot order. Reproducing the
  saved index's answers after a round trip cannot be proved without storing the
  slot permutation and the pending list and re-validating them, so it is not
  attempted: search results after load equal those of a JSON load, and an index
  built after load equals one built after a JSON load.
* **The int8 copy** (derived; section 10).
* Warnings, the pool, per-store settings.

### 11.4 Measured (indicative)

| 384-d | rows | JSON save / size / load | snapshot save / size / load |
|---|---|---|---|
| 100k | 100,000 | 1.7 s / 451 MiB / 2.5 s | 0.21 s / 150 MiB / 0.34 s |
| 1M | 1,000,000 | 49.8 s / 4513 MiB / 30.3 s | 3.55 s / 1503 MiB / 4.33 s |

The snapshot is 1.03x the raw f32 matrix (1465 MiB at 1M), 3.0x smaller than the
JSON, loads 7x faster and saves 14x faster. It is **not** mmap-class: faiss and
hnswlib save in 0.06-0.2 s because they write one buffer; this loader reads,
checksums (SHA-256 ~2 GB/s here, ~0.7 s of the 4.3 s), copies each row into the
chunked matrix, computes its norm and builds the key map, which is most of the
time. Making load mmap-fast would need a layout that is the in-memory layout
(aligned chunks, no per-row metadata interleaved), a different trade-off than the
portability chosen here.

Ports: write `.snap` byte-for-byte as in 11.2 or skip the feature; a port that does
not know the file simply loads the JSON. A port reading a `.snap` must apply every
rejection rule above and fall back to the JSON on any failure.

## 12. HNSW (opt-in, beside IVF)

`AnnConfig.mode = Hnsw` (`AnnConfig::hnsw()`) replaces IVF as the approximate
index; exact search stays the default and IVF is untouched. Code:
`src/hnsw.rs`. Like IVF it is built lazily by the first search that wants it (or
`build_ann_index`), is **never persisted** (a loaded store rebuilds it; the graph
file, the JSON sidecar and the graph digest are byte-identical with it on or
off), and is used only for a bounded `top_k` on a store of at least `min_size`
live rows. `exact`, `recall_check`, `set_scan_threads` and the pool mean what
they mean for IVF. Why: at equal recall@10 of about 0.95 on real text-384
vectors, IVF answers a single query about 4x (100k) to 10x (1M) slower than a
graph walk, because a graph touches far fewer rows per query than the probed
cells do.

### 12.1 Parameters

| knob | default | meaning |
|---|---|---|
| `hnsw.m` | 16 | links per node on layers >= 1; layer 0 holds `2m` (2..64) |
| `hnsw.ef_construction` | 200 | candidates while linking a row (used as `max(efc, m)`) |
| `hnsw.ef_search` | auto | candidates per query, `max(ef, k)`; auto by live rows: 64 (<10k), 96 (<200k), 128 (<600k), 192 |
| `seed` | 0 | seeds the level generator (shared with IVF's k-means seed) |
| `min_size` | 50,000 | same threshold as IVF: below it, or `top_k >= live`, search is exact |

`ef_search` can be changed without a rebuild (`set_ann_ef_search`) or for one
call (`similarity_search_vector_ef`).

### 12.2 Layout: links only, no second copy of any vector

Scores are computed from the one stored matrix with the section 2 kernels
(1 query x 4 rows, then 1 x 1), so a hit's score is **bit-identical** to the
exact scan's, and ties order by slot as everywhere else. The graph holds, per
table slot:

* `level[slot]` (u8; 255 = not in the graph),
* layer 0: one block of `2m + 1` u32 (link count, then links), in chunks of
  `CHUNK_ROWS` nodes so growth never copies,
* `upper_at[slot]` (u32) into an arena holding, for a node of level `L`, `L`
  consecutive blocks of `m + 1` u32 for layers `1..L`,
* the entry slot and its level.

That is `4(2m+1) + 5` bytes plus about `4(m+1)/m` of upper layers per node:
**146 B/node at m = 16**, 9.5% of the raw 384-d vectors; the whole store
(matrix, metadata, graph) measured **1.18x** the raw vector bytes at 1M x 384.
Searches also use a per-thread visited bitmap (1 bit per slot, cleared by
touched words, so a query pays for what it visited).

### 12.3 Level draw and search (portable, integers only)

* **Level**: `level = 0; while level < 24 and rng.below(m) == 0: level += 1`
  with the SplitMix64 generator of the IVF code (`state += 0x9E3779B97F4A7C15`,
  then the usual mix; `below(n)` = high 64 bits of `next * n`), seeded `seed XOR
  0x4857535748575357`, one draw sequence in insertion order. So
  `P(level >= l) = m^-l`, and no logarithm is needed (ports must not use one).
* **Order** ("better is greater"): higher score, then lower slot. Every heap and
  every tie in the graph uses it.
* **Query**: greedy descent (ef = 1) from the entry through layers `top..1`,
  then a best-first search of layer 0: candidates in a max-heap, the best `ef`
  *admitted* nodes in a min-heap; pop the best candidate; stop when it is worse
  than the worst admitted node and `ef` are kept; score its unvisited links
  (4 at a time) and push those that beat the worst kept (or any, while fewer
  than `ef` are kept). **Admitted** = live (norm not NaN), type code matches the
  filter, score >= threshold. Tombstones and non-admitted nodes are routed
  through but never kept. NaN scores rank last. The answer is the best `k` kept,
  sorted by (score desc, slot asc).
* **Filters** (type, threshold, a graph `node_filter` gate): if fewer than `k`
  are kept, retry with `ef x 4`, repeating; give up to the **exact scan** when
  `ef >= live` (first try) / `2 ef >= live` (later tries), or when one search
  scores more than `max(live / 8, 16 ef)` nodes (a filter so selective that
  scanning is cheaper). The graph-gate path asks for the best `ef x scale`
  admitted nodes and widens the same way.
* **Batch**: queries are spread over the pool, each runs the single-query path
  above, so a batch answer equals the one-query answer bit for bit. (No
  cell-major sharing exists for a graph: see 12.8.)
* **Concurrency**: a search takes `&self` plus a shared read lock on the lazily
  built graph, like IVF; all scratch is per thread. Writes are `&mut self`.

### 12.4 Linking (one routine for build, insert and update)

`link(jobs)`: for each job (a new row with its drawn level, or an existing node
to relink) **plan** against the graph as it is, then **apply** the plans in job
order.

* **Plan** (read-only, parallel over jobs): descend greedily to the job's level;
  on each layer `min(level, top)..0` run the layer search with `ef_construction`
  admitting only live nodes other than the job itself, and pass the kept set as
  entry points to the next layer down; add the other *new* jobs of the same
  batch with level >= the layer as extra candidates (scored by brute force; this
  is how a batch sees itself); sort candidates best first; pick `m` with the
  **diversity heuristic**: keep a candidate only if it scores no better with any
  already-kept node than with the base (`sim(c, kept) > sim(c, base)` rejects);
  with `<= m` candidates, keep all.
* **Apply**: write each job's lists (new nodes get their blocks; an update
  overwrites its own lists); then add the reverse links. Reverse links are
  grouped by (layer, target) in a fixed order; a target whose list stays within
  capacity (`2m` on layer 0, `m` above) just appends (deduplicated); an overflow
  first drops tombstones, and if the list is still over capacity **re-prunes**
  it with the same heuristic over the old links plus the new ones. Groups are independent, so
  they run in parallel and are written back in group order. The entry point is
  the first new node of the highest level.
* A job with a non-finite norm is not linked (the exact scan never returns it
  either).

### 12.5 Maintenance

* **Build**: insert live rows in slot order in batches of `size / 16` (1 at the
  start, at most 512), each batch one `link` call. Because plans read a frozen
  graph and are applied in order, **the graph depends on the seed, the data and
  the insertion order, not on the thread count** (checked by the tests). The
  price is that rows of one batch find each other only by brute force, not
  through the graph; at most 6% of the graph, this costs no measurable recall
  (the recall-vs-ef curve equals hnswlib's on text-384 at 100k).
* **Insert**: the write that stores a row links it (`link` of one job; the Rust
  code still runs its reverse-link groups on the pool, the C# and C++ ports
  run a lone insert serially and measured no loss). No pending list, so a search never waits for a fold and never
  scans an unindexed tail. Cost 0.4-1.5 ms per insert at 100k-1M x 384
  (dominated by the layer search at `ef_construction` 200 and the re-prunes).
* **Update** (same slot, new vector): the node keeps its level; its own links
  are recomputed from the new vector and the new neighbours link back. Links
  other nodes held to the old position stay: they are only hints, every score
  comes from the current vector, and a list drops them when it next overflows.
  Counts as churn.
* **Delete**: tombstone only (the table's NaN norm already hides the row). The
  node and its links stay, so routes through it keep working; its floats are
  still in the matrix until compaction. A tombstone leaves a list when that list
  is re-pruned. No work at delete time (about a microsecond).
* **Compaction** (the table's, > 25% dead, or explicit): the graph follows the
  table's old-slot -> new-slot map in one serial pass: tombstones leave, every
  link is renumbered, and a node that lost links takes up the links of the
  tombstones it pointed to ("bridging": walk its lost neighbours in order and
  append their surviving links not already present, up to capacity, no distance
  computed). The entry point becomes the first node of the highest level. An
  O(n x m) pause, the same order as the table's own compaction.
* **Rebuild**: updates plus deletes since the build exceeding the node count
  mark the graph stale; the next search that needs it rebuilds it under the
  write lock (the build time below; searches wait). Call
  `build_ann_index(true)` off the hot path to choose when. Inserts do not count.
* Dropping the index when live rows fall below `min_size` (or 2), `clear`, and
  `set_ann` follow IVF.

### 12.6 Invariants (added to section 7; checked by `verify_invariants`)

8. `level.len() == slots`; every live row with a finite norm is in the graph;
   tombstoned rows may be (until compaction).
9. A node's lists exist for layers `0..=level`, hold no more than the layer's
   capacity, no self link, no duplicate, and every target is in the graph with
   level >= the layer.
10. The entry point is in the graph at the top level, and no node is higher.

### 12.7 Measured (real text-384, MiniLM, HotpotQA; `examples/hnsw_bench.rs`)

32-thread laptop with other work running, so **timings are indicative**; recall
and memory are exact. Single-thread loops pinned to one CPU, `k = 10`, 1,000
real HotpotQA questions, ground truth the float64 cosine top-10
(`benchmarks/competitive/`), `m` 16, `ef_construction` 200, build on all 32
threads. IVF is the default `AnnConfig` with `scan_threads(1)` in the 1T loop.

| | 100,000 rows | 1,000,000 rows |
|---|---|---|
| HNSW build | 4.9-6.1 s | 142-170 s |
| IVF build | 0.6-1.2 s | 8.4 s |
| graph size | 13.8 MB (145 B/node) | 139 MB (146 B/node) |
| whole store / raw vectors | 1.18x | 1.18x |
| insert / update (median, p99) | 0.44 / 0.81 ms | 0.65-1.5 / 1.4-2.7 ms |
| delete | < 2 us | < 2 us |
| HNSW recall@10 by `ef` | 64: .915, 96: .948, 128: .963, 192: .980 | 96: .906, 128: .929, 192: .955, 256: .969 |
| **HNSW** at the auto `ef` (96 / 192): recall, QPS 1T, QPS batch | .948, 5,021, 28,925 | .955, 1,539-1,782, 6,225 |
| **IVF** near the same recall (`nprobe` 101 / 200): recall, QPS 1T, QPS batch | .943, 1,186, 49,996 | .941, 159, 8,570 |
| single-thread speed-up over IVF | about 4x | about 10x |

For reference, `benchmarks/competitive/` measured hnswlib at recall 0.949, 3,563
QPS (100k, `ef` 96) and 1,486 QPS at recall 0.95 (1M) on a quiet machine, and its
`ef`-vs-recall curve matches the one above (0.916 at 64, 0.964 at 128), so the
graph quality is the same; the batch column is where IVF is not beaten (12.8).

### 12.8 Not done, on purpose

* **No persistence.** The graph is rebuilt after load, like IVF. A saved graph
  would change the file format and the digest contract; loading 1M rows takes
  seconds, rebuilding the graph minutes.
* **Slow build.** HNSW cannot be built as fast as k-means cells. The batched
  parallel build is the mitigation; incremental inserts are the other (build once
  from the loaded rows, then writes keep it current).
* **Update leaves stale in-links** (12.5) and delete leaves tombstones until
  compaction; heavy churn rebuilds instead of repairing.
* **No cell-major batch.** IVF's batch scores each probed cell once for all
  queries that probe it; a graph has no such shared structure, so a batch is
  throughput-parallel single queries. At 100k text-384 (recall about 0.95) IVF's
  batch measured faster than HNSW's; at 1M they are comparable. Choose HNSW for
  single-query latency and QPS, IVF for fast builds, small memory and big
  batches.
* **No quantisation, no filter-aware graph (ACORN style)**: a selective type
  filter widens `ef` and may end in the scan.
* **No online graph repair** other than the bridging at compaction.

### 12.9 Ports and cross-checks

C++ (`isongraph_vector_hnsw.hpp`) and C# (`HnswIndex.cs`) implement 12.3-12.5
from this text. Checked with `examples/hnsw_xcheck.rs` and the matching C++ /
C# dumpers on integer-generated data (5-6 parameter sets, n 5k-50k, m 2-32, through
build, writes and compaction): the C# port gave identical ids, order and graph
statistics in all six sets; C++ gave identical ids in four of five sets and one
query of 200 differed by one id in the fifth. With the C++ scorer temporarily
replaced by Rust's 8-lane fused order, ids **and** score bits were identical in
every phase, so the portable rules above are sufficient to reproduce Rust's
graph. What is not portable is the score itself: section 2 lets a port pick its
summation order, and Rust's `dot` is fused (`mul_add`) on AVX2+FMA hosts and
unfused on the portable fallback, so identical graphs across ports (or across
Rust hosts) hold only up to last-bit near-ties. Scores are bit-identical to the
exact scan **within a port**, not across ports.

Notes for readers: (a) the plan selects `m` neighbours on layer 0, not `2m`;
forward lists start at `m` and only reverse links grow them toward `2m`. (b) The
visit budget `max(live / 8, 16 ef)` sends graph search to the exact scan often on
small or hard stores (a 12,000 x 24-d Gaussian store sent 66% of queries to the
scan at ef 64), so tests and benchmarks must count `approximate` searches, not
just recall. (c) `Sc` orders with `f32::total_cmp` (-0.0 below +0.0) while the
final sort and the exact scan treat +-0 as equal and break ties by slot; the
C++ and C# ports use the exact-scan rule everywhere; the difference needs a
score of exactly +-0.0.

TypeScript/JavaScript (`ann: 'hnsw'`, single-threaded) matched Rust's ids in 6
of 6 sets; a pure-Python port also matched 6 of 6 but is not shipped (1.6-1.7x
IVF). Two findings from those ports: (d) a tombstoned row's floats must stay in
the table until compaction, because routing through tombstones scores them (a
Python table that zeroed deleted rows broke the match, 4 of 6 until fixed);
(e) the TS table pops trailing tombstones and reuses the slot, which Rust and
C# never do, so after deleting the newest row slot numbers (hence tie order and
level draws) can differ from Rust's; the graph keeps such nodes as dead ghosts
until the slot is reused or compaction. C++'s HNSW scores rows with a
lane-per-row SSE2 kernel and cached norms: same ascending sums per row, so the
same bits as the scalar scorer; the graph visits exactly Rust's number of rows
at equal `ef` (1,850 per query at ef 96, 100k real text).

### 12.10 Portable vs language-specific

*Portable (copy exactly)*: the level draw (integer, seeded), the (score, slot)
order, the layout sizes, the plan/apply split with in-batch brute-force
candidates and fixed apply order (this is what makes the build deterministic and
parallel in any language), the diversity heuristic and overflow re-prune, the
widening/fallback rules, tombstone-then-bridge compaction, the rebuild rule.
*Language-specific*: threads and the pool (plans are independent; any parallel
map with ordered results works; a single-threaded port is just slower), the
visited set (bitmap per thread, or a stamp array), prefetching, heaps, SIMD
(section 2: one fixed summation order everywhere).
