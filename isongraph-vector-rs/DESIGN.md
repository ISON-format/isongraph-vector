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
