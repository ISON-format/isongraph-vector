//! Search for [`EmbeddingStore`]: the exact scan, the batched scan, and the
//! opt-in IVF-flat approximate index built on top of the store's matrix (the
//! opt-in HNSW graph index lives in `hnsw.rs`; `AnnConfig::mode` picks one).
//! `DESIGN.md` is the specification; this file is its Rust realisation.
//!
//! # The matrix
//!
//! The store's vectors live once, in a [`VectorTable`] (see `table.rs`): a
//! chunked row-major `f32` matrix with a norm and a type code per row. A search
//! is a pass over that memory: a dot product per row, a bounded heap for the
//! best `k`, no per-row allocation, no hashing, **no lock**. Writes need
//! `&mut self`, so a search can never overlap one (see `shared.rs` for the
//! reader/writer wrapper).
//!
//! # One definition of a score
//!
//! Every cosine in the crate (single query, batch, index probe, fold, k-means)
//! is `dot / (|row| * |query|)` where `dot` has one fixed summation order: a
//! single 8-lane accumulator walked in ascending order, a fixed horizontal-sum
//! tree, then the scalar tail. Kernels differ only in how many (query, row)
//! pairs they compute at once; none changes the order inside a pair, so a
//! batched answer is bit-identical to the one-query answer, and the answer does
//! not depend on thread count.
//!
//! # The IVF-flat index
//!
//! Opt-in through [`AnnConfig`]. Spherical k-means (seeded, on a sample)
//! partitions the rows into `nlist` cells by direction; a query scores the
//! `nprobe` cells whose centroids are nearest and only the rows inside them,
//! with exactly the cosine the exact scan uses. The index stores no vectors:
//! per-cell lists of slot numbers, a cell number per slot, and the centroids.
//! New and updated rows wait in a short `pending` list that every query scores
//! exactly; the write that makes it [`FOLD_BATCH`] long folds it into cells on
//! the spot, so a search never has a backlog to pay for. The index is built
//! lazily by the first search that wants it (or by `build_ann_index`) and is
//! never persisted.
//!
//! The behaviour deliberately mirrors the Python port's `ann="ivf"`.

use crate::hnsw::HnswConfig;
use crate::pool::SearchPool;
use crate::table::{Rows, VectorTable, CHUNK_ROWS};
use crate::{EmbeddingEncoder, EmbeddingError, EmbeddingStore, Query, SimilarityResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::time::Instant;

/// Below this many vectors the index is not used and search stays exact.
pub const ANN_MIN_SIZE: usize = 50_000;
/// Rows that are new or rewritten since the last fold wait in `pending`, scored
/// exactly by every query. The write that makes the list this long assigns all
/// of them to cells. Smaller is a shorter write pause and a cheaper query-side
/// exact tail; larger is fewer, bigger pauses.
pub const FOLD_BATCH: usize = 256;
/// Retrain once writes plus deletes exceed this fraction of the trained size.
const RETRAIN_FRACTION: f64 = 0.5;
const KMEANS_ITERATIONS: usize = 8;
const SAMPLE_PER_CELL: usize = 40;
/// Threads one single-query scan uses when none is configured (memory
/// bandwidth, not compute, limits it; see `workers`).
const SCAN_AUTO_THREADS: usize = 8;
/// Rows per scan task. Divides the table's chunk size, so a block never
/// straddles chunks.
const BLOCK_ROWS: usize = 1024;
/// A scan spreads over threads only past this many floats per thread.
const PAR_MIN_FLOATS: usize = 1 << 21;
/// The same for the rows of probed cells, which cost more per float.
const PAR_MIN_CELL_FLOATS: usize = 1 << 20;
/// Points per unit of parallel k-means work; a multiple of `TILE`.
const NEAR_BLOCK: usize = 256;
/// Points per tile of the assignment kernel (the tile stays in L2).
const TILE: usize = 64;
/// Queries scored together against one block of rows in a batch.
const BATCH_TILE: usize = 16;
/// Queries handled per pass of a big batch (bounds the per-thread heaps).
const BATCH_CHUNK: usize = 4096;

const NO_CELL: i32 = -1;
const PENDING: i32 = -2;

// =============================================================================
// Public types
// =============================================================================

/// Settings for the opt-in approximate index (IVF-flat by default; HNSW with
/// `mode: AnnMode::Hnsw`, see [`AnnConfig::hnsw`]).
///
/// Off unless given to [`EmbeddingStore::set_ann`]; with it off every search
/// is exact, as before. With it on, a *bounded* top-k search
/// (`top_k < number of vectors`) on a store of at least `min_size` vectors is
/// answered from the index. The scores are the same cosine values the exact
/// scan gives, but a true neighbour in an unprobed cell can be missing:
/// measure it with [`EmbeddingStore::recall_check`] on your own queries.
///
/// Policy (the Python port's, except that folding is eager):
///
/// * the index is built by the first search that wants it (or by
///   `build_ann_index`), from k-means on a sample of up to `40 * nlist` live
///   vectors, and is never persisted;
/// * new and updated vectors are scored exactly until [`FOLD_BATCH`] of them
///   are pending, then the write that reaches that count folds them into their
///   nearest cells;
/// * a delete just stops the row being returned;
/// * once writes plus deletes since training exceed 50% of the trained size
///   the next search retrains the index (with an automatic `nlist`, re-chosen
///   for the new size);
/// * a clear, or a store that shrinks below `min_size`, drops the index;
/// * a filter (`node_type`, `threshold`, a graph `node_filter`) applies to the
///   probed rows; if fewer than `top_k` survive the probe is widened 4x and
///   tried again, ending in the exact scan, so a selective filter costs time
///   rather than results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnConfig {
    /// Cells in the index; `None` picks about `4 * sqrt(N)` (16 to 65,536)
    /// when the index is built.
    pub nlist: Option<usize>,
    /// Cells scanned per query; `None` picks about 5% of `nlist`. More cells
    /// is higher recall and slower; `nprobe >= nlist` is the exact scan.
    pub nprobe: Option<usize>,
    /// Below this many vectors search stays exact (an index only pays off on
    /// large stores). Default 50,000.
    pub min_size: usize,
    /// Seed for the k-means initialisation (IVF) or the level draws (HNSW): the
    /// same data in the same order gives the same index.
    pub seed: u64,
    /// Which index answers bounded searches. `Ivf` (the default) keeps every
    /// pre-existing behaviour; `Hnsw` is the graph index of `hnsw.rs`.
    pub mode: AnnMode,
    /// HNSW settings; ignored unless `mode` is `Hnsw`.
    pub hnsw: HnswConfig,
}

/// Which approximate index an [`AnnConfig`] asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnnMode {
    /// IVF-flat: spherical k-means cells (the default).
    #[default]
    Ivf,
    /// A hierarchical navigable small-world graph: slower to build, several
    /// times faster per query at equal recall on large stores.
    Hnsw,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            nlist: None,
            nprobe: None,
            min_size: ANN_MIN_SIZE,
            seed: 0,
            mode: AnnMode::Ivf,
            hnsw: HnswConfig::default(),
        }
    }
}

impl AnnConfig {
    fn validate(&self) -> Result<(), EmbeddingError> {
        if self.nlist == Some(0) {
            return Err(EmbeddingError::InvalidOption(
                "ann nlist must be at least 1".to_string(),
            ));
        }
        if self.nprobe == Some(0) {
            return Err(EmbeddingError::InvalidOption(
                "ann nprobe must be at least 1".to_string(),
            ));
        }
        self.hnsw.validate()
    }

    /// Settings for the HNSW index with its defaults (M 16, ef_construction
    /// 200, ef_search chosen per query), otherwise as [`AnnConfig::default`].
    pub fn hnsw() -> Self {
        Self {
            mode: AnnMode::Hnsw,
            ..Self::default()
        }
    }
}

/// How many searches each path has answered since the store was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnnSearchCounts {
    /// Answered from the index.
    pub approximate: u64,
    /// Answered by the full scan while an index was configured.
    pub exact: u64,
    /// Answered from the index after widening the probe at least once.
    pub widened: u64,
    /// Times the index was (re)trained.
    pub builds: u64,
}

/// What the approximate index looks like right now; see
/// [`EmbeddingStore::ann_stats`]. Never builds anything.
#[derive(Clone, Debug, Default)]
pub struct AnnStats {
    /// An [`AnnConfig`] is set.
    pub enabled: bool,
    /// An index exists and will answer.
    pub active: bool,
    pub nlist: Option<usize>,
    /// Cells scanned per query (the effective value, once an index exists).
    pub nprobe: Option<usize>,
    /// Vectors the centroids were trained on.
    pub trained_n: Option<usize>,
    /// Rows scored exactly until they are folded into cells (always fewer than
    /// [`FOLD_BATCH`] between writes).
    pub pending: usize,
    /// Writes plus deletes since training.
    pub churn: usize,
    pub build_seconds: Option<f64>,
    /// Memory the index holds (it stores no vectors).
    pub index_bytes: usize,
    pub min_size: Option<usize>,
    pub searches: AnnSearchCounts,
    /// Fold steps run by writes since the index was built.
    pub folds: u64,
    /// Total time those steps took.
    pub fold_seconds: f64,
    /// Which index is configured (`Ivf` when none is).
    pub mode: AnnMode,
    /// HNSW only: the `ef_search` a query uses when it gives none (before the
    /// `max(k)` rule): the configured value, else the size-based default.
    pub ef_search: Option<usize>,
    /// HNSW only: highest layer of the graph.
    pub max_level: Option<usize>,
    /// HNSW only: nodes in the graph (live rows plus tombstones awaiting
    /// compaction).
    pub graph_nodes: usize,
    /// HNSW only: tombstones still in the graph.
    pub graph_dead: usize,
    /// HNSW only: rows linked into the graph by writes since the build, and the
    /// time that took.
    pub inserts: u64,
    pub insert_seconds: f64,
}

/// Where a store's memory goes; see [`EmbeddingStore::memory_stats`]. The
/// vectors are held once, so `matrix_bytes` is the only part that scales with
/// the dimension.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    /// Row slots in use, tombstones included.
    pub rows: usize,
    pub live: usize,
    pub dead: usize,
    pub chunks: usize,
    /// Bytes allocated for the matrix (`f32` rows).
    pub matrix_bytes: usize,
    /// Bytes the live vectors need at minimum: `live * dim * 4`.
    pub live_vector_bytes: usize,
    /// Norms, type codes, per-row metadata, keys, text, the key index.
    pub metadata_bytes: usize,
    /// The approximate index, if built.
    pub index_bytes: usize,
}

impl MemoryStats {
    pub fn total_bytes(&self) -> usize {
        self.matrix_bytes + self.metadata_bytes + self.index_bytes
    }
}

/// Recall of the approximate index against the exact scan, over some queries.
///
/// `recall` is the mean over queries of `|approximate top-k & exact top-k| /
/// |exact top-k|` (1.0 = identical sets); `min_recall` is the worst single
/// query. `approximate` is false when the index was not actually consulted (it
/// is off, the store is smaller than `min_size`, or `nprobe >= nlist`), in
/// which case recall is 1.0 by construction. Times are per-query medians in
/// milliseconds.
#[derive(Clone, Debug)]
pub struct RecallReport {
    pub recall: f64,
    pub min_recall: f64,
    pub queries: usize,
    pub k: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub approximate: bool,
    /// HNSW only: the effective `ef_search` (0 otherwise).
    pub ef_search: usize,
    pub ann_ms: f64,
    pub exact_ms: f64,
}

/// The search counters, updated without a lock.
#[derive(Default)]
pub(crate) struct Counters {
    approximate: AtomicU64,
    exact: AtomicU64,
    widened: AtomicU64,
    pub(crate) builds: AtomicU64,
}

impl Counters {
    /// One search answered from the index (`widened` if it needed a wider probe).
    pub(crate) fn note_approximate(&self, widened: bool) {
        self.approximate.fetch_add(1, Ordering::Relaxed);
        if widened {
            self.widened.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn snapshot(&self) -> AnnSearchCounts {
        AnnSearchCounts {
            approximate: self.approximate.load(Ordering::Relaxed),
            exact: self.exact.load(Ordering::Relaxed),
            widened: self.widened.load(Ordering::Relaxed),
            builds: self.builds.load(Ordering::Relaxed),
        }
    }
}

// =============================================================================
// Small helpers
// =============================================================================

/// SplitMix64: tiny, seedable, and the same on every platform.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// `count` distinct values of `items`, by a partial Fisher-Yates.
    fn choose<T: Copy>(&mut self, items: &[T], count: usize) -> Vec<T> {
        let mut pool = items.to_vec();
        let count = count.min(pool.len());
        for i in 0..count {
            let j = i + self.below(pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(count);
        pool
    }
}

/// Round half to even, as Python's `round` does, so the automatic `nlist` and
/// `nprobe` match the reference port on the half-way cases.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if floor % 2.0 == 0.0 {
        floor
    } else {
        floor + 1.0
    }
}

fn auto_nlist(n: usize) -> usize {
    (round_half_even(4.0 * (n as f64).sqrt()).max(16.0) as usize).min(65_536)
}

/// How a step may use the pool: the store's configured thread count (0 = auto)
/// and the pool to run on.
#[derive(Clone, Copy)]
pub(crate) struct Par<'a> {
    pub(crate) configured: usize,
    pub(crate) pool: &'a Option<Arc<SearchPool>>,
}

impl Par<'_> {
    /// Threads available: the pool's size.
    pub(crate) fn size(&self) -> usize {
        self.pool
            .as_ref()
            .map_or_else(SearchPool::default_threads, |p| p.threads())
    }

    /// Participants for `tasks` units of work: the configured count, or with 0
    /// the pool size up to `auto_cap`. A single-query scan is limited by memory
    /// bandwidth, not compute; on the machine this was tuned on eight threads
    /// were within a few percent of thirty-two with a much steadier tail.
    pub(crate) fn workers(&self, tasks: usize, auto_cap: usize) -> usize {
        let want = if self.configured == 0 {
            self.size().min(auto_cap)
        } else {
            self.configured
        };
        want.min(tasks).max(1)
    }

    pub(crate) fn run(&self, participants: usize, work: &(dyn Fn(usize) + Sync)) {
        if participants <= 1 {
            work(0);
            return;
        }
        match self.pool {
            Some(pool) => pool.run(participants, work),
            None => SearchPool::global().run(participants, work),
        }
    }
}

// =============================================================================
// Kernels
//
// One family, `block::<Q, R, F>`: Q queries against R rows at once, returning
// the Q x R dot products. Each (query, row) pair is summed in the same order
// whatever Q and R are (see the module docs), so callers pick the shape that
// suits the loop they are in. `F` selects fused multiply-add; the `true`
// instantiation is only ever reached through a function that enabled the
// feature, so `mul_add` compiles to one instruction, never a library call.
// =============================================================================

#[inline(always)]
fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        // (l0+l4, l1+l5, l2+l6, l3+l7), then (t0+t2) + (t1+t3): the tree the
        // portable path spells out.
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// `Q` queries against `R` rows.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; every slice must be exactly `dim`
    /// long (the callers slice rows to `dim`).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn block<const Q: usize, const R: usize>(
        q: [&[f32]; Q],
        r: [&[f32]; R],
        dim: usize,
    ) -> [[f32; R]; Q] {
        debug_assert!(q.iter().all(|s| s.len() == dim) && r.iter().all(|s| s.len() == dim));
        let qp: [*const f32; Q] = std::array::from_fn(|i| q[i].as_ptr());
        let rp: [*const f32; R] = std::array::from_fn(|i| r[i].as_ptr());
        let mut acc = [[_mm256_setzero_ps(); R]; Q];
        let body = dim & !7;
        let mut i = 0;
        while i < body {
            let mut rv = [_mm256_setzero_ps(); R];
            for j in 0..R {
                rv[j] = _mm256_loadu_ps(rp[j].add(i));
            }
            for a in 0..Q {
                let qv = _mm256_loadu_ps(qp[a].add(i));
                for j in 0..R {
                    acc[a][j] = _mm256_fmadd_ps(qv, rv[j], acc[a][j]);
                }
            }
            i += 8;
        }
        let mut out = [[0.0f32; R]; Q];
        for a in 0..Q {
            for j in 0..R {
                let mut s = hsum(acc[a][j]);
                let mut t = body;
                while t < dim {
                    s += *qp[a].add(t) * *rp[j].add(t);
                    t += 1;
                }
                out[a][j] = s;
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Define `$name` as a run-time dispatch between the AVX2+FMA and the portable
/// instantiation of `$imp`.
macro_rules! dispatch {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:ident) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn fast($($arg: $ty),*) $(-> $ret)? {
                    $imp::<true>($($arg),*)
                }
                if has_avx2_fma() {
                    // SAFETY: the CPU reports AVX2 and FMA, which is all `fast`
                    // requires; it touches nothing but its arguments.
                    return unsafe { fast($($arg),*) };
                }
            }
            $imp::<false>($($arg),*)
        }
    };
}

/// `Q` queries against `R` rows, each exactly `dim` floats.
#[inline(always)]
fn block<const Q: usize, const R: usize, const F: bool>(
    q: [&[f32]; Q],
    r: [&[f32]; R],
) -> [[f32; R]; Q] {
    let dim = q[0].len();
    #[cfg(target_arch = "x86_64")]
    if F {
        // the kernel reads `dim` floats from every slice: make that a fact, not a hope
        assert!(q.iter().all(|s| s.len() == dim) && r.iter().all(|s| s.len() == dim));
        // SAFETY: `F` is true only in the instantiation reached through a
        // `dispatch!` wrapper, which has checked for AVX2 and FMA.
        return unsafe { x86::block::<Q, R>(q, r, dim) };
    }
    const LANES: usize = 8;
    let body = dim - dim % LANES;
    let mut acc = [[[0.0f32; LANES]; R]; Q];
    let mut i = 0;
    while i < body {
        for a in 0..Q {
            for j in 0..R {
                for l in 0..LANES {
                    acc[a][j][l] = fma::<F>(q[a][i + l], r[j][i + l], acc[a][j][l]);
                }
            }
        }
        i += LANES;
    }
    let mut out = [[0.0f32; R]; Q];
    for a in 0..Q {
        for j in 0..R {
            let v = &acc[a][j];
            let t = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
            let mut s = (t[0] + t[2]) + (t[1] + t[3]);
            for x in body..dim {
                s += q[a][x] * r[j][x];
            }
            out[a][j] = s;
        }
    }
    out
}

/// Dot product in the crate's one summation order.
#[inline(always)]
fn dot_impl<const F: bool>(a: &[f32], b: &[f32]) -> f32 {
    block::<1, 1, F>([a], [b])[0][0]
}

dispatch! {
    pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 = dot_impl
}

/// `out[i] = dot(query, row i of mat)`.
#[inline(always)]
fn dots_impl<const F: bool>(query: &[f32], mat: &[f32], dim: usize, out: &mut [f32]) {
    let n = out.len();
    let row = |i: usize| &mat[i * dim..(i + 1) * dim];
    let mut i = 0;
    while i + 4 <= n {
        let d = block::<1, 4, F>([query], [row(i), row(i + 1), row(i + 2), row(i + 3)]);
        out[i..i + 4].copy_from_slice(&d[0]);
        i += 4;
    }
    while i < n {
        out[i] = block::<1, 1, F>([query], [row(i)])[0][0];
        i += 1;
    }
}

dispatch! {
    fn dots_into(query: &[f32], mat: &[f32], dim: usize, out: &mut [f32]) = dots_impl
}

/// For each point in `pts` (row-major, `out.len()` rows of `dim`), the index of
/// the centroid with the largest dot product (the first on a tie). Points are
/// handled in tiles that stay in cache while the centroids stream past. Every
/// point's dot products use the canonical order, so its answer does not depend
/// on which tile or block it is in.
#[inline(always)]
fn nearest_impl<const F: bool>(
    pts: &[f32],
    dim: usize,
    cents: &[f32],
    nlist: usize,
    out: &mut [i32],
) {
    let m = out.len();
    let point = |p: usize| &pts[p * dim..(p + 1) * dim];
    let cent = |c: usize| &cents[c * dim..(c + 1) * dim];
    let mut best_s = [f32::NEG_INFINITY; TILE];
    let mut best_i = [0i32; TILE];
    let mut p0 = 0;
    while p0 < m {
        let t = (m - p0).min(TILE);
        best_s[..t].fill(f32::NEG_INFINITY);
        best_i[..t].fill(0);
        let mut c = 0;
        while c + 4 <= nlist {
            let group = [cent(c), cent(c + 1), cent(c + 2), cent(c + 3)];
            let mut p = 0;
            while p + 2 <= t {
                let d = block::<2, 4, F>([point(p0 + p), point(p0 + p + 1)], group);
                for (pi, row) in d.iter().enumerate() {
                    for (ci, &s) in row.iter().enumerate() {
                        if s > best_s[p + pi] {
                            best_s[p + pi] = s;
                            best_i[p + pi] = (c + ci) as i32;
                        }
                    }
                }
                p += 2;
            }
            if p < t {
                let d = block::<1, 4, F>([point(p0 + p)], group);
                for (ci, &s) in d[0].iter().enumerate() {
                    if s > best_s[p] {
                        best_s[p] = s;
                        best_i[p] = (c + ci) as i32;
                    }
                }
            }
            c += 4;
        }
        while c < nlist {
            let mut p = 0;
            while p + 2 <= t {
                let d = block::<2, 1, F>([point(p0 + p), point(p0 + p + 1)], [cent(c)]);
                for pi in 0..2 {
                    if d[pi][0] > best_s[p + pi] {
                        best_s[p + pi] = d[pi][0];
                        best_i[p + pi] = c as i32;
                    }
                }
                p += 2;
            }
            if p < t {
                let s = block::<1, 1, F>([point(p0 + p)], [cent(c)])[0][0];
                if s > best_s[p] {
                    best_s[p] = s;
                    best_i[p] = c as i32;
                }
            }
            c += 1;
        }
        out[p0..p0 + t].copy_from_slice(&best_i[..t]);
        p0 += t;
    }
}

dispatch! {
    fn nearest_block(pts: &[f32], dim: usize, cents: &[f32], nlist: usize, out: &mut [i32]) = nearest_impl
}

/// `nearest_block` over `m = out.len()` points spread over threads in fixed
/// blocks. `points(lo, len)` returns the contiguous floats of points
/// `lo..lo + len`; a block never straddles a table chunk because the block
/// sizes divide the chunk size.
fn nearest_parallel<'a>(
    points: &(dyn Fn(usize, usize) -> &'a [f32] + Sync),
    dim: usize,
    cents: &[f32],
    nlist: usize,
    out: &mut [i32],
    par: Par,
) {
    let m = out.len();
    // big jobs in NEAR_BLOCK units; a small one (a fold) is cut finer, so that
    // every thread of the pool gets a piece
    let block = if m >= 8 * NEAR_BLOCK {
        NEAR_BLOCK
    } else {
        (m / par.size().max(1)).next_power_of_two().clamp(16, TILE)
    };
    let blocks = m.div_ceil(block);
    let work = m.saturating_mul(nlist).saturating_mul(dim);
    let threads = if work < (1 << 24) {
        1
    } else {
        par.workers(blocks, usize::MAX)
    };
    if threads <= 1 {
        for (b, chunk) in out.chunks_mut(block).enumerate() {
            nearest_block(points(b * block, chunk.len()), dim, cents, nlist, chunk);
        }
        return;
    }
    let queue = Mutex::new(out.chunks_mut(block).enumerate());
    par.run(threads, &|_| loop {
        let next = queue.lock().ok().and_then(|mut q| q.next());
        let Some((b, chunk)) = next else { break };
        nearest_block(points(b * block, chunk.len()), dim, cents, nlist, chunk);
    });
}

// =============================================================================
// Top-k selection
// =============================================================================

/// A scored row. Ordered by score, then by arrival (lower slot first).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hit {
    pub score: f32,
    pub slot: u32,
}

/// `a` is a worse answer than `b`.
#[inline(always)]
fn worse(a: Hit, b: Hit) -> bool {
    a.score < b.score || (a.score == b.score && a.slot > b.slot)
}

/// The best `k` rows above a threshold: a min-heap whose root is the worst row
/// kept, so most rows are rejected with one comparison against `bar`. Asked
/// for at least as many rows as can exist, it just collects them.
pub(crate) struct TopK {
    k: usize,
    collect_all: bool,
    /// Rows must score at least this much: the threshold until the heap is
    /// full, then the worst score kept.
    bar: f32,
    heap: Vec<Hit>,
}

impl TopK {
    /// `bound` is the most rows that can ever be offered.
    pub fn new(k: usize, threshold: f32, bound: usize) -> Self {
        let collect_all = k >= bound;
        Self {
            k,
            collect_all,
            bar: threshold,
            heap: Vec::with_capacity(if collect_all { 0 } else { k.min(1 << 16) }),
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline(always)]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(a >= b)` is deliberately true for NaN
    pub fn offer(&mut self, score: f32, slot: u32) {
        // False for NaN as well, so tombstones (norm NaN) and corrupt rows
        // drop out here.
        if !(score >= self.bar) {
            return;
        }
        self.accept(Hit { score, slot });
    }

    #[inline(never)]
    fn accept(&mut self, hit: Hit) {
        if self.collect_all {
            self.heap.push(hit);
        } else if self.heap.len() < self.k {
            self.heap.push(hit);
            self.sift_up(self.heap.len() - 1);
            if self.heap.len() == self.k {
                self.bar = self.heap[0].score;
            }
        } else if worse(self.heap[0], hit) {
            self.heap[0] = hit;
            self.sift_down(0);
            self.bar = self.heap[0].score;
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if worse(self.heap[i], self.heap[parent]) {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let len = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < len && worse(self.heap[l], self.heap[m]) {
                m = l;
            }
            if r < len && worse(self.heap[r], self.heap[m]) {
                m = r;
            }
            if m == i {
                break;
            }
            self.heap.swap(i, m);
            i = m;
        }
    }

    pub fn merge(&mut self, other: TopK) {
        for hit in other.heap {
            self.offer(hit.score, hit.slot);
        }
    }

    /// Best first; equal scores in arrival order.
    pub fn into_sorted(mut self) -> Vec<Hit> {
        self.heap.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.slot.cmp(&b.slot))
        });
        self.heap
    }
}

// =============================================================================
// Scoring
// =============================================================================

pub(crate) struct QueryVec<'a> {
    pub(crate) v: &'a [f32],
    pub(crate) norm: f32,
}

impl<'a> QueryVec<'a> {
    pub(crate) fn new(v: &'a [f32]) -> Self {
        Self {
            v,
            norm: dot_f32(v, v).sqrt(),
        }
    }

    /// A direction to rank cells by.
    pub(crate) fn has_direction(&self) -> bool {
        self.norm > 0.0 && self.norm.is_finite()
    }
}

/// Cosine from a dot product and the two norms. A zero-length row scores 0, a
/// tombstone (norm NaN) scores NaN and is dropped by the collector.
#[inline(always)]
pub(crate) fn cosine(dot: f32, row_norm: f32, query_norm: f32) -> f32 {
    let denom = row_norm * query_norm;
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Rows scored per kernel call by a single query.
const QW: usize = 4;

/// Collects rows `QW` at a time so one query is scored against `QW` rows per
/// kernel call (`QW` independent dependency chains and memory streams).
struct Quad<'a> {
    slots: [usize; QW],
    rows: [&'a [f32]; QW],
    m: usize,
}

impl<'a> Quad<'a> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            slots: [0; QW],
            rows: [&[]; QW],
            m: 0,
        }
    }

    /// Queue `slot`, looking its row up in the (chunked) matrix.
    #[inline(always)]
    fn push<const F: bool>(
        &mut self,
        rows: &Rows<'a>,
        q: &QueryVec,
        slot: usize,
        sink: &mut impl FnMut(usize, f32),
    ) {
        self.push_row::<F>(rows.norms, q, slot, rows.row(slot), sink);
    }

    /// Queue `slot` whose row the caller already has in hand.
    #[inline(always)]
    fn push_row<const F: bool>(
        &mut self,
        norms: &[f32],
        q: &QueryVec,
        slot: usize,
        row: &'a [f32],
        sink: &mut impl FnMut(usize, f32),
    ) {
        self.slots[self.m] = slot;
        self.rows[self.m] = row;
        self.m += 1;
        if self.m == QW {
            self.m = 0;
            let d = block::<1, QW, F>([q.v], self.rows);
            for i in 0..QW {
                sink(self.slots[i], cosine(d[0][i], norms[self.slots[i]], q.norm));
            }
        }
    }

    #[inline(always)]
    fn finish<const F: bool>(
        &mut self,
        rows: &Rows,
        q: &QueryVec,
        sink: &mut impl FnMut(usize, f32),
    ) {
        for i in 0..self.m {
            let slot = self.slots[i];
            let d = block::<1, 1, F>([q.v], [self.rows[i]]);
            sink(slot, cosine(d[0][0], rows.norms[slot], q.norm));
        }
        self.m = 0;
    }
}

/// Score rows `lo..hi`.
#[inline(always)]
fn scan_range_impl<const F: bool>(
    rows: Rows,
    q: &QueryVec,
    code: Option<u32>,
    lo: usize,
    hi: usize,
    top: &mut TopK,
) {
    let mut quad = Quad::new();
    let mut sink = |slot: usize, score: f32| top.offer(score, slot as u32);
    // walk the matrix one contiguous run (the rest of a chunk) at a time, so a
    // row is a slice of the run rather than two dependent lookups
    let dim = rows.dim;
    let mut at = lo;
    while at < hi {
        let (flat, count) = rows.run(at, hi);
        for i in 0..count {
            let slot = at + i;
            if let Some(code) = code {
                if rows.codes[slot] != code {
                    continue;
                }
            }
            quad.push_row::<F>(
                rows.norms,
                q,
                slot,
                &flat[i * dim..(i + 1) * dim],
                &mut sink,
            );
        }
        at += count;
    }
    quad.finish::<F>(&rows, q, &mut sink);
}

dispatch! {
    fn scan_range(rows: Rows, q: &QueryVec, code: Option<u32>, lo: usize, hi: usize, top: &mut TopK) = scan_range_impl
}

/// `out[i]` = cosine of the query with row `slots[i]`, computed exactly as the
/// scan computes it (one query against four rows at a time, then singly), so a
/// score from here is bit-identical to the scan's. A tombstone scores NaN.
#[inline(always)]
fn score_slots_impl<const F: bool>(rows: Rows, q: &QueryVec, slots: &[u32], out: &mut [f32]) {
    let row = |i: usize| rows.row(slots[i] as usize);
    let norm = |i: usize| rows.norms[slots[i] as usize];
    let mut i = 0;
    while i + QW <= slots.len() {
        let d = block::<1, QW, F>([q.v], [row(i), row(i + 1), row(i + 2), row(i + 3)]);
        for j in 0..QW {
            out[i + j] = cosine(d[0][j], norm(i + j), q.norm);
        }
        i += QW;
    }
    while i < slots.len() {
        let d = block::<1, 1, F>([q.v], [row(i)]);
        out[i] = cosine(d[0][0], norm(i), q.norm);
        i += 1;
    }
}

dispatch! {
    pub(crate) fn score_slots(rows: Rows, q: &QueryVec, slots: &[u32], out: &mut [f32]) = score_slots_impl
}

/// Score every live row (for the score map) and hand `(slot, score)` on.
#[inline(always)]
fn score_all_impl<const F: bool>(rows: Rows, q: &QueryVec, n: usize, out: &mut Vec<(u32, f32)>) {
    let mut quad = Quad::new();
    let mut sink = |slot: usize, score: f32| out.push((slot as u32, score));
    for slot in 0..n {
        if rows.norms[slot].is_nan() {
            continue;
        }
        quad.push::<F>(&rows, q, slot, &mut sink);
    }
    quad.finish::<F>(&rows, q, &mut sink);
}

dispatch! {
    fn score_all(rows: Rows, q: &QueryVec, n: usize, out: &mut Vec<(u32, f32)>) = score_all_impl
}

/// The parts of an [`Ivf`] a cell scan reads.
#[derive(Clone, Copy)]
struct Cells<'a> {
    lists: &'a [Vec<u32>],
    pending: &'a [u32],
}

/// Score the rows of the `chosen` cells, and every pending row if asked.
#[inline(always)]
fn scan_cells_impl<const F: bool>(
    rows: Rows,
    cells: Cells,
    chosen: &[u32],
    with_pending: bool,
    q: &QueryVec,
    code: Option<u32>,
    top: &mut TopK,
) {
    let mut quad = Quad::new();
    let mut sink = |slot: usize, score: f32| top.offer(score, slot as u32);
    for &c in chosen {
        for &slot in &cells.lists[c as usize] {
            let slot = slot as usize;
            if let Some(code) = code {
                if rows.codes[slot] != code {
                    continue;
                }
            }
            quad.push::<F>(&rows, q, slot, &mut sink);
        }
    }
    if with_pending {
        for &slot in cells.pending {
            let slot = slot as usize;
            if let Some(code) = code {
                if rows.codes[slot] != code {
                    continue;
                }
            }
            quad.push::<F>(&rows, q, slot, &mut sink);
        }
    }
    quad.finish::<F>(&rows, q, &mut sink);
}

dispatch! {
    fn scan_cells(rows: Rows, cells: Cells, chosen: &[u32], with_pending: bool, q: &QueryVec, code: Option<u32>, top: &mut TopK) = scan_cells_impl
}

/// A tile of queries against one block of rows (`slots`, already filtered by
/// type): four queries against two rows per kernel call, so a row loaded from
/// L2 is used by four queries and a query loaded from L1 by two rows.
#[inline(always)]
fn batch_block_impl<const F: bool>(rows: Rows, qs: &[&QueryVec], slots: &[u32], tops: &mut [TopK]) {
    let mut base = 0;
    for group in qs.chunks(4) {
        if group.len() == 4 {
            let qv = [group[0].v, group[1].v, group[2].v, group[3].v];
            let mut i = 0;
            while i + 2 <= slots.len() {
                let (s0, s1) = (slots[i] as usize, slots[i + 1] as usize);
                let d = block::<4, 2, F>(qv, [rows.row(s0), rows.row(s1)]);
                for a in 0..4 {
                    let top = &mut tops[base + a];
                    top.offer(cosine(d[a][0], rows.norms[s0], group[a].norm), s0 as u32);
                    top.offer(cosine(d[a][1], rows.norms[s1], group[a].norm), s1 as u32);
                }
                i += 2;
            }
            if i < slots.len() {
                let s0 = slots[i] as usize;
                let d = block::<4, 1, F>(qv, [rows.row(s0)]);
                for a in 0..4 {
                    tops[base + a].offer(cosine(d[a][0], rows.norms[s0], group[a].norm), s0 as u32);
                }
            }
        } else {
            for (a, q) in group.iter().enumerate() {
                let top = &mut tops[base + a];
                let mut quad = Quad::new();
                let mut sink = |slot: usize, score: f32| top.offer(score, slot as u32);
                for &slot in slots {
                    quad.push::<F>(&rows, q, slot as usize, &mut sink);
                }
                quad.finish::<F>(&rows, q, &mut sink);
            }
        }
        base += group.len();
    }
}

dispatch! {
    fn batch_block(rows: Rows, qs: &[&QueryVec], slots: &[u32], tops: &mut [TopK]) = batch_block_impl
}

// =============================================================================
// The IVF index
// =============================================================================

pub(crate) struct Ivf {
    /// `nlist` unit vectors, row-major.
    centroids: Vec<f32>,
    nlist: usize,
    dim: usize,
    /// Per row slot: its cell, `PENDING`, or `NO_CELL` (deleted).
    assign: Vec<i32>,
    /// Per row slot: its position in its cell's list (or in `pending`), so a
    /// row leaves a list in O(1).
    pos: Vec<u32>,
    /// The slots of each cell. No stale entries: a list holds exactly the live,
    /// folded rows assigned to it.
    lists: Vec<Vec<u32>>,
    /// Live rows with no cell yet; every query scores them exactly.
    pending: Vec<u32>,
    churn: usize,
    trained_n: usize,
    build_seconds: f64,
    folds: u64,
    fold_seconds: f64,
}

impl Ivf {
    /// Train centroids on a sample, then give every live row its cell.
    fn build(table: &VectorTable, nlist: usize, seed: u64, par: Par) -> Ivf {
        let started = Instant::now();
        let (n, dim) = (table.slots(), table.dim);
        let live: Vec<u32> = table.live_slots().map(|s| s as u32).collect();
        let total = live.len();
        let nlist = nlist.clamp(1, total.max(1));
        let mut rng = Rng(seed);

        let sample_n = total.min(SAMPLE_PER_CELL * nlist);
        let sample: Vec<u32> = if sample_n == total {
            live
        } else {
            let mut picked = rng.choose(&live, sample_n);
            picked.sort_unstable();
            picked
        };

        // Unit-length copies of the sample (a zero row stays as it is).
        let mut x = vec![0.0f32; sample_n * dim];
        for (i, &slot) in sample.iter().enumerate() {
            let norm = table.norm(slot as usize);
            let scale = if norm > 0.0 && norm.is_finite() {
                1.0 / norm
            } else {
                1.0
            };
            for (dst, src) in x[i * dim..(i + 1) * dim]
                .iter_mut()
                .zip(table.row(slot as usize))
            {
                *dst = src * scale;
            }
        }

        let sample_ids: Vec<usize> = (0..sample_n).collect();
        let mut centroids = vec![0.0f32; nlist * dim];
        for (c, &i) in rng.choose(&sample_ids, nlist).iter().enumerate() {
            centroids[c * dim..(c + 1) * dim].copy_from_slice(&x[i * dim..(i + 1) * dim]);
        }

        let mut cell = vec![0i32; sample_n];
        let mut sums = vec![0.0f64; nlist * dim];
        let mut counts = vec![0u32; nlist];
        for _ in 0..KMEANS_ITERATIONS {
            nearest_parallel(
                &|lo, len| &x[lo * dim..(lo + len) * dim],
                dim,
                &centroids,
                nlist,
                &mut cell,
                par,
            );
            sums.fill(0.0);
            counts.fill(0);
            for (i, &c) in cell.iter().enumerate() {
                let c = c as usize;
                counts[c] += 1;
                for (s, v) in sums[c * dim..(c + 1) * dim]
                    .iter_mut()
                    .zip(&x[i * dim..(i + 1) * dim])
                {
                    *s += *v as f64;
                }
            }
            for c in 0..nlist {
                let sum = &sums[c * dim..(c + 1) * dim];
                let length = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
                let target = &mut centroids[c * dim..(c + 1) * dim];
                if counts[c] == 0 || length == 0.0 || !length.is_finite() {
                    // A dead centroid is re-seeded on a random sample point.
                    let i = rng.below(sample_n);
                    target.copy_from_slice(&x[i * dim..(i + 1) * dim]);
                } else {
                    for (t, s) in target.iter_mut().zip(sum) {
                        *t = (*s / length) as f32;
                    }
                }
            }
        }
        drop(x);

        // Every row (dead ones too, to keep the kernel's blocks fixed) gets its
        // nearest cell; the dead are then taken out again.
        let mut assign = vec![0i32; n];
        nearest_parallel(
            &|lo, len| table.run(lo, lo + len).0,
            dim,
            &centroids,
            nlist,
            &mut assign,
            par,
        );
        for (slot, a) in assign.iter_mut().enumerate() {
            if !table.is_live(slot) {
                *a = NO_CELL;
            }
        }
        let mut index = Ivf {
            centroids,
            nlist,
            dim,
            assign,
            pos: vec![0; n],
            lists: Vec::new(),
            pending: Vec::new(),
            churn: 0,
            trained_n: total,
            build_seconds: 0.0,
            folds: 0,
            fold_seconds: 0.0,
        };
        index.relist();
        index.build_seconds = started.elapsed().as_secs_f64();
        index
    }

    /// Rebuild every cell list from `assign` (slot order within a cell).
    fn relist(&mut self) {
        let mut counts = vec![0usize; self.nlist];
        for &c in &self.assign {
            if c >= 0 {
                counts[c as usize] += 1;
            }
        }
        self.lists = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        for (slot, &c) in self.assign.iter().enumerate() {
            if c >= 0 {
                let list = &mut self.lists[c as usize];
                self.pos[slot] = list.len() as u32;
                list.push(slot as u32);
            }
        }
    }

    /// Past the churn that calls for retraining.
    fn stale(&self) -> bool {
        self.churn as f64 > RETRAIN_FRACTION * self.trained_n as f64
    }

    /// Take `slot` out of whatever list holds it.
    fn detach(&mut self, slot: u32) {
        let s = slot as usize;
        let Some(&cell) = self.assign.get(s) else {
            return;
        };
        let at = self.pos[s] as usize;
        let list = match cell {
            NO_CELL => return,
            PENDING => &mut self.pending,
            c => &mut self.lists[c as usize],
        };
        let last = list.len() - 1;
        list.swap_remove(at);
        if at != last {
            self.pos[list[at] as usize] = at as u32;
        }
        self.assign[s] = NO_CELL;
    }

    /// A row was written (new, or its vector replaced): it waits in `pending`.
    fn note_put(&mut self, slot: u32) {
        let s = slot as usize;
        if s >= self.assign.len() {
            self.assign.resize(s + 1, NO_CELL);
            self.pos.resize(s + 1, 0);
        }
        self.detach(slot);
        self.assign[s] = PENDING;
        self.pos[s] = self.pending.len() as u32;
        self.pending.push(slot);
        self.churn += 1;
    }

    fn note_drop(&mut self, slot: u32) {
        self.detach(slot);
        self.churn += 1;
    }

    /// Give every pending row its nearest cell.
    fn fold(&mut self, table: &VectorTable, par: Par) {
        if self.pending.is_empty() {
            return;
        }
        let started = Instant::now();
        let dim = self.dim;
        let slots = std::mem::take(&mut self.pending);
        let mut gathered = Vec::with_capacity(slots.len() * dim);
        for &s in &slots {
            gathered.extend_from_slice(table.row(s as usize));
        }
        let mut cells = vec![0i32; slots.len()];
        nearest_parallel(
            &|lo, len| &gathered[lo * dim..(lo + len) * dim],
            dim,
            &self.centroids,
            self.nlist,
            &mut cells,
            par,
        );
        for (&s, &c) in slots.iter().zip(&cells) {
            let list = &mut self.lists[c as usize];
            self.assign[s as usize] = c;
            self.pos[s as usize] = list.len() as u32;
            list.push(s);
        }
        self.folds += 1;
        self.fold_seconds += started.elapsed().as_secs_f64();
    }

    /// Follow the table's compaction (`live` = old slots kept, in order).
    fn compact(&mut self, live: &[u32]) {
        let moved: Vec<i32> = live.iter().map(|&old| self.assign[old as usize]).collect();
        let pending: Vec<u32> = self
            .pending
            .iter()
            .filter_map(|old| live.binary_search(old).ok().map(|new| new as u32))
            .collect();
        self.assign = moved;
        self.pos = vec![0; live.len()];
        self.relist();
        for (i, &slot) in pending.iter().enumerate() {
            self.pos[slot as usize] = i as u32;
        }
        self.pending = pending;
    }

    /// Every invariant of the index against the table it indexes.
    fn check(&self, table: &VectorTable, pending_cap: usize) -> Result<(), String> {
        let n = table.slots();
        if self.assign.len() != n || self.pos.len() != n {
            return Err(format!(
                "index covers {} rows (pos {}), table has {n}",
                self.assign.len(),
                self.pos.len()
            ));
        }
        if self.pending.len() >= pending_cap {
            return Err(format!(
                "{} rows pending, the cap is {pending_cap}",
                self.pending.len()
            ));
        }
        let mut seen = 0usize;
        for (c, list) in self.lists.iter().enumerate() {
            for (at, &slot) in list.iter().enumerate() {
                let s = slot as usize;
                if self.assign[s] != c as i32 || self.pos[s] as usize != at {
                    return Err(format!(
                        "slot {slot} listed in cell {c} at {at} but assigned {} at {}",
                        self.assign[s], self.pos[s]
                    ));
                }
                if !table.is_live(s) {
                    return Err(format!("dead slot {slot} still in cell {c}"));
                }
                seen += 1;
            }
        }
        for (at, &slot) in self.pending.iter().enumerate() {
            let s = slot as usize;
            if self.assign[s] != PENDING || self.pos[s] as usize != at || !table.is_live(s) {
                return Err(format!("pending slot {slot} is inconsistent"));
            }
            seen += 1;
        }
        for slot in 0..n {
            let live = table.is_live(slot);
            match (live, self.assign[slot]) {
                (false, NO_CELL) => {}
                (false, a) => return Err(format!("dead slot {slot} assigned {a}")),
                (true, NO_CELL) => return Err(format!("live slot {slot} is in no cell")),
                (true, _) => {}
            }
        }
        if seen != table.live() {
            return Err(format!("{seen} rows in lists, {} live", table.live()));
        }
        Ok(())
    }

    fn nbytes(&self) -> usize {
        self.centroids.len() * 4
            + self.assign.len() * 4
            + self.pos.len() * 4
            + self
                .lists
                .iter()
                .map(|l| l.capacity() * 4 + 24)
                .sum::<usize>()
            + self.pending.capacity() * 4
    }

    /// The `lists` cells whose centroids are most similar to the query
    /// (ascending, for memory locality).
    fn choose_cells(&self, q: &QueryVec, lists: usize) -> Vec<u32> {
        if lists >= self.nlist {
            return (0..self.nlist as u32).collect();
        }
        let mut scores = vec![0.0f32; self.nlist];
        dots_into(q.v, &self.centroids, self.dim, &mut scores);
        let mut order: Vec<u32> = (0..self.nlist as u32).collect();
        order.select_nth_unstable_by(lists - 1, |&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then(a.cmp(&b))
        });
        order.truncate(lists);
        order.sort_unstable();
        order
    }
}

fn effective_nprobe(cfg: &AnnConfig, nlist: usize) -> usize {
    cfg.nprobe
        .unwrap_or_else(|| (round_half_even(0.05 * nlist as f64) as usize).max(1))
        .min(nlist)
}

// =============================================================================
// The store's search surface
// =============================================================================

/// Matches from a search, best first, resolved to results one at a time (so a
/// caller that stops early never builds the rest). Returned by
/// `EmbeddingStore::ranked_hits`.
pub(crate) struct RankedHits<'a> {
    store: &'a EmbeddingStore,
    hits: std::vec::IntoIter<Hit>,
}

impl Iterator for RankedHits<'_> {
    type Item = SimilarityResult;

    fn next(&mut self) -> Option<SimilarityResult> {
        for hit in self.hits.by_ref() {
            if let Some(result) = self.store.resolve_hit(hit) {
                return Some(result);
            }
        }
        None
    }
}

impl EmbeddingStore {
    /// A store whose bounded searches use the approximate index.
    pub fn with_ann(
        encoder: Option<Box<dyn EmbeddingEncoder>>,
        config: AnnConfig,
    ) -> Result<Self, EmbeddingError> {
        let mut store = Self::new(encoder);
        store.set_ann(Some(config))?;
        Ok(store)
    }

    /// Switch the approximate index on (with these settings) or, with `None`,
    /// off. Any index already built is dropped. Nothing is saved: a saved
    /// graph or sidecar is the same with or without it.
    pub fn set_ann(&mut self, config: Option<AnnConfig>) -> Result<(), EmbeddingError> {
        if let Some(config) = &config {
            config.validate()?;
        }
        self.ann = config;
        *self.ivf_mut() = None;
        *self.hnsw_mut() = None;
        Ok(())
    }

    /// The approximate-index settings, or `None` while search is exact.
    pub fn ann(&self) -> Option<&AnnConfig> {
        self.ann.as_ref()
    }

    /// Change how many cells a query scans (`None` = about 5% of `nlist`).
    /// Takes effect on the next search; the index is not rebuilt.
    pub fn set_ann_nprobe(&mut self, nprobe: Option<usize>) -> Result<(), EmbeddingError> {
        if nprobe == Some(0) {
            return Err(EmbeddingError::InvalidOption(
                "ann nprobe must be at least 1".to_string(),
            ));
        }
        match self.ann.as_mut() {
            Some(config) => {
                config.nprobe = nprobe;
                Ok(())
            }
            None => Err(EmbeddingError::InvalidOption(
                "this store has no approximate index; call set_ann first".to_string(),
            )),
        }
    }

    /// The most threads one search, batch, fold or index build may use (0, the
    /// default, picks automatically: a single scan up to 8, a batch or a build
    /// the whole pool; 1 never touches the pool). The answer does not depend
    /// on it.
    pub fn set_scan_threads(&mut self, threads: usize) {
        self.scan_threads = threads;
    }

    /// The configured thread cap (0 = automatic).
    pub fn scan_threads(&self) -> usize {
        self.scan_threads
    }

    /// Use this pool instead of the process-wide one (`SearchPool::global`),
    /// e.g. to give a store its own fixed number of workers. `None` goes back
    /// to the shared pool.
    pub fn set_search_pool(&mut self, pool: Option<Arc<SearchPool>>) {
        self.pool = pool;
    }

    /// The pool this store runs parallel work on (the shared one by default).
    pub fn search_pool(&self) -> Arc<SearchPool> {
        self.pool.clone().unwrap_or_else(SearchPool::global)
    }

    pub(crate) fn par(&self) -> Par<'_> {
        Par {
            configured: self.scan_threads,
            pool: &self.pool,
        }
    }

    pub(crate) fn ivf_mut(&mut self) -> &mut Option<Ivf> {
        self.ivf.get_mut().unwrap_or_else(|e| e.into_inner())
    }

    fn ivf_read(&self) -> RwLockReadGuard<'_, Option<Ivf>> {
        self.ivf.read().unwrap_or_else(|e| e.into_inner())
    }

    // -- write-side upkeep of the index (the caller holds `&mut self`) --------

    /// A row was written: it joins `pending`, and a full `pending` is folded
    /// now, by the write that filled it.
    pub(crate) fn index_row_written(&mut self, slot: u32) {
        self.hnsw_row_written(slot);
        let (table, pool, configured) = (&self.table, &self.pool, self.scan_threads);
        let guard = self.ivf.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(ivf) = guard.as_mut() {
            ivf.note_put(slot);
            if ivf.pending.len() >= FOLD_BATCH {
                ivf.fold(table, Par { configured, pool });
            }
        }
    }

    /// A row was deleted (and the table compacted if it was a quarter dead).
    pub(crate) fn index_row_removed(&mut self, slot: u32) {
        if let Some(ivf) = self.ivf_mut().as_mut() {
            ivf.note_drop(slot);
        }
        self.hnsw_row_removed();
        if self.table.should_compact() {
            self.compact_now();
        }
        if let Some(cfg) = &self.ann {
            if self.table.live() < cfg.min_size.max(2) {
                *self.ivf_mut() = None;
                *self.hnsw_mut() = None;
            }
        }
    }

    /// Squeeze the tombstones out of the matrix now (it happens by itself once
    /// a quarter of the rows are dead). Order-preserving, so results do not
    /// change; a pause proportional to the store's size.
    pub fn compact(&mut self) {
        if self.table.dead() > 0 {
            self.compact_now();
        }
    }

    fn compact_now(&mut self) {
        let live = self.table.compact();
        if let Some(ivf) = self.ivf_mut().as_mut() {
            ivf.compact(&live);
        }
        self.hnsw_compact(&live);
    }

    // -- the index, as the searches see it ------------------------------------

    /// A read lock on a usable index, building or retraining first if the
    /// policy says so; `None` means "answer exactly" (the store is under
    /// `min_size`). Concurrent searches share the read lock; only a build
    /// takes the write lock.
    fn ready_index(&self, cfg: &AnnConfig) -> Option<RwLockReadGuard<'_, Option<Ivf>>> {
        if self.table.live() < cfg.min_size.max(2) {
            return None;
        }
        loop {
            {
                let guard = self.ivf_read();
                if guard.as_ref().is_some_and(|i| !i.stale()) {
                    return Some(guard);
                }
            }
            let mut guard = self.ivf.write().unwrap_or_else(|e| e.into_inner());
            if guard.as_ref().is_none_or(|i| i.stale()) {
                let nlist = cfg.nlist.unwrap_or_else(|| auto_nlist(self.table.live()));
                *guard = Some(Ivf::build(&self.table, nlist, cfg.seed, self.par()));
                self.counts.builds.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Top `k` from the index, or `None` to fall back to the scan.
    ///
    /// `scale` 1 honours `nprobe` as configured (`nprobe >= nlist` is the
    /// scan). Later widening steps multiply it by 4 and give up on the index
    /// once they would touch half the cells, where scanning everything costs
    /// about the same. `threads` is the cap for scanning one query's cells.
    #[allow(clippy::too_many_arguments)]
    fn approximate_topk(
        &self,
        cfg: &AnnConfig,
        ivf: &Ivf,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
        par: Par,
    ) -> Option<Vec<Hit>> {
        if !q.has_direction() {
            return None;
        }
        let mut scale = 1usize;
        loop {
            let top = self.probe(cfg, ivf, q, code, threshold, k, scale, par)?;
            if top.len() >= k {
                self.counts.approximate.fetch_add(1, Ordering::Relaxed);
                if scale > 1 {
                    self.counts.widened.fetch_add(1, Ordering::Relaxed);
                }
                return Some(top.into_sorted());
            }
            scale = scale.saturating_mul(4);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn probe(
        &self,
        cfg: &AnnConfig,
        ivf: &Ivf,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
        scale: usize,
        par: Par,
    ) -> Option<TopK> {
        let nprobe = effective_nprobe(cfg, ivf.nlist);
        let lists = nprobe.saturating_mul(scale);
        let exhausted = if scale == 1 {
            lists >= ivf.nlist
        } else {
            lists.saturating_mul(2) >= ivf.nlist
        };
        if exhausted {
            return None;
        }
        let chosen = ivf.choose_cells(q, lists);
        let cells = Cells {
            lists: &ivf.lists,
            pending: &ivf.pending,
        };
        Some(scan_probed(
            &self.table,
            cells,
            &chosen,
            q,
            code,
            threshold,
            k,
            par,
        ))
    }

    /// Every row the probed cells yield (for a caller that filters further),
    /// best first; `None` to fall back to the scan.
    fn approximate_ranked(
        &self,
        cfg: &AnnConfig,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        scale: usize,
    ) -> Option<Vec<Hit>> {
        if !q.has_direction() {
            return None;
        }
        if cfg.mode == AnnMode::Hnsw {
            return self.hnsw_ranked(cfg, q, code, threshold, scale);
        }
        let guard = self.ready_index(cfg)?;
        let ivf = guard.as_ref()?;
        let top = self.probe(cfg, ivf, q, code, threshold, usize::MAX, scale, self.par())?;
        self.counts.approximate.fetch_add(1, Ordering::Relaxed);
        if scale > 1 {
            self.counts.widened.fetch_add(1, Ordering::Relaxed);
        }
        Some(top.into_sorted())
    }

    // -- exact scans ----------------------------------------------------------

    /// The full scan, in int8 + exact rerank when that is on and applies.
    fn exact_hits(&self, q: &QueryVec, code: Option<u32>, threshold: f32, k: usize) -> Vec<Hit> {
        match self.quant_exact(q, code, threshold, k) {
            Some(hits) => hits,
            None => self.exact_hits_f32(q, code, threshold, k),
        }
    }

    /// The f32 full scan: every live row, spread over threads for a big store.
    fn exact_hits_f32(
        &self,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
    ) -> Vec<Hit> {
        let table = &self.table;
        let (n, dim) = (table.slots(), table.dim);
        let rows = table.rows();
        let live = table.live();
        let blocks = n.div_ceil(BLOCK_ROWS);
        let par = self.par();
        let threads = if par.configured == 0 {
            par.workers(blocks, SCAN_AUTO_THREADS)
                .min((n * dim / PAR_MIN_FLOATS).max(1))
        } else {
            par.workers(blocks, SCAN_AUTO_THREADS)
        };
        if threads <= 1 {
            let mut top = TopK::new(k, threshold, live);
            scan_range(rows, q, code, 0, n, &mut top);
            return top.into_sorted();
        }
        let next = AtomicUsize::new(0);
        let parts = Mutex::new(Vec::with_capacity(threads));
        par.run(threads, &|_| {
            let mut top = TopK::new(k, threshold, live);
            loop {
                let b = next.fetch_add(1, Ordering::Relaxed);
                if b >= blocks {
                    break;
                }
                scan_range(
                    rows,
                    q,
                    code,
                    b * BLOCK_ROWS,
                    ((b + 1) * BLOCK_ROWS).min(n),
                    &mut top,
                );
            }
            parts.lock().unwrap_or_else(|e| e.into_inner()).push(top);
        });
        let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut merged = parts.pop().expect("at least one participant");
        for part in parts {
            merged.merge(part);
        }
        merged.into_sorted()
    }

    /// Many queries against the whole matrix at once: the work is cut into
    /// (block of rows) x (tile of queries) tasks claimed from one counter, with
    /// all tiles of a block adjacent so threads share the block in cache.
    fn exact_batch_hits(
        &self,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        k: usize,
    ) -> Vec<Vec<Hit>> {
        if let Some(hits) = self.quant_exact_batch(qs, code, threshold, k) {
            return hits;
        }
        let table = &self.table;
        let n = table.slots();
        let rows = table.rows();
        let live = table.live();
        let blocks = n.div_ceil(BLOCK_ROWS);
        let tiles = qs.len().div_ceil(BATCH_TILE);
        let total = blocks * tiles;
        let par = self.par();
        let threads = par.workers(total, usize::MAX);
        let next = AtomicUsize::new(0);
        let parts: Mutex<Vec<Vec<TopK>>> = Mutex::new(Vec::with_capacity(threads));
        par.run(threads, &|_| {
            let mut tops: Vec<TopK> = Vec::new();
            let mut slots: Vec<u32> = Vec::with_capacity(BLOCK_ROWS);
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= total {
                    break;
                }
                if tops.is_empty() {
                    tops = (0..qs.len())
                        .map(|_| TopK::new(k, threshold, live))
                        .collect();
                }
                let (block, tile) = (t / tiles, t % tiles);
                let (lo, hi) = (block * BLOCK_ROWS, ((block + 1) * BLOCK_ROWS).min(n));
                slots.clear();
                match code {
                    None => slots.extend(lo as u32..hi as u32),
                    Some(c) => {
                        slots.extend((lo..hi).filter(|&s| rows.codes[s] == c).map(|s| s as u32))
                    }
                }
                let (q_lo, q_hi) = (tile * BATCH_TILE, ((tile + 1) * BATCH_TILE).min(qs.len()));
                let tile_queries: Vec<&QueryVec> = qs[q_lo..q_hi].iter().collect();
                batch_block(rows, &tile_queries, &slots, &mut tops[q_lo..q_hi]);
            }
            if !tops.is_empty() {
                parts.lock().unwrap_or_else(|e| e.into_inner()).push(tops);
            }
        });
        let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
        let Some(mut merged) = parts.pop() else {
            return qs.iter().map(|_| Vec::new()).collect();
        };
        for part in parts {
            for (m, p) in merged.iter_mut().zip(part) {
                m.merge(p);
            }
        }
        merged.into_iter().map(TopK::into_sorted).collect()
    }

    // -- results --------------------------------------------------------------

    fn resolve_hit(&self, hit: Hit) -> Option<SimilarityResult> {
        let slot = hit.slot as usize;
        let meta = self.table.meta(slot)?;
        Some(SimilarityResult {
            node_id: self.table.node_id(slot)?,
            score: hit.score,
            text: meta.text.to_string(),
        })
    }

    fn materialise(&self, hits: &[Hit]) -> Vec<SimilarityResult> {
        hits.iter()
            .filter_map(|&hit| self.resolve_hit(hit))
            .collect()
    }

    /// The type filter as a code: `Ok(None)` no filter, `Err(())` a type the
    /// store has never held (nothing can match).
    fn type_filter(&self, node_type: Option<&str>) -> Result<Option<u32>, ()> {
        match node_type {
            None => Ok(None),
            Some(t) => self.table.type_code(t).map(Some).ok_or(()),
        }
    }

    // -- public search surface ------------------------------------------------

    /// Best `k` matches for a vector, from the index when one is configured
    /// and `exact` is false, else from the full scan.
    pub(crate) fn search_vector(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
        exact: bool,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.search_vector_with(query_vector, top_k, node_type, threshold, exact, None)
    }

    /// `search_vector` with an optional per-call HNSW `ef_search`.
    pub(crate) fn search_vector_with(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
        exact: bool,
        ef: Option<usize>,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.check_query(query_vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let live = self.table.live();
        let Ok(code) = self.type_filter(node_type) else {
            return Ok(Vec::new());
        };
        if live == 0 {
            return Ok(Vec::new());
        }
        let q = QueryVec::new(query_vector);
        let hits = self.hits_for(&q, code, threshold, top_k, exact, ef);
        Ok(self.materialise(&hits))
    }

    fn hits_for(
        &self,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        top_k: usize,
        exact: bool,
        ef: Option<usize>,
    ) -> Vec<Hit> {
        if let Some(cfg) = &self.ann {
            if exact || top_k >= self.table.live() {
                self.counts.exact.fetch_add(1, Ordering::Relaxed);
            } else {
                let approx = if cfg.mode == AnnMode::Hnsw {
                    self.ready_hnsw(cfg).and_then(|guard| {
                        let h = guard.as_ref()?;
                        self.hnsw_topk(cfg, h, q, code, threshold, top_k, ef)
                    })
                } else {
                    self.ready_index(cfg).and_then(|guard| {
                        let ivf = guard.as_ref()?;
                        self.approximate_topk(cfg, ivf, q, code, threshold, top_k, self.par())
                    })
                };
                match approx {
                    Some(hits) => return hits,
                    None => {
                        self.counts.exact.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        self.exact_hits(q, code, threshold, top_k)
    }

    /// Search with many vector queries in one call, best `top_k` for each,
    /// in query order. Each answer is **identical** (same ids, same bit-exact
    /// scores, same order) to `similarity_search_vector` on that query alone;
    /// the batch is only faster, because the work is spread over the thread
    /// pool and, for the exact scan, queries share every pass over the matrix.
    ///
    /// With an approximate index configured, queries are answered from it
    /// exactly as the single-query path would. The first bad query (wrong
    /// dimension, empty) fails the whole call.
    pub fn similarity_search_vector_batch<V: AsRef<[f32]> + Sync>(
        &self,
        queries: &[V],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        self.search_batch(queries, top_k, node_type, threshold, false)
    }

    /// Like [`similarity_search_vector_batch`](Self::similarity_search_vector_batch)
    /// but always the full scan, even with an approximate index configured.
    pub fn similarity_search_vector_exact_batch<V: AsRef<[f32]> + Sync>(
        &self,
        queries: &[V],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        self.search_batch(queries, top_k, node_type, threshold, true)
    }

    /// Search with many text queries in one call (encoded together with the
    /// encoder's `encode_batch`), then as `similarity_search_vector_batch`.
    pub fn similarity_search_batch(
        &self,
        queries: &[&str],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        let vectors = self.resolve_text_batch(queries)?;
        self.search_batch(&vectors, top_k, node_type, threshold, false)
    }

    /// Text batch, always the full scan.
    pub fn similarity_search_exact_batch(
        &self,
        queries: &[&str],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        let vectors = self.resolve_text_batch(queries)?;
        self.search_batch(&vectors, top_k, node_type, threshold, true)
    }

    fn resolve_text_batch(&self, queries: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let encoder = self.encoder.as_ref().ok_or(EmbeddingError::NoEncoder)?;
        self.check_model_compat()?;
        Ok(encoder.encode_batch(queries))
    }

    fn search_batch<V: AsRef<[f32]> + Sync>(
        &self,
        queries: &[V],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
        exact: bool,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        for q in queries {
            self.check_query(q.as_ref())?;
        }
        let empty = || queries.iter().map(|_| Vec::new()).collect::<Vec<_>>();
        if top_k == 0 || queries.is_empty() {
            return Ok(empty());
        }
        let live = self.table.live();
        let Ok(code) = self.type_filter(node_type) else {
            return Ok(empty());
        };
        if live == 0 {
            return Ok(empty());
        }
        if top_k >= live {
            // "everything" answers hold every row: no shared heaps to bound
            return queries
                .iter()
                .map(|q| self.search_vector(q.as_ref(), top_k, node_type, threshold, exact))
                .collect();
        }
        let mut out: Vec<Vec<SimilarityResult>> = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(BATCH_CHUNK) {
            let qs: Vec<QueryVec> = chunk.iter().map(|q| QueryVec::new(q.as_ref())).collect();
            let hits = self.batch_hits(&qs, code, threshold, top_k, exact);
            out.extend(hits.iter().map(|h| self.materialise(h)));
        }
        Ok(out)
    }

    fn batch_hits(
        &self,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        top_k: usize,
        exact: bool,
    ) -> Vec<Vec<Hit>> {
        let mut answers: Vec<Option<Vec<Hit>>> = (0..qs.len()).map(|_| None).collect();
        if let Some(cfg) = &self.ann {
            if exact {
                self.counts
                    .exact
                    .fetch_add(qs.len() as u64, Ordering::Relaxed);
            } else if cfg.mode == AnnMode::Hnsw {
                if let Some(guard) = self.ready_hnsw(cfg) {
                    if let Some(h) = guard.as_ref() {
                        self.hnsw_batch(cfg, h, qs, code, threshold, top_k, &mut answers);
                    }
                    let fallbacks = answers.iter().filter(|a| a.is_none()).count() as u64;
                    self.counts.exact.fetch_add(fallbacks, Ordering::Relaxed);
                } else {
                    self.counts
                        .exact
                        .fetch_add(qs.len() as u64, Ordering::Relaxed);
                }
            } else if let Some(guard) = self.ready_index(cfg) {
                if let Some(ivf) = guard.as_ref() {
                    if !self.quant_approximate_batch(
                        cfg,
                        ivf,
                        qs,
                        code,
                        threshold,
                        top_k,
                        &mut answers,
                    ) {
                        self.approximate_batch(cfg, ivf, qs, code, threshold, top_k, &mut answers);
                    }
                }
                let fallbacks = answers.iter().filter(|a| a.is_none()).count() as u64;
                self.counts.exact.fetch_add(fallbacks, Ordering::Relaxed);
            } else {
                self.counts
                    .exact
                    .fetch_add(qs.len() as u64, Ordering::Relaxed);
            }
        }
        // whatever the index did not answer goes through the batched scan
        let todo: Vec<usize> = (0..qs.len()).filter(|&i| answers[i].is_none()).collect();
        if !todo.is_empty() {
            let rest: Vec<QueryVec> = todo
                .iter()
                .map(|&i| QueryVec {
                    v: qs[i].v,
                    norm: qs[i].norm,
                })
                .collect();
            let exact_hits = self.exact_batch_hits(&rest, code, threshold, top_k);
            for (i, hits) in todo.into_iter().zip(exact_hits) {
                answers[i] = Some(hits);
            }
        }
        answers.into_iter().map(|a| a.unwrap_or_default()).collect()
    }

    /// Answer a batch from the index, **cell-major**: work out which cells each
    /// query probes, invert that into "the queries that probe cell c", and score
    /// each probed cell once against all of its queries (the same 4-query x
    /// 2-row kernel the exact batch uses), instead of once per query. The rows
    /// scored for a query are exactly the rows the single-query probe scores
    /// (its cells plus the pending rows), so the answer is identical. A query
    /// that comes up short (a selective filter) is re-run on the single-query
    /// path, which widens.
    #[allow(clippy::too_many_arguments)]
    fn approximate_batch(
        &self,
        cfg: &AnnConfig,
        ivf: &Ivf,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        top_k: usize,
        answers: &mut [Option<Vec<Hit>>],
    ) {
        let nprobe = effective_nprobe(cfg, ivf.nlist);
        if nprobe >= ivf.nlist {
            return; // probing everything is the exact scan
        }
        let rows = self.table.rows();
        let live = self.table.live();
        let par = self.par();

        // 1. the cells each query probes
        let mut chosen: Vec<Option<Vec<u32>>> = (0..qs.len()).map(|_| None).collect();
        {
            let next = AtomicUsize::new(0);
            let done: Mutex<Vec<(usize, Vec<u32>)>> = Mutex::new(Vec::new());
            par.run(par.workers(qs.len(), usize::MAX), &|_| {
                let mut local = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= qs.len() {
                        break;
                    }
                    if qs[i].has_direction() {
                        local.push((i, ivf.choose_cells(&qs[i], nprobe)));
                    }
                }
                done.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
            });
            for (i, cells) in done.into_inner().unwrap_or_else(|e| e.into_inner()) {
                chosen[i] = Some(cells);
            }
        }

        // 2. invert: the queries of each cell; the extra last entry is the
        //    pending rows, which every query scores
        let mut by_cell: Vec<Vec<u32>> = vec![Vec::new(); ivf.nlist + 1];
        for (i, cells) in chosen.iter().enumerate() {
            if let Some(cells) = cells {
                for &c in cells {
                    by_cell[c as usize].push(i as u32);
                }
                by_cell[ivf.nlist].push(i as u32);
            }
        }
        let members_of = |c: usize| -> &[u32] {
            if c == ivf.nlist {
                &ivf.pending
            } else {
                &ivf.lists[c]
            }
        };
        let tasks: Vec<usize> = (0..=ivf.nlist)
            .filter(|&c| !by_cell[c].is_empty() && !members_of(c).is_empty())
            .collect();

        // 3. score every probed cell once against all of its queries
        let next = AtomicUsize::new(0);
        let parts: Mutex<Vec<Vec<TopK>>> = Mutex::new(Vec::new());
        par.run(par.workers(tasks.len(), usize::MAX), &|_| {
            let mut tops: Vec<TopK> = Vec::new();
            let mut slots: Vec<u32> = Vec::new();
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= tasks.len() {
                    break;
                }
                if tops.is_empty() {
                    tops = (0..qs.len())
                        .map(|_| TopK::new(top_k, threshold, live))
                        .collect();
                }
                let c = tasks[t];
                slots.clear();
                slots.extend(
                    members_of(c)
                        .iter()
                        .copied()
                        .filter(|&s| code.is_none_or(|code| rows.codes[s as usize] == code)),
                );
                for group in by_cell[c].chunks(4) {
                    let refs: Vec<&QueryVec> = group.iter().map(|&i| &qs[i as usize]).collect();
                    let mut taken: Vec<TopK> = group
                        .iter()
                        .map(|&i| std::mem::replace(&mut tops[i as usize], TopK::new(0, 0.0, 0)))
                        .collect();
                    batch_block(rows, &refs, &slots, &mut taken);
                    for (&i, top) in group.iter().zip(taken) {
                        tops[i as usize] = top;
                    }
                }
            }
            if !tops.is_empty() {
                parts.lock().unwrap_or_else(|e| e.into_inner()).push(tops);
            }
        });
        let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut merged: Vec<TopK> = match parts.pop() {
            Some(first) => first,
            None => (0..qs.len())
                .map(|_| TopK::new(top_k, threshold, live))
                .collect(),
        };
        for part in parts {
            for (a, b) in merged.iter_mut().zip(part) {
                a.merge(b);
            }
        }

        // 4. answers; the short ones go back through the single-query path
        let mut short = Vec::new();
        for (i, top) in merged.into_iter().enumerate() {
            if chosen[i].is_none() {
                continue;
            }
            if top.len() >= top_k {
                self.counts.approximate.fetch_add(1, Ordering::Relaxed);
                answers[i] = Some(top.into_sorted());
            } else {
                short.push(i);
            }
        }
        let one_thread = Par {
            configured: 1,
            pool: &self.pool,
        };
        let next = AtomicUsize::new(0);
        let redone: Mutex<Vec<(usize, Option<Vec<Hit>>)>> = Mutex::new(Vec::new());
        par.run(par.workers(short.len(), usize::MAX), &|_| {
            let mut local = Vec::new();
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= short.len() {
                    break;
                }
                let i = short[j];
                local.push((
                    i,
                    self.approximate_topk(cfg, ivf, &qs[i], code, threshold, top_k, one_thread),
                ));
            }
            redone
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(local);
        });
        for (i, hits) in redone.into_inner().unwrap_or_else(|e| e.into_inner()) {
            answers[i] = hits;
        }
    }

    /// Every match above `threshold`, best first, as a lazy iterator. With
    /// `scale: Some(s)` and an index configured, only the rows of the probed
    /// cells (`nprobe * s` of them); the flag says whether that was so (false
    /// means the list is exact and there is nothing left to widen to). Used by
    /// the graph's filtered search, which widens until enough matches pass its
    /// filter.
    pub(crate) fn ranked_hits(
        &self,
        query_vector: &[f32],
        node_type: Option<&str>,
        threshold: f32,
        scale: Option<usize>,
    ) -> Result<(RankedHits<'_>, bool), EmbeddingError> {
        self.check_query(query_vector)?;
        let empty = || RankedHits {
            store: self,
            hits: Vec::new().into_iter(),
        };
        let Ok(code) = self.type_filter(node_type) else {
            return Ok((empty(), false));
        };
        if self.table.live() == 0 {
            return Ok((empty(), false));
        }
        let q = QueryVec::new(query_vector);
        let mut approximate = None;
        if let (Some(cfg), Some(scale)) = (&self.ann, scale) {
            approximate = self.approximate_ranked(cfg, &q, code, threshold, scale);
            if approximate.is_none() {
                self.counts.exact.fetch_add(1, Ordering::Relaxed);
            }
        }
        let was_approximate = approximate.is_some();
        let hits = approximate.unwrap_or_else(|| self.exact_hits(&q, code, threshold, usize::MAX));
        Ok((
            RankedHits {
                store: self,
                hits: hits.into_iter(),
            },
            was_approximate,
        ))
    }

    /// Cosine score of every stored vector, keyed by `type:id`.
    pub(crate) fn scan_scores(
        &self,
        query_vector: &[f32],
    ) -> Result<HashMap<String, f32>, EmbeddingError> {
        self.check_query(query_vector)?;
        let q = QueryVec::new(query_vector);
        let table = &self.table;
        let mut scored = Vec::with_capacity(table.live());
        score_all(table.rows(), &q, table.slots(), &mut scored);
        let mut scores = HashMap::with_capacity(scored.len());
        for (slot, score) in scored {
            if let Some(meta) = table.meta(slot as usize) {
                scores.insert(meta.key.to_string(), score);
            }
        }
        Ok(scores)
    }

    /// Like `similarity_search_vector`, but always the full scan, even with an
    /// approximate index configured.
    pub fn similarity_search_vector_exact(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.search_vector(query_vector, top_k, node_type, threshold, true)
    }

    /// Like `similarity_search`, but always the full scan.
    pub fn similarity_search_exact(
        &self,
        query: &str,
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let query_vector = self.resolve_query(Query::Text(query))?;
        self.similarity_search_vector_exact(&query_vector, top_k, node_type, threshold)
    }

    /// Build the approximate index now instead of on the first search.
    ///
    /// Returns [`ann_stats`](Self::ann_stats). `force` retrains even if an
    /// index exists. Does nothing (and says so in the stats) while the store
    /// is smaller than `min_size`. Call this off the hot path after loading a
    /// big store: the first search that needs the index otherwise pays for it.
    ///
    /// # Errors
    ///
    /// `InvalidOption` when the store has no [`AnnConfig`].
    pub fn build_ann_index(&self, force: bool) -> Result<AnnStats, EmbeddingError> {
        let Some(cfg) = &self.ann else {
            return Err(EmbeddingError::InvalidOption(
                "this store has no approximate index; call set_ann first".to_string(),
            ));
        };
        if cfg.mode == AnnMode::Hnsw {
            if force {
                *self.hnsw.write().unwrap_or_else(|e| e.into_inner()) = None;
            }
            let _ = self.ready_hnsw(cfg);
            return Ok(self.ann_stats());
        }
        if force {
            *self.ivf.write().unwrap_or_else(|e| e.into_inner()) = None;
        }
        if self.ready_index(cfg).is_some() {
            let mut guard = self.ivf.write().unwrap_or_else(|e| e.into_inner());
            if let Some(ivf) = guard.as_mut() {
                ivf.fold(&self.table, self.par());
            }
        }
        Ok(self.ann_stats())
    }

    /// What the approximate index looks like right now. Never builds anything.
    pub fn ann_stats(&self) -> AnnStats {
        if self.hnsw_is_mode() {
            return self.hnsw_ann_stats();
        }
        let guard = self.ivf_read();
        let ivf = guard.as_ref();
        AnnStats {
            enabled: self.ann.is_some(),
            active: ivf.is_some(),
            nlist: ivf.map(|i| i.nlist),
            nprobe: match (ivf, &self.ann) {
                (Some(i), Some(cfg)) => Some(effective_nprobe(cfg, i.nlist)),
                (None, Some(cfg)) => cfg.nprobe,
                _ => None,
            },
            trained_n: ivf.map(|i| i.trained_n),
            pending: ivf.map_or(0, |i| i.pending.len()),
            churn: ivf.map_or(0, |i| i.churn),
            build_seconds: ivf.map(|i| i.build_seconds),
            index_bytes: ivf.map_or(0, Ivf::nbytes),
            min_size: self.ann.as_ref().map(|c| c.min_size),
            searches: self.counts.snapshot(),
            folds: ivf.map_or(0, |i| i.folds),
            fold_seconds: ivf.map_or(0.0, |i| i.fold_seconds),
            mode: AnnMode::Ivf,
            ef_search: None,
            max_level: None,
            graph_nodes: 0,
            graph_dead: 0,
            inserts: 0,
            insert_seconds: 0.0,
        }
    }

    /// Check every internal invariant (matrix shape, key index, per-model
    /// counts, the approximate index against the matrix) and describe the first
    /// one that fails. For tests and debugging; it walks the whole store.
    pub fn verify_invariants(&self) -> Result<(), String> {
        self.table.check()?;
        let by_model: usize = self.model_counts.values().sum();
        if by_model != self.table.live() {
            return Err(format!(
                "model counts add up to {by_model}, {} live rows",
                self.table.live()
            ));
        }
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for slot in self.table.live_slots() {
            if let Some(meta) = self.table.meta(slot) {
                *seen.entry(&meta.model).or_default() += 1;
            }
        }
        for (model, count) in &self.model_counts {
            if seen.get(model.as_str()) != Some(count) {
                return Err(format!(
                    "model {model:?} counted {count}, rows say {:?}",
                    seen.get(model.as_str())
                ));
            }
        }
        if let Some(ivf) = self.ivf_read().as_ref() {
            ivf.check(&self.table, FOLD_BATCH)?;
        }
        if let Some(h) = self.hnsw_read().as_ref() {
            h.check(&self.table)?;
        }
        Ok(())
    }

    /// Where the store's memory goes. The vectors are held once; compare
    /// `matrix_bytes` with `live_vector_bytes`.
    pub fn memory_stats(&self) -> MemoryStats {
        let t = self.table.bytes();
        MemoryStats {
            rows: self.table.slots(),
            live: self.table.live(),
            dead: self.table.dead(),
            chunks: self.table.chunk_count(),
            matrix_bytes: t.matrix,
            live_vector_bytes: self.table.live() * self.table.dim * 4,
            metadata_bytes: t.per_row + t.keys_and_text + t.key_index,
            index_bytes: self.ivf_read().as_ref().map_or(0, Ivf::nbytes)
                + self.hnsw_read().as_ref().map_or(0, |h| h.nbytes()),
        }
    }

    /// Recall@k of the approximate index against the exact scan.
    ///
    /// Runs every query both ways and compares the *sets* of top-k nodes.
    /// Queries are texts or vectors. Use real queries drawn from your own
    /// workload: recall depends on how the data clusters, and a score measured
    /// on unrelated queries or data does not transfer. Raise `nprobe` until
    /// the recall you need holds.
    ///
    /// Without an index in use (none configured, a store under `min_size`, or
    /// `nprobe >= nlist`) both answers are the exact scan, recall is 1.0 and
    /// `approximate` is false.
    pub fn recall_check<'q, I, Q>(
        &self,
        queries: I,
        k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<RecallReport, EmbeddingError>
    where
        I: IntoIterator<Item = Q>,
        Q: Into<Query<'q>>,
    {
        if k == 0 {
            return Err(EmbeddingError::InvalidOption(
                "k must be positive".to_string(),
            ));
        }
        let mut recalls: Vec<f64> = Vec::new();
        let mut ann_times: Vec<f64> = Vec::new();
        let mut exact_times: Vec<f64> = Vec::new();
        let mut used = false;
        let mut count = 0usize;
        for query in queries {
            count += 1;
            let vector = self.resolve_query(query.into())?;
            let before = self.ann_stats().searches.approximate;
            let started = Instant::now();
            let found = self.search_vector(&vector, k, node_type, threshold, false)?;
            ann_times.push(started.elapsed().as_secs_f64());
            used |= self.ann_stats().searches.approximate > before;
            let started = Instant::now();
            let truth = self.search_vector(&vector, k, node_type, threshold, true)?;
            exact_times.push(started.elapsed().as_secs_f64());
            if !truth.is_empty() {
                let wanted: std::collections::HashSet<&crate::NodeId> =
                    truth.iter().map(|r| &r.node_id).collect();
                let hit = found.iter().filter(|r| wanted.contains(&r.node_id)).count();
                recalls.push(hit as f64 / wanted.len() as f64);
            }
        }
        let stats = self.ann_stats();
        let median = |mut v: Vec<f64>| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(f64::total_cmp);
            let mid = v.len() / 2;
            if v.len() % 2 == 1 {
                v[mid]
            } else {
                0.5 * (v[mid - 1] + v[mid])
            }
        };
        Ok(RecallReport {
            recall: if recalls.is_empty() {
                1.0
            } else {
                recalls.iter().sum::<f64>() / recalls.len() as f64
            },
            min_recall: recalls.iter().copied().fold(1.0, f64::min),
            queries: count,
            k,
            nlist: stats.nlist.unwrap_or(0),
            nprobe: stats.nprobe.unwrap_or(0),
            approximate: used,
            ef_search: stats.ef_search.unwrap_or(0),
            ann_ms: 1000.0 * median(ann_times),
            exact_ms: 1000.0 * median(exact_times),
        })
    }
}

/// Score the probed cells (and the pending rows). The rows of a cell are
/// scattered through the matrix, so one thread is limited by how many cache
/// lines it can have in flight, not by bandwidth: a large probe is shared out
/// a few cells at a time. The answer does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
fn scan_probed(
    table: &VectorTable,
    cells: Cells,
    chosen: &[u32],
    q: &QueryVec,
    code: Option<u32>,
    threshold: f32,
    k: usize,
    par: Par,
) -> TopK {
    const CELLS_PER_TASK: usize = 4;
    if let Some(top) = quant::scan_probed(table, cells, chosen, q, code, threshold, k, par) {
        return top;
    }
    let rows = table.rows();
    let live = table.live();
    let candidates: usize = chosen
        .iter()
        .map(|&c| cells.lists[c as usize].len())
        .sum::<usize>()
        + cells.pending.len();
    let tasks = chosen.len().div_ceil(CELLS_PER_TASK);
    let threads = if par.configured == 0 {
        par.workers(tasks, SCAN_AUTO_THREADS)
            .min((candidates * table.dim / PAR_MIN_CELL_FLOATS).max(1))
    } else {
        par.workers(tasks, SCAN_AUTO_THREADS)
    };
    let mut top = TopK::new(k, threshold, live);
    if threads <= 1 {
        scan_cells(rows, cells, chosen, true, q, code, &mut top);
        return top;
    }
    let next = AtomicUsize::new(0);
    let parts = Mutex::new(Vec::with_capacity(threads));
    par.run(threads, &|_| {
        let mut local = TopK::new(k, threshold, live);
        loop {
            let t = next.fetch_add(1, Ordering::Relaxed);
            if t >= tasks {
                break;
            }
            let lo = t * CELLS_PER_TASK;
            let hi = (lo + CELLS_PER_TASK).min(chosen.len());
            // the pending rows ride with the first task only
            scan_cells(rows, cells, &chosen[lo..hi], t == 0, q, code, &mut local);
        }
        parts.lock().unwrap_or_else(|e| e.into_inner()).push(local);
    });
    for part in parts.into_inner().unwrap_or_else(|e| e.into_inner()) {
        top.merge(part);
    }
    top
}

/// Counters, the pool and the lazily built index: the search-side state an
/// [`EmbeddingStore`] carries besides its table.
pub(crate) type IvfSlot = RwLock<Option<Ivf>>;

#[allow(dead_code, clippy::manual_is_multiple_of)]
const _: () = assert!(CHUNK_ROWS % BLOCK_ROWS == 0 && CHUNK_ROWS % NEAR_BLOCK == 0);

// Opt-in int8 first-pass scan with an exact rerank (see `quant.rs`). Declared
// after `dispatch!` so the child can use that macro.
#[path = "quant.rs"]
pub(crate) mod quant;

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect()
    }

    #[test]
    fn the_dot_kernel_matches_a_plain_sum_at_awkward_lengths() {
        for len in [1usize, 3, 7, 8, 31, 32, 33, 100, 384, 1000] {
            let a = pseudo(len, 1);
            let b = pseudo(len, 2);
            let want = naive_dot(&a, &b);
            assert!((dot_f32(&a, &b) as f64 - want).abs() < 1e-4, "len {len}");
            assert!(
                (dot_impl::<false>(&a, &b) as f64 - want).abs() < 1e-4,
                "len {len}"
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn every_kernel_shape_gives_bit_identical_dot_products() {
        // The promise behind "a batch equals the single queries exactly": a
        // (query, row) pair is summed in the same order whatever block shape
        // computes it.
        fn check<const F: bool>(dim: usize) {
            let q: Vec<Vec<f32>> = (0..4).map(|i| pseudo(dim, 10 + i)).collect();
            let r: Vec<Vec<f32>> = (0..4).map(|i| pseudo(dim, 20 + i)).collect();
            let one = |a: usize, j: usize| block::<1, 1, F>([&q[a]], [&r[j]])[0][0];
            let b42 = block::<4, 2, F>([&q[0], &q[1], &q[2], &q[3]], [&r[0], &r[1]]);
            let b14 = block::<1, 4, F>([&q[2]], [&r[0], &r[1], &r[2], &r[3]]);
            let b24 = block::<2, 4, F>([&q[1], &q[3]], [&r[0], &r[1], &r[2], &r[3]]);
            let b41 = block::<4, 1, F>([&q[0], &q[1], &q[2], &q[3]], [&r[3]]);
            for a in 0..4 {
                for j in 0..2 {
                    assert_eq!(b42[a][j].to_bits(), one(a, j).to_bits(), "4x2 dim {dim}");
                }
                assert_eq!(b41[a][0].to_bits(), one(a, 3).to_bits(), "4x1 dim {dim}");
            }
            for j in 0..4 {
                assert_eq!(b14[0][j].to_bits(), one(2, j).to_bits(), "1x4 dim {dim}");
                assert_eq!(b24[0][j].to_bits(), one(1, j).to_bits(), "2x4 dim {dim}");
                assert_eq!(b24[1][j].to_bits(), one(3, j).to_bits(), "2x4 dim {dim}");
            }
        }
        for dim in [1usize, 5, 8, 9, 32, 100, 384] {
            check::<false>(dim);
            // the fused-multiply-add instantiation is only valid where the CPU
            // has it; the dispatching wrapper checks
            let wrapper_ok = {
                let a = pseudo(dim, 1);
                let b = pseudo(dim, 2);
                let fused = dot_f32(&a, &b);
                let plain = dot_impl::<false>(&a, &b);
                (fused - plain).abs() < 1e-4
            };
            assert!(wrapper_ok);
        }
    }

    #[test]
    fn the_assignment_kernel_matches_a_plain_argmax() {
        for (dim, nlist, m) in [
            (5usize, 1usize, 3usize),
            (9, 7, 130),
            (32, 10, 65),
            (384, 13, 70),
        ] {
            let pts = pseudo(m * dim, 3);
            let cents = pseudo(nlist * dim, 4);
            let mut got = vec![0i32; m];
            nearest_block(&pts, dim, &cents, nlist, &mut got);
            // the answer for a point does not depend on its neighbours
            let mut alone = vec![0i32; m];
            for p in 0..m {
                nearest_block(
                    &pts[p * dim..(p + 1) * dim],
                    dim,
                    &cents,
                    nlist,
                    &mut alone[p..p + 1],
                );
            }
            assert_eq!(got, alone);
            let pool = None;
            let par = Par {
                configured: 1,
                pool: &pool,
            };
            let mut serial = vec![0i32; m];
            nearest_parallel(
                &|lo, len| &pts[lo * dim..(lo + len) * dim],
                dim,
                &cents,
                nlist,
                &mut serial,
                par,
            );
            assert_eq!(got, serial);
            for p in 0..m {
                let x = &pts[p * dim..(p + 1) * dim];
                let scores: Vec<f64> = (0..nlist)
                    .map(|c| naive_dot(x, &cents[c * dim..(c + 1) * dim]))
                    .collect();
                let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // an f32 near-tie may legitimately pick the runner-up
                assert!(
                    scores[got[p] as usize] >= best - 1e-4,
                    "dim {dim} nlist {nlist} point {p}"
                );
            }
        }
    }

    #[test]
    fn top_k_keeps_the_best_and_breaks_ties_by_arrival() {
        let mut top = TopK::new(3, 0.0, 100);
        for (slot, score) in [0.5f32, 0.9, 0.9, 0.1, 0.9, 0.7, f32::NAN, -1.0, 0.9]
            .into_iter()
            .enumerate()
        {
            top.offer(score, slot as u32);
        }
        let got: Vec<(u32, f32)> = top
            .into_sorted()
            .iter()
            .map(|h| (h.slot, h.score))
            .collect();
        assert_eq!(got, vec![(1, 0.9), (2, 0.9), (4, 0.9)]);

        // asked for more than can exist: collects everything above the bar
        let mut all = TopK::new(usize::MAX, 0.3, 10);
        for (slot, score) in [0.2f32, 0.4, 0.4, 0.9].into_iter().enumerate() {
            all.offer(score, slot as u32);
        }
        let got: Vec<u32> = all.into_sorted().iter().map(|h| h.slot).collect();
        assert_eq!(got, vec![3, 1, 2]);
    }

    #[test]
    fn merging_partial_results_equals_one_pass() {
        let scores = pseudo(500, 9);
        let mut whole = TopK::new(17, -1.0, 500);
        let (mut a, mut b) = (TopK::new(17, -1.0, 500), TopK::new(17, -1.0, 500));
        for (slot, &s) in scores.iter().enumerate() {
            // quantise so there are plenty of ties
            let s = (s * 20.0).round() / 20.0;
            whole.offer(s, slot as u32);
            if slot % 2 == 0 {
                a.offer(s, slot as u32);
            } else {
                b.offer(s, slot as u32);
            }
        }
        a.merge(b);
        let w: Vec<(u32, f32)> = whole
            .into_sorted()
            .iter()
            .map(|h| (h.slot, h.score))
            .collect();
        let m: Vec<(u32, f32)> = a.into_sorted().iter().map(|h| (h.slot, h.score)).collect();
        assert_eq!(w, m);
    }

    #[test]
    fn rounding_matches_pythons_round() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
        assert_eq!(auto_nlist(100), 40);
        assert_eq!(auto_nlist(4), 16);
        assert_eq!(auto_nlist(1_000_000), 4000);
        assert_eq!(auto_nlist(usize::MAX / 4), 65_536);
        assert_eq!(effective_nprobe(&AnnConfig::default(), 4000), 200);
        assert_eq!(effective_nprobe(&AnnConfig::default(), 10), 1);
        assert_eq!(effective_nprobe(&AnnConfig::default(), 50), 2);
        assert_eq!(
            effective_nprobe(
                &AnnConfig {
                    nprobe: Some(99),
                    ..AnnConfig::default()
                },
                64
            ),
            64
        );
    }
}
