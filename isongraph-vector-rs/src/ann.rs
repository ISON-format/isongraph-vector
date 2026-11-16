//! Fast search for [`EmbeddingStore`]: a contiguous scan cache for exact
//! search, and the opt-in IVF-flat approximate index built on top of it.
//!
//! # The scan cache
//!
//! The store keeps one [`EmbeddingRecord`] per node (that is the public,
//! `HashMap`-backed view). The first search additionally decodes every vector
//! into one row-major `f32` matrix with a norm, a type code and a liveness flag
//! per row. A search is then a single pass over contiguous memory: a dot
//! product per row, a bounded heap for the best `k`, no per-row allocation and
//! no hashing. Writes patch the matrix in place (an insert appends a row, an
//! update overwrites one, a delete tombstones one; a quarter of dead rows
//! triggers a compaction), so a write never throws the cache away.
//!
//! The matrix is a second copy of the vectors. It exists only once a search has
//! asked for it and is never written anywhere.
//!
//! Rows are ordered by key when the cache is first built and by arrival after
//! that, and ties between equal scores come back in that order, so a search
//! returns the same answer on every run.
//!
//! # The IVF-flat index
//!
//! Opt-in through [`AnnConfig`]. Spherical k-means (seeded, on a sample)
//! partitions the rows into `nlist` cells by direction; a query scores the
//! `nprobe` cells whose centroids are nearest and only the rows inside them,
//! with exactly the cosine the exact scan uses. The index stores no vectors:
//! it is a cell number per row, the rows grouped by cell, and the centroids.
//! It follows the cache through writes and deletes instead of being rebuilt
//! (see [`AnnConfig`] for the policy), is built lazily, and is never persisted.
//!
//! The behaviour deliberately mirrors the Python port's `ann="ivf"`.

use crate::{EmbeddingEncoder, EmbeddingError, EmbeddingStore, Query, SimilarityResult};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

/// Below this many vectors the index is not used and search stays exact.
pub const ANN_MIN_SIZE: usize = 50_000;
/// Pending rows are folded into cells at this many ...
const FOLD_MIN: usize = 512;
/// ... and at least this fraction of the indexed rows.
const FOLD_FRACTION: f64 = 0.02;
/// Retrain once writes plus deletes exceed this fraction of the trained size.
const RETRAIN_FRACTION: f64 = 0.5;
const KMEANS_ITERATIONS: usize = 8;
const SAMPLE_PER_CELL: usize = 40;
/// Threads a scan uses when none is configured (see `workers`).
const SCAN_AUTO_THREADS: usize = 8;
/// Floats per scan block handed to a worker (2 MiB).
const BLOCK_FLOATS: usize = 1 << 19;
/// A scan spreads over threads only past this many floats per thread.
const PAR_MIN_FLOATS: usize = 1 << 21;
/// The same for the rows of probed cells, which cost more per float.
const PAR_MIN_CELL_FLOATS: usize = 1 << 20;
/// Points per unit of parallel k-means work; a multiple of `TILE`.
const NEAR_BLOCK: usize = 256;
/// Points per tile of the assignment kernel (the tile stays in L2).
const TILE: usize = 64;

// =============================================================================
// Public types
// =============================================================================

/// Settings for the opt-in approximate (IVF-flat) index.
///
/// Off unless given to [`EmbeddingStore::set_ann`]; with it off every search
/// is exact, as before. With it on, a *bounded* top-k search
/// (`top_k < number of vectors`) on a store of at least `min_size` vectors is
/// answered from the index. The scores are the same cosine values the exact
/// scan gives, but a true neighbour in an unprobed cell can be missing:
/// measure it with [`EmbeddingStore::recall_check`] on your own queries.
///
/// Policy (identical to the Python port):
///
/// * the index is built by the first search that wants it (or by
///   `build_ann_index`), from k-means on a sample of up to `40 * nlist` live
///   vectors, and is never persisted;
/// * new and updated vectors are scored exactly until at least
///   `max(512, 2%)` of the indexed rows are pending, then folded into their
///   nearest cell;
/// * a delete just stops the row being returned;
/// * once writes plus deletes since training exceed 50% of the trained size
///   the index is retrained (with an automatic `nlist`, re-chosen for the new
///   size);
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
    /// Seed for the k-means initialisation: the same data gives the same index.
    pub seed: u64,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            nlist: None,
            nprobe: None,
            min_size: ANN_MIN_SIZE,
            seed: 0,
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
        Ok(())
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
    /// Rows scored exactly until they are folded into cells.
    pub pending: usize,
    /// Writes plus deletes since training.
    pub churn: usize,
    pub build_seconds: Option<f64>,
    /// Memory on top of the cached vectors.
    pub index_bytes: usize,
    pub min_size: Option<usize>,
    pub searches: AnnSearchCounts,
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
    pub ann_ms: f64,
    pub exact_ms: f64,
}

// =============================================================================
// Small helpers
// =============================================================================

/// SplitMix64: tiny, seedable, and the same on every platform.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
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

fn auto_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Workers for `tasks` units of work: the configured count, or with 0 the
/// number of CPUs up to `auto_cap`. Memory bandwidth, not compute, limits a
/// scan, and on the machine this was tuned on eight threads were within a few
/// percent of thirty-two with a much steadier tail.
fn workers(configured: usize, tasks: usize, auto_cap: usize) -> usize {
    let want = if configured == 0 {
        auto_threads().min(auto_cap)
    } else {
        configured
    };
    want.min(tasks).max(1)
}

// =============================================================================
// Kernels
//
// Each hot routine is written once as an `#[inline(always)]` generic over
// `FMA`, then instantiated twice: with AVX2 + FMA enabled (picked at run time
// when the CPU has them) and with whatever the build targets. The `FMA = true`
// instantiation is only ever reached through a function that enables the
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

/// Explicit AVX2 + FMA kernels. The portable loops below do not reliably
/// vectorise (measured: about scalar speed), so the fast instantiation calls
/// these. Only reachable through the `dispatch!` wrappers, which check the CPU.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA; `a` and `b` must have equal length.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                s1,
            );
            s2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                s2,
            );
            s3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                s3,
            );
            i += 32;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            i += 8;
        }
        let mut total = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
        while i < n {
            total += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        total
    }

    /// Two points against four centroids.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; all six slices must have equal length.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_2x4(x0: &[f32], x1: &[f32], c: [&[f32]; 4]) -> [[f32; 4]; 2] {
        let n = x0.len();
        debug_assert!(x1.len() == n && c.iter().all(|s| s.len() == n));
        let (p0, p1) = (x0.as_ptr(), x1.as_ptr());
        let cp = [c[0].as_ptr(), c[1].as_ptr(), c[2].as_ptr(), c[3].as_ptr()];
        let z = _mm256_setzero_ps();
        let (mut a00, mut a01, mut a02, mut a03) = (z, z, z, z);
        let (mut a10, mut a11, mut a12, mut a13) = (z, z, z, z);
        let mut i = 0;
        while i + 8 <= n {
            let v0 = _mm256_loadu_ps(p0.add(i));
            let v1 = _mm256_loadu_ps(p1.add(i));
            let y0 = _mm256_loadu_ps(cp[0].add(i));
            a00 = _mm256_fmadd_ps(v0, y0, a00);
            a10 = _mm256_fmadd_ps(v1, y0, a10);
            let y1 = _mm256_loadu_ps(cp[1].add(i));
            a01 = _mm256_fmadd_ps(v0, y1, a01);
            a11 = _mm256_fmadd_ps(v1, y1, a11);
            let y2 = _mm256_loadu_ps(cp[2].add(i));
            a02 = _mm256_fmadd_ps(v0, y2, a02);
            a12 = _mm256_fmadd_ps(v1, y2, a12);
            let y3 = _mm256_loadu_ps(cp[3].add(i));
            a03 = _mm256_fmadd_ps(v0, y3, a03);
            a13 = _mm256_fmadd_ps(v1, y3, a13);
            i += 8;
        }
        let mut out = [
            [hsum(a00), hsum(a01), hsum(a02), hsum(a03)],
            [hsum(a10), hsum(a11), hsum(a12), hsum(a13)],
        ];
        while i < n {
            for (ci, base) in cp.iter().enumerate() {
                let y = *base.add(i);
                out[0][ci] += *p0.add(i) * y;
                out[1][ci] += *p1.add(i) * y;
            }
            i += 1;
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
    ($(#[$meta:meta])* fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:ident) => {
        $(#[$meta])*
        fn $name($($arg: $ty),*) $(-> $ret)? {
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

/// Dot product with 32 independent partial sums, so the loop is limited by
/// memory rather than by the latency of one running total.
#[inline(always)]
fn dot_impl<const F: bool>(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if F {
        // SAFETY: `F` is true only in the instantiation reached through a
        // `dispatch!` wrapper, which has checked for AVX2 and FMA.
        return unsafe { x86::dot(a, b) };
    }
    const LANES: usize = 8;
    const UNROLL: usize = 4;
    let mut acc = [[0.0f32; LANES]; UNROLL];
    let mut ca = a.chunks_exact(LANES * UNROLL);
    let mut cb = b.chunks_exact(LANES * UNROLL);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for (u, lanes) in acc.iter_mut().enumerate() {
            for (l, slot) in lanes.iter_mut().enumerate() {
                let i = u * LANES + l;
                *slot = fma::<F>(x[i], y[i], *slot);
            }
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail = fma::<F>(*x, *y, tail);
    }
    let mut s = [0.0f32; LANES];
    for (l, slot) in s.iter_mut().enumerate() {
        *slot = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }
    (((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))) + tail
}

dispatch! {
    fn dot_f32(a: &[f32], b: &[f32]) -> f32 = dot_impl
}

/// `out[i] = dot(query, row i of mat)`.
#[inline(always)]
fn dots_impl<const F: bool>(query: &[f32], mat: &[f32], dim: usize, out: &mut [f32]) {
    for (slot, row) in out.iter_mut().zip(mat.chunks_exact(dim)) {
        *slot = dot_impl::<F>(query, row);
    }
}

dispatch! {
    fn dots_into(query: &[f32], mat: &[f32], dim: usize, out: &mut [f32]) = dots_impl
}

/// Two points against four centroids, sharing every load: eight running
/// vector sums instead of one, a quarter of the memory traffic per multiply.
#[inline(always)]
fn dot_2x4<const F: bool>(x0: &[f32], x1: &[f32], c: [&[f32]; 4]) -> [[f32; 4]; 2] {
    #[cfg(target_arch = "x86_64")]
    if F {
        // SAFETY: as in `dot_impl`.
        return unsafe { x86::dot_2x4(x0, x1, c) };
    }
    const LANES: usize = 8;
    let dim = x0.len();
    let body = dim - dim % LANES;
    let mut acc = [[[0.0f32; LANES]; 4]; 2];
    let mut d = 0;
    while d < body {
        let a0 = &x0[d..d + LANES];
        let a1 = &x1[d..d + LANES];
        for (ci, centroid) in c.iter().enumerate() {
            let y = &centroid[d..d + LANES];
            for l in 0..LANES {
                acc[0][ci][l] = fma::<F>(a0[l], y[l], acc[0][ci][l]);
                acc[1][ci][l] = fma::<F>(a1[l], y[l], acc[1][ci][l]);
            }
        }
        d += LANES;
    }
    let mut out = [[0.0f32; 4]; 2];
    for (p, x) in [x0, x1].into_iter().enumerate() {
        for (ci, centroid) in c.iter().enumerate() {
            let a = &acc[p][ci];
            let mut s = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            for t in body..dim {
                s = fma::<F>(x[t], centroid[t], s);
            }
            out[p][ci] = s;
        }
    }
    out
}

/// For each point in `pts` (row-major, `out.len()` rows of `dim`), the index of
/// the centroid with the largest dot product (the first on a tie). Points are
/// handled in tiles that stay in cache while the centroids stream past.
#[inline(always)]
fn nearest_impl<const F: bool>(
    pts: &[f32],
    dim: usize,
    cents: &[f32],
    nlist: usize,
    out: &mut [i32],
) {
    let m = out.len();
    let mut best_s = [f32::NEG_INFINITY; TILE];
    let mut best_i = [0i32; TILE];
    let mut p0 = 0;
    while p0 < m {
        let t = (m - p0).min(TILE);
        best_s[..t].fill(f32::NEG_INFINITY);
        best_i[..t].fill(0);
        let mut c = 0;
        while c + 4 <= nlist {
            let group = [
                &cents[c * dim..(c + 1) * dim],
                &cents[(c + 1) * dim..(c + 2) * dim],
                &cents[(c + 2) * dim..(c + 3) * dim],
                &cents[(c + 3) * dim..(c + 4) * dim],
            ];
            let mut p = 0;
            while p + 2 <= t {
                let x0 = &pts[(p0 + p) * dim..(p0 + p + 1) * dim];
                let x1 = &pts[(p0 + p + 1) * dim..(p0 + p + 2) * dim];
                let d = dot_2x4::<F>(x0, x1, group);
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
                let x = &pts[(p0 + p) * dim..(p0 + p + 1) * dim];
                for (ci, centroid) in group.iter().enumerate() {
                    let s = dot_impl::<F>(x, centroid);
                    if s > best_s[p] {
                        best_s[p] = s;
                        best_i[p] = (c + ci) as i32;
                    }
                }
            }
            c += 4;
        }
        while c < nlist {
            let centroid = &cents[c * dim..(c + 1) * dim];
            for p in 0..t {
                let x = &pts[(p0 + p) * dim..(p0 + p + 1) * dim];
                let s = dot_impl::<F>(x, centroid);
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

/// `nearest_block` spread over threads in fixed-size blocks, so the answer for
/// a point never depends on how many threads there are.
fn nearest_parallel(
    pts: &[f32],
    dim: usize,
    cents: &[f32],
    nlist: usize,
    out: &mut [i32],
    threads: usize,
) {
    let blocks = out.len().div_ceil(NEAR_BLOCK);
    let work = out.len().saturating_mul(nlist).saturating_mul(dim);
    let threads = if work < (1 << 24) {
        1
    } else {
        workers(threads, blocks, usize::MAX)
    };
    if threads <= 1 {
        nearest_block(pts, dim, cents, nlist, out);
        return;
    }
    let queue = Mutex::new(out.chunks_mut(NEAR_BLOCK).enumerate());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let next = queue.lock().ok().and_then(|mut q| q.next());
                let Some((b, chunk)) = next else { break };
                let lo = b * NEAR_BLOCK;
                nearest_block(
                    &pts[lo * dim..(lo + chunk.len()) * dim],
                    dim,
                    cents,
                    nlist,
                    chunk,
                );
            });
        }
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
    v: &'a [f32],
    norm: f32,
}

impl<'a> QueryVec<'a> {
    fn new(v: &'a [f32]) -> Self {
        Self {
            v,
            norm: dot_f32(v, v).sqrt(),
        }
    }

    /// A direction to rank cells by.
    fn has_direction(&self) -> bool {
        self.norm > 0.0 && self.norm.is_finite()
    }
}

/// The matrix side of the cache, borrowed for a scan.
#[derive(Clone, Copy)]
struct Rows<'a> {
    data: &'a [f32],
    norms: &'a [f32],
    codes: &'a [u32],
    dim: usize,
}

/// Cosine of `q` against one row. A zero-length row scores 0, a tombstone
/// (norm NaN) scores NaN and is dropped by the collector.
#[inline(always)]
fn score_row<const F: bool>(q: &QueryVec, row: &[f32], norm: f32) -> f32 {
    let dot = dot_impl::<F>(q.v, row);
    let denom = norm * q.norm;
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
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
    let dim = rows.dim;
    for slot in lo..hi {
        if let Some(code) = code {
            if rows.codes[slot] != code {
                continue;
            }
        }
        let row = &rows.data[slot * dim..(slot + 1) * dim];
        top.offer(score_row::<F>(q, row, rows.norms[slot]), slot as u32);
    }
}

dispatch! {
    fn scan_range(rows: Rows, q: &QueryVec, code: Option<u32>, lo: usize, hi: usize, top: &mut TopK) = scan_range_impl
}

/// The parts of an [`Ivf`] a cell scan reads.
#[derive(Clone, Copy)]
struct Cells<'a> {
    order: &'a [u32],
    offsets: &'a [usize],
    assign: &'a [i32],
    pending: &'a BTreeSet<u32>,
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
    let dim = rows.dim;
    let visit = |slot: usize, top: &mut TopK| {
        if let Some(code) = code {
            if rows.codes[slot] != code {
                return;
            }
        }
        let row = &rows.data[slot * dim..(slot + 1) * dim];
        top.offer(score_row::<F>(q, row, rows.norms[slot]), slot as u32);
    };
    for &c in chosen {
        let c = c as usize;
        for &slot in &cells.order[cells.offsets[c]..cells.offsets[c + 1]] {
            // Entries can be stale: a row deleted or rewritten since the
            // lists were last regrouped no longer belongs to this cell.
            if cells.assign[slot as usize] == c as i32 {
                visit(slot as usize, top);
            }
        }
    }
    if with_pending {
        for &slot in cells.pending {
            visit(slot as usize, top);
        }
    }
}

dispatch! {
    fn scan_cells(rows: Rows, cells: Cells, chosen: &[u32], with_pending: bool, q: &QueryVec, code: Option<u32>, top: &mut TopK) = scan_cells_impl
}

// =============================================================================
// The scan cache
// =============================================================================

/// Row-major snapshot of every vector, patched in place by writes.
pub(crate) struct ScanCache {
    dim: usize,
    data: Vec<f32>,
    /// Euclidean norm per row; NaN marks a tombstone.
    norms: Vec<f32>,
    codes: Vec<u32>,
    alive: Vec<bool>,
    keys: Vec<Option<Arc<str>>>,
    slot_of: HashMap<Arc<str>, u32>,
    code_of: HashMap<String, u32>,
    /// Slots in use, tombstones included.
    n: usize,
    dead: usize,
}

impl ScanCache {
    fn empty(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            norms: Vec::new(),
            codes: Vec::new(),
            alive: Vec::new(),
            keys: Vec::new(),
            slot_of: HashMap::new(),
            code_of: HashMap::new(),
            n: 0,
            dead: 0,
        }
    }

    fn build(records: &HashMap<String, crate::EmbeddingRecord>, dim: usize) -> Self {
        let mut cache = Self::empty(dim);
        let mut keys: Vec<&String> = records.keys().collect();
        keys.sort_unstable();
        cache.data.reserve_exact(keys.len() * dim);
        cache.norms.reserve_exact(keys.len());
        cache.codes.reserve_exact(keys.len());
        cache.alive.reserve_exact(keys.len());
        cache.keys.reserve_exact(keys.len());
        cache.slot_of.reserve(keys.len());
        for key in keys {
            let record = &records[key];
            cache.put(key, &record.node_id.node_type, &record.vector);
        }
        cache
    }

    fn live(&self) -> usize {
        self.n - self.dead
    }

    fn row(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    fn rows(&self) -> Rows<'_> {
        Rows {
            data: &self.data,
            norms: &self.norms,
            codes: &self.codes,
            dim: self.dim,
        }
    }

    fn code_for(&mut self, node_type: &str) -> u32 {
        if let Some(&code) = self.code_of.get(node_type) {
            return code;
        }
        let code = self.code_of.len() as u32;
        self.code_of.insert(node_type.to_string(), code);
        code
    }

    /// Insert or overwrite the row for `key`; returns its slot.
    fn put(&mut self, key: &str, node_type: &str, vector: &[f32]) -> u32 {
        let code = self.code_for(node_type);
        let norm = dot_f32(vector, vector).sqrt();
        if let Some(&slot) = self.slot_of.get(key) {
            let s = slot as usize;
            self.data[s * self.dim..(s + 1) * self.dim].copy_from_slice(vector);
            self.norms[s] = norm;
            self.codes[s] = code;
            return slot;
        }
        let slot = self.n as u32;
        let key: Arc<str> = Arc::from(key);
        self.data.extend_from_slice(vector);
        self.norms.push(norm);
        self.codes.push(code);
        self.alive.push(true);
        self.keys.push(Some(key.clone()));
        self.slot_of.insert(key, slot);
        self.n += 1;
        slot
    }

    /// Tombstone the row for `key`; returns its slot.
    fn remove(&mut self, key: &str) -> Option<u32> {
        let slot = self.slot_of.remove(key)?;
        let s = slot as usize;
        self.norms[s] = f32::NAN;
        self.alive[s] = false;
        self.keys[s] = None;
        self.dead += 1;
        Some(slot)
    }

    fn should_compact(&self) -> bool {
        self.dead > 0 && self.dead * 4 > self.n
    }

    /// Drop the tombstones, keeping row order. Returns the old slot of every
    /// surviving row, in order, so an index can follow the move.
    fn compact(&mut self) -> Vec<u32> {
        let dim = self.dim;
        let live: Vec<u32> = (0..self.n as u32)
            .filter(|&s| self.alive[s as usize])
            .collect();
        for (new, &old) in live.iter().enumerate() {
            let old = old as usize;
            if new != old {
                self.data.copy_within(old * dim..(old + 1) * dim, new * dim);
                self.norms[new] = self.norms[old];
                self.codes[new] = self.codes[old];
                self.keys.swap(new, old);
                if let Some(key) = &self.keys[new] {
                    self.slot_of.insert(key.clone(), new as u32);
                }
            }
        }
        let kept = live.len();
        self.data.truncate(kept * dim);
        self.norms.truncate(kept);
        self.codes.truncate(kept);
        self.keys.truncate(kept);
        self.alive.clear();
        self.alive.resize(kept, true);
        self.n = kept;
        self.dead = 0;
        live
    }
}

// =============================================================================
// The IVF index
// =============================================================================

pub(crate) struct Ivf {
    /// `nlist` unit vectors, row-major.
    centroids: Vec<f32>,
    nlist: usize,
    dim: usize,
    /// Cell per row slot; -1 = in no cell (deleted, or pending).
    assign: Vec<i32>,
    /// Row slots grouped by cell, ascending within a cell.
    order: Vec<u32>,
    /// `order[offsets[c]..offsets[c + 1]]` is cell `c`.
    offsets: Vec<usize>,
    /// Live rows with no cell yet; every query scores them exactly.
    pending: BTreeSet<u32>,
    churn: usize,
    trained_n: usize,
    build_seconds: f64,
    /// Per-query scratch, kept so a search allocates nothing in its hot path.
    cell_scores: Vec<f32>,
    cell_order: Vec<u32>,
    chosen: Vec<u32>,
}

impl Ivf {
    /// Train centroids on a sample, then give every live row its cell.
    fn build(cache: &ScanCache, nlist: usize, seed: u64, threads: usize) -> Ivf {
        let started = Instant::now();
        let (n, dim) = (cache.n, cache.dim);
        let live: Vec<u32> = (0..n as u32).filter(|&s| cache.alive[s as usize]).collect();
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
            let norm = cache.norms[slot as usize];
            let scale = if norm > 0.0 && norm.is_finite() {
                1.0 / norm
            } else {
                1.0
            };
            for (dst, src) in x[i * dim..(i + 1) * dim]
                .iter_mut()
                .zip(cache.row(slot as usize))
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
            nearest_parallel(&x, dim, &centroids, nlist, &mut cell, threads);
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
            &cache.data[..n * dim],
            dim,
            &centroids,
            nlist,
            &mut assign,
            threads,
        );
        for (a, alive) in assign.iter_mut().zip(&cache.alive) {
            if !alive {
                *a = -1;
            }
        }
        let mut index = Ivf {
            centroids,
            nlist,
            dim,
            assign,
            order: Vec::new(),
            offsets: Vec::new(),
            pending: BTreeSet::new(),
            churn: 0,
            trained_n: total,
            build_seconds: 0.0,
            cell_scores: vec![0.0; nlist],
            cell_order: Vec::new(),
            chosen: Vec::new(),
        };
        index.relist(n);
        index.build_seconds = started.elapsed().as_secs_f64();
        index
    }

    /// Regroup the rows by cell (a stable counting sort of `assign`).
    fn relist(&mut self, n: usize) {
        if self.assign.len() < n {
            self.assign.resize(n, -1);
        }
        let mut offsets = vec![0usize; self.nlist + 1];
        for &c in &self.assign[..n] {
            if c >= 0 {
                offsets[c as usize + 1] += 1;
            }
        }
        for c in 0..self.nlist {
            offsets[c + 1] += offsets[c];
        }
        let mut cursor = offsets.clone();
        let mut order = vec![0u32; offsets[self.nlist]];
        for (slot, &c) in self.assign[..n].iter().enumerate() {
            if c >= 0 {
                let at = &mut cursor[c as usize];
                order[*at] = slot as u32;
                *at += 1;
            }
        }
        self.order = order;
        self.offsets = offsets;
    }

    /// Give every pending row its nearest cell and regroup.
    fn fold(&mut self, cache: &ScanCache, threads: usize) {
        if !self.pending.is_empty() {
            let dim = self.dim;
            let slots: Vec<u32> = self.pending.iter().copied().collect();
            let mut gathered = Vec::with_capacity(slots.len() * dim);
            for &s in &slots {
                gathered.extend_from_slice(cache.row(s as usize));
            }
            let mut cells = vec![0i32; slots.len()];
            nearest_parallel(
                &gathered,
                dim,
                &self.centroids,
                self.nlist,
                &mut cells,
                threads,
            );
            for (&s, &c) in slots.iter().zip(&cells) {
                self.assign[s as usize] = c;
            }
            self.pending.clear();
        }
        self.relist(cache.n);
    }

    fn note_put(&mut self, slot: u32) {
        let s = slot as usize;
        if s >= self.assign.len() {
            self.assign.resize(s + 1, -1);
        }
        self.assign[s] = -1;
        self.pending.insert(slot);
        self.churn += 1;
    }

    fn note_drop(&mut self, slot: u32) {
        if let Some(a) = self.assign.get_mut(slot as usize) {
            *a = -1;
        }
        self.pending.remove(&slot);
        self.churn += 1;
    }

    /// Follow the cache's compaction (`live` = old slots kept, in order).
    fn compact(&mut self, live: &[u32]) {
        let moved: Vec<i32> = live.iter().map(|&old| self.assign[old as usize]).collect();
        self.assign = moved;
        self.pending = self
            .pending
            .iter()
            .filter_map(|old| live.binary_search(old).ok().map(|new| new as u32))
            .collect();
        self.relist(live.len());
    }

    fn nbytes(&self) -> usize {
        self.centroids.len() * 4
            + self.assign.len() * 4
            + self.order.len() * 4
            + self.offsets.len() * std::mem::size_of::<usize>()
    }

    /// The `lists` cells whose centroids are most similar to the query, into
    /// `self.chosen` (ascending, for memory locality).
    fn choose_cells(&mut self, q: &QueryVec, lists: usize) {
        self.chosen.clear();
        if lists >= self.nlist {
            self.chosen.extend(0..self.nlist as u32);
            return;
        }
        dots_into(q.v, &self.centroids, self.dim, &mut self.cell_scores);
        self.cell_order.clear();
        self.cell_order.extend(0..self.nlist as u32);
        let scores = &self.cell_scores;
        self.cell_order.select_nth_unstable_by(lists - 1, |&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then(a.cmp(&b))
        });
        self.chosen.extend_from_slice(&self.cell_order[..lists]);
        self.chosen.sort_unstable();
    }
}

// =============================================================================
// State kept behind the store's lock
// =============================================================================

#[derive(Default)]
pub(crate) struct ScanState {
    cache: Option<ScanCache>,
    ivf: Option<Ivf>,
    counts: AnnSearchCounts,
}

impl ScanState {
    /// Forget everything derived from the vectors (the counters stay).
    pub(crate) fn reset(&mut self) {
        self.cache = None;
        self.ivf = None;
    }

    /// A vector was written: patch the cache and the index if they exist.
    pub(crate) fn note_put(&mut self, key: &str, node_type: &str, vector: &[f32]) {
        let Some(cache) = self.cache.as_mut() else {
            return;
        };
        if cache.dim != vector.len() {
            self.reset();
            return;
        }
        let slot = cache.put(key, node_type, vector);
        if let Some(ivf) = self.ivf.as_mut() {
            ivf.note_put(slot);
        }
    }

    /// A vector was removed.
    pub(crate) fn note_remove(&mut self, key: &str) {
        let Some(cache) = self.cache.as_mut() else {
            return;
        };
        let Some(slot) = cache.remove(key) else {
            return;
        };
        if let Some(ivf) = self.ivf.as_mut() {
            ivf.note_drop(slot);
        }
        if cache.should_compact() {
            let live = cache.compact();
            if let Some(ivf) = self.ivf.as_mut() {
                ivf.compact(&live);
            }
        }
    }

    /// Build or maintain the index as the policy says; false = answer exactly.
    fn index_ready(&mut self, cfg: &AnnConfig, threads: usize) -> bool {
        let Some(cache) = self.cache.as_ref() else {
            return false;
        };
        if cache.live() < cfg.min_size.max(2) {
            self.ivf = None;
            return false;
        }
        if self
            .ivf
            .as_ref()
            .is_some_and(|i| i.churn as f64 > RETRAIN_FRACTION * i.trained_n as f64)
        {
            self.ivf = None;
        }
        match self.ivf.as_mut() {
            None => {
                let nlist = cfg.nlist.unwrap_or_else(|| auto_nlist(cache.live()));
                self.ivf = Some(Ivf::build(cache, nlist, cfg.seed, threads));
                self.counts.builds += 1;
            }
            Some(ivf) => {
                let fold_at = (FOLD_MIN as f64).max(FOLD_FRACTION * ivf.trained_n as f64);
                if ivf.pending.len() as f64 >= fold_at {
                    ivf.fold(cache, threads);
                }
            }
        }
        true
    }

    /// Rows scored from the probed cells, or `None` meaning "use the scan".
    ///
    /// `scale` 1 honours `nprobe` as configured (`nprobe >= nlist` is the
    /// scan). Later steps multiply it by 4 and give up on the index once they
    /// would touch half the cells, where scanning everything costs about the
    /// same.
    #[allow(clippy::too_many_arguments)]
    fn probe(
        &mut self,
        cfg: &AnnConfig,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
        scale: usize,
        threads: usize,
    ) -> Option<TopK> {
        let cache = self.cache.as_ref()?;
        let ivf = self.ivf.as_mut()?;
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
        ivf.choose_cells(q, lists);
        let cells = Cells {
            order: &ivf.order,
            offsets: &ivf.offsets,
            assign: &ivf.assign,
            pending: &ivf.pending,
        };
        Some(scan_probed(
            cache,
            cells,
            &ivf.chosen,
            q,
            code,
            threshold,
            k,
            threads,
        ))
    }

    /// Top `k` from the index, or `None` to fall back to the scan.
    fn approximate_topk(
        &mut self,
        cfg: &AnnConfig,
        threads: usize,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
    ) -> Option<Vec<Hit>> {
        if !q.has_direction() || !self.index_ready(cfg, threads) {
            return None;
        }
        let mut scale = 1usize;
        loop {
            let top = self.probe(cfg, q, code, threshold, k, scale, threads)?;
            if top.len() >= k {
                self.counts.approximate += 1;
                if scale > 1 {
                    self.counts.widened += 1;
                }
                return Some(top.into_sorted());
            }
            scale = scale.saturating_mul(4);
        }
    }

    /// Every row the probed cells yield (for a caller that filters further),
    /// best first; `None` to fall back to the scan.
    fn approximate_ranked(
        &mut self,
        cfg: &AnnConfig,
        threads: usize,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        scale: usize,
    ) -> Option<Vec<Hit>> {
        if !q.has_direction() || !self.index_ready(cfg, threads) {
            return None;
        }
        let top = self.probe(cfg, q, code, threshold, usize::MAX, scale, threads)?;
        self.counts.approximate += 1;
        if scale > 1 {
            self.counts.widened += 1;
        }
        Some(top.into_sorted())
    }
}

/// Score the probed cells (and the pending rows). The rows of a cell are
/// scattered through the matrix, so one thread is limited by how many cache
/// lines it can have in flight, not by bandwidth: a large probe is shared out
/// a few cells at a time. The answer does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
fn scan_probed(
    cache: &ScanCache,
    cells: Cells,
    chosen: &[u32],
    q: &QueryVec,
    code: Option<u32>,
    threshold: f32,
    k: usize,
    configured_threads: usize,
) -> TopK {
    const CELLS_PER_TASK: usize = 4;
    let rows = cache.rows();
    let candidates: usize = chosen
        .iter()
        .map(|&c| cells.offsets[c as usize + 1] - cells.offsets[c as usize])
        .sum::<usize>()
        + cells.pending.len();
    let tasks = chosen.len().div_ceil(CELLS_PER_TASK);
    let threads = if configured_threads == 0 {
        workers(0, tasks, SCAN_AUTO_THREADS)
            .min((candidates * cache.dim / PAR_MIN_CELL_FLOATS).max(1))
    } else {
        workers(configured_threads, tasks, SCAN_AUTO_THREADS)
    };
    let mut top = TopK::new(k, threshold, cache.live());
    if threads <= 1 {
        scan_cells(rows, cells, chosen, true, q, code, &mut top);
        return top;
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<TopK> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut local = TopK::new(k, threshold, cache.live());
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
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("scan worker panicked"))
            .collect()
    });
    for part in parts {
        top.merge(part);
    }
    top
}

fn effective_nprobe(cfg: &AnnConfig, nlist: usize) -> usize {
    cfg.nprobe
        .unwrap_or_else(|| (round_half_even(0.05 * nlist as f64) as usize).max(1))
        .min(nlist)
}

/// The full scan: every live row, spread over threads for a big store.
fn exact_hits(
    cache: &ScanCache,
    q: &QueryVec,
    code: Option<u32>,
    threshold: f32,
    k: usize,
    configured_threads: usize,
) -> Vec<Hit> {
    let (n, dim) = (cache.n, cache.dim);
    let rows = cache.rows();
    let block = (BLOCK_FLOATS / dim.max(1)).max(64);
    let blocks = n.div_ceil(block);
    let threads = if configured_threads == 0 {
        workers(0, blocks, SCAN_AUTO_THREADS).min((n * dim / PAR_MIN_FLOATS).max(1))
    } else {
        workers(configured_threads, blocks, SCAN_AUTO_THREADS)
    };
    if threads <= 1 {
        let mut top = TopK::new(k, threshold, cache.live());
        scan_range(rows, q, code, 0, n, &mut top);
        return top.into_sorted();
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<TopK> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut top = TopK::new(k, threshold, cache.live());
                    loop {
                        let b = next.fetch_add(1, Ordering::Relaxed);
                        if b >= blocks {
                            break;
                        }
                        scan_range(rows, q, code, b * block, ((b + 1) * block).min(n), &mut top);
                    }
                    top
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("scan worker panicked"))
            .collect()
    });
    let mut parts = parts.into_iter();
    let mut merged = parts.next().expect("at least one worker");
    for part in parts {
        merged.merge(part);
    }
    merged.into_sorted()
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
        self.scan_state_mut().ivf = None;
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

    /// Threads a full scan may use (0, the default, picks automatically from
    /// the store's size; 1 never spawns). The answer does not depend on it.
    pub fn set_scan_threads(&mut self, threads: usize) {
        self.scan_threads = threads;
    }

    /// The configured scan thread count (0 = automatic).
    pub fn scan_threads(&self) -> usize {
        self.scan_threads
    }

    /// Lock the derived search state. A thread that panicked while holding it
    /// may have left it half-patched, so a poisoned lock rebuilds from scratch.
    fn scan_state(&self) -> MutexGuard<'_, ScanState> {
        self.scan.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            guard.reset();
            self.scan.clear_poison();
            guard
        })
    }

    pub(crate) fn scan_state_mut(&mut self) -> &mut ScanState {
        if self.scan.is_poisoned() {
            self.scan.clear_poison();
            self.scan
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
        }
        self.scan.get_mut().unwrap_or_else(|e| e.into_inner())
    }

    /// Build the cache on first use.
    fn ensure_cache<'s>(&self, state: &'s mut ScanState) -> &'s mut ScanCache {
        state
            .cache
            .get_or_insert_with(|| ScanCache::build(&self.embeddings, self.dimension.unwrap_or(0)))
    }

    fn resolve_hit(&self, hit: Hit) -> Option<SimilarityResult> {
        let state = self.scan_state();
        let key = state
            .cache
            .as_ref()?
            .keys
            .get(hit.slot as usize)?
            .as_ref()?;
        let record = self.embeddings.get(&**key)?;
        Some(SimilarityResult {
            node_id: record.node_id.clone(),
            score: hit.score,
            text: record.text.clone(),
        })
    }

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
        self.check_query(query_vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut guard = self.scan_state();
        let state = &mut *guard;
        self.ensure_cache(state);
        let cache = state.cache.as_ref().expect("cache was just built");
        let live = cache.live();
        let code = match node_type {
            None => None,
            Some(t) => match cache.code_of.get(t) {
                Some(&code) => Some(code),
                None => return Ok(Vec::new()),
            },
        };
        if live == 0 {
            return Ok(Vec::new());
        }
        let q = QueryVec::new(query_vector);

        let mut hits = None;
        if let Some(cfg) = &self.ann {
            if exact || top_k >= live {
                state.counts.exact += 1;
            } else {
                hits = state.approximate_topk(cfg, self.scan_threads, &q, code, threshold, top_k);
                if hits.is_none() {
                    state.counts.exact += 1;
                }
            }
        }
        let cache = state.cache.as_ref().expect("cache was just built");
        let hits = hits
            .unwrap_or_else(|| exact_hits(cache, &q, code, threshold, top_k, self.scan_threads));
        Ok(self.materialise(cache, &hits))
    }

    fn materialise(&self, cache: &ScanCache, hits: &[Hit]) -> Vec<SimilarityResult> {
        hits.iter()
            .filter_map(|hit| {
                let key = cache.keys[hit.slot as usize].as_ref()?;
                let record = self.embeddings.get(&**key)?;
                Some(SimilarityResult {
                    node_id: record.node_id.clone(),
                    score: hit.score,
                    text: record.text.clone(),
                })
            })
            .collect()
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
        let mut guard = self.scan_state();
        let state = &mut *guard;
        self.ensure_cache(state);
        let cache = state.cache.as_ref().expect("cache was just built");
        let code = match node_type {
            None => None,
            Some(t) => match cache.code_of.get(t) {
                Some(&code) => Some(code),
                None => return Ok((empty(), false)),
            },
        };
        if cache.live() == 0 {
            return Ok((empty(), false));
        }
        let q = QueryVec::new(query_vector);
        let mut approximate = None;
        if let (Some(cfg), Some(scale)) = (&self.ann, scale) {
            approximate =
                state.approximate_ranked(cfg, self.scan_threads, &q, code, threshold, scale);
            if approximate.is_none() {
                state.counts.exact += 1;
            }
        }
        let was_approximate = approximate.is_some();
        let cache = state.cache.as_ref().expect("cache was just built");
        let hits = approximate.unwrap_or_else(|| {
            exact_hits(cache, &q, code, threshold, usize::MAX, self.scan_threads)
        });
        Ok((
            RankedHits {
                store: self,
                hits: hits.into_iter(),
            },
            was_approximate,
        ))
    }

    /// Cosine score of every stored vector, keyed by `type:id`, from the scan
    /// cache.
    pub(crate) fn scan_scores(
        &self,
        query_vector: &[f32],
    ) -> Result<HashMap<String, f32>, EmbeddingError> {
        self.check_query(query_vector)?;
        let mut guard = self.scan_state();
        let state = &mut *guard;
        self.ensure_cache(state);
        let cache = state.cache.as_ref().expect("cache was just built");
        let q = QueryVec::new(query_vector);
        let mut scores = HashMap::with_capacity(cache.live());
        for slot in 0..cache.n {
            if let Some(key) = &cache.keys[slot] {
                let norm = cache.norms[slot];
                let denom = norm * q.norm;
                let dot = dot_f32(q.v, cache.row(slot));
                let score = if denom == 0.0 { 0.0 } else { dot / denom };
                scores.insert(key.to_string(), score);
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
    /// is smaller than `min_size`.
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
        {
            let mut guard = self.scan_state();
            let state = &mut *guard;
            self.ensure_cache(state);
            if force {
                state.ivf = None;
            }
            if state.index_ready(cfg, self.scan_threads) {
                let cache = state.cache.as_ref().expect("cache was just built");
                if let Some(ivf) = state.ivf.as_mut() {
                    if !ivf.pending.is_empty() {
                        ivf.fold(cache, self.scan_threads);
                    }
                }
            }
        }
        Ok(self.ann_stats())
    }

    /// What the approximate index looks like right now. Never builds anything.
    pub fn ann_stats(&self) -> AnnStats {
        let guard = self.scan_state();
        let ivf = guard.ivf.as_ref();
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
            searches: guard.counts,
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
            ann_ms: 1000.0 * median(ann_times),
            exact_ms: 1000.0 * median(exact_times),
        })
    }
}

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
            let mut serial = vec![0i32; m];
            nearest_parallel(&pts, dim, &cents, nlist, &mut serial, 1);
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
