//! Opt-in int8 scalar-quantised scan with an exact f32 rerank.
//!
//! This is a child module of `ann` (it reuses the kernels, `TopK`, `Quad` and
//! the pool wrapper without widening their visibility). `DESIGN.md` section 11
//! is the specification.
//!
//! # What is stored
//!
//! With [`ScanPrecision::Int8`] the table keeps, beside every f32 row, an int8
//! copy of it and a per-row scale: `scale = max|x| / 127`, `q = round(x /
//! scale)` clamped to +-127 (symmetric, no zero point; an all-zero or
//! non-finite row has scale 0 and codes 0). The copy lives in chunks that
//! mirror the f32 matrix chunk for chunk, so a write, an update, a delete and a
//! compaction maintain it in the same place and at the same moment as the f32
//! row: it can never be stale. Cost: `dim` bytes + 4 per row = 0.25x the f32
//! matrix (+ 4 bytes/row).
//!
//! # How a search uses it
//!
//! 1. The query is quantised the same way.
//! 2. The scan (full scans and IVF probed cells alike) scores rows with an
//!    **integer** dot product `sum(q_i * r_i)` (exact in i32, so it does not
//!    depend on summation order, lane count, kernel or thread) and turns it
//!    into an approximate cosine `dot * row_scale * query_scale / (|row| *
//!    |query|)` using the exact f32 norms (a tombstone's NaN norm still drops
//!    it).
//! 3. The best `k * rerank_factor` rows by approximate score are kept.
//! 4. Those rows are re-scored with the exact f32 kernel (the same one the
//!    f32 scan uses, same summation order) and the best `k` of them are
//!    returned. **Returned scores are exact cosines**; only the choice of
//!    candidates is approximate, so a result differs from the f32 scan only by
//!    a true top-k row that ranked below `k * rerank_factor` by approximate
//!    score.
//!
//! Because step 2 is integer arithmetic and step 3 selects by the total order
//! (score, slot), the candidate set depends on neither threading, nor tiling,
//! nor batching: **a batch answer is bit-identical to the one-query answer**.
//!
//! When `k * rerank_factor` is not smaller than the number of rows that would
//! be scored (a huge `k`, a small store, the graph's unbounded filtered search)
//! the int8 pass could not skip anything, so the f32 path answers.

#[cfg(target_arch = "x86_64")]
use super::has_avx2_fma; // named by the `dispatch!` expansions below
use super::{
    cosine, Cells, Hit, Ivf, Par, Quad, QueryVec, TopK, BATCH_TILE, BLOCK_ROWS,
    PAR_MIN_CELL_FLOATS, PAR_MIN_FLOATS, SCAN_AUTO_THREADS,
};
use crate::table::{Rows, VectorTable, CHUNK_MASK, CHUNK_ROWS, CHUNK_SHIFT};
use crate::{AnnConfig, EmbeddingError, EmbeddingStore};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The rerank factor `ScanPrecision::int8()` uses.
pub const DEFAULT_RERANK_FACTOR: usize = 4;

/// First-pass candidates are kept down to `threshold - THRESHOLD_SLACK`, so a
/// row whose exact score passes the threshold is not lost to quantisation noise
/// (the noise is about 0.01 for unit vectors at 384-d; the exact threshold is
/// applied at the rerank).
const THRESHOLD_SLACK: f32 = 0.05;

/// How the scan scores rows. Set with [`EmbeddingStore::set_scan_precision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScanPrecision {
    /// Score every scanned row with the exact f32 kernel (the default).
    #[default]
    F32,
    /// First-pass scan on a per-row int8 copy, then re-score the best
    /// `k * rerank_factor` rows in f32. Adds 0.25x the raw vector memory.
    Int8 {
        /// Candidates kept per requested result (>= 1).
        rerank_factor: usize,
    },
}

impl ScanPrecision {
    /// `Int8` with [`DEFAULT_RERANK_FACTOR`].
    pub fn int8() -> Self {
        ScanPrecision::Int8 {
            rerank_factor: DEFAULT_RERANK_FACTOR,
        }
    }
}

// =============================================================================
// Quantisation
// =============================================================================

/// Quantise `x` into `out` (same length) and return the scale. Deterministic:
/// the table's copy and a query use this one function.
pub(crate) fn quantize(x: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), out.len());
    let mut m = 0.0f32;
    for &v in x {
        m = m.max(v.abs()); // f32::max ignores a NaN operand
    }
    if m.is_nan() || m <= 0.0 || !m.is_finite() {
        out.fill(0);
        return 0.0;
    }
    let inv = 127.0 / m;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    m / 127.0
}

/// A quantised query: signed codes, their absolute values (the unsigned
/// operand of the AVX2 kernel) and the scale.
pub(crate) struct QQuery {
    q: Vec<i8>,
    abs: Vec<i8>,
    scale: f32,
}

impl QQuery {
    fn new(v: &[f32]) -> Self {
        let mut q = vec![0i8; v.len()];
        let scale = quantize(v, &mut q);
        let abs = q.iter().map(|c| c.abs()).collect();
        Self { q, abs, scale }
    }
}

// =============================================================================
// The int8 copy
// =============================================================================

/// Int8 rows and scales, slot for slot with the table's f32 matrix.
pub(crate) struct QuantRows {
    chunks: Vec<Vec<i8>>,
    scales: Vec<f32>,
    pub rerank_factor: usize,
}

/// A read-only view for scan kernels.
#[derive(Clone, Copy)]
pub(crate) struct QRows<'a> {
    chunks: &'a [Vec<i8>],
    scales: &'a [f32],
    dim: usize,
}

impl<'a> QRows<'a> {
    #[inline(always)]
    fn row(&self, slot: usize) -> &'a [i8] {
        let chunk: &'a Vec<i8> = &self.chunks[slot >> CHUNK_SHIFT];
        let at = (slot & CHUNK_MASK) * self.dim;
        &chunk[at..at + self.dim]
    }

    #[inline(always)]
    fn run(&self, slot: usize, hi: usize) -> (&'a [i8], usize) {
        let count = (CHUNK_ROWS - (slot & CHUNK_MASK)).min(hi - slot);
        let at = (slot & CHUNK_MASK) * self.dim;
        let chunk: &'a Vec<i8> = &self.chunks[slot >> CHUNK_SHIFT];
        (&chunk[at..at + count * self.dim], count)
    }
}

impl QuantRows {
    pub fn new(rerank_factor: usize) -> Self {
        Self {
            chunks: Vec::new(),
            scales: Vec::new(),
            rerank_factor,
        }
    }

    pub fn clear(&mut self) {
        self.chunks = Vec::new();
        self.scales = Vec::new();
    }

    fn view(&self, dim: usize) -> QRows<'_> {
        QRows {
            chunks: &self.chunks,
            scales: &self.scales,
            dim,
        }
    }

    /// Quantise every row of the f32 matrix (tombstones included, so slots
    /// line up).
    pub fn rebuild(&mut self, rows: &Rows, n: usize) {
        self.clear();
        for slot in 0..n {
            self.push(slot, rows.row(slot));
        }
    }

    /// Append the row for `slot` (which must be the next slot).
    pub fn push(&mut self, slot: usize, vector: &[f32]) {
        debug_assert_eq!(slot, self.scales.len());
        if slot & CHUNK_MASK == 0 {
            self.chunks.push(if self.chunks.is_empty() {
                Vec::new()
            } else {
                Vec::with_capacity(CHUNK_ROWS * vector.len())
            });
        }
        let chunk = self.chunks.last_mut().expect("a chunk exists");
        let at = chunk.len();
        chunk.resize(at + vector.len(), 0);
        let scale = quantize(vector, &mut chunk[at..]);
        self.scales.push(scale);
    }

    /// Overwrite the row for an existing `slot`.
    pub fn set(&mut self, slot: usize, vector: &[f32]) {
        let dim = vector.len();
        let at = (slot & CHUNK_MASK) * dim;
        self.scales[slot] = quantize(vector, &mut self.chunks[slot >> CHUNK_SHIFT][at..at + dim]);
    }

    /// Follow the table's order-preserving compaction (`live` = old slot of
    /// every survivor).
    pub fn compact(&mut self, live: &[u32], dim: usize) {
        for (new, &old) in live.iter().enumerate() {
            let old = old as usize;
            if new == old {
                continue;
            }
            let (oc, oa) = (old >> CHUNK_SHIFT, (old & CHUNK_MASK) * dim);
            let (nc, na) = (new >> CHUNK_SHIFT, (new & CHUNK_MASK) * dim);
            if oc == nc {
                self.chunks[oc].copy_within(oa..oa + dim, na);
            } else {
                let (head, tail) = self.chunks.split_at_mut(oc);
                head[nc][na..na + dim].copy_from_slice(&tail[0][oa..oa + dim]);
            }
            self.scales[new] = self.scales[old];
        }
        let kept = live.len();
        self.scales.truncate(kept);
        let chunks = kept.div_ceil(CHUNK_ROWS);
        self.chunks.truncate(chunks);
        if let Some(last) = self.chunks.last_mut() {
            last.truncate((kept - (chunks - 1) * CHUNK_ROWS) * dim);
        }
    }

    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity()).sum::<usize>() + self.scales.capacity() * 4
    }

    /// The copy agrees with the f32 matrix: same shape, and every row (dead
    /// ones too) is exactly what `quantize` makes of its f32 row.
    pub fn check(&self, rows: &Rows, n: usize) -> Result<(), String> {
        if self.scales.len() != n {
            return Err(format!(
                "int8 copy has {} scales for {n} rows",
                self.scales.len()
            ));
        }
        if self.chunks.len() != n.div_ceil(CHUNK_ROWS) {
            return Err(format!(
                "int8 copy has {} chunks for {n} rows",
                self.chunks.len()
            ));
        }
        let total: usize = self.chunks.iter().map(|c| c.len()).sum();
        if total != n * rows.dim {
            return Err(format!(
                "int8 copy holds {total} codes, expected {}",
                n * rows.dim
            ));
        }
        let view = self.view(rows.dim);
        let mut fresh = vec![0i8; rows.dim];
        for slot in 0..n {
            let scale = quantize(rows.row(slot), &mut fresh);
            if scale.to_bits() != self.scales[slot].to_bits() || view.row(slot) != &fresh[..] {
                return Err(format!(
                    "int8 row {slot} is not the quantisation of its f32 row"
                ));
            }
        }
        Ok(())
    }
}

// =============================================================================
// Kernels: integer dot products
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86q {
    use std::arch::x86_64::*;

    /// `R` int8 rows against one query: `sum(q_i * r_i)` per row, exact.
    ///
    /// `sign(r, q) * |q|` is `r * q` with the unsigned operand `maddubs`
    /// wants; a pair sum is at most 2 * 127 * 127 < 2^15, so the 16-bit step
    /// cannot saturate (codes are clamped to +-127, never -128).
    ///
    /// # Safety
    /// The CPU must support AVX2; `q`, `abs` and every row must be `dim` long.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn dots<const R: usize>(
        q: *const i8,
        abs: *const i8,
        rows: [*const i8; R],
        dim: usize,
    ) -> [i32; R] {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); R];
        let body = dim & !31;
        let mut i = 0;
        while i < body {
            let qv = _mm256_loadu_si256(q.add(i) as *const __m256i);
            let av = _mm256_loadu_si256(abs.add(i) as *const __m256i);
            for j in 0..R {
                let rv = _mm256_loadu_si256(rows[j].add(i) as *const __m256i);
                let pairs = _mm256_maddubs_epi16(av, _mm256_sign_epi8(rv, qv));
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(pairs, ones));
            }
            i += 32;
        }
        let mut out = [0i32; R];
        for j in 0..R {
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(acc[j]),
                _mm256_extracti128_si256(acc[j], 1),
            );
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            let mut sum = _mm_cvtsi128_si32(s);
            let mut t = body;
            while t < dim {
                sum += *q.add(t) as i32 * *rows[j].add(t) as i32;
                t += 1;
            }
            out[j] = sum;
        }
        out
    }
}

/// The portable integer dot (also the specification the AVX2 kernel is tested
/// against).
#[inline(always)]
pub(crate) fn dot_i8_portable(q: &[i8], r: &[i8]) -> i32 {
    q.iter().zip(r).map(|(a, b)| *a as i32 * *b as i32).sum()
}

/// `R` rows against one quantised query. `F` selects the AVX2 kernel; the
/// `true` instantiation is only reached through a `dispatch!` wrapper that
/// checked the CPU.
#[inline(always)]
fn dots<const R: usize, const F: bool>(qq: &QQuery, rows: [&[i8]; R]) -> [i32; R] {
    let dim = qq.q.len();
    #[cfg(target_arch = "x86_64")]
    if F {
        // the kernel reads `dim` bytes from every operand: make that a fact
        assert!(qq.abs.len() == dim && rows.iter().all(|r| r.len() == dim));
        // SAFETY: `F` is true only in the instantiation reached through a
        // `dispatch!` wrapper, which has checked for AVX2 (and FMA); the
        // assert above guarantees every pointer has `dim` readable bytes.
        return unsafe {
            x86q::dots::<R>(
                qq.q.as_ptr(),
                qq.abs.as_ptr(),
                std::array::from_fn(|j| rows[j].as_ptr()),
                dim,
            )
        };
    }
    std::array::from_fn(|j| dot_i8_portable(&qq.q, rows[j]))
}

/// Int8 dot of two vectors by whichever kernel the CPU has (for tests).
#[cfg(test)]
pub(crate) fn dot_i8(q: &[i8], r: &[i8]) -> i32 {
    let qq = QQuery {
        q: q.to_vec(),
        abs: q.iter().map(|c| c.abs()).collect(),
        scale: 1.0,
    };
    dot_i8_dispatch(&qq, r)
}

#[cfg(test)]
dispatch! {
    fn dot_i8_dispatch(qq: &QQuery, r: &[i8]) -> i32 = dot_one_impl
}

#[cfg(test)]
#[inline(always)]
fn dot_one_impl<const F: bool>(qq: &QQuery, r: &[i8]) -> i32 {
    dots::<1, F>(qq, [r])[0]
}

/// Approximate cosine of a row from its integer dot.
#[inline(always)]
fn approx(d: i32, row_scale: f32, q_scale: f32, row_norm: f32, q_norm: f32) -> f32 {
    cosine(d as f32 * row_scale * q_scale, row_norm, q_norm)
}

/// Everything a first-pass kernel reads.
#[derive(Clone, Copy)]
struct View<'a> {
    rows: Rows<'a>,
    q: QRows<'a>,
}

/// Four-at-a-time batching of int8 rows (the f32 `Quad`'s counterpart).
struct Quad8<'a> {
    slots: [usize; 4],
    rows: [&'a [i8]; 4],
    m: usize,
}

impl<'a> Quad8<'a> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            slots: [0; 4],
            rows: [&[]; 4],
            m: 0,
        }
    }

    #[inline(always)]
    fn push<const F: bool>(
        &mut self,
        v: &View,
        qq: &QQuery,
        q: &QueryVec,
        slot: usize,
        row: &'a [i8],
        top: &mut TopK,
    ) {
        self.slots[self.m] = slot;
        self.rows[self.m] = row;
        self.m += 1;
        if self.m == 4 {
            self.m = 0;
            let d = dots::<4, F>(qq, self.rows);
            for (&s, &dot) in self.slots.iter().zip(&d) {
                top.offer(
                    approx(dot, v.q.scales[s], qq.scale, v.rows.norms[s], q.norm),
                    s as u32,
                );
            }
        }
    }

    #[inline(always)]
    fn finish<const F: bool>(&mut self, v: &View, qq: &QQuery, q: &QueryVec, top: &mut TopK) {
        for i in 0..self.m {
            let s = self.slots[i];
            let d = dots::<1, F>(qq, [self.rows[i]])[0];
            top.offer(
                approx(d, v.q.scales[s], qq.scale, v.rows.norms[s], q.norm),
                s as u32,
            );
        }
        self.m = 0;
    }
}

/// First pass over rows `lo..hi`.
#[inline(always)]
fn scan_range_impl<const F: bool>(
    v: View,
    qq: &QQuery,
    q: &QueryVec,
    code: Option<u32>,
    lo: usize,
    hi: usize,
    top: &mut TopK,
) {
    let mut quad = Quad8::new();
    let dim = v.q.dim;
    let mut at = lo;
    while at < hi {
        let (flat, count) = v.q.run(at, hi);
        for i in 0..count {
            let slot = at + i;
            if let Some(code) = code {
                if v.rows.codes[slot] != code {
                    continue;
                }
            }
            quad.push::<F>(&v, qq, q, slot, &flat[i * dim..(i + 1) * dim], top);
        }
        at += count;
    }
    quad.finish::<F>(&v, qq, q, top);
}

dispatch! {
    fn scan_range(v: View, qq: &QQuery, q: &QueryVec, code: Option<u32>, lo: usize, hi: usize, top: &mut TopK) = scan_range_impl
}

/// First pass over the rows of the `chosen` cells (and the pending rows).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scan_cells_impl<const F: bool>(
    v: View,
    cells: Cells,
    chosen: &[u32],
    with_pending: bool,
    qq: &QQuery,
    q: &QueryVec,
    code: Option<u32>,
    top: &mut TopK,
) {
    let mut quad = Quad8::new();
    let mut visit = |slot: u32| {
        let slot = slot as usize;
        if let Some(code) = code {
            if v.rows.codes[slot] != code {
                return;
            }
        }
        quad.push::<F>(&v, qq, q, slot, v.q.row(slot), top);
    };
    for &c in chosen {
        for &slot in &cells.lists[c as usize] {
            visit(slot);
        }
    }
    if with_pending {
        for &slot in cells.pending {
            visit(slot);
        }
    }
    quad.finish::<F>(&v, qq, q, top);
}

dispatch! {
    #[allow(clippy::too_many_arguments)]
    fn scan_cells(v: View, cells: Cells, chosen: &[u32], with_pending: bool, qq: &QQuery, q: &QueryVec, code: Option<u32>, top: &mut TopK) = scan_cells_impl
}

/// First pass over an explicit list of slots (a cell's rows).
#[inline(always)]
fn scan_slots_impl<const F: bool>(
    v: View,
    slots: &[u32],
    qq: &QQuery,
    q: &QueryVec,
    top: &mut TopK,
) {
    let mut quad = Quad8::new();
    for &slot in slots {
        quad.push::<F>(&v, qq, q, slot as usize, v.q.row(slot as usize), top);
    }
    quad.finish::<F>(&v, qq, q, top);
}

dispatch! {
    fn scan_slots(v: View, slots: &[u32], qq: &QQuery, q: &QueryVec, top: &mut TopK) = scan_slots_impl
}

/// Exact f32 scores of `slots`, offered to `top`.
#[inline(always)]
fn rerank_impl<const F: bool>(rows: Rows, q: &QueryVec, slots: &[u32], top: &mut TopK) {
    let mut quad = Quad::new();
    let mut sink = |slot: usize, score: f32| top.offer(score, slot as u32);
    for &slot in slots {
        quad.push::<F>(&rows, q, slot as usize, &mut sink);
    }
    quad.finish::<F>(&rows, q, &mut sink);
}

dispatch! {
    fn rerank(rows: Rows, q: &QueryVec, slots: &[u32], top: &mut TopK) = rerank_impl
}

/// The exact top `k` of a first-pass candidate heap.
fn rerank_candidates(rows: Rows, q: &QueryVec, cand: TopK, threshold: f32, k: usize) -> TopK {
    let mut slots: Vec<u32> = cand.into_sorted().into_iter().map(|h| h.slot).collect();
    slots.sort_unstable(); // walk the matrix in order
    let mut top = TopK::new(k, threshold, slots.len());
    rerank(rows, q, &slots, &mut top);
    top
}

// =============================================================================
// Searches (called from the hooks in ann.rs)
// =============================================================================

/// First-pass scan of a probed set of cells, then the exact rerank. `None`
/// when int8 cannot skip any work here (the caller scores in f32).
#[allow(clippy::too_many_arguments)]
pub(super) fn scan_probed(
    table: &VectorTable,
    cells: Cells,
    chosen: &[u32],
    q: &QueryVec,
    code: Option<u32>,
    threshold: f32,
    k: usize,
    par: Par,
) -> Option<TopK> {
    const CELLS_PER_TASK: usize = 4;
    let qr = table.quant()?;
    if !q.has_direction() {
        return None;
    }
    let candidates: usize = chosen
        .iter()
        .map(|&c| cells.lists[c as usize].len())
        .sum::<usize>()
        + cells.pending.len();
    let kc = k.saturating_mul(qr.rerank_factor);
    if kc >= candidates {
        return None;
    }
    let dim = table.dim;
    let view = View {
        rows: table.rows(),
        q: qr.view(dim),
    };
    let qq = QQuery::new(q.v);
    let live = table.live();
    let bar = threshold - THRESHOLD_SLACK;
    let tasks = chosen.len().div_ceil(CELLS_PER_TASK);
    let threads = if par.configured == 0 {
        par.workers(tasks, SCAN_AUTO_THREADS)
            .min((candidates * dim / PAR_MIN_CELL_FLOATS).max(1))
    } else {
        par.workers(tasks, SCAN_AUTO_THREADS)
    };
    let mut top = TopK::new(kc, bar, live);
    if threads <= 1 {
        scan_cells(view, cells, chosen, true, &qq, q, code, &mut top);
    } else {
        let next = AtomicUsize::new(0);
        let parts = Mutex::new(Vec::with_capacity(threads));
        par.run(threads, &|_| {
            let mut local = TopK::new(kc, bar, live);
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= tasks {
                    break;
                }
                let lo = t * CELLS_PER_TASK;
                let hi = (lo + CELLS_PER_TASK).min(chosen.len());
                scan_cells(
                    view,
                    cells,
                    &chosen[lo..hi],
                    t == 0,
                    &qq,
                    q,
                    code,
                    &mut local,
                );
            }
            parts.lock().unwrap_or_else(|e| e.into_inner()).push(local);
        });
        for part in parts.into_inner().unwrap_or_else(|e| e.into_inner()) {
            top.merge(part);
        }
    }
    Some(rerank_candidates(view.rows, q, top, threshold, k))
}

impl EmbeddingStore {
    /// Choose how scans score rows. `Int8` builds an int8 copy of every row
    /// now (one pass over the matrix) and keeps it current through every
    /// write; `F32` drops it. Searches are unchanged in what they return
    /// (exact cosines) but, with `Int8`, a bounded search may miss a true
    /// top-`k` row that ranks below `k * rerank_factor` by the int8 score:
    /// measure with [`int8_recall_report`](Self::int8_recall_report).
    ///
    /// # Errors
    /// `InvalidOption` for a rerank factor of 0.
    pub fn set_scan_precision(&mut self, precision: ScanPrecision) -> Result<(), EmbeddingError> {
        match precision {
            ScanPrecision::F32 => self.table.set_quant(None),
            ScanPrecision::Int8 { rerank_factor } => {
                if rerank_factor == 0 {
                    return Err(EmbeddingError::InvalidOption(
                        "int8 rerank_factor must be at least 1".to_string(),
                    ));
                }
                self.table.set_quant(Some(rerank_factor));
            }
        }
        Ok(())
    }

    /// The current scan precision.
    pub fn scan_precision(&self) -> ScanPrecision {
        match self.table.quant() {
            None => ScanPrecision::F32,
            Some(q) => ScanPrecision::Int8 {
                rerank_factor: q.rerank_factor,
            },
        }
    }

    /// Bytes held by the int8 copy (0 under `F32`).
    pub fn quant_bytes(&self) -> usize {
        self.table.quant().map_or(0, QuantRows::bytes)
    }

    /// Full scan with the int8 first pass and exact rerank, for a given
    /// `factor`. `None` when it cannot apply (no int8 copy, a query with no
    /// direction, or `k * factor` not smaller than the live rows).
    pub(super) fn q8_exact_hits(
        &self,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
        factor: usize,
    ) -> Option<Vec<Hit>> {
        let table = &self.table;
        let qr = table.quant()?;
        let live = table.live();
        let kc = k.saturating_mul(factor);
        if kc >= live || !q.has_direction() {
            return None;
        }
        let (n, dim) = (table.slots(), table.dim);
        let view = View {
            rows: table.rows(),
            q: qr.view(dim),
        };
        let qq = QQuery::new(q.v);
        let bar = threshold - THRESHOLD_SLACK;
        let blocks = n.div_ceil(BLOCK_ROWS);
        let par = self.par();
        let threads = if par.configured == 0 {
            par.workers(blocks, SCAN_AUTO_THREADS)
                .min((n * dim / PAR_MIN_FLOATS).max(1))
        } else {
            par.workers(blocks, SCAN_AUTO_THREADS)
        };
        let cand = if threads <= 1 {
            let mut top = TopK::new(kc, bar, live);
            scan_range(view, &qq, q, code, 0, n, &mut top);
            top
        } else {
            let next = AtomicUsize::new(0);
            let parts = Mutex::new(Vec::with_capacity(threads));
            par.run(threads, &|_| {
                let mut top = TopK::new(kc, bar, live);
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= blocks {
                        break;
                    }
                    let (lo, hi) = (b * BLOCK_ROWS, ((b + 1) * BLOCK_ROWS).min(n));
                    scan_range(view, &qq, q, code, lo, hi, &mut top);
                }
                parts.lock().unwrap_or_else(|e| e.into_inner()).push(top);
            });
            let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
            let mut merged = parts.pop().expect("at least one participant");
            for part in parts {
                merged.merge(part);
            }
            merged
        };
        Some(rerank_candidates(view.rows, q, cand, threshold, k).into_sorted())
    }

    /// The hook `exact_hits` calls first: int8 when it applies.
    pub(super) fn quant_exact(
        &self,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
    ) -> Option<Vec<Hit>> {
        let factor = self.table.quant()?.rerank_factor;
        self.q8_exact_hits(q, code, threshold, k, factor)
    }

    /// The hook `exact_batch_hits` calls first: queries x blocks tasks, a
    /// block of int8 rows (<= 512 KB at 512-d) staying in cache across a tile
    /// of 16 queries. `None` when int8 cannot apply.
    pub(super) fn quant_exact_batch(
        &self,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        k: usize,
    ) -> Option<Vec<Vec<Hit>>> {
        let table = &self.table;
        let qr = table.quant()?;
        let live = table.live();
        let kc = k.saturating_mul(qr.rerank_factor);
        if kc >= live || qs.is_empty() || !qs.iter().all(QueryVec::has_direction) {
            return None;
        }
        let (n, dim) = (table.slots(), table.dim);
        let view = View {
            rows: table.rows(),
            q: qr.view(dim),
        };
        let qqs: Vec<QQuery> = qs.iter().map(|q| QQuery::new(q.v)).collect();
        let bar = threshold - THRESHOLD_SLACK;
        let blocks = n.div_ceil(BLOCK_ROWS);
        let tiles = qs.len().div_ceil(BATCH_TILE);
        let total = blocks * tiles;
        let par = self.par();
        let threads = par.workers(total, usize::MAX);
        let next = AtomicUsize::new(0);
        let parts: Mutex<Vec<Vec<TopK>>> = Mutex::new(Vec::with_capacity(threads));
        par.run(threads, &|_| {
            let mut tops: Vec<TopK> = Vec::new();
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= total {
                    break;
                }
                if tops.is_empty() {
                    tops = (0..qs.len()).map(|_| TopK::new(kc, bar, live)).collect();
                }
                let (block, tile) = (t / tiles, t % tiles);
                let (lo, hi) = (block * BLOCK_ROWS, ((block + 1) * BLOCK_ROWS).min(n));
                let (q_lo, q_hi) = (tile * BATCH_TILE, ((tile + 1) * BATCH_TILE).min(qs.len()));
                for i in q_lo..q_hi {
                    scan_range(view, &qqs[i], &qs[i], code, lo, hi, &mut tops[i]);
                }
            }
            if !tops.is_empty() {
                parts.lock().unwrap_or_else(|e| e.into_inner()).push(tops);
            }
        });
        let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut merged = parts.pop()?;
        for part in parts {
            for (m, p) in merged.iter_mut().zip(part) {
                m.merge(p);
            }
        }
        // rerank every query exactly, in parallel
        let cands: Vec<Mutex<Option<TopK>>> =
            merged.into_iter().map(|t| Mutex::new(Some(t))).collect();
        let next = AtomicUsize::new(0);
        let done: Mutex<Vec<(usize, Vec<Hit>)>> = Mutex::new(Vec::with_capacity(qs.len()));
        par.run(par.workers(qs.len(), usize::MAX), &|_| {
            let mut local = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= qs.len() {
                    break;
                }
                let cand = cands[i]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                    .expect("each query is reranked once");
                local.push((
                    i,
                    rerank_candidates(view.rows, &qs[i], cand, threshold, k).into_sorted(),
                ));
            }
            done.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
        });
        let mut out: Vec<Vec<Hit>> = vec![Vec::new(); qs.len()];
        for (i, hits) in done.into_inner().unwrap_or_else(|e| e.into_inner()) {
            out[i] = hits;
        }
        Some(out)
    }

    /// The hook `batch_hits` calls before the f32 cell-major batch. Like it,
    /// this works **cell-major**: each probed cell's int8 rows are scored once
    /// against every query that probes the cell (the rows stay in cache), into
    /// per-query candidate heaps of `k * rerank_factor`; then each query's
    /// candidates are re-scored exactly. A query's candidate set is the top
    /// `kc` of exactly the rows the single-query probe scans, under the same
    /// total order, so the answer is identical to the single-query answer. A
    /// query that comes up short, or whose probe is too small for int8 to skip
    /// anything, is re-run on the single-query path. Returns whether it handled
    /// the batch (even by leaving `answers` empty for the exact scan).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quant_approximate_batch(
        &self,
        cfg: &AnnConfig,
        ivf: &Ivf,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        top_k: usize,
        answers: &mut [Option<Vec<Hit>>],
    ) -> bool {
        let table = &self.table;
        let Some(qr) = table.quant() else {
            return false;
        };
        let nprobe = super::effective_nprobe(cfg, ivf.nlist);
        if nprobe >= ivf.nlist {
            return true; // probing everything is the exact scan
        }
        let kc = top_k.saturating_mul(qr.rerank_factor);
        let live = table.live();
        let view = View {
            rows: table.rows(),
            q: qr.view(table.dim),
        };
        let bar = threshold - THRESHOLD_SLACK;
        let par = self.par();

        // 1. the cells each query probes (queries with no direction go to the
        //    exact scan, as in the f32 batch)
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
        // a probe with no more rows than `kc` cannot be thinned by int8: the
        // single-query path answers it in f32
        let mut single = Vec::new();
        for (i, cells) in chosen.iter_mut().enumerate() {
            if let Some(c) = cells {
                let rows: usize = c
                    .iter()
                    .map(|&c| ivf.lists[c as usize].len())
                    .sum::<usize>()
                    + ivf.pending.len();
                if kc >= rows {
                    single.push(i);
                    *cells = None;
                }
            }
        }

        // 2. invert: the queries of each cell (the extra last entry is the
        //    pending rows, which every query scores)
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
        let qqs: Vec<Option<QQuery>> = chosen
            .iter()
            .zip(qs)
            .map(|(c, q)| c.as_ref().map(|_| QQuery::new(q.v)))
            .collect();
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
                    tops = (0..qs.len()).map(|_| TopK::new(kc, bar, live)).collect();
                }
                let c = tasks[t];
                slots.clear();
                slots.extend(
                    members_of(c)
                        .iter()
                        .copied()
                        .filter(|&s| code.is_none_or(|code| view.rows.codes[s as usize] == code)),
                );
                for &i in &by_cell[c] {
                    let i = i as usize;
                    let qq = qqs[i]
                        .as_ref()
                        .expect("a probing query has a quantised copy");
                    scan_slots(view, &slots, qq, &qs[i], &mut tops[i]);
                }
            }
            if !tops.is_empty() {
                parts.lock().unwrap_or_else(|e| e.into_inner()).push(tops);
            }
        });
        let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut merged: Vec<TopK> = match parts.pop() {
            Some(first) => first,
            None => (0..qs.len()).map(|_| TopK::new(kc, bar, live)).collect(),
        };
        for part in parts {
            for (a, b) in merged.iter_mut().zip(part) {
                a.merge(b);
            }
        }

        // 4. exact rerank; the short ones go back through the single path
        for (i, cand) in merged.into_iter().enumerate() {
            if chosen[i].is_none() {
                continue;
            }
            let top = rerank_candidates(view.rows, &qs[i], cand, threshold, top_k);
            if top.len() >= top_k {
                self.counts.approximate.fetch_add(1, Ordering::Relaxed);
                answers[i] = Some(top.into_sorted());
            } else {
                single.push(i);
            }
        }
        let one_thread = Par {
            configured: 1,
            pool: &self.pool,
        };
        let next = AtomicUsize::new(0);
        let redone: Mutex<Vec<(usize, Option<Vec<Hit>>)>> = Mutex::new(Vec::new());
        par.run(par.workers(single.len(), usize::MAX), &|_| {
            let mut local = Vec::new();
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= single.len() {
                    break;
                }
                let i = single[j];
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
        true
    }

    /// Recall of the int8 scan against the exact f32 scan, at each of the
    /// given rerank factors, over `queries` (use real queries from your own
    /// workload; recall depends on the data).
    ///
    /// Full scans only (an IVF probe adds its own approximation; measure that
    /// with [`recall_check`](Self::recall_check)). For each factor the report
    /// gives the mean and worst per-query overlap between the int8 top `k` and
    /// the f32 top `k`, whether every returned score is bit-identical to the
    /// exact cosine of that row, and median per-query times.
    ///
    /// # Errors
    /// `InvalidOption` unless the store is on `ScanPrecision::Int8`, `k` is 0,
    /// or a factor is 0; a query of the wrong width.
    pub fn int8_recall_report<V: AsRef<[f32]>>(
        &self,
        queries: &[V],
        k: usize,
        node_type: Option<&str>,
        threshold: f32,
        rerank_factors: &[usize],
    ) -> Result<Vec<Int8Recall>, EmbeddingError> {
        if self.table.quant().is_none() {
            return Err(EmbeddingError::InvalidOption(
                "scan precision is not Int8; call set_scan_precision first".to_string(),
            ));
        }
        if k == 0 || rerank_factors.contains(&0) {
            return Err(EmbeddingError::InvalidOption(
                "k and every rerank factor must be positive".to_string(),
            ));
        }
        for q in queries {
            self.check_query(q.as_ref())?;
        }
        let Ok(code) = self.type_filter(node_type) else {
            return Ok(Vec::new());
        };
        let qvs: Vec<QueryVec> = queries.iter().map(|q| QueryVec::new(q.as_ref())).collect();
        let live = self.table.live();
        let mut truth: Vec<Vec<Hit>> = Vec::new();
        let mut f32_times = Vec::new();
        for q in &qvs {
            let started = Instant::now();
            truth.push(self.exact_hits_f32(q, code, threshold, k));
            f32_times.push(started.elapsed().as_secs_f64());
        }
        let f32_ms = 1000.0 * median(f32_times);
        let mut out = Vec::new();
        for &factor in rerank_factors {
            let (mut recalls, mut times) = (Vec::new(), Vec::new());
            let mut exact = true;
            for (q, truth) in qvs.iter().zip(&truth) {
                let started = Instant::now();
                let found = self
                    .q8_exact_hits(q, code, threshold, k, factor)
                    .unwrap_or_else(|| self.exact_hits_f32(q, code, threshold, k));
                times.push(started.elapsed().as_secs_f64());
                let wanted: std::collections::HashSet<u32> = truth.iter().map(|h| h.slot).collect();
                if !wanted.is_empty() {
                    let hit = found.iter().filter(|h| wanted.contains(&h.slot)).count();
                    recalls.push(hit as f64 / wanted.len() as f64);
                }
                for h in &found {
                    let slot = h.slot as usize;
                    let d = crate::ann::dot_f32(q.v, self.table.row(slot));
                    exact &=
                        cosine(d, self.table.norm(slot), q.norm).to_bits() == h.score.to_bits();
                }
            }
            out.push(Int8Recall {
                rerank_factor: factor,
                k,
                queries: qvs.len(),
                recall: if recalls.is_empty() {
                    1.0
                } else {
                    recalls.iter().sum::<f64>() / recalls.len() as f64
                },
                min_recall: recalls.iter().copied().fold(1.0, f64::min),
                scores_exact: exact,
                applies: k.saturating_mul(factor) < live,
                int8_ms: 1000.0 * median(times),
                f32_ms,
            });
        }
        Ok(out)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
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
}

/// Recall of the int8 scan at one rerank factor; see
/// [`EmbeddingStore::int8_recall_report`].
#[derive(Clone, Debug)]
pub struct Int8Recall {
    pub rerank_factor: usize,
    pub k: usize,
    pub queries: usize,
    /// Mean over queries of `|int8 top-k & f32 top-k| / |f32 top-k|`.
    pub recall: f64,
    /// The worst single query.
    pub min_recall: f64,
    /// Every returned score was bit-identical to the exact f32 cosine of its
    /// row (true by construction; this checks it).
    pub scores_exact: bool,
    /// `k * rerank_factor` is below the row count, so the int8 pass really
    /// ran (otherwise the answer is the f32 scan and recall is 1.0).
    pub applies: bool,
    /// Median per-query time of the int8 search, milliseconds.
    pub int8_ms: f64,
    /// Median per-query time of the f32 scan, milliseconds.
    pub f32_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn quantize_is_symmetric_and_bounded() {
        let mut out = vec![0i8; 4];
        let s = quantize(&[1.0, -0.5, 0.0, 0.25], &mut out);
        assert_eq!(out, vec![127, -64, 0, 32]);
        assert!((s - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(quantize(&[0.0; 4], &mut out), 0.0);
        assert_eq!(out, vec![0; 4]);
        assert_eq!(quantize(&[f32::NAN, 1.0, 0.0, 0.0], &mut out), 1.0 / 127.0);
        assert_eq!(out[0], 0);
        assert_eq!(quantize(&[f32::INFINITY, 1.0, 0.0, 0.0], &mut out), 0.0);
        assert!(out.iter().all(|&c| c != i8::MIN));
    }

    #[test]
    fn avx2_kernel_matches_the_portable_dot_exactly() {
        let mut rng = Rng(7);
        for dim in [1usize, 7, 16, 31, 32, 33, 63, 64, 100, 384, 385, 1000] {
            for trial in 0..20 {
                let mut gen = |extreme: bool| -> Vec<i8> {
                    (0..dim)
                        .map(|_| {
                            if extreme {
                                if rng.next() & 1 == 0 {
                                    127
                                } else {
                                    -127
                                }
                            } else {
                                ((rng.next() % 255) as i32 - 127) as i8
                            }
                        })
                        .collect()
                };
                let extreme = trial % 5 == 0;
                let q = gen(extreme);
                let rows: Vec<Vec<i8>> = (0..4).map(|_| gen(extreme)).collect();
                let qq = QQuery {
                    abs: q.iter().map(|c| c.abs()).collect(),
                    q: q.clone(),
                    scale: 1.0,
                };
                let r4 = [&rows[0][..], &rows[1][..], &rows[2][..], &rows[3][..]];
                let want: Vec<i32> = rows.iter().map(|r| dot_i8_portable(&q, r)).collect();
                assert_eq!(dots::<4, false>(&qq, r4).to_vec(), want);
                assert_eq!(dot_i8_dispatch(&qq, &rows[0]), want[0], "dim {dim}");
                assert_eq!(dot_i8(&q, &rows[1]), want[1]);
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma() {
                    #[target_feature(enable = "avx2,fma")]
                    unsafe fn fast(qq: &QQuery, r4: [&[i8]; 4]) -> [i32; 4] {
                        dots::<4, true>(qq, r4)
                    }
                    // SAFETY: AVX2 and FMA were just detected.
                    let got = unsafe { fast(&qq, r4) };
                    assert_eq!(got.to_vec(), want, "dim {dim} trial {trial}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_kernel_refuses_a_short_row() {
        if !has_avx2_fma() {
            return;
        }
        #[target_feature(enable = "avx2,fma")]
        unsafe fn fast(qq: &QQuery, r: [&[i8]; 1]) -> [i32; 1] {
            dots::<1, true>(qq, r)
        }
        let qq = QQuery::new(&[0.5; 64]);
        let short = [1i8; 63];
        let result = std::panic::catch_unwind(|| {
            // SAFETY: AVX2 detected; the call must panic in the assert before
            // any read.
            unsafe { fast(&qq, [&short[..]]) }
        });
        assert!(result.is_err());
    }
}
