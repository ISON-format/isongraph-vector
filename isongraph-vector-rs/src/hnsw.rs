//! The opt-in HNSW approximate index (`AnnMode::Hnsw`), beside the IVF-flat one
//! in `ann.rs`. `DESIGN.md` section 10 is the specification; this file is its
//! Rust realisation.
//!
//! # What it is
//!
//! A hierarchical navigable small-world graph over the store's matrix. The
//! graph holds **slot numbers and links only**: every score is computed from
//! the one stored copy of each vector with the crate's canonical kernels, so a
//! hit's score is bit-identical to the exact scan's. Layer 0 holds every node
//! with up to `2 * M` links; a node also lives in layers `1..=level`
//! (`P(level >= l) = M^-l`, drawn from a seeded integer generator) with up to
//! `M` links per layer. A query descends greedily from the entry point to layer
//! 0, then runs a best-first search with `ef` candidates there.
//!
//! # Maintenance
//!
//! * **Insert** happens inside the write that stored the row (no pending list):
//!   search for neighbours, choose `M` with the diversity heuristic, link both
//!   ways, re-prune any neighbour that overflowed. A few hundred microseconds to
//!   a few milliseconds at a million rows.
//! * **Update** (same slot, new vector): the node keeps its level; its own links
//!   are recomputed from the new vector and the new neighbours link back. Links
//!   that other nodes held to the old position are left (they are only routing
//!   hints; scores always come from the current vector) and are trimmed the next
//!   time those lists overflow. Updates count as churn.
//! * **Delete** tombstones: the table's NaN norm stops the row from ever being
//!   returned, but the graph keeps the node so routes through it still work (its
//!   floats stay in the matrix until compaction). A tombstone is dropped from a
//!   list whenever that list is re-pruned, and wholesale at compaction.
//! * **Compaction** (the table's, at 25% dead) drops tombstones from the graph,
//!   renumbers every link, and refills a node's lost links from the links of the
//!   tombstones it pointed to ("bridging"), without computing a distance.
//! * **Rebuild**: once updates plus deletes exceed the number of nodes, the next
//!   search that needs the index rebuilds it.
//!
//! # Determinism and parallel build
//!
//! A build inserts in slot order in batches that grow with the graph
//! (`size / 16`, at most 512). Within a batch every member is planned against
//! the same frozen graph in parallel, and the plans are applied in member order,
//! so the result depends on the seed and the insertion order, never on the
//! thread count. Members of one batch see each other by brute force.
//!
//! The index is never persisted; a loaded store builds it on its first search.

use crate::ann::{
    cosine, dot_f32, score_slots, AnnConfig, AnnMode, AnnStats, Hit, Par, QueryVec, Rng,
};
use crate::table::{Rows, VectorTable, CHUNK_MASK, CHUNK_ROWS, CHUNK_SHIFT};
use crate::{EmbeddingError, EmbeddingStore};
use std::cell::RefCell;
use std::cmp::{Ordering as Cmp, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard};
use std::time::Instant;

/// Links per node on layers above 0 (layer 0 allows twice this).
pub const HNSW_M: usize = 16;
/// Candidates kept while inserting.
pub const HNSW_EF_CONSTRUCTION: usize = 200;

const ABSENT: u8 = u8::MAX;
const NONE: u32 = u32::MAX;
const MAX_LEVEL: usize = 24;
/// Largest batch the parallel build plans against one frozen graph.
const BUILD_BATCH: usize = 512;
/// A batch is at most `size / BUILD_RATIO` nodes, so members are few next to
/// the graph they are planned against.
const BUILD_RATIO: usize = 16;
/// Updates plus deletes beyond this multiple of the node count call for a
/// rebuild.
const REBUILD_FRACTION: f64 = 1.0;
/// Most links a layer can hold: `2 * M` with `M` at most 64.
const MAX_CAP: usize = 128;

/// Settings for the HNSW index; see [`AnnConfig::hnsw`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HnswConfig {
    /// Links per node on layers above 0 (layer 0 holds up to `2 * m`). More is
    /// better recall and more memory. 2 to 64; default 16.
    pub m: usize,
    /// Candidates kept while linking a new row. More is a better graph and a
    /// slower build and write. Default 200 (never less than `m`).
    pub ef_construction: usize,
    /// Candidates kept per query (never less than the `top_k` asked for). More
    /// is higher recall and slower. `None` picks by store size, aiming at about
    /// 0.95 recall@10 on 384-d text embeddings: 64 under 10,000 rows, 96 under
    /// 200,000, 128 under 600,000, 192 above (measured 0.95 at 100,000 rows
    /// with 96 and 1,000,000 with 192; measure yours with `recall_check`).
    pub ef_search: Option<usize>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: HNSW_M,
            ef_construction: HNSW_EF_CONSTRUCTION,
            ef_search: None,
        }
    }
}

impl HnswConfig {
    pub(crate) fn validate(&self) -> Result<(), EmbeddingError> {
        if !(2..=MAX_CAP / 2).contains(&self.m) {
            return Err(EmbeddingError::InvalidOption(format!(
                "hnsw m must be between 2 and {}",
                MAX_CAP / 2
            )));
        }
        if self.ef_construction == 0 {
            return Err(EmbeddingError::InvalidOption(
                "hnsw ef_construction must be at least 1".to_string(),
            ));
        }
        if self.ef_search == Some(0) {
            return Err(EmbeddingError::InvalidOption(
                "hnsw ef_search must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

// =============================================================================
// Scores and small helpers
// =============================================================================

/// A scored node. The order is "better is greater": higher score, then lower
/// slot (the exact scan's tie-break).
#[derive(Clone, Copy, Debug)]
struct Sc {
    score: f32,
    slot: u32,
}

impl PartialEq for Sc {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Cmp::Equal
    }
}
impl Eq for Sc {}
impl PartialOrd for Sc {
    fn partial_cmp(&self, other: &Self) -> Option<Cmp> {
        Some(self.cmp(other))
    }
}
impl Ord for Sc {
    fn cmp(&self, other: &Self) -> Cmp {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

/// Euclidean norm of a row, for tombstones too (their table norm is NaN).
fn node_norm(rows: &Rows, slot: usize) -> f32 {
    let n = rows.norms[slot];
    if n.is_nan() {
        let row = rows.row(slot);
        dot_f32(row, row).sqrt()
    } else {
        n
    }
}

fn node_query<'a>(rows: &Rows<'a>, slot: usize) -> QueryVec<'a> {
    QueryVec {
        v: rows.row(slot),
        norm: node_norm(rows, slot),
    }
}

/// Cosine of the query with each of `slots`. Live rows score exactly as the
/// scan scores them; tombstones, which the table marks NaN, are scored from
/// their still-present floats so the search can route through them; anything
/// that stays NaN (a corrupt row) ranks last.
fn score_into(rows: Rows, q: &QueryVec, slots: &[u32], out: &mut [f32]) {
    score_slots(rows, q, slots, out);
    for (i, &s) in slots.iter().enumerate() {
        if out[i].is_nan() {
            out[i] = if rows.norms[s as usize].is_nan() {
                let row = rows.row(s as usize);
                let sc = cosine(dot_f32(q.v, row), dot_f32(row, row).sqrt(), q.norm);
                if sc.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    sc
                }
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

fn score_one(rows: Rows, q: &QueryVec, slot: u32) -> Sc {
    let mut out = [0.0f32];
    score_into(rows, q, &[slot], &mut out);
    Sc {
        score: out[0],
        slot,
    }
}

#[inline(always)]
fn prefetch(row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let p = row.as_ptr() as *const i8;
        // SAFETY: a prefetch is a hint; it never faults, whatever the address.
        #[allow(unused_unsafe)]
        unsafe {
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p.wrapping_add(64), _MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = row;
}

/// Which nodes a layer-0 search may return (the others are only routed
/// through).
#[derive(Clone, Copy)]
struct Admit {
    code: Option<u32>,
    threshold: f32,
    exclude: u32,
}

impl Admit {
    #[inline(always)]
    fn ok(&self, rows: &Rows, slot: u32, score: f32) -> bool {
        let s = slot as usize;
        slot != self.exclude
            && !rows.norms[s].is_nan()
            && score >= self.threshold
            && self.code.is_none_or(|c| rows.codes[s] == c)
    }
}

/// The marks of the nodes a search has touched. Thread-local and reused: a
/// query clears only the words it set, so it costs what it visited, and the
/// memory is one bit per slot per searching thread.
#[derive(Default)]
struct Visited {
    bits: Vec<u64>,
    words: Vec<u32>,
}

impl Visited {
    fn begin(&mut self, slots: usize) {
        self.clear();
        let need = slots.div_ceil(64);
        if self.bits.len() < need {
            self.bits.resize(need, 0);
        }
    }

    fn clear(&mut self) {
        for &w in &self.words {
            self.bits[w as usize] = 0;
        }
        self.words.clear();
    }

    /// Mark `slot`; true if it already was.
    #[inline(always)]
    fn test_set(&mut self, slot: u32) -> bool {
        let w = (slot >> 6) as usize;
        let m = 1u64 << (slot & 63);
        let cur = self.bits[w];
        if cur & m != 0 {
            return true;
        }
        if cur == 0 {
            self.words.push(w as u32);
        }
        self.bits[w] = cur | m;
        false
    }
}

thread_local! {
    static VISITED: RefCell<Visited> = RefCell::new(Visited::default());
}

fn with_visited<R>(slots: usize, f: impl FnOnce(&mut Visited) -> R) -> R {
    VISITED.with(|v| {
        let mut v = v.borrow_mut();
        v.begin(slots);
        let out = f(&mut v);
        v.clear();
        out
    })
}

/// What a layer search returns.
pub(crate) struct Found {
    /// Admitted nodes, best first.
    pub hits: Vec<Hit>,
    /// The search gave up after scoring too many nodes (a filter so selective
    /// that scanning is cheaper).
    pub budget_hit: bool,
}

// =============================================================================
// The graph
// =============================================================================

pub(crate) struct Hnsw {
    m: usize,
    m0: usize,
    ef_c: usize,
    /// Highest layer of each slot's node, or `ABSENT`.
    levels: Vec<u8>,
    /// Layer 0: `CHUNK_ROWS` nodes per chunk, `m0 + 1` words each (the link
    /// count, then the links).
    l0: Vec<Vec<u32>>,
    /// Per slot: where its upper-layer blocks start in `upper`, or `NONE`.
    upper_at: Vec<u32>,
    /// A node of level `L` owns `L` consecutive blocks of `m + 1` words.
    upper: Vec<u32>,
    entry: u32,
    top: u8,
    rng: Rng,
    /// Nodes in the graph, tombstones included.
    members: usize,
    churn: usize,
    build_seconds: f64,
    inserts: u64,
    insert_seconds: f64,
}

/// One node to link: a new row, or an existing node whose vector changed.
#[derive(Clone, Copy)]
struct Job {
    slot: u32,
    level: u8,
    relink: bool,
}

impl Hnsw {
    fn new(cfg: &AnnConfig) -> Hnsw {
        let h = cfg.hnsw;
        Hnsw {
            m: h.m,
            m0: 2 * h.m,
            ef_c: h.ef_construction.max(h.m),
            levels: Vec::new(),
            l0: Vec::new(),
            upper_at: Vec::new(),
            upper: Vec::new(),
            entry: NONE,
            top: 0,
            rng: Rng(cfg.seed ^ 0x4857_5357_4857_5357),
            members: 0,
            churn: 0,
            build_seconds: 0.0,
            inserts: 0,
            insert_seconds: 0.0,
        }
    }

    fn cap(&self, level: usize) -> usize {
        if level == 0 {
            self.m0
        } else {
            self.m
        }
    }

    /// Cover slot `s` (new slots start absent, with empty link blocks).
    fn ensure_slot(&mut self, s: usize) {
        while self.levels.len() <= s {
            self.levels.push(ABSENT);
            self.upper_at.push(NONE);
        }
        while self.l0.len() <= s >> CHUNK_SHIFT {
            self.l0.push(vec![0u32; CHUNK_ROWS * (self.m0 + 1)]);
        }
    }

    fn block_at(&self, s: usize, level: usize) -> (bool, usize) {
        if level == 0 {
            (true, (s & CHUNK_MASK) * (self.m0 + 1))
        } else {
            (
                false,
                self.upper_at[s] as usize + (level - 1) * (self.m + 1),
            )
        }
    }

    /// The links of `s` on `level` (`s` must have that layer).
    #[inline(always)]
    fn list(&self, s: usize, level: usize) -> &[u32] {
        let (zero, at) = self.block_at(s, level);
        let block = if zero {
            &self.l0[s >> CHUNK_SHIFT][at..at + self.m0 + 1]
        } else {
            &self.upper[at..at + self.m + 1]
        };
        &block[1..1 + block[0] as usize]
    }

    fn set_list(&mut self, s: usize, level: usize, items: &[u32]) {
        debug_assert!(items.len() <= self.cap(level));
        let (zero, at) = self.block_at(s, level);
        let (m0, m) = (self.m0, self.m);
        let block = if zero {
            &mut self.l0[s >> CHUNK_SHIFT][at..at + m0 + 1]
        } else {
            &mut self.upper[at..at + m + 1]
        };
        block[0] = items.len() as u32;
        block[1..1 + items.len()].copy_from_slice(items);
    }

    fn add_node(&mut self, s: usize, level: u8) {
        self.ensure_slot(s);
        debug_assert_eq!(self.levels[s], ABSENT);
        self.levels[s] = level;
        if level >= 1 {
            self.upper_at[s] = self.upper.len() as u32;
            let grown = self.upper.len() + level as usize * (self.m + 1);
            self.upper.resize(grown, 0);
        }
        self.members += 1;
    }

    /// `P(level >= l) = m^-l`, from integer draws only (portable).
    fn draw_level(&mut self) -> u8 {
        let mut level = 0usize;
        while level < MAX_LEVEL && self.rng.below(self.m) == 0 {
            level += 1;
        }
        level as u8
    }

    fn in_graph(&self, s: usize) -> bool {
        self.levels.get(s).is_some_and(|&l| l != ABSENT)
    }

    pub(crate) fn stale(&self) -> bool {
        self.churn as f64 > REBUILD_FRACTION * self.members.max(1) as f64
    }

    // -- search -----------------------------------------------------------------

    /// Walk to the best node reachable by single steps on `level`.
    fn greedy(&self, rows: Rows, q: &QueryVec, mut cur: Sc, level: usize) -> Sc {
        let mut scores = [0.0f32; MAX_CAP];
        loop {
            let list = self.list(cur.slot as usize, level);
            if list.is_empty() {
                return cur;
            }
            for &x in list {
                prefetch(rows.row(x as usize));
            }
            score_into(rows, q, list, &mut scores[..list.len()]);
            let mut best = cur;
            for (i, &x) in list.iter().enumerate() {
                let c = Sc {
                    score: scores[i],
                    slot: x,
                };
                if c > best {
                    best = c;
                }
            }
            if best == cur {
                return cur;
            }
            cur = best;
        }
    }

    /// Best-first search of one layer from `eps`, keeping the `ef` best
    /// admitted nodes. Tombstones and non-admitted nodes are routed through but
    /// never kept. Stops when the best unexpanded node is worse than the worst
    /// kept one (once `ef` are kept), or after `budget` nodes were scored.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        rows: Rows,
        q: &QueryVec,
        eps: &[Sc],
        ef: usize,
        level: usize,
        admit: &Admit,
        budget: usize,
        visited: &mut Visited,
    ) -> Found {
        visited.clear();
        let mut cand: BinaryHeap<Sc> = BinaryHeap::new();
        let mut kept: BinaryHeap<Reverse<Sc>> = BinaryHeap::with_capacity(ef + 1);
        for &e in eps {
            if !visited.test_set(e.slot) {
                cand.push(e);
                if admit.ok(&rows, e.slot, e.score) {
                    kept.push(Reverse(e));
                    if kept.len() > ef {
                        kept.pop();
                    }
                }
            }
        }
        let mut fresh: Vec<u32> = Vec::with_capacity(self.m0);
        let mut scores = [0.0f32; MAX_CAP];
        let mut scored = 0usize;
        let mut budget_hit = false;
        while let Some(c) = cand.pop() {
            if kept.len() >= ef {
                if let Some(Reverse(worst)) = kept.peek() {
                    if c < *worst {
                        break;
                    }
                }
            }
            fresh.clear();
            for &x in self.list(c.slot as usize, level) {
                if !visited.test_set(x) {
                    fresh.push(x);
                    prefetch(rows.row(x as usize));
                }
            }
            if fresh.is_empty() {
                continue;
            }
            score_into(rows, q, &fresh, &mut scores[..fresh.len()]);
            scored += fresh.len();
            for (i, &x) in fresh.iter().enumerate() {
                let node = Sc {
                    score: scores[i],
                    slot: x,
                };
                let promising = kept.len() < ef || kept.peek().is_some_and(|w| node > w.0);
                if promising {
                    cand.push(node);
                    if admit.ok(&rows, x, node.score) {
                        kept.push(Reverse(node));
                        if kept.len() > ef {
                            kept.pop();
                        }
                    }
                }
            }
            if scored > budget {
                budget_hit = true;
                break;
            }
        }
        let mut best: Vec<Sc> = kept.into_iter().map(|r| r.0).collect();
        best.sort_unstable_by(|a, b| b.cmp(a));
        Found {
            hits: best
                .into_iter()
                .map(|s| Hit {
                    score: s.score,
                    slot: s.slot,
                })
                .collect(),
            budget_hit,
        }
    }

    /// The `ef` best admitted nodes for a query, best first (ties by slot).
    pub(crate) fn search(
        &self,
        rows: Rows,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        ef: usize,
        budget: usize,
    ) -> Found {
        if self.entry == NONE {
            return Found {
                hits: Vec::new(),
                budget_hit: false,
            };
        }
        with_visited(self.levels.len(), |visited| {
            let mut cur = score_one(rows, q, self.entry);
            for level in (1..=self.top as usize).rev() {
                cur = self.greedy(rows, q, cur, level);
            }
            let admit = Admit {
                code,
                threshold,
                exclude: NONE,
            };
            let mut found = self.search_layer(rows, q, &[cur], ef, 0, &admit, budget, visited);
            // best first, ties by slot, exactly as the scan orders them
            found.hits.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Cmp::Equal)
                    .then(a.slot.cmp(&b.slot))
            });
            found
        })
    }

    // -- linking ------------------------------------------------------------------

    /// The diversity heuristic: walk `cands` (best first) and keep one only if
    /// it is closer to the base than to every node already kept, until `cap`
    /// are kept. Fewer than `cap` candidates are all kept.
    fn select(&self, rows: Rows, cands: &[Sc], cap: usize) -> Vec<u32> {
        if cands.len() <= cap {
            return cands.iter().map(|c| c.slot).collect();
        }
        let mut chosen: Vec<u32> = Vec::with_capacity(cap);
        let mut scores = [0.0f32; MAX_CAP];
        for c in cands {
            if chosen.len() >= cap {
                break;
            }
            let q = node_query(&rows, c.slot as usize);
            score_into(rows, &q, &chosen, &mut scores[..chosen.len()]);
            if !scores[..chosen.len()].iter().any(|&s| s > c.score) {
                chosen.push(c.slot);
            }
        }
        chosen
    }

    /// Neighbours chosen for `job` on every layer, against the graph as it is
    /// now plus the other new members of its batch (`intra`).
    fn plan(
        &self,
        rows: Rows,
        job: &Job,
        intra: &[(u32, u8)],
        visited: &mut Visited,
    ) -> Vec<Vec<u32>> {
        let x = job.slot;
        let level = job.level as usize;
        let q = node_query(&rows, x as usize);
        let admit = Admit {
            code: None,
            threshold: f32::NEG_INFINITY,
            exclude: x,
        };
        let top = self.top as usize;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); level + 1];
        let mut eps: Vec<Sc> = Vec::new();
        if self.entry != NONE {
            let mut cur = score_one(rows, &q, self.entry);
            for l in ((level + 1)..=top).rev() {
                cur = self.greedy(rows, &q, cur, l);
            }
            eps.push(cur);
        }
        for l in (0..=level).rev() {
            let mut cands: Vec<Sc> = Vec::new();
            if l <= top && !eps.is_empty() {
                let found =
                    self.search_layer(rows, &q, &eps, self.ef_c, l, &admit, usize::MAX, visited);
                cands = found
                    .hits
                    .iter()
                    .map(|h| Sc {
                        score: h.score,
                        slot: h.slot,
                    })
                    .collect();
                if !cands.is_empty() {
                    eps = cands.clone();
                }
            }
            let members: Vec<u32> = intra
                .iter()
                .filter(|&&(s, lv)| s != x && lv as usize >= l)
                .map(|&(s, _)| s)
                .collect();
            if !members.is_empty() {
                let mut scores = vec![0.0f32; members.len()];
                score_into(rows, &q, &members, &mut scores);
                cands.extend(
                    members
                        .iter()
                        .zip(&scores)
                        .map(|(&slot, &score)| Sc { score, slot }),
                );
                cands.sort_unstable_by(|a, b| b.cmp(a));
            }
            lists[l] = self.select(rows, &cands, self.m);
        }
        lists
    }

    /// `t`'s list on `layer` with `incoming` added, pruned to the layer's
    /// capacity when it overflows (tombstones are dropped then). `None` if
    /// nothing changes.
    fn grown_list(&self, rows: Rows, layer: usize, t: u32, incoming: &[u32]) -> Option<Vec<u32>> {
        let cap = self.cap(layer);
        let cur = self.list(t as usize, layer);
        let mut all: Vec<u32> = cur.to_vec();
        for &n in incoming {
            if n != t && !all.contains(&n) {
                all.push(n);
            }
        }
        if all.len() == cur.len() {
            return None;
        }
        if all.len() <= cap {
            return Some(all);
        }
        all.retain(|&s| !rows.norms[s as usize].is_nan());
        if all.len() <= cap {
            return Some(all);
        }
        let q = node_query(&rows, t as usize);
        let mut scores = vec![0.0f32; all.len()];
        score_into(rows, &q, &all, &mut scores);
        let mut cands: Vec<Sc> = all
            .iter()
            .zip(&scores)
            .map(|(&slot, &score)| Sc { score, slot })
            .collect();
        cands.sort_unstable_by(|a, b| b.cmp(a));
        Some(self.select(rows, &cands, cap))
    }

    /// Link `jobs` into the graph: plan every job against the current graph (in
    /// parallel, read-only), then apply the plans in job order.
    fn link(&mut self, table: &VectorTable, jobs: &[Job], par: Par) {
        let rows = table.rows();
        let slots = self.levels.len();
        let intra: Vec<(u32, u8)> = jobs
            .iter()
            .filter(|j| !j.relink)
            .map(|j| (j.slot, j.level))
            .collect();

        let plans: Vec<Vec<Vec<u32>>> = {
            let this: &Hnsw = self;
            let workers = if jobs.len() <= 1 {
                1
            } else {
                par.workers(jobs.len(), usize::MAX)
            };
            if workers <= 1 {
                jobs.iter()
                    .map(|j| with_visited(slots, |v| this.plan(rows, j, &intra, v)))
                    .collect()
            } else {
                let next = AtomicUsize::new(0);
                let done: Mutex<Vec<(usize, Vec<Vec<u32>>)>> = Mutex::new(Vec::new());
                par.run(workers, &|_| {
                    let mut local = Vec::new();
                    with_visited(slots, |v| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        local.push((i, this.plan(rows, &jobs[i], &intra, v)));
                    });
                    done.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
                });
                let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
                done.sort_unstable_by_key(|d| d.0);
                done.into_iter().map(|d| d.1).collect()
            }
        };

        // forward links
        for (job, lists) in jobs.iter().zip(&plans) {
            if !job.relink {
                self.add_node(job.slot as usize, job.level);
            }
            for (l, list) in lists.iter().enumerate() {
                self.set_list(job.slot as usize, l, list);
            }
        }

        // backward links, grouped by (layer, target) in a fixed order
        let mut adds: Vec<(u8, u32, u32)> = Vec::new();
        for (job, lists) in jobs.iter().zip(&plans) {
            for (l, list) in lists.iter().enumerate() {
                for &nb in list {
                    if !rows.norms[nb as usize].is_nan() {
                        adds.push((l as u8, nb, job.slot));
                    }
                }
            }
        }
        adds.sort_by_key(|a| (a.0, a.1));
        let mut groups: Vec<(usize, usize)> = Vec::new(); // (start, end) into `adds`
        let mut at = 0;
        while at < adds.len() {
            let mut end = at + 1;
            while end < adds.len() && adds[end].0 == adds[at].0 && adds[end].1 == adds[at].1 {
                end += 1;
            }
            groups.push((at, end));
            at = end;
        }
        let grown: Vec<Option<Vec<u32>>> = {
            let this: &Hnsw = self;
            let one = |g: &(usize, usize)| {
                let (layer, target) = (adds[g.0].0 as usize, adds[g.0].1);
                let incoming: Vec<u32> = adds[g.0..g.1].iter().map(|a| a.2).collect();
                this.grown_list(rows, layer, target, &incoming)
            };
            let workers = if groups.len() <= 1 {
                1
            } else {
                par.workers(groups.len(), usize::MAX)
            };
            if workers <= 1 {
                groups.iter().map(one).collect()
            } else {
                let next = AtomicUsize::new(0);
                let done: Mutex<Vec<(usize, Option<Vec<u32>>)>> = Mutex::new(Vec::new());
                par.run(workers, &|_| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= groups.len() {
                            break;
                        }
                        local.push((i, one(&groups[i])));
                    }
                    done.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
                });
                let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
                done.sort_unstable_by_key(|d| d.0);
                done.into_iter().map(|d| d.1).collect()
            }
        };
        for (g, list) in groups.iter().zip(grown) {
            if let Some(list) = list {
                self.set_list(adds[g.0].1 as usize, adds[g.0].0 as usize, &list);
            }
        }

        // the entry point is the first node of the highest level
        for job in jobs.iter().filter(|j| !j.relink) {
            if self.entry == NONE || job.level > self.top {
                self.entry = job.slot;
                self.top = job.level;
            }
        }
    }

    /// A graph over every live row of `table`, built in slot order.
    pub(crate) fn build(table: &VectorTable, cfg: &AnnConfig, par: Par) -> Hnsw {
        let started = Instant::now();
        let mut h = Hnsw::new(cfg);
        let n = table.slots();
        if n > 0 {
            h.ensure_slot(n - 1);
        }
        let order: Vec<u32> = table
            .live_slots()
            .filter(|&s| table.norm(s).is_finite())
            .map(|s| s as u32)
            .collect();
        let mut at = 0;
        let mut jobs: Vec<Job> = Vec::new();
        while at < order.len() {
            let size = (at / BUILD_RATIO).clamp(1, BUILD_BATCH);
            let end = (at + size).min(order.len());
            jobs.clear();
            for &slot in &order[at..end] {
                let level = h.draw_level();
                jobs.push(Job {
                    slot,
                    level,
                    relink: false,
                });
            }
            h.link(table, &jobs, par);
            at = end;
        }
        h.build_seconds = started.elapsed().as_secs_f64();
        h
    }

    /// A row was written (new, or its vector replaced).
    fn note_put(&mut self, table: &VectorTable, slot: u32, par: Par) {
        let started = Instant::now();
        let s = slot as usize;
        self.ensure_slot(s);
        if !table.norm(s).is_finite() {
            if self.in_graph(s) {
                self.churn += 1;
            }
            return;
        }
        let job = if self.in_graph(s) {
            self.churn += 1;
            Job {
                slot,
                level: self.levels[s],
                relink: true,
            }
        } else {
            Job {
                slot,
                level: self.draw_level(),
                relink: false,
            }
        };
        self.link(table, &[job], par);
        self.inserts += 1;
        self.insert_seconds += started.elapsed().as_secs_f64();
    }

    fn note_drop(&mut self) {
        self.churn += 1;
    }

    /// Follow the table's compaction (`live` = old slots kept, in order):
    /// tombstones leave the graph, links are renumbered, and a node that lost
    /// links takes up the links of the tombstones it pointed to.
    fn compact(&mut self, live: &[u32]) {
        let old_n = self.levels.len();
        let mut new_of = vec![NONE; old_n];
        for (new, &old) in live.iter().enumerate() {
            if (old as usize) < old_n {
                new_of[old as usize] = new as u32;
            }
        }
        let mut fresh = Hnsw {
            m: self.m,
            m0: self.m0,
            ef_c: self.ef_c,
            levels: Vec::new(),
            l0: Vec::new(),
            upper_at: Vec::new(),
            upper: Vec::new(),
            entry: NONE,
            top: 0,
            rng: Rng(self.rng.0),
            members: 0,
            churn: self.churn,
            build_seconds: self.build_seconds,
            inserts: self.inserts,
            insert_seconds: self.insert_seconds,
        };
        if !live.is_empty() {
            fresh.ensure_slot(live.len() - 1);
        }
        let mut lost: Vec<u32> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        for (new, &old) in live.iter().enumerate() {
            let old = old as usize;
            let level = self.levels.get(old).copied().unwrap_or(ABSENT);
            // Only live rows survive; a tombstone has no new slot, so every
            // surviving node is either a live graph member or absent.
            if level == ABSENT {
                continue;
            }
            fresh.add_node(new, level);
            for l in 0..=level as usize {
                let cap = self.cap(l);
                out.clear();
                lost.clear();
                for &nb in self.list(old, l) {
                    match new_of.get(nb as usize) {
                        Some(&n) if n != NONE => out.push(n),
                        _ => lost.push(nb),
                    }
                }
                'refill: for &d in &lost {
                    if out.len() >= cap {
                        break;
                    }
                    let d = d as usize;
                    if d >= old_n || (self.levels[d] as usize) < l || self.levels[d] == ABSENT {
                        continue;
                    }
                    for &w in self.list(d, l) {
                        let nw = new_of.get(w as usize).copied().unwrap_or(NONE);
                        if nw != NONE && nw as usize != new && !out.contains(&nw) {
                            out.push(nw);
                            if out.len() >= cap {
                                break 'refill;
                            }
                        }
                    }
                }
                fresh.set_list(new, l, &out);
            }
            if fresh.entry == NONE || level > fresh.top {
                fresh.entry = new as u32;
                fresh.top = level;
            }
        }
        *self = fresh;
    }

    /// Every invariant of the graph against the table it indexes.
    pub(crate) fn check(&self, table: &VectorTable) -> Result<(), String> {
        let n = table.slots();
        if self.levels.len() != n || self.upper_at.len() != n {
            return Err(format!(
                "graph covers {} rows, table has {n}",
                self.levels.len()
            ));
        }
        if self.l0.len() != n.div_ceil(CHUNK_ROWS) {
            return Err(format!("{} link chunks for {n} rows", self.l0.len()));
        }
        let mut members = 0usize;
        let mut max_level = 0u8;
        for s in 0..n {
            let level = self.levels[s];
            if level == ABSENT {
                if table.is_live(s) && table.norm(s).is_finite() {
                    return Err(format!("live slot {s} is not in the graph"));
                }
                continue;
            }
            members += 1;
            max_level = max_level.max(level);
            for l in 0..=level as usize {
                let list = self.list(s, l);
                if list.len() > self.cap(l) {
                    return Err(format!("slot {s} layer {l} holds {} links", list.len()));
                }
                let mut seen: Vec<u32> = list.to_vec();
                seen.sort_unstable();
                if seen.windows(2).any(|w| w[0] == w[1]) {
                    return Err(format!("slot {s} layer {l} links a node twice"));
                }
                for &nb in list {
                    let t = nb as usize;
                    if t >= n || nb as usize == s {
                        return Err(format!("slot {s} layer {l} links {nb}"));
                    }
                    if self.levels[t] == ABSENT || (self.levels[t] as usize) < l {
                        return Err(format!(
                            "slot {s} layer {l} links {nb}, which is not on that layer"
                        ));
                    }
                }
            }
        }
        if members != self.members {
            return Err(format!(
                "member counter {} but {members} nodes",
                self.members
            ));
        }
        if members == 0 {
            if self.entry != NONE {
                return Err("an empty graph has an entry point".to_string());
            }
        } else if self.entry == NONE
            || self.levels[self.entry as usize] != self.top
            || max_level != self.top
        {
            return Err(format!(
                "entry {} (level {:?}) does not sit on top level {} (highest node {max_level})",
                self.entry,
                self.levels.get(self.entry as usize),
                self.top
            ));
        }
        Ok(())
    }

    /// Nodes whose rows are tombstones.
    fn dead(&self, table: &VectorTable) -> usize {
        (0..self.levels.len())
            .filter(|&s| self.levels[s] != ABSENT && table.norm(s).is_nan())
            .count()
    }

    pub(crate) fn nbytes(&self) -> usize {
        self.levels.capacity()
            + self.l0.iter().map(|c| c.capacity() * 4).sum::<usize>()
            + self.upper_at.capacity() * 4
            + self.upper.capacity() * 4
    }
}

// =============================================================================
// The store's side
// =============================================================================

/// The graph slot an [`EmbeddingStore`] carries: built lazily by a search,
/// maintained by writes through `&mut self`, never persisted.
pub(crate) type HnswSlot = RwLock<Option<Hnsw>>;

/// `ef_search` by store size when none is configured (see
/// [`HnswConfig::ef_search`]).
pub(crate) fn auto_ef(live: usize) -> usize {
    match live {
        0..=9_999 => 64,
        10_000..=199_999 => 96,
        200_000..=599_999 => 128,
        _ => 192,
    }
}

/// The `ef` a query uses: the per-call override, else the configured value,
/// else the size-based default; never below `k`.
fn effective_ef(cfg: &AnnConfig, over: Option<usize>, k: usize, live: usize) -> usize {
    over.or(cfg.hnsw.ef_search)
        .unwrap_or_else(|| auto_ef(live))
        .max(k)
        .max(1)
}

/// Scoring more than this many nodes costs more than the exact scan.
fn budget_for(live: usize, ef: usize) -> usize {
    (live / 8).max(ef.saturating_mul(16))
}

impl EmbeddingStore {
    pub(crate) fn hnsw_mut(&mut self) -> &mut Option<Hnsw> {
        self.hnsw.get_mut().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn hnsw_read(&self) -> RwLockReadGuard<'_, Option<Hnsw>> {
        self.hnsw.read().unwrap_or_else(|e| e.into_inner())
    }

    /// A row was written: link it into the graph now.
    pub(crate) fn hnsw_row_written(&mut self, slot: u32) {
        let (table, pool, configured) = (&self.table, &self.pool, self.scan_threads);
        let guard = self.hnsw.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.note_put(table, slot, Par { configured, pool });
        }
    }

    pub(crate) fn hnsw_row_removed(&mut self) {
        if let Some(h) = self.hnsw_mut().as_mut() {
            h.note_drop();
        }
    }

    pub(crate) fn hnsw_compact(&mut self, live: &[u32]) {
        if let Some(h) = self.hnsw_mut().as_mut() {
            h.compact(live);
        }
    }

    /// A read lock on a usable graph, building (or rebuilding when stale)
    /// first; `None` means "answer exactly" (the store is under `min_size`).
    pub(crate) fn ready_hnsw(&self, cfg: &AnnConfig) -> Option<RwLockReadGuard<'_, Option<Hnsw>>> {
        if self.table.live() < cfg.min_size.max(2) {
            return None;
        }
        loop {
            {
                let guard = self.hnsw_read();
                if guard.as_ref().is_some_and(|h| !h.stale()) {
                    return Some(guard);
                }
            }
            let mut guard = self.hnsw.write().unwrap_or_else(|e| e.into_inner());
            if guard.as_ref().is_none_or(|h| h.stale()) {
                *guard = Some(Hnsw::build(&self.table, cfg, self.par()));
                self.counts.builds.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Top `k` from the graph, or `None` to fall back to the scan. Widens `ef`
    /// 4x when a filter leaves fewer than `k`, and gives up (scan) once `ef`
    /// would cover half the store or the search scores more nodes than the scan
    /// would cost.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hnsw_topk(
        &self,
        cfg: &AnnConfig,
        h: &Hnsw,
        q: &QueryVec,
        code: Option<u32>,
        threshold: f32,
        k: usize,
        over: Option<usize>,
    ) -> Option<Vec<Hit>> {
        if !q.has_direction() {
            return None;
        }
        let live = self.table.live();
        let base = effective_ef(cfg, over, k, live);
        let rows = self.table.rows();
        let mut scale = 1usize;
        loop {
            let ef = base.saturating_mul(scale);
            let exhausted = if scale == 1 {
                ef >= live
            } else {
                ef.saturating_mul(2) >= live
            };
            if exhausted {
                return None;
            }
            let found = h.search(rows, q, code, threshold, ef, budget_for(live, ef));
            if found.budget_hit {
                return None;
            }
            if found.hits.len() >= k {
                let mut hits = found.hits;
                hits.truncate(k);
                self.counts.note_approximate(scale > 1);
                return Some(hits);
            }
            scale = scale.saturating_mul(4);
        }
    }

    /// Every row the widened search keeps (for a caller that filters further),
    /// best first; `None` to fall back to the scan.
    pub(crate) fn hnsw_ranked(
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
        let guard = self.ready_hnsw(cfg)?;
        let h = guard.as_ref()?;
        let live = self.table.live();
        let ef = effective_ef(cfg, None, 1, live).saturating_mul(scale);
        let exhausted = if scale == 1 {
            ef >= live
        } else {
            ef.saturating_mul(2) >= live
        };
        if exhausted {
            return None;
        }
        let found = h.search(
            self.table.rows(),
            q,
            code,
            threshold,
            ef,
            budget_for(live, ef),
        );
        if found.budget_hit {
            return None;
        }
        self.counts.note_approximate(scale > 1);
        Some(found.hits)
    }

    /// Answer a batch from the graph: queries are spread over the pool and each
    /// runs the single-query path, so each answer is the one-query answer bit
    /// for bit. Queries it cannot answer stay `None` for the exact batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hnsw_batch(
        &self,
        cfg: &AnnConfig,
        h: &Hnsw,
        qs: &[QueryVec],
        code: Option<u32>,
        threshold: f32,
        k: usize,
        answers: &mut [Option<Vec<Hit>>],
    ) {
        let par = self.par();
        let next = AtomicUsize::new(0);
        let done: Mutex<Vec<(usize, Option<Vec<Hit>>)>> = Mutex::new(Vec::new());
        par.run(par.workers(qs.len(), usize::MAX), &|_| {
            let mut local = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= qs.len() {
                    break;
                }
                local.push((i, self.hnsw_topk(cfg, h, &qs[i], code, threshold, k, None)));
            }
            done.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
        });
        for (i, hits) in done.into_inner().unwrap_or_else(|e| e.into_inner()) {
            answers[i] = hits;
        }
    }

    /// Change `ef_search` (`None` = chosen by store size). Takes effect on the next
    /// search; the graph is not rebuilt.
    pub fn set_ann_ef_search(&mut self, ef_search: Option<usize>) -> Result<(), EmbeddingError> {
        if ef_search == Some(0) {
            return Err(EmbeddingError::InvalidOption(
                "hnsw ef_search must be at least 1".to_string(),
            ));
        }
        match self.ann.as_mut() {
            Some(config) => {
                config.hnsw.ef_search = ef_search;
                Ok(())
            }
            None => Err(EmbeddingError::InvalidOption(
                "this store has no approximate index; call set_ann first".to_string(),
            )),
        }
    }

    /// Like `similarity_search_vector`, with `ef_search` for this call only
    /// (HNSW mode; ignored by the exact scan and by IVF).
    pub fn similarity_search_vector_ef(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
        ef_search: usize,
    ) -> Result<Vec<crate::SimilarityResult>, EmbeddingError> {
        if ef_search == 0 {
            return Err(EmbeddingError::InvalidOption(
                "hnsw ef_search must be at least 1".to_string(),
            ));
        }
        self.search_vector_with(
            query_vector,
            top_k,
            node_type,
            threshold,
            false,
            Some(ef_search),
        )
    }

    /// Graph facts for `ann_stats`.
    pub(crate) fn hnsw_stats(&self) -> HnswStats {
        let guard = self.hnsw_read();
        match guard.as_ref() {
            Some(h) => HnswStats {
                max_level: Some(h.top as usize),
                nodes: h.members,
                dead: h.dead(&self.table),
                churn: h.churn,
                build_seconds: Some(h.build_seconds),
                bytes: h.nbytes(),
                inserts: h.inserts,
                insert_seconds: h.insert_seconds,
                built: true,
            },
            None => HnswStats::default(),
        }
    }

    pub(crate) fn hnsw_is_mode(&self) -> bool {
        self.ann.as_ref().is_some_and(|c| c.mode == AnnMode::Hnsw)
    }

    /// `ann_stats` for a store in HNSW mode.
    pub(crate) fn hnsw_ann_stats(&self) -> AnnStats {
        let g = self.hnsw_stats();
        let cfg = self.ann.as_ref();
        AnnStats {
            enabled: cfg.is_some(),
            active: g.built,
            nlist: None,
            nprobe: None,
            trained_n: g.built.then_some(g.nodes),
            pending: 0,
            churn: g.churn,
            build_seconds: g.build_seconds,
            index_bytes: g.bytes,
            min_size: cfg.map(|c| c.min_size),
            searches: self.counts.snapshot(),
            folds: 0,
            fold_seconds: 0.0,
            mode: AnnMode::Hnsw,
            ef_search: cfg.map(|c| {
                c.hnsw
                    .ef_search
                    .unwrap_or_else(|| auto_ef(self.table.live()))
            }),
            max_level: g.max_level,
            graph_nodes: g.nodes,
            graph_dead: g.dead,
            inserts: g.inserts,
            insert_seconds: g.insert_seconds,
        }
    }
}

#[derive(Default)]
pub(crate) struct HnswStats {
    pub max_level: Option<usize>,
    pub nodes: usize,
    pub dead: usize,
    pub churn: usize,
    pub build_seconds: Option<f64>,
    pub bytes: usize,
    pub inserts: u64,
    pub insert_seconds: f64,
    pub built: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn table(n: usize, dim: usize) -> VectorTable {
        let model: Arc<str> = Arc::from("m");
        let mut rng = Rng(99);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.below(2001) as f32 - 1000.0) / 1000.0)
                    .collect()
            })
            .collect();
        let mut t = VectorTable::new();
        for i in 0..n {
            let c = &centers[rng.below(centers.len())];
            let v: Vec<f32> = c
                .iter()
                .map(|x| x + (rng.below(2001) as f32 - 1000.0) / 2500.0)
                .collect();
            t.put(&format!("doc:{i}"), "doc", &v, "", &model);
        }
        t
    }

    fn cfg() -> AnnConfig {
        AnnConfig {
            min_size: 0,
            ..AnnConfig::hnsw()
        }
    }

    fn same_graph(a: &Hnsw, b: &Hnsw) -> bool {
        a.levels == b.levels
            && a.l0 == b.l0
            && a.upper_at == b.upper_at
            && a.upper == b.upper
            && a.entry == b.entry
            && a.top == b.top
            && a.members == b.members
    }

    #[test]
    fn the_built_graph_does_not_depend_on_the_thread_count() {
        let t = table(3000, 12);
        let pool = None;
        let one = Hnsw::build(
            &t,
            &cfg(),
            Par {
                configured: 1,
                pool: &pool,
            },
        );
        let auto = Hnsw::build(
            &t,
            &cfg(),
            Par {
                configured: 0,
                pool: &pool,
            },
        );
        let three = Hnsw::build(
            &t,
            &cfg(),
            Par {
                configured: 3,
                pool: &pool,
            },
        );
        one.check(&t).unwrap();
        assert!(same_graph(&one, &auto), "auto threads changed the graph");
        assert!(same_graph(&one, &three), "three threads changed the graph");
        let again = Hnsw::build(
            &t,
            &cfg(),
            Par {
                configured: 0,
                pool: &pool,
            },
        );
        assert!(same_graph(&one, &again));
        let other = Hnsw::build(
            &t,
            &AnnConfig { seed: 1, ..cfg() },
            Par {
                configured: 1,
                pool: &pool,
            },
        );
        assert!(!same_graph(&one, &other), "the seed must matter");
    }

    #[test]
    fn levels_follow_a_geometric_law() {
        let mut h = Hnsw::new(&cfg());
        let draws = 160_000usize;
        let mut counts = [0usize; 6];
        for _ in 0..draws {
            let l = h.draw_level() as usize;
            for c in counts.iter_mut().take(l.min(5) + 1).skip(1) {
                *c += 1; // draws with level >= index
            }
        }
        // P(level >= 1) = 1/16, P(level >= 2) = 1/256, ...
        let p1 = counts[1] as f64 / draws as f64;
        let p2 = counts[2] as f64 / draws as f64;
        assert!((p1 - 1.0 / 16.0).abs() < 0.004, "{p1}");
        assert!((p2 - 1.0 / 256.0).abs() < 0.001, "{p2}");
    }

    /// Nodes reachable from the entry point on layer 0, live ones counted.
    fn reachable_live(h: &Hnsw, t: &VectorTable) -> usize {
        let mut seen = vec![false; t.slots()];
        let mut stack = vec![h.entry as usize];
        seen[h.entry as usize] = true;
        while let Some(s) = stack.pop() {
            for &nb in h.list(s, 0) {
                if !seen[nb as usize] {
                    seen[nb as usize] = true;
                    stack.push(nb as usize);
                }
            }
        }
        (0..t.slots()).filter(|&s| seen[s] && t.is_live(s)).count()
    }

    #[test]
    fn compaction_renumbers_and_bridges_over_the_tombstones() {
        let mut t = table(4000, 12);
        let pool = None;
        let par = Par {
            configured: 0,
            pool: &pool,
        };
        let mut h = Hnsw::build(&t, &cfg(), par);
        assert_eq!(reachable_live(&h, &t), 4000);
        // kill every third row, but keep the entry point's row alive
        for i in (0..4000).step_by(3) {
            if i as u32 != h.entry {
                t.remove(&format!("doc:{i}"));
            }
        }
        for _ in 0..3 {
            h.note_drop();
        }
        let live = t.compact();
        h.compact(&live);
        h.check(&t).unwrap();
        assert_eq!(h.members, t.live());
        assert_eq!(h.dead(&t), 0);
        let reach = reachable_live(&h, &t);
        assert!(
            reach * 100 >= t.live() * 99,
            "{reach} of {} live rows reachable after compaction",
            t.live()
        );
        // lists were refilled from the tombstones' links: the mean degree stays
        // within 15% of what it was
        let degree = |h: &Hnsw| -> f64 {
            let total: usize = (0..h.levels.len()).map(|s| h.list(s, 0).len()).sum();
            total as f64 / h.levels.len() as f64
        };
        let fresh = Hnsw::build(&t, &cfg(), par);
        assert!(
            degree(&h) >= 0.85 * degree(&fresh),
            "{} vs {}",
            degree(&h),
            degree(&fresh)
        );
    }

    #[test]
    fn writes_after_a_build_keep_every_invariant() {
        let mut t = table(1500, 10);
        let pool = None;
        let par = Par {
            configured: 0,
            pool: &pool,
        };
        let mut h = Hnsw::build(&t, &cfg(), par);
        let model: Arc<str> = Arc::from("m");
        let mut rng = Rng(5);
        for i in 0..400 {
            let v: Vec<f32> = (0..10)
                .map(|_| (rng.below(2001) as f32 - 1000.0) / 1000.0)
                .collect();
            match i % 4 {
                0 | 1 => {
                    let (slot, _) = t.put(&format!("doc:{}", 2000 + i), "doc", &v, "", &model);
                    h.note_put(&t, slot, par);
                }
                2 => {
                    let (slot, old) =
                        t.put(&format!("doc:{}", rng.below(1500)), "doc", &v, "", &model);
                    if old.is_some() {
                        h.note_put(&t, slot, par);
                    }
                }
                _ => {
                    if t.remove(&format!("doc:{}", rng.below(1500))).is_some() {
                        h.note_drop();
                    }
                }
            }
        }
        h.check(&t).unwrap();
        assert!(h.inserts > 0 && h.churn > 0);
    }
}
