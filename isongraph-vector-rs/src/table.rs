//! The single source of truth for a store's vectors: one contiguous matrix.
//!
//! A [`VectorTable`] holds every vector exactly once, in row-major chunks of
//! [`CHUNK_ROWS`] rows, with the per-row facts a scan needs (norm, type code)
//! in parallel arrays and the per-row metadata a result needs (key, text,
//! model) beside them. A record is *a slot number*: nothing else owns a copy of
//! the floats. Searches read the table directly; the approximate index stores
//! only slot numbers.
//!
//! * Rows are in arrival order. An update overwrites its slot, an insert takes
//!   the next one, a delete tombstones one (norm NaN, metadata dropped) and a
//!   quarter of dead rows triggers an order-preserving compaction.
//! * The matrix is chunked so growth never copies what is already stored (a
//!   single `Vec` would copy up to the whole matrix on a doubling) and so a
//!   scan splits into cache-sized blocks that never straddle a chunk.

use crate::ann::dot_f32;
use crate::{EmbeddingRecord, NodeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Rows per chunk (a power of two: slot -> chunk is a shift).
pub(crate) const CHUNK_SHIFT: usize = 10;
pub(crate) const CHUNK_ROWS: usize = 1 << CHUNK_SHIFT;
pub(crate) const CHUNK_MASK: usize = CHUNK_ROWS - 1;

/// What a live row remembers besides its floats.
pub(crate) struct RowMeta {
    /// `"<type>:<id>"`, the record's key (shared with the key index).
    pub key: Arc<str>,
    pub text: Box<str>,
    pub model: Arc<str>,
}

/// The matrix and everything indexed by slot.
pub(crate) struct VectorTable {
    pub dim: usize,
    chunks: Vec<Vec<f32>>,
    /// Euclidean norm per row; NaN marks a tombstone.
    norms: Vec<f32>,
    /// Node-type code per row (index into `types`).
    codes: Vec<u32>,
    meta: Vec<Option<RowMeta>>,
    index: HashMap<Arc<str>, u32>,
    types: Vec<Arc<str>>,
    code_of: HashMap<Arc<str>, u32>,
    /// Slots in use, tombstones included.
    n: usize,
    dead: usize,
}

/// A read-only view of the matrix side, handed to scan kernels.
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub chunks: &'a [Vec<f32>],
    pub norms: &'a [f32],
    pub codes: &'a [u32],
    pub dim: usize,
}

impl<'a> Rows<'a> {
    #[inline(always)]
    pub fn row(&self, slot: usize) -> &'a [f32] {
        let chunk: &'a Vec<f32> = &self.chunks[slot >> CHUNK_SHIFT];
        let at = (slot & CHUNK_MASK) * self.dim;
        &chunk[at..at + self.dim]
    }

    /// The contiguous rows from `slot` to the end of its chunk or `hi`,
    /// whichever comes first: `(floats, row count)`.
    #[inline(always)]
    pub fn run(&self, slot: usize, hi: usize) -> (&'a [f32], usize) {
        let count = (CHUNK_ROWS - (slot & CHUNK_MASK)).min(hi - slot);
        let at = (slot & CHUNK_MASK) * self.dim;
        let chunk: &'a Vec<f32> = &self.chunks[slot >> CHUNK_SHIFT];
        (&chunk[at..at + count * self.dim], count)
    }
}

impl VectorTable {
    pub fn new() -> Self {
        Self {
            dim: 0,
            chunks: Vec::new(),
            norms: Vec::new(),
            codes: Vec::new(),
            meta: Vec::new(),
            index: HashMap::new(),
            types: Vec::new(),
            code_of: HashMap::new(),
            n: 0,
            dead: 0,
        }
    }

    /// Forget every row (the type codes are kept; they are tiny and stable).
    pub fn clear(&mut self) {
        self.chunks = Vec::new();
        self.norms = Vec::new();
        self.codes = Vec::new();
        self.meta = Vec::new();
        self.index = HashMap::new();
        self.n = 0;
        self.dead = 0;
        self.dim = 0;
    }

    /// Rows in use, tombstones included. Slots are `0..slots()`.
    pub fn slots(&self) -> usize {
        self.n
    }

    pub fn live(&self) -> usize {
        self.n - self.dead
    }

    pub fn dead(&self) -> usize {
        self.dead
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn rows(&self) -> Rows<'_> {
        Rows {
            chunks: &self.chunks,
            norms: &self.norms,
            codes: &self.codes,
            dim: self.dim,
        }
    }

    pub fn row(&self, slot: usize) -> &[f32] {
        self.rows().row(slot)
    }

    pub fn norm(&self, slot: usize) -> f32 {
        self.norms[slot]
    }

    pub fn is_live(&self, slot: usize) -> bool {
        self.meta.get(slot).is_some_and(|m| m.is_some())
    }

    pub fn meta(&self, slot: usize) -> Option<&RowMeta> {
        self.meta.get(slot)?.as_ref()
    }

    pub fn slot_of(&self, key: &str) -> Option<u32> {
        self.index.get(key).copied()
    }

    /// The slot's contiguous run of rows from `slot` to the end of its chunk
    /// or `hi`, whichever is first: `(rows as a flat slice, row count)`.
    pub fn run(&self, slot: usize, hi: usize) -> (&[f32], usize) {
        let in_chunk = CHUNK_ROWS - (slot & CHUNK_MASK);
        let count = in_chunk.min(hi - slot);
        let at = (slot & CHUNK_MASK) * self.dim;
        (
            &self.chunks[slot >> CHUNK_SHIFT][at..at + count * self.dim],
            count,
        )
    }

    pub fn type_code(&self, node_type: &str) -> Option<u32> {
        self.code_of.get(node_type).copied()
    }

    fn code_for(&mut self, node_type: &str) -> u32 {
        if let Some(&code) = self.code_of.get(node_type) {
            return code;
        }
        let code = self.types.len() as u32;
        let name: Arc<str> = Arc::from(node_type);
        self.types.push(name.clone());
        self.code_of.insert(name, code);
        code
    }

    pub fn node_id(&self, slot: usize) -> Option<NodeId> {
        let meta = self.meta(slot)?;
        let ty = &self.types[self.codes[slot] as usize];
        Some(NodeId::new(&**ty, &meta.key[ty.len() + 1..]))
    }

    pub fn record(&self, slot: usize) -> Option<EmbeddingRecord> {
        let meta = self.meta(slot)?;
        Some(EmbeddingRecord {
            id: meta.key.to_string(),
            node_id: self.node_id(slot)?,
            vector: self.row(slot).to_vec(),
            text: meta.text.to_string(),
            model: meta.model.to_string(),
        })
    }

    /// Live slots in arrival order.
    pub fn live_slots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(|&s| self.meta[s].is_some())
    }

    /// Insert or overwrite the row for `key`. Returns its slot and the model
    /// of the record it replaced, if any.
    pub fn put(
        &mut self,
        key: &str,
        node_type: &str,
        vector: &[f32],
        text: &str,
        model: &Arc<str>,
    ) -> (u32, Option<Arc<str>>) {
        if self.live() == 0 && self.dim != vector.len() {
            // empty (possibly after deletes): the first vector fixes the width
            self.clear();
            self.dim = vector.len();
        }
        debug_assert_eq!(self.dim, vector.len());
        let code = self.code_for(node_type);
        let norm = dot_f32(vector, vector).sqrt();
        if let Some(&slot) = self.index.get(key) {
            let s = slot as usize;
            let at = (s & CHUNK_MASK) * self.dim;
            self.chunks[s >> CHUNK_SHIFT][at..at + self.dim].copy_from_slice(vector);
            self.norms[s] = norm;
            self.codes[s] = code;
            let meta = self.meta[s].as_mut().expect("indexed rows are live");
            meta.text = text.into();
            let old = std::mem::replace(&mut meta.model, model.clone());
            return (slot, Some(old));
        }
        let slot = self.n;
        if slot & CHUNK_MASK == 0 {
            // later chunks are reserved whole so they never reallocate; the
            // first grows like a Vec so a tiny store stays tiny
            self.chunks.push(if self.chunks.is_empty() {
                Vec::new()
            } else {
                Vec::with_capacity(CHUNK_ROWS * self.dim)
            });
        }
        self.chunks
            .last_mut()
            .expect("a chunk exists")
            .extend_from_slice(vector);
        self.norms.push(norm);
        self.codes.push(code);
        let key: Arc<str> = Arc::from(key);
        self.meta.push(Some(RowMeta {
            key: key.clone(),
            text: text.into(),
            model: model.clone(),
        }));
        self.index.insert(key, slot as u32);
        self.n += 1;
        (slot as u32, None)
    }

    /// Tombstone the row for `key`. Returns its slot and the model it had.
    pub fn remove(&mut self, key: &str) -> Option<(u32, Arc<str>)> {
        let slot = self.index.remove(key)?;
        let s = slot as usize;
        self.norms[s] = f32::NAN;
        self.dead += 1;
        let meta = self.meta[s].take().expect("indexed rows are live");
        Some((slot, meta.model))
    }

    pub fn should_compact(&self) -> bool {
        self.dead > 0 && self.dead * 4 > self.n
    }

    /// Drop the tombstones, keeping row order. Returns the old slot of every
    /// surviving row, in order, so an index can follow the move.
    pub fn compact(&mut self) -> Vec<u32> {
        let dim = self.dim;
        let live: Vec<u32> = (0..self.n as u32)
            .filter(|&s| self.meta[s as usize].is_some())
            .collect();
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
                // new < old, so the destination chunk is the earlier one
                let (head, tail) = self.chunks.split_at_mut(oc);
                head[nc][na..na + dim].copy_from_slice(&tail[0][oa..oa + dim]);
            }
            self.norms[new] = self.norms[old];
            self.codes[new] = self.codes[old];
            self.meta.swap(new, old);
            if let Some(meta) = &self.meta[new] {
                if let Some(slot) = self.index.get_mut(&*meta.key) {
                    *slot = new as u32;
                }
            }
        }
        let kept = live.len();
        self.norms.truncate(kept);
        self.codes.truncate(kept);
        self.meta.truncate(kept);
        let chunks = kept.div_ceil(CHUNK_ROWS);
        self.chunks.truncate(chunks);
        if let Some(last) = self.chunks.last_mut() {
            last.truncate((kept - (chunks - 1) * CHUNK_ROWS) * dim);
        }
        self.n = kept;
        self.dead = 0;
        live
    }

    /// Every structural invariant of the table, or the first one broken.
    pub fn check(&self) -> Result<(), String> {
        let n = self.n;
        if self.norms.len() != n || self.codes.len() != n || self.meta.len() != n {
            return Err(format!(
                "parallel arrays disagree: n={n} norms={} codes={} meta={}",
                self.norms.len(),
                self.codes.len(),
                self.meta.len()
            ));
        }
        let floats: usize = self.chunks.iter().map(|c| c.len()).sum();
        if floats != n * self.dim {
            return Err(format!(
                "matrix holds {floats} floats, expected {}",
                n * self.dim
            ));
        }
        if self.chunks.len() != n.div_ceil(CHUNK_ROWS) {
            return Err(format!("{} chunks for {n} rows", self.chunks.len()));
        }
        for (i, chunk) in self.chunks.iter().enumerate() {
            if i + 1 < self.chunks.len() && chunk.len() != CHUNK_ROWS * self.dim {
                return Err(format!("chunk {i} is not full ({} floats)", chunk.len()));
            }
        }
        let dead = self.meta.iter().filter(|m| m.is_none()).count();
        if dead != self.dead {
            return Err(format!("dead counter {} but {dead} tombstones", self.dead));
        }
        if self.index.len() != n - dead {
            return Err(format!(
                "index holds {} keys for {} live rows",
                self.index.len(),
                n - dead
            ));
        }
        for (slot, meta) in self.meta.iter().enumerate() {
            match meta {
                None => {
                    if !self.norms[slot].is_nan() {
                        return Err(format!("tombstone {slot} has norm {}", self.norms[slot]));
                    }
                }
                Some(m) => {
                    if self.index.get(&*m.key).copied() != Some(slot as u32) {
                        return Err(format!("key {:?} does not map back to slot {slot}", m.key));
                    }
                    let ty = self
                        .types
                        .get(self.codes[slot] as usize)
                        .ok_or_else(|| format!("slot {slot} has an unknown type code"))?;
                    if !m.key.starts_with(&**ty) || m.key.as_bytes().get(ty.len()) != Some(&b':') {
                        return Err(format!("key {:?} does not start with type {ty:?}", m.key));
                    }
                }
            }
        }
        Ok(())
    }

    /// Bytes held by the table, by part (capacity, i.e. what is allocated).
    pub fn bytes(&self) -> TableBytes {
        let matrix: usize = self.chunks.iter().map(|c| c.capacity() * 4).sum();
        let per_row_arrays = self.norms.capacity() * 4
            + self.codes.capacity() * 4
            + self.meta.capacity() * std::mem::size_of::<Option<RowMeta>>();
        let mut strings = 0usize;
        for m in self.meta.iter().flatten() {
            // key (shared with the index), text, and the allocator's headers
            strings += 16 + m.key.len().max(1) + m.text.len();
        }
        // hashbrown: 1 control byte + (Arc<str>, u32) per bucket, ~7/8 load
        let index = self.index.capacity() * (std::mem::size_of::<(Arc<str>, u32)>() + 1);
        TableBytes {
            matrix,
            per_row: per_row_arrays,
            keys_and_text: strings,
            key_index: index,
        }
    }
}

/// Where a table's memory goes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TableBytes {
    pub matrix: usize,
    pub per_row: usize,
    pub keys_and_text: usize,
    pub key_index: usize,
}
