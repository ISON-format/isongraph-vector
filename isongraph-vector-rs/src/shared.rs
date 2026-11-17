//! Readers and a writer on different threads.
//!
//! [`EmbeddingStore`] is already safe to *read* from any number of threads
//! (`&self` searches never lock the data), and its writes need `&mut self`, so
//! a search cannot overlap a write: the borrow checker is the isolation
//! mechanism. [`SharedEmbeddingStore`] is the thin layer that lets different
//! threads hold the two roles at once: a cloneable handle around
//! `Arc<RwLock<EmbeddingStore>>`.
//!
//! * A search holds the read lock for the length of the scan. Any number run
//!   at once.
//! * A write holds the write lock for one `put`/`remove`: a row copy and a few
//!   index updates, plus (every [`FOLD_BATCH`](crate::FOLD_BATCH) writes) one
//!   bounded fold. Encoding a text happens *before* the lock is taken.
//! * A reader therefore sees the store before a write or after it, never in
//!   between; a writer waits for the scans in flight (milliseconds) and holds
//!   readers off for microseconds.
//!
//! Rejected alternative: copy-on-write snapshots (readers pin an `Arc` of the
//! data, writers publish a new one). With a matrix of gigabytes a writer cannot
//! copy it per write, and chunked structural sharing buys a reader nothing the
//! lock does not already give, at the price of a far harder-to-port design.

use crate::{EmbeddingError, EmbeddingRecord, EmbeddingStore, NodeId, SimilarityResult};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A cloneable, thread-safe handle to one [`EmbeddingStore`]. See the module
/// docs for the guarantees.
#[derive(Clone)]
pub struct SharedEmbeddingStore {
    inner: Arc<RwLock<EmbeddingStore>>,
}

impl SharedEmbeddingStore {
    pub fn new(store: EmbeddingStore) -> Self {
        Self {
            inner: Arc::new(RwLock::new(store)),
        }
    }

    /// Shared access, for calling any `&self` method (several threads at once).
    /// A poisoned lock (a thread panicked inside a write, which is a bug) is
    /// still handed out rather than turning every later call into a panic.
    pub fn read(&self) -> RwLockReadGuard<'_, EmbeddingStore> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Exclusive access, for any `&mut self` method. Keep it short.
    pub fn write(&self) -> RwLockWriteGuard<'_, EmbeddingStore> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Add or replace one embedding. With no vector the text is encoded first,
    /// under the read lock, so a slow encoder never holds up the writer queue.
    pub fn add(
        &self,
        node_id: NodeId,
        text: &str,
        vector: Option<Vec<f32>>,
    ) -> Result<String, EmbeddingError> {
        let vector = match vector {
            Some(vector) => vector,
            None => self.read().encode(text)?,
        };
        self.write().add(node_id, text, Some(vector))
    }

    pub fn remove(&self, node_id: &NodeId) -> bool {
        self.write().remove(node_id)
    }

    pub fn get(&self, node_id: &NodeId) -> Option<EmbeddingRecord> {
        self.read().get(node_id)
    }

    pub fn count(&self) -> usize {
        self.read().count()
    }

    pub fn similarity_search_vector(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.read()
            .similarity_search_vector(query_vector, top_k, node_type, threshold)
    }

    pub fn similarity_search_vector_batch<V: AsRef<[f32]> + Sync>(
        &self,
        queries: &[V],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<Vec<SimilarityResult>>, EmbeddingError> {
        self.read()
            .similarity_search_vector_batch(queries, top_k, node_type, threshold)
    }
}
