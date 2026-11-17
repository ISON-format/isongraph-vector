//! ISONGraph Vector - Semantic Graph Extension for Rust
//!
//! Extends ISONGraph with embedding-based similarity search and semantic traversal.
//!
//! # Example
//!
//! ```rust
//! use isongraph_vector_rs::{SemanticGraph, MockEncoder, PropertyValue};
//!
//! let mut graph = SemanticGraph::new("semantic", Box::new(MockEncoder::new(384)));
//!
//! // Property values are ISON values, so a property can be text, a number or
//! // a bool. add_node_with_embed still takes plain &str values.
//! graph.add_node(
//!     "person", "1",
//!     vec![("name", PropertyValue::String("Alice".into()))],
//!     Some("Alice is an engineer"),
//! ).unwrap();
//! let results = graph.similarity_search("engineer", 5, None, -1.0).unwrap();
//! ```
//!
//! Author: Mahesh Vaikri

use ison_graph_rs::{Direction, Edge, ISONGraph, Node, Value};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

mod ann;
pub use ann::{
    AnnConfig, AnnMode, AnnSearchCounts, AnnStats, MemoryStats, RecallReport, ANN_MIN_SIZE,
    FOLD_BATCH,
};
mod hnsw;
pub use hnsw::{HnswConfig, HNSW_EF_CONSTRUCTION, HNSW_M};
mod pool;
pub use pool::SearchPool;
mod shared;
pub use shared::SharedEmbeddingStore;
mod table;
use table::VectorTable;
mod digest;
pub use digest::{
    graph_digest, graph_digest_of_text, graph_digest_with_int_ids, GRAPH_DIGEST_BASIS,
};

/// Property values are ISON values, re-exported so callers do not need a
/// direct dependency on `ison-graph` to build a property list.
pub use ison_graph_rs::Value as PropertyValue;

/// Node identity, re-exported for the same reason: it appears in every store
/// signature, so a caller who never touches `ISONGraph` directly would
/// otherwise still need `ison-graph` in their Cargo.toml just to name one.
pub use ison_graph_rs::NodeId;

/// SQLite-backed storage, behind the `sqlite` feature.
#[cfg(feature = "sqlite")]
pub mod sqlite_store;
#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteEmbeddingStore;

/// HTTP embeddings endpoint as an encoder, behind the `http-encoder` feature.
/// Texts are sent to the configured URL.
#[cfg(feature = "http-encoder")]
pub mod http_encoder;
#[cfg(feature = "http-encoder")]
pub use http_encoder::{HttpEncoder, HttpEncoderError};

pub const VERSION: &str = "1.0.0";

/// Portable embedding payload shared by every language port, so a store built
/// in one can be searched by another.
pub const EMBEDDING_FORMAT: &str = "ison-embeddings";
pub const EMBEDDING_FORMAT_VERSION: u32 = 1;

/// Suffix appended to a saved graph's path to name the embeddings sidecar
/// (`graph.ison` -> `graph.ison.embeddings.json`). Identical in every port.
pub const EMBEDDING_SIDECAR_SUFFIX: &str = ".embeddings.json";

/// Model names that carry no information about which encoder made a vector:
/// `"provided"` (a vector handed in while the store has no encoder) and
/// `"unknown"` (payloads from ports that do not record a model). Vectors under
/// either are never judged stale by model unless asked to be.
pub const UNKNOWN_MODELS: [&str; 2] = ["unknown", "provided"];

fn is_unknown_model(model: &str) -> bool {
    UNKNOWN_MODELS.contains(&model)
}

/// The message for a text query whose encoder is not the stored vectors'
/// model, and the key that de-duplicates its warning (one per model pair).
/// Sorts `foreign` in place. Shared by the in-memory and SQLite stores so the
/// wording cannot drift between them.
pub(crate) fn mismatch_report(current: &str, foreign: &mut [&String]) -> (String, String) {
    foreign.sort();
    let listed = foreign
        .iter()
        .map(|m| format!("'{m}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let message = format!(
        "encoder model '{current}' differs from the model that produced the stored \
         vectors ({listed}); scores compare vectors from different spaces. Re-embed \
         with reembed_stale() or search with the original encoder."
    );
    let key = format!(
        "{current}\u{0}{}",
        foreign
            .iter()
            .map(|m| m.as_str())
            .collect::<Vec<_>>()
            .join("\u{0}")
    );
    (message, key)
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("No encoder provided")]
    NoEncoder,
    #[error("Graph error: {0}")]
    GraphError(String),
    /// A vector was stored or queried with the wrong number of dimensions.
    ///
    /// Cosine similarity over a mismatched pair returns a plausible-looking
    /// number rather than an error, so a store that quietly accepted mixed
    /// widths would degrade silently. A store pins its dimension to the first
    /// vector it sees.
    #[error("{what} has {got} dimensions, store holds {expected}")]
    DimensionMismatch {
        what: String,
        got: usize,
        expected: usize,
    },
    #[error("{0} is empty")]
    EmptyVector(String),
    #[error("blend must be in [0, 1], got {0}")]
    InvalidBlend(f32),
    #[error("embedding payload error: {0}")]
    Payload(String),
    /// The SQLite-backed store failed at the database layer.
    #[error("storage error: {0}")]
    Storage(String),
    /// Reading or writing a file (such as the embeddings sidecar) failed.
    #[error("I/O error: {0}")]
    Io(String),
    /// A text query was encoded by a different model than the stored vectors.
    ///
    /// Two models of the same dimension give a plausible-looking cosine score
    /// between vectors that live in unrelated spaces, so the ranking is noise
    /// rather than an error. Returned only with `ModelMismatch::Error`; the
    /// default is a warning.
    #[error("{0}")]
    ModelMismatch(String),
    /// A graph file and its embeddings sidecar are not from the same save:
    /// the sidecar's recorded `graph_digest` does not match the graph read.
    /// Returned by `SemanticGraph::load` with `GraphMismatch::Error` (default).
    #[error("{0}")]
    GraphEmbeddingMismatch(String),
    /// An option had a value the operation cannot honour.
    #[error("invalid option: {0}")]
    InvalidOption(String),
}

/// What a *text* query does when the encoder's model differs from the model
/// that produced the stored vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelMismatch {
    /// One warning per distinct pair of models, per store (the default).
    #[default]
    Warn,
    /// Fail with `EmbeddingError::ModelMismatch`.
    Error,
    /// Say nothing.
    Ignore,
}

/// What `SemanticGraph::load` does when the sidecar's recorded graph digest
/// does not match the graph file (a sidecar from a different save, or a graph
/// edited since).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GraphMismatch {
    /// Fail with `EmbeddingError::GraphEmbeddingMismatch` (the default).
    #[default]
    Error,
    /// Warn, and load the sidecar's vectors anyway.
    Warn,
    /// Discard the sidecar's vectors and re-embed every node with the encoder.
    Reembed,
}

// =============================================================================
// Embedding Encoder Trait
// =============================================================================

pub trait EmbeddingEncoder: Send + Sync {
    fn dimension(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<f32>;
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Name of the model behind this encoder, stamped on every vector it
    /// writes. It decides whether stored vectors are *stale* for this encoder
    /// and whether a text query is searching a foreign vector space. The
    /// default, `"unknown"`, makes no claim.
    fn model_name(&self) -> &str {
        "unknown"
    }

    /// Optional free-form encoder/model version (a revision, a weights hash),
    /// recorded in the sidecar as `encoder_version`.
    fn version(&self) -> Option<&str> {
        None
    }
}

// =============================================================================
// Mock Encoder
// =============================================================================

/// Deterministic encoder with no dependencies, for tests and offline work.
///
/// The same text produces the same vector in *every* language port
/// (Python/TypeScript/JavaScript/Rust/C++), so a store built by one can be
/// searched by another: a 32-bit FNV-1a hash over the UTF-8 bytes seeds an
/// xorshift32 generator, and the top 24 bits of each state become a value in
/// `[-1, 1)`.
///
/// This replaces a scheme that emitted the *low* 16 bits of an LCG. The low
/// bits of an LCG evolve independently of the high bits, so every vector was
/// determined by `seed mod 65536` - with exact `u32` arithmetic that meant 280
/// outright collisions in 5,000 distinct texts, with colliding texts scoring a
/// perfect 1.0 against each other.
///
/// These are pseudo-embeddings with no semantic structure. Use them to test
/// plumbing, never to judge relevance quality.
pub struct MockEncoder {
    dimension: usize,
    model_name: String,
    version: Option<String>,
}

impl MockEncoder {
    const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;

    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        Self {
            dimension,
            model_name: format!("mock-{dimension}"),
            version: None,
        }
    }

    /// Report a different model name (default `mock-<dimension>`), to stand in
    /// for a second model of the same width.
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Report an encoder version, recorded in the sidecar.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

impl EmbeddingEncoder for MockEncoder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let mut state = Self::FNV_OFFSET_BASIS;
        for byte in text.as_bytes() {
            state = (state ^ (*byte as u32)).wrapping_mul(Self::FNV_PRIME);
        }
        // xorshift32 has no way out of a zero state; steer off it.
        if state == 0 {
            state = 0x9e37_79b9;
        }

        let mut values = Vec::with_capacity(self.dimension);
        for _ in 0..self.dimension {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Top 24 bits -> [-1, 1)
            values.push((state >> 8) as f32 / 8_388_608.0 - 1.0);
        }
        values
    }
}

// =============================================================================
// Embedding Record
// =============================================================================

#[derive(Clone, Debug)]
pub struct EmbeddingRecord {
    pub id: String,
    pub node_id: NodeId,
    pub vector: Vec<f32>,
    pub text: String,
    /// Model that produced the vector (`"unknown"` / `"provided"` when it is
    /// not known; see `UNKNOWN_MODELS`).
    pub model: String,
}

// =============================================================================
// Similarity Result
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimilarityResult {
    pub node_id: NodeId,
    pub score: f32,
    pub text: String,
}

// =============================================================================
// Portable payload
// =============================================================================

/// A node id as it crosses the wire.
///
/// Ports whose node ids are always strings write `"id": "007"`; the ports that
/// keep an integer id write `"id": 1`. Both have to land on the same Rust
/// string id, so accept either form.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingId {
    Str(String),
    Int(i64),
}

impl EmbeddingId {
    pub fn as_string(&self) -> String {
        match self {
            EmbeddingId::Str(s) => s.clone(),
            EmbeddingId::Int(i) => i.to_string(),
        }
    }
}

/// One entry of the portable embedding payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedEmbedding {
    #[serde(rename = "type")]
    pub node_type: String,
    pub id: EmbeddingId,
    #[serde(default = "default_id_type")]
    pub id_type: String,
    #[serde(default)]
    pub text: String,
    /// Model that produced the vector; absent in payloads from ports that do
    /// not record one, which import as `"unknown"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub vector: Vec<f32>,
}

fn default_id_type() -> String {
    "str".to_string()
}

/// Payload written by `export_embeddings`, readable by every language port.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedEmbeddings {
    pub format: String,
    pub version: u32,
    pub dimension: Option<usize>,
    pub count: usize,
    pub embeddings: Vec<ExportedEmbedding>,
    // Optional metadata, appended after `embeddings` as flat fields. Version 1
    // of the payload is unchanged: readers that do not know these ignore them,
    // and absent values are omitted rather than written as null.
    /// Model name of the encoder that wrote the payload.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_string"
    )]
    pub encoder_model: Option<String>,
    /// Version of that encoder, when it has one.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_string"
    )]
    pub encoder_version: Option<String>,
    /// `graph_digest` of the graph this payload was saved with.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_string"
    )]
    pub graph_digest: Option<String>,
    /// Which serialization `graph_digest` hashes (`GRAPH_DIGEST_BASIS`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_string"
    )]
    pub graph_digest_basis: Option<String>,
}

/// Read an optional metadata field as text, whatever scalar another port
/// wrote it as (a version may arrive as the number 2): metadata must never be
/// the reason a payload fails to import.
fn lenient_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    Ok(match value {
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Number(n)) => Some(n.to_string()),
        Some(serde_json::Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    })
}

/// Metadata of the most recently imported payload (empty before any import).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportMeta {
    pub encoder_model: Option<String>,
    pub encoder_version: Option<String>,
    pub graph_digest: Option<String>,
    pub graph_digest_basis: Option<String>,
}

// =============================================================================
// Embedding Store
// =============================================================================

/// A search query: text to be encoded, or a vector to search with directly.
///
/// `&str`, `&String`, `&[f32]`, `&Vec<f32>` and `&[f32; N]` all convert, so
/// the `*_with` methods take `"some text"` or `&vector` alike. Only text
/// queries are checked against the stored vectors' model.
#[derive(Clone, Copy, Debug)]
pub enum Query<'a> {
    Text(&'a str),
    Vector(&'a [f32]),
}

impl<'a> From<&'a str> for Query<'a> {
    fn from(text: &'a str) -> Self {
        Query::Text(text)
    }
}
impl<'a> From<&'a String> for Query<'a> {
    fn from(text: &'a String) -> Self {
        Query::Text(text.as_str())
    }
}
impl<'a> From<&'a [f32]> for Query<'a> {
    fn from(vector: &'a [f32]) -> Self {
        Query::Vector(vector)
    }
}
impl<'a> From<&'a Vec<f32>> for Query<'a> {
    fn from(vector: &'a Vec<f32>) -> Self {
        Query::Vector(vector.as_slice())
    }
}
impl<'a, const N: usize> From<&'a [f32; N]> for Query<'a> {
    fn from(vector: &'a [f32; N]) -> Self {
        Query::Vector(vector.as_slice())
    }
}

/// Write `contents` to `path` so a reader sees the old file or the new one.
///
/// The bytes go to a temp file in the same directory (same filesystem, so the
/// rename is not a copy), are flushed to disk, and are then renamed into
/// place. A crash mid-write leaves the previous file intact instead of a
/// truncated one, and a failed write removes its temp file.
pub(crate) fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), EmbeddingError> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let io_err = |what: &str, e: std::io::Error| {
        EmbeddingError::Io(format!("{} {}: {}", what, path.display(), e))
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let tmp = path.with_file_name(format!(
        ".{}.{}.{}.tmp",
        name,
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));

    let result = (|| -> Result<(), EmbeddingError> {
        let mut file = std::fs::File::create(&tmp).map_err(|e| io_err("writing", e))?;
        file.write_all(contents).map_err(|e| io_err("writing", e))?;
        file.flush().map_err(|e| io_err("writing", e))?;
        // A filesystem that cannot fsync still gets the atomic rename.
        let _ = file.sync_all();
        drop(file);
        std::fs::rename(&tmp, path).map_err(|e| io_err("replacing", e))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Vectors, their text and model, searchable by cosine similarity.
///
/// The vectors live once, in a contiguous matrix; a record is a row of it (see
/// `DESIGN.md`). Searches take `&self`, never lock the data and may run from
/// any number of threads at once; writes take `&mut self`, so a search can
/// never see a half-applied write. To read and write from different threads,
/// share the store through [`SharedEmbeddingStore`] (a reader/writer lock with
/// short write sections).
pub struct EmbeddingStore {
    /// The single copy of every vector, plus the per-row metadata and the key
    /// index. Everything else derived from it (the approximate index) refers to
    /// rows by slot number.
    table: VectorTable,
    /// One shared string per distinct model name, so a row holds a pointer.
    model_names: HashMap<String, Arc<str>>,
    encoder: Option<Box<dyn EmbeddingEncoder>>,
    dimension: Option<usize>,
    version: Option<String>,
    on_model_mismatch: ModelMismatch,
    /// Vectors per model, kept in step with `embeddings`.
    model_counts: HashMap<String, usize>,
    /// Model pairs already warned about (one warning per pair).
    warned: Mutex<HashSet<String>>,
    /// Every warning this store has issued, so callers and tests can read them.
    warnings: Mutex<Vec<String>>,
    last_import_meta: ImportMeta,
    /// The opt-in approximate index settings (`None` = every search exact).
    ann: Option<AnnConfig>,
    /// The most threads a search, batch, fold or build may use (0 = automatic).
    scan_threads: usize,
    /// The pool parallel work runs on (`None` = the process-wide one).
    pool: Option<Arc<SearchPool>>,
    /// The approximate index: built lazily by a search (the only time the lock
    /// is written during a search), maintained by writes through `&mut self`
    /// (no locking), never persisted.
    ivf: ann::IvfSlot,
    /// The HNSW graph (same life cycle as `ivf`; only one of the two is ever
    /// built, the one `ann.mode` names).
    hnsw: hnsw::HnswSlot,
    counts: ann::Counters,
}

impl EmbeddingStore {
    pub fn new(encoder: Option<Box<dyn EmbeddingEncoder>>) -> Self {
        Self {
            table: VectorTable::new(),
            model_names: HashMap::new(),
            encoder,
            dimension: None,
            version: None,
            on_model_mismatch: ModelMismatch::Warn,
            model_counts: HashMap::new(),
            warned: Mutex::new(HashSet::new()),
            warnings: Mutex::new(Vec::new()),
            last_import_meta: ImportMeta::default(),
            ann: None,
            scan_threads: 0,
            pool: None,
            ivf: RwLock::new(None),
            hnsw: RwLock::new(None),
            counts: ann::Counters::default(),
        }
    }

    /// Vector length this store holds, or `None` while it is still empty.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    pub fn set_encoder(&mut self, encoder: Box<dyn EmbeddingEncoder>) {
        self.encoder = Some(encoder);
    }

    /// The encoder in use, if any.
    pub fn encoder(&self) -> Option<&dyn EmbeddingEncoder> {
        self.encoder.as_deref()
    }

    /// Detach and return the encoder. Vectors added afterwards are recorded as
    /// `"provided"`.
    pub fn take_encoder(&mut self) -> Option<Box<dyn EmbeddingEncoder>> {
        self.encoder.take()
    }

    /// Encoder/model version recorded in the sidecar: the explicit one set with
    /// `set_version`, else the encoder's own, else `None`.
    pub fn version(&self) -> Option<String> {
        self.version.clone().or_else(|| {
            self.encoder
                .as_ref()
                .and_then(|e| e.version().map(String::from))
        })
    }

    /// Set (or, with `None`, clear) the explicit version; an explicit version
    /// wins over the encoder's.
    pub fn set_version(&mut self, version: Option<String>) {
        self.version = version;
    }

    pub fn on_model_mismatch(&self) -> ModelMismatch {
        self.on_model_mismatch
    }

    pub fn set_on_model_mismatch(&mut self, mode: ModelMismatch) {
        self.on_model_mismatch = mode;
    }

    /// Every warning this store has issued (model mismatch, and the notices
    /// `SemanticGraph::load` raises). Each is also printed to stderr.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.lock().map(|w| w.clone()).unwrap_or_default()
    }

    pub fn clear_warnings(&self) {
        if let Ok(mut w) = self.warnings.lock() {
            w.clear();
        }
    }

    pub(crate) fn warn(&self, message: String) {
        eprintln!("warning: {message}");
        if let Ok(mut w) = self.warnings.lock() {
            w.push(message);
        }
    }

    /// Metadata of the most recent imported sidecar (empty before any).
    pub fn last_import_meta(&self) -> &ImportMeta {
        &self.last_import_meta
    }

    fn node_id_to_key(node_id: &NodeId) -> String {
        format!("{}:{}", node_id.node_type, node_id.id)
    }

    fn check_dimension(&mut self, vector: &[f32], what: &str) -> Result<(), EmbeddingError> {
        if vector.is_empty() {
            return Err(EmbeddingError::EmptyVector(what.to_string()));
        }
        match self.dimension {
            None => {
                self.dimension = Some(vector.len());
                Ok(())
            }
            Some(expected) if expected == vector.len() => Ok(()),
            Some(expected) => Err(EmbeddingError::DimensionMismatch {
                what: what.to_string(),
                got: vector.len(),
                expected,
            }),
        }
    }

    fn check_query(&self, vector: &[f32]) -> Result<(), EmbeddingError> {
        if vector.is_empty() {
            return Err(EmbeddingError::EmptyVector("query vector".to_string()));
        }
        match self.dimension {
            Some(expected) if expected != vector.len() => Err(EmbeddingError::DimensionMismatch {
                what: "query vector".to_string(),
                got: vector.len(),
                expected,
            }),
            _ => Ok(()),
        }
    }

    // -- model lifecycle ------------------------------------------------------

    /// Model name stamped on a vector written now: the encoder's, or
    /// `"provided"` when the store has none.
    fn write_model(&self) -> String {
        self.encoder
            .as_ref()
            .map(|e| e.model_name().to_string())
            .unwrap_or_else(|| "provided".to_string())
    }

    /// Which models produced the stored vectors: `{model_name: count}`.
    ///
    /// `"unknown"` / `"provided"` appear as-is (vectors imported from a port
    /// that records no model, or handed in with no encoder attached).
    pub fn models(&self) -> HashMap<String, usize> {
        self.model_counts.clone()
    }

    /// The encoder's model name, or `None` when there is no encoder or it does
    /// not say (`"unknown"` / `"provided"`).
    fn current_model(&self) -> Option<String> {
        let name = self.encoder.as_ref()?.model_name();
        if name.is_empty() || is_unknown_model(name) {
            None
        } else {
            Some(name.to_string())
        }
    }

    /// Complain when a text query's encoder is not the stored vectors' model.
    fn check_model_compat(&self) -> Result<(), EmbeddingError> {
        if self.on_model_mismatch == ModelMismatch::Ignore {
            return Ok(());
        }
        let current = match self.current_model() {
            Some(current) => current,
            None => return Ok(()),
        };
        let mut foreign: Vec<&String> = self
            .model_counts
            .keys()
            .filter(|m| !is_unknown_model(m) && **m != current)
            .collect();
        if foreign.is_empty() {
            return Ok(());
        }
        let (message, key) = mismatch_report(&current, &mut foreign);
        if self.on_model_mismatch == ModelMismatch::Error {
            return Err(EmbeddingError::ModelMismatch(message));
        }
        let first_time = self
            .warned
            .lock()
            .map(|mut set| set.insert(key))
            .unwrap_or(false);
        if first_time {
            self.warn(message);
        }
        Ok(())
    }

    /// Node ids whose stored vector does not belong to the current encoder,
    /// sorted by type and id.
    ///
    /// A vector is stale when its recorded model differs from `model`
    /// (default: the encoder's model name) or its length differs from
    /// `dimension` (default: the encoder's dimension). Vectors whose model is
    /// `"unknown"` / `"provided"` cannot be judged by model and are skipped
    /// unless `include_unknown` is true (a dimension mismatch still counts).
    /// With no encoder and no arguments nothing can be judged and the result
    /// is empty.
    pub fn stale(
        &self,
        model: Option<&str>,
        dimension: Option<usize>,
        include_unknown: bool,
    ) -> Vec<NodeId> {
        let model = model.map(String::from).or_else(|| self.current_model());
        let dimension = dimension.or_else(|| self.encoder.as_ref().map(|e| e.dimension()));
        let by_dim = dimension.is_some_and(|d| self.table.dim != d);
        let mut out: Vec<NodeId> = self
            .table
            .live_slots()
            .filter(|&slot| {
                let meta = self.table.meta(slot).expect("live rows have metadata");
                let by_model = model.as_ref().is_some_and(|m| {
                    *meta.model != **m && (include_unknown || !is_unknown_model(&meta.model))
                });
                by_model || by_dim
            })
            .filter_map(|slot| self.table.node_id(slot))
            .collect();
        out.sort_by(|a, b| (&a.node_type, &a.id).cmp(&(&b.node_type, &b.id)));
        out
    }

    /// Encode one text with the store's encoder (no model-compatibility
    /// check; that is for queries, see `resolve_query`).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let encoder = self.encoder.as_ref().ok_or(EmbeddingError::NoEncoder)?;
        Ok(encoder.encode(text))
    }

    /// Turn a query into a vector. Text needs an encoder, and is checked
    /// against the stored vectors' model; a vector passes through untouched.
    pub fn resolve_query(&self, query: Query) -> Result<Vec<f32>, EmbeddingError> {
        match query {
            Query::Vector(vector) => Ok(vector.to_vec()),
            Query::Text(text) => {
                let encoder = self.encoder.as_ref().ok_or(EmbeddingError::NoEncoder)?;
                self.check_model_compat()?;
                Ok(encoder.encode(text))
            }
        }
    }

    // -- writes ---------------------------------------------------------------

    /// The shared string for a model name.
    fn intern_model(&mut self, model: &str) -> Arc<str> {
        if let Some(name) = self.model_names.get(model) {
            return name.clone();
        }
        let name: Arc<str> = Arc::from(model);
        self.model_names.insert(model.to_string(), name.clone());
        name
    }

    /// One fewer record carries `model`.
    fn uncount_model(&mut self, model: &str) {
        if let Some(n) = self.model_counts.get_mut(model) {
            *n -= 1;
            if *n == 0 {
                self.model_counts.remove(model);
            }
        }
    }

    /// Insert or replace one record, keeping the per-model counts in step. An
    /// update overwrites the row in place (same slot); an insert takes the next
    /// one. The write is complete when this returns: the row, its metadata, the
    /// key index and the approximate index (including any fold it triggers).
    fn put(&mut self, node_id: NodeId, text: &str, vector: Vec<f32>, model: &str) -> String {
        let key = Self::node_id_to_key(&node_id);
        let model_name = self.intern_model(model);
        let (slot, replaced) = self
            .table
            .put(&key, &node_id.node_type, &vector, text, &model_name);
        if let Some(old) = replaced {
            self.uncount_model(&old);
        }
        *self.model_counts.entry(model.to_string()).or_insert(0) += 1;
        self.index_row_written(slot);
        key
    }

    /// Remove a record: tombstone its row and take it out of the index.
    fn forget(&mut self, key: &str) -> bool {
        match self.table.remove(key) {
            Some((slot, model)) => {
                self.uncount_model(&model);
                self.index_row_removed(slot);
                true
            }
            None => false,
        }
    }

    /// Store a vector copied from another store, keeping the model it was
    /// made by (rather than stamping it with this store's encoder).
    pub(crate) fn add_with_model(
        &mut self,
        node_id: NodeId,
        text: &str,
        vector: Vec<f32>,
        model: &str,
    ) -> Result<String, EmbeddingError> {
        self.check_dimension(&vector, "vector")?;
        Ok(self.put(node_id, text, vector, model))
    }

    pub fn add(
        &mut self,
        node_id: NodeId,
        text: &str,
        vector: Option<Vec<f32>>,
    ) -> Result<String, EmbeddingError> {
        let vector = match vector {
            Some(v) => v,
            None => match &self.encoder {
                Some(enc) => enc.encode(text),
                None => return Err(EmbeddingError::NoEncoder),
            },
        };
        self.check_dimension(&vector, "vector")?;
        let model = self.write_model();
        Ok(self.put(node_id, text, vector, &model))
    }

    /// Add many embeddings, encoding their texts in one batched call.
    pub fn add_batch(&mut self, entries: &[(NodeId, &str)]) -> Result<usize, EmbeddingError> {
        if entries.is_empty() {
            return Ok(0);
        }
        let texts: Vec<&str> = entries.iter().map(|(_, text)| *text).collect();
        let vectors = match &self.encoder {
            Some(enc) => enc.encode_batch(&texts),
            None => return Err(EmbeddingError::NoEncoder),
        };
        if vectors.len() != entries.len() {
            return Err(EmbeddingError::Payload(format!(
                "encoder returned {} vectors for {} texts",
                vectors.len(),
                entries.len()
            )));
        }
        for vector in &vectors {
            self.check_dimension(vector, "vector")?;
        }
        let model = self.write_model();
        for ((node_id, text), vector) in entries.iter().zip(vectors) {
            self.put(node_id.clone(), text, vector, &model);
        }
        Ok(entries.len())
    }

    /// The record for a node, copied out of the store (the vector is a row of
    /// the shared matrix, so a record is built on request; use
    /// [`get_vector`](Self::get_vector) to borrow it instead).
    pub fn get(&self, node_id: &NodeId) -> Option<EmbeddingRecord> {
        let key = Self::node_id_to_key(node_id);
        self.table.record(self.table.slot_of(&key)? as usize)
    }

    /// The stored vector of a node, borrowed from the matrix.
    pub fn get_vector(&self, node_id: &NodeId) -> Option<&[f32]> {
        let key = Self::node_id_to_key(node_id);
        Some(self.table.row(self.table.slot_of(&key)? as usize))
    }

    pub fn remove(&mut self, node_id: &NodeId) -> bool {
        let key = Self::node_id_to_key(node_id);
        self.forget(&key)
    }

    pub fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let query_vector = self.resolve_query(Query::Text(query))?;
        self.similarity_search_vector(&query_vector, top_k, node_type, threshold)
    }

    /// Best `top_k` matches for a vector, best first; equal scores come back
    /// in a stable order (the order the vectors were stored in).
    ///
    /// `top_k = usize::MAX` returns every match. With an approximate index
    /// configured (`set_ann`) a bounded `top_k` on a large enough store is
    /// answered from it; use `similarity_search_vector_exact` to force the
    /// full scan.
    pub fn similarity_search_vector(
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.search_vector(query_vector, top_k, node_type, threshold, false)
    }

    /// Cosine score for every embedded node, keyed by `type:id`.
    ///
    /// One scan serving both seed selection and the per-node relevance that
    /// `semantic_multi_hop` blends in, instead of two. Always the full scan.
    pub fn score_map(&self, query_vector: &[f32]) -> Result<HashMap<String, f32>, EmbeddingError> {
        self.scan_scores(query_vector)
    }

    pub fn count(&self) -> usize {
        self.table.live()
    }

    /// Remove every embedding and forget the pinned dimension, so the next
    /// vector may have a different length. Use `clear_with(false)` to keep the
    /// store refusing vectors of the old length.
    pub fn clear(&mut self) {
        self.clear_with(true);
    }

    /// Remove every embedding. With `reset_dimension` the pinned vector length
    /// is forgotten too (switching to an encoder of a different dimension);
    /// without it the store keeps refusing vectors of the wrong length even
    /// while empty.
    pub fn clear_with(&mut self, reset_dimension: bool) {
        self.table.clear();
        self.model_counts.clear();
        *self.ivf_mut() = None;
        *self.hnsw_mut() = None;
        if reset_dimension {
            self.dimension = None;
        }
    }

    /// Serialize every embedding into the portable cross-language payload.
    pub fn export_embeddings(&self) -> ExportedEmbeddings {
        self.export_embeddings_with(None)
    }

    /// `export_embeddings`, tying the payload to one graph by recording its
    /// `graph_digest` (see `graph_digest`) and the basis it was taken on.
    ///
    /// Version 1 of the payload is unchanged. The optional metadata trails the
    /// `embeddings` array: `encoder_model` and `encoder_version` when known,
    /// and `graph_digest` / `graph_digest_basis` when a digest is given.
    pub fn export_embeddings_with(&self, graph_digest: Option<&str>) -> ExportedEmbeddings {
        let mut embeddings: Vec<ExportedEmbedding> = self
            .table
            .live_slots()
            .filter_map(|slot| {
                let meta = self.table.meta(slot)?;
                let node_id = self.table.node_id(slot)?;
                Some(ExportedEmbedding {
                    node_type: node_id.node_type,
                    id: EmbeddingId::Str(node_id.id),
                    // Rust node ids are always strings; the other ports use this
                    // to tell a numeric id from a numeric-looking string one.
                    id_type: "str".to_string(),
                    text: meta.text.to_string(),
                    model: Some(meta.model.to_string()),
                    vector: self.table.row(slot).to_vec(),
                })
            })
            .collect();
        embeddings.sort_by(|a, b| {
            (a.node_type.clone(), a.id.as_string()).cmp(&(b.node_type.clone(), b.id.as_string()))
        });

        ExportedEmbeddings {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_FORMAT_VERSION,
            dimension: self.dimension,
            count: embeddings.len(),
            embeddings,
            encoder_model: self.current_model(),
            encoder_version: self.version(),
            graph_digest: graph_digest.map(String::from),
            graph_digest_basis: graph_digest.map(|_| GRAPH_DIGEST_BASIS.to_string()),
        }
    }

    /// Load a payload produced by `export_embeddings` (in any language port).
    /// Fields it does not know are ignored; the metadata it does know is kept
    /// in `last_import_meta`.
    pub fn import_embeddings(
        &mut self,
        payload: &ExportedEmbeddings,
        replace: bool,
    ) -> Result<usize, EmbeddingError> {
        if payload.format != EMBEDDING_FORMAT {
            return Err(EmbeddingError::Payload(format!(
                "not an {} payload: format={}",
                EMBEDDING_FORMAT, payload.format
            )));
        }
        if payload.version != EMBEDDING_FORMAT_VERSION {
            return Err(EmbeddingError::Payload(format!(
                "unsupported {} version {} (this build reads version {})",
                EMBEDDING_FORMAT, payload.version, EMBEDDING_FORMAT_VERSION
            )));
        }
        self.last_import_meta = ImportMeta {
            encoder_model: payload.encoder_model.clone(),
            encoder_version: payload.encoder_version.clone(),
            graph_digest: payload.graph_digest.clone(),
            graph_digest_basis: payload.graph_digest_basis.clone(),
        };
        if replace {
            self.clear_with(true);
            self.dimension = payload.dimension;
        }

        let mut written = 0;
        for item in &payload.embeddings {
            let node_id = NodeId::new(&item.node_type, item.id.as_string());
            self.check_dimension(&item.vector, "vector")?;
            let model = item.model.as_deref().unwrap_or("unknown");
            self.put(node_id, &item.text, item.vector.clone(), model);
            written += 1;
        }
        Ok(written)
    }

    /// Serialize the portable payload to a JSON string.
    pub fn to_json(&self) -> Result<String, EmbeddingError> {
        serde_json::to_string(&self.export_embeddings())
            .map_err(|e| EmbeddingError::Payload(e.to_string()))
    }

    /// Load embeddings from a JSON payload written by any language port.
    pub fn from_json(&mut self, json: &str, replace: bool) -> Result<usize, EmbeddingError> {
        let payload: ExportedEmbeddings =
            serde_json::from_str(json).map_err(|e| EmbeddingError::Payload(e.to_string()))?;
        self.import_embeddings(&payload, replace)
    }

    /// Write every embedding to a JSON sidecar file, atomically. Returns the
    /// count.
    pub fn save_embeddings(&self, path: impl AsRef<Path>) -> Result<usize, EmbeddingError> {
        self.save_embeddings_with(path, None)
    }

    /// `save_embeddings`, tying the sidecar to one graph via its digest.
    ///
    /// The file is written to a temp file and renamed into place, so a crash
    /// never leaves a half-written sidecar.
    pub fn save_embeddings_with(
        &self,
        path: impl AsRef<Path>,
        graph_digest: Option<&str>,
    ) -> Result<usize, EmbeddingError> {
        let payload = self.export_embeddings_with(graph_digest);
        let json =
            serde_json::to_string(&payload).map_err(|e| EmbeddingError::Payload(e.to_string()))?;
        atomic_write(path.as_ref(), json.as_bytes())?;
        Ok(payload.count)
    }

    /// Read a JSON sidecar written by `save_embeddings` (in any language port).
    pub fn load_embeddings(
        &mut self,
        path: impl AsRef<Path>,
        replace: bool,
    ) -> Result<usize, EmbeddingError> {
        let json = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            EmbeddingError::Io(format!("reading {}: {}", path.as_ref().display(), e))
        })?;
        self.from_json(&json, replace)
    }
}

// =============================================================================
// Semantic Search Result
// =============================================================================

#[derive(Clone, Debug)]
pub struct SemanticSearchResult {
    pub node_id: NodeId,
    pub score: f32,
    pub hop_count: usize,
    pub path: Vec<NodeId>,
}

/// Options for `semantic_multi_hop`.
#[derive(Clone, Debug)]
pub struct MultiHopOptions<'a> {
    pub rel_type: Option<&'a str>,
    pub max_hops: usize,
    pub top_k_seeds: usize,
    pub top_k_results: usize,
    pub direction: Direction,
    pub decay: f32,
    pub threshold: f32,
    /// How much of the relevance of a node itself to mix into its traversal
    /// score, in `[0, 1]`. 0 (the default) keeps pure seed-and-decay scoring.
    /// Raise it so a node found two hops out but highly relevant in its own
    /// right is not buried under a closer, unrelated neighbour. Hop-0 seeds
    /// score the same either way.
    pub blend: f32,
}

impl Default for MultiHopOptions<'_> {
    fn default() -> Self {
        Self {
            rel_type: None,
            max_hops: 2,
            top_k_seeds: 5,
            top_k_results: 10,
            direction: Direction::Out,
            decay: 0.8,
            threshold: 0.3,
            blend: 0.0,
        }
    }
}

// -- node filtering -----------------------------------------------------------

/// Which nodes a search may return, and (when `strict`) walk through.
///
/// `node_filter` and `node_types` combine with AND. A node passes when its
/// type is in `node_types` (an empty list passes nothing) *and*
/// `node_filter(node)` is true. A reference with no node in the graph cannot
/// be shown to `node_filter`, so it fails whenever a `node_filter` is set.
///
/// * Seeds must pass: the top seeds are chosen *among* passing nodes, so a
///   filtered-out best match never occupies a seed slot.
/// * Only passing nodes are returned.
/// * By default a node that fails can still be walked *through*: it is not
///   returned, but it still costs a hop and its decay, and a returned node's
///   `path` may list it. With `strict` a failing node is never entered, so
///   every node on every returned path passes.
///
/// ```
/// use isongraph_vector_rs::NodeFilter;
/// let only_people = NodeFilter::new().types(["person"]).strict();
/// ```
#[derive(Clone, Default)]
pub struct NodeFilter<'a> {
    /// Predicate on the node; `None` accepts every node.
    pub node_filter: Option<&'a dyn Fn(&Node) -> bool>,
    /// Node types allowed; `None` allows every type.
    pub node_types: Option<Vec<String>>,
    /// Make the filter a hard wall for traversal too (`strict_filter`).
    pub strict: bool,
}

impl std::fmt::Debug for NodeFilter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeFilter")
            .field("node_filter", &self.node_filter.map(|_| "<fn>"))
            .field("node_types", &self.node_types)
            .field("strict", &self.strict)
            .finish()
    }
}

impl<'a> NodeFilter<'a> {
    /// A filter that passes every node.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allow only these node types.
    pub fn types<I, S>(mut self, types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.node_types = Some(types.into_iter().map(Into::into).collect());
        self
    }

    /// Allow only nodes for which `predicate` is true.
    pub fn predicate(mut self, predicate: &'a dyn Fn(&Node) -> bool) -> Self {
        self.node_filter = Some(predicate);
        self
    }

    /// Never enter a failing node while traversing.
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// The `id -> passes` predicate over `graph`, or `None` when neither a
    /// predicate nor a type list is set (so unfiltered calls take exactly the
    /// unfiltered code path). Answers are memoized for the life of the gate.
    fn gate<'g>(&'g self, graph: &'g ISONGraph) -> Option<Gate<'g>> {
        if self.node_filter.is_none() && self.node_types.is_none() {
            return None;
        }
        Some(Gate {
            graph,
            types: self
                .node_types
                .as_ref()
                .map(|types| types.iter().map(String::as_str).collect()),
            filter: self.node_filter,
            memo: RefCell::new(HashMap::new()),
        })
    }
}

struct Gate<'g> {
    graph: &'g ISONGraph,
    types: Option<HashSet<&'g str>>,
    filter: Option<&'g dyn Fn(&Node) -> bool>,
    memo: RefCell<HashMap<NodeId, bool>>,
}

impl Gate<'_> {
    fn passes(&self, id: &NodeId) -> bool {
        if let Some(cached) = self.memo.borrow().get(id) {
            return *cached;
        }
        let mut ok = self
            .types
            .as_ref()
            .is_none_or(|types| types.contains(id.node_type.as_str()));
        if ok {
            if let Some(filter) = self.filter {
                ok = self
                    .graph
                    .get_node(&id.node_type, &id.id)
                    .map(filter)
                    .unwrap_or(false);
            }
        }
        self.memo.borrow_mut().insert(id.clone(), ok);
        ok
    }
}

/// Options for `SemanticGraph::similarity_search_with`.
#[derive(Clone, Debug)]
pub struct SimilarityOptions<'a> {
    /// Matches to return; `None` returns every match.
    pub top_k: Option<usize>,
    /// Keep only this node type.
    pub node_type: Option<&'a str>,
    /// Minimum similarity.
    pub threshold: f32,
    /// Applied *before* `top_k` is taken, so a filtered-out best match does
    /// not cost a slot.
    pub filter: NodeFilter<'a>,
    /// Force the full scan even when the store has an approximate index
    /// (`set_ann`); no effect otherwise.
    pub exact: bool,
}

impl Default for SimilarityOptions<'_> {
    fn default() -> Self {
        Self {
            top_k: Some(10),
            node_type: None,
            threshold: 0.0,
            filter: NodeFilter::default(),
            exact: false,
        }
    }
}

/// Options for `SemanticGraph::semantic_multi_hop_search`: the plain
/// multi-hop options plus edge weighting and node filtering.
#[derive(Clone, Debug, Default)]
pub struct SearchOptions<'a> {
    pub hops: MultiHopOptions<'a>,
    /// Name of an edge property whose numeric value multiplies the decay of
    /// every hop that crosses that edge (clamped to `[0, 1]`; a missing,
    /// non-numeric, bool or NaN value counts as 1.0). A node is scored by its
    /// *best* route, which with weights can be a longer one than the
    /// fewest-hops route. `None` ignores edge properties entirely.
    pub weight_field: Option<&'a str>,
    pub filter: NodeFilter<'a>,
}

impl<'a> From<MultiHopOptions<'a>> for SearchOptions<'a> {
    fn from(hops: MultiHopOptions<'a>) -> Self {
        Self {
            hops,
            ..Self::default()
        }
    }
}

/// Options for `SemanticGraph::semantic_path_with`.
#[derive(Clone, Debug)]
pub struct PathOptions<'a> {
    pub rel_type: Option<&'a str>,
    pub max_hops: usize,
    pub top_k_seeds: usize,
    pub direction: Direction,
    pub decay: f32,
    pub threshold: f32,
    /// Seeds are chosen among passing nodes and the target must pass or the
    /// result is `None`; with `strict` the path may use passing nodes only.
    pub filter: NodeFilter<'a>,
}

impl Default for PathOptions<'_> {
    fn default() -> Self {
        Self {
            rel_type: None,
            max_hops: 5,
            top_k_seeds: 3,
            direction: Direction::Out,
            decay: 0.9,
            threshold: 0.0,
            filter: NodeFilter::default(),
        }
    }
}

/// Options for `SemanticGraph::semantic_subgraph_with`.
#[derive(Clone, Debug)]
pub struct SubgraphOptions<'a> {
    /// How the nodes are selected (`semantic_multi_hop_search`).
    pub search: SearchOptions<'a>,
    /// Name for the returned graph; `None` uses `<name>_subgraph`.
    pub name: Option<&'a str>,
    /// Also add *connector* nodes so the selected nodes fall into as few
    /// connected components as possible. Components are weak (edge direction
    /// ignored when judging the result), but connector routes honour
    /// `rel_type` and `direction`. Each round adds the interior of the shortest
    /// route between two components (greedy; not a minimum Steiner tree),
    /// until one component remains or no route fits `connect_max_hops`.
    /// Nothing but those interior nodes is added, and they are not marked.
    ///
    /// With `connect` the result can hold **more than** `top_k_results`
    /// nodes: that caps the *selected* nodes, and connectors are extra.
    /// Without `strict` a connector may be a node that fails the filter (it is
    /// only a bridge); with it, connectors must pass too.
    pub connect: bool,
    /// Longest route, counted in edges, that may join two components. The
    /// default 2 allows one connector node per join; 3 allows two. A direct
    /// edge needs no connector (it is already an induced edge). Must be >= 1.
    pub connect_max_hops: usize,
}

impl Default for SubgraphOptions<'_> {
    fn default() -> Self {
        Self {
            search: SearchOptions::default(),
            name: None,
            connect: false,
            connect_max_hops: 2,
        }
    }
}

/// An edge's numeric `weight_field` property clamped to `[0, 1]`.
///
/// A missing, non-numeric (including bool) or NaN value counts as 1.0, so an
/// unweighted edge never penalises a hop.
fn edge_weight(edge: &Edge, weight_field: &str) -> f32 {
    let value = match edge.properties.get(weight_field) {
        Some(Value::Int(i)) => *i as f64,
        Some(Value::Float(f)) => *f,
        _ => return 1.0,
    };
    if value.is_nan() {
        return 1.0;
    }
    value.clamp(0.0, 1.0) as f32
}

/// (neighbor, connecting edge) pairs per node, for traversals that need the
/// edge as well as the neighbor.
struct Adjacency<'g> {
    out: HashMap<&'g NodeId, Vec<(&'g NodeId, &'g Edge)>>,
    inn: HashMap<&'g NodeId, Vec<(&'g NodeId, &'g Edge)>>,
}

impl<'g> Adjacency<'g> {
    fn build(graph: &'g ISONGraph, rel_type: Option<&str>) -> Self {
        let mut rel_types: Vec<&String> = graph.edge_types();
        rel_types.sort();
        let mut out: HashMap<&NodeId, Vec<(&NodeId, &Edge)>> = HashMap::new();
        let mut inn: HashMap<&NodeId, Vec<(&NodeId, &Edge)>> = HashMap::new();
        for rel in rel_types {
            if rel_type.is_some_and(|wanted| wanted != rel) {
                continue;
            }
            for edge in graph.edges_of_type(rel) {
                out.entry(&edge.source)
                    .or_default()
                    .push((&edge.target, edge));
                inn.entry(&edge.target)
                    .or_default()
                    .push((&edge.source, edge));
            }
        }
        Self { out, inn }
    }

    fn adjacent(&self, id: &NodeId, direction: Direction) -> Vec<(&'g NodeId, &'g Edge)> {
        let mut pairs = Vec::new();
        if matches!(direction, Direction::Out | Direction::Both) {
            if let Some(list) = self.out.get(id) {
                pairs.extend(list.iter().copied());
            }
        }
        if matches!(direction, Direction::In | Direction::Both) {
            if let Some(list) = self.inn.get(id) {
                pairs.extend(list.iter().copied());
            }
        }
        pairs
    }
}

// =============================================================================
// Semantic Graph
// =============================================================================

/// Options for `SemanticGraph::load_with`.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// File format passed to the base serializer (`"ison"` or `"isonl"`);
    /// `None` picks it from the file extension.
    pub format: Option<String>,
    /// Restore the `<path>.embeddings.json` sidecar when it exists.
    pub embeddings: bool,
    /// Auto-embed nodes added to the loaded graph.
    pub auto_embed: bool,
    /// Fields concatenated for auto-embedding; `None` uses the defaults
    /// (`name`, `description`, `content`, `text`).
    pub embed_fields: Option<Vec<String>>,
    /// What to do when the sidecar carries a graph digest that does not match
    /// the graph file. A sidecar without a digest (older releases, other
    /// ports), or one recorded on a basis this build does not compute, is
    /// loaded as before and never checked.
    pub on_mismatch: GraphMismatch,
    /// Encoder/model version for the loaded graph's store. If it (or the
    /// encoder's own version) and the sidecar's `encoder_version` are both set
    /// and differ, a warning is issued.
    pub version: Option<String>,
    /// What a text search does when the encoder's model differs from the
    /// stored vectors' model.
    pub on_model_mismatch: ModelMismatch,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            format: None,
            embeddings: true,
            auto_embed: true,
            embed_fields: None,
            on_mismatch: GraphMismatch::Error,
            version: None,
            on_model_mismatch: ModelMismatch::Warn,
        }
    }
}

pub struct SemanticGraph {
    graph: ISONGraph,
    embedding_store: EmbeddingStore,
    auto_embed: bool,
    embed_fields: Vec<String>,
}

impl SemanticGraph {
    pub fn new(name: &str, encoder: Box<dyn EmbeddingEncoder>) -> Self {
        Self {
            graph: ISONGraph::new(name),
            embedding_store: EmbeddingStore::new(Some(encoder)),
            auto_embed: true,
            embed_fields: vec![
                "name".to_string(),
                "description".to_string(),
                "content".to_string(),
                "text".to_string(),
            ],
        }
    }

    pub fn with_options(
        name: &str,
        encoder: Box<dyn EmbeddingEncoder>,
        auto_embed: bool,
        embed_fields: Vec<String>,
    ) -> Self {
        Self {
            graph: ISONGraph::new(name),
            embedding_store: EmbeddingStore::new(Some(encoder)),
            auto_embed,
            embed_fields,
        }
    }

    pub fn graph(&self) -> &ISONGraph {
        &self.graph
    }

    /// Add an edge between two existing nodes.
    ///
    /// Edges carry no text, so they never touch the embedding store. This and
    /// `remove_edge` are the only structural mutations the underlying graph
    /// lets through: node changes go through `add_node`, `update_node` and
    /// `remove_node`, which keep each node's vector in step. (There is
    /// deliberately no `graph_mut()`: it would let a caller change a node's
    /// text, or remove the node, behind the embedding store's back.)
    pub fn add_edge(
        &mut self,
        rel_type: &str,
        source: (&str, &str),
        target: (&str, &str),
        properties: Vec<(&str, Value)>,
    ) -> Result<(), EmbeddingError> {
        self.graph
            .add_edge(rel_type, source, target, properties)
            .map_err(|e| EmbeddingError::GraphError(e.to_string()))
    }

    /// Remove an edge. Nodes and their embeddings are untouched.
    pub fn remove_edge(
        &mut self,
        rel_type: &str,
        source: (&str, &str),
        target: (&str, &str),
    ) -> Result<(), EmbeddingError> {
        self.graph
            .remove_edge(rel_type, source, target)
            .map_err(|e| EmbeddingError::GraphError(e.to_string()))
    }

    pub fn embedding_store(&self) -> &EmbeddingStore {
        &self.embedding_store
    }

    pub fn embedding_store_mut(&mut self) -> &mut EmbeddingStore {
        &mut self.embedding_store
    }

    /// The encoder in use, if any.
    pub fn encoder(&self) -> Option<&dyn EmbeddingEncoder> {
        self.embedding_store.encoder()
    }

    /// Switch the approximate (IVF-flat) index on, with these settings, or off
    /// with `None`. See `AnnConfig`. Nothing is saved: a saved graph, its
    /// sidecar and its digest are the same with or without it, and a loaded
    /// graph builds the index again on its first search.
    pub fn set_ann(&mut self, config: Option<AnnConfig>) -> Result<(), EmbeddingError> {
        self.embedding_store.set_ann(config)
    }

    /// The approximate-index settings, or `None` while search is exact.
    pub fn ann(&self) -> Option<&AnnConfig> {
        self.embedding_store.ann()
    }

    /// Build the approximate index now (see `EmbeddingStore::build_ann_index`).
    pub fn build_ann_index(&self, force: bool) -> Result<AnnStats, EmbeddingError> {
        self.embedding_store.build_ann_index(force)
    }

    /// What the approximate index looks like right now.
    pub fn ann_stats(&self) -> AnnStats {
        self.embedding_store.ann_stats()
    }

    /// Recall@k of the approximate index against the exact scan (see
    /// `EmbeddingStore::recall_check`).
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
        self.embedding_store
            .recall_check(queries, k, node_type, threshold)
    }

    /// Install an encoder, replacing any current one. Vectors already stored
    /// keep the model that made them; see `stale_embeddings`.
    pub fn set_encoder(&mut self, encoder: Box<dyn EmbeddingEncoder>) {
        self.embedding_store.set_encoder(encoder);
    }

    /// Set the encoder/model version recorded in the sidecar (an explicit
    /// version wins over the encoder's own).
    pub fn set_version(&mut self, version: Option<String>) {
        self.embedding_store.set_version(version);
    }

    /// Choose what a text search does when the encoder's model differs from
    /// the stored vectors' model.
    pub fn set_on_model_mismatch(&mut self, mode: ModelMismatch) {
        self.embedding_store.set_on_model_mismatch(mode);
    }

    /// Concatenate the configured embed_fields present in a property list.
    /// Concatenate the configured embed fields present in a property bag.
    ///
    /// Property values are typed, so they are rendered through Display: a
    /// string comes through unquoted, a number or bool as its literal text.
    /// Nulls are skipped rather than embedded as the word "null".
    fn embed_text_for(&self, properties: &HashMap<String, Value>) -> String {
        let parts: Vec<String> = self
            .embed_fields
            .iter()
            .filter_map(|f| properties.get(f.as_str()))
            .filter(|v| !matches!(v, Value::Null))
            .map(|v| v.to_string())
            .filter(|text| !text.is_empty())
            .collect();
        parts.join(" ")
    }

    /// Add a node, embedding it from `embed_text` or (when `auto_embed` is on)
    /// from its properties.
    pub fn add_node(
        &mut self,
        node_type: &str,
        node_id: &str,
        properties: Vec<(&str, Value)>,
        embed_text: Option<&str>,
    ) -> Result<NodeId, EmbeddingError> {
        // Clone properties before handing them to the graph
        let props_for_embed: HashMap<String, Value> = properties
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();

        self.graph
            .add_node(node_type, node_id, properties)
            .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;

        let result_node_id = NodeId::new(node_type, node_id);

        let text = match embed_text {
            Some(text) => Some(text.to_string()),
            None if self.auto_embed => {
                let text = self.embed_text_for(&props_for_embed);
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
            None => None,
        };

        if let Some(text) = text {
            self.embed_node(&result_node_id, &text)?;
        }

        Ok(result_node_id)
    }

    /// Kept for callers written against the original name, and against the
    /// string-valued properties ISONGraph used before 1.4.0. Values are lifted
    /// into `Value::String`, so existing call sites keep compiling.
    pub fn add_node_with_embed(
        &mut self,
        node_type: &str,
        node_id: &str,
        properties: Vec<(&str, &str)>,
        embed_text: Option<&str>,
    ) -> Result<NodeId, EmbeddingError> {
        let typed: Vec<(&str, Value)> = properties
            .into_iter()
            .map(|(k, v)| (k, Value::String(v.to_string())))
            .collect();
        self.add_node(node_type, node_id, typed, embed_text)
    }

    pub fn embed_node(&mut self, node_id: &NodeId, text: &str) -> Result<String, EmbeddingError> {
        self.embedding_store.add(node_id.clone(), text, None)
    }

    /// Embed many nodes in a single batched encoder call.
    pub fn embed_nodes(&mut self, items: &[(NodeId, &str)]) -> Result<usize, EmbeddingError> {
        self.embedding_store.add_batch(items)
    }

    pub fn get_embedding(&self, node_id: &NodeId) -> Option<EmbeddingRecord> {
        self.embedding_store.get(node_id)
    }

    /// Update node properties, keeping the embedding in step with them.
    ///
    /// With `auto_embed` on, a change to any of the embed fields re-embeds the
    /// node from its new (merged) properties, so searches never rank it by
    /// text it no longer carries. If the new properties leave nothing to
    /// embed, the stale vector is dropped. Updates that touch no embed field,
    /// and graphs with `auto_embed` off (whose vectors the caller manages),
    /// leave the embedding alone.
    pub fn update_node(
        &mut self,
        node_type: &str,
        node_id: &str,
        properties: Vec<(&str, Value)>,
    ) -> Result<(), EmbeddingError> {
        let touches_embed_field = properties
            .iter()
            .any(|(k, _)| self.embed_fields.iter().any(|f| f == k));

        self.graph
            .update_node(node_type, node_id, properties)
            .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;

        if self.auto_embed && touches_embed_field {
            let text = {
                let node = self
                    .graph
                    .get_node(node_type, node_id)
                    .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;
                self.embed_text_for(&node.properties)
            };
            let id = NodeId::new(node_type, node_id);
            if text.is_empty() {
                self.embedding_store.remove(&id);
            } else {
                self.embed_node(&id, &text)?;
            }
        }
        Ok(())
    }

    pub fn remove_node(&mut self, node_type: &str, node_id: &str) -> Result<(), EmbeddingError> {
        let id = NodeId::new(node_type, node_id);
        self.embedding_store.remove(&id);
        self.graph
            .remove_node(node_type, node_id)
            .map_err(|e| EmbeddingError::GraphError(e.to_string()))
    }

    pub fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.embedding_store
            .similarity_search(query, top_k, node_type, threshold)
    }

    /// Top-k similarity search among the nodes that pass `gate`.
    ///
    /// With a gate the store cannot truncate first: the best k *unfiltered*
    /// matches may all be filtered out, so matches are walked best first and
    /// filtered until k have passed.
    ///
    /// With the store's approximate index the candidates are the probed cells
    /// only. If fewer than k of them pass the gate, the probe is widened (4x
    /// each step) and the search retried, ending in the full scan, so a
    /// selective filter costs time rather than results.
    fn seed_search(
        &self,
        query_vector: &[f32],
        top_k: Option<usize>,
        node_type: Option<&str>,
        threshold: f32,
        gate: Option<&Gate>,
        exact: bool,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let store = &self.embedding_store;
        let Some(gate) = gate else {
            return store.search_vector(
                query_vector,
                top_k.unwrap_or(usize::MAX),
                node_type,
                threshold,
                exact,
            );
        };
        if top_k == Some(0) {
            return Ok(Vec::new());
        }
        let mut scale = (store.ann().is_some() && !exact && top_k.is_some()).then_some(1usize);
        loop {
            let (hits, approximate) =
                store.ranked_hits(query_vector, node_type, threshold, scale)?;
            let mut results = Vec::new();
            for hit in hits {
                if gate.passes(&hit.node_id) {
                    results.push(hit);
                    if top_k.is_some_and(|k| results.len() >= k) {
                        break;
                    }
                }
            }
            if !approximate || top_k.is_some_and(|k| results.len() >= k) {
                return Ok(results);
            }
            scale = scale.map(|s| s.saturating_mul(4));
        }
    }

    /// Find similar nodes by embedding, for a text or vector query, with an
    /// optional node filter.
    ///
    /// The filter is applied *before* `top_k` is taken, so you still get
    /// `top_k` matching nodes when that many exist. An embedding whose node is
    /// not in the graph fails a `node_filter` (but not a bare `node_types`).
    pub fn similarity_search_with<'q>(
        &self,
        query: impl Into<Query<'q>>,
        options: &SimilarityOptions,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let query_vector = self.embedding_store.resolve_query(query.into())?;
        let gate = options.filter.gate(&self.graph);
        self.seed_search(
            &query_vector,
            options.top_k,
            options.node_type,
            options.threshold,
            gate.as_ref(),
            options.exact,
        )
    }

    /// Find nodes semantically closest to one already in the graph.
    ///
    /// The "more like this" query - no text and no encoder call: it searches
    /// with the stored vector of the node and drops the node itself from the
    /// results.
    pub fn similar_to_node(
        &self,
        node_id: &NodeId,
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let vector = match self.embedding_store.get_vector(node_id) {
            Some(vector) => vector,
            None => return Ok(vec![]),
        };
        let mut results = self.embedding_store.similarity_search_vector(
            vector,
            top_k + 1,
            node_type,
            threshold,
        )?;
        results.retain(|r| &r.node_id != node_id);
        results.truncate(top_k);
        Ok(results)
    }

    /// Semantic multi-hop search.
    ///
    /// 1. Find top-k similar nodes as seeds (hop 0)
    /// 2. Traverse the graph following relationships
    /// 3. Score by `max(seed_similarity, 0) * decay^hop_count`, optionally
    ///    blended with the relevance of each node itself
    pub fn semantic_multi_hop_with(
        &self,
        query: &str,
        options: &MultiHopOptions,
    ) -> Result<Vec<SemanticSearchResult>, EmbeddingError> {
        self.semantic_multi_hop_search(
            query,
            &SearchOptions {
                hops: options.clone(),
                ..SearchOptions::default()
            },
        )
    }

    /// Semantic multi-hop search for a text or vector query, with edge
    /// weighting and node filtering (see `SearchOptions` and `NodeFilter`).
    ///
    /// Scores are `max(seed_similarity, 0) * prod(decay * weight)` over the
    /// hops, where `weight` is 1.0 unless `weight_field` is set. A node is
    /// scored by its best route.
    pub fn semantic_multi_hop_search<'q>(
        &self,
        query: impl Into<Query<'q>>,
        options: &SearchOptions,
    ) -> Result<Vec<SemanticSearchResult>, EmbeddingError> {
        let hops = &options.hops;
        if !(0.0..=1.0).contains(&hops.blend) {
            return Err(EmbeddingError::InvalidBlend(hops.blend));
        }

        let query_vector = self.embedding_store.resolve_query(query.into())?;
        let gate = options.filter.gate(&self.graph);
        let gate = gate.as_ref();
        let strict = options.filter.strict;

        // Blending needs a score for every node, so one exact scan serves the
        // per-node relevance (seeds come from the top-k search either way).
        let scores = if hops.blend > 0.0 {
            self.embedding_store.score_map(&query_vector)?
        } else {
            HashMap::new()
        };

        let seeds = self.seed_search(
            &query_vector,
            Some(hops.top_k_seeds),
            None,
            hops.threshold,
            gate,
            // Blending scores every node anyway, so seeds come from the same
            // exact scan rather than from the approximate index.
            hops.blend > 0.0,
        )?;
        if seeds.is_empty() {
            return Ok(vec![]);
        }

        let combined = |base: f32, key: &str| -> f32 {
            if hops.blend <= 0.0 {
                base
            } else {
                (1.0 - hops.blend) * base + hops.blend * scores.get(key).copied().unwrap_or(0.0)
            }
        };

        // Edge weights need the connecting edge, which `neighbors` does not
        // return; without a weight field the plain neighbor lookup is used.
        let weight_field = options.weight_field;
        let adjacency = weight_field.map(|_| Adjacency::build(&self.graph, hops.rel_type));
        let adjacent = |id: &NodeId| -> Vec<(NodeId, Option<&Edge>)> {
            match &adjacency {
                Some(adjacency) => adjacency
                    .adjacent(id, hops.direction)
                    .into_iter()
                    .map(|(neighbor, edge)| (neighbor.clone(), Some(edge)))
                    .collect(),
                None => self
                    .graph
                    .neighbors(&id.as_ref(), hops.rel_type, hops.direction)
                    .into_iter()
                    .map(|neighbor| (neighbor, None))
                    .collect(),
            }
        };

        // Results in first-seen order, so equal scores sort the same way on
        // every run.
        let mut results: Vec<SemanticSearchResult> = Vec::new();
        let mut index: HashMap<NodeId, usize> = HashMap::new();
        let upsert =
            |results: &mut Vec<SemanticSearchResult>,
             index: &mut HashMap<NodeId, usize>,
             result: SemanticSearchResult| match index.get(&result.node_id) {
                Some(&at) => results[at] = result,
                None => {
                    index.insert(result.node_id.clone(), results.len());
                    results.push(result);
                }
            };

        for seed in &seeds {
            // Cosine similarity is signed, and `base * decay ** hops` only
            // decays a non-negative base: multiplying a negative seed score by
            // 0.8 moves it toward zero, so everything it reached outranked the
            // seed itself and relevance grew with distance.
            let traversal_base = seed.score.max(0.0);

            // A seed is a hop-0 match on its own merit. An earlier traversal
            // may already have recorded it with a decayed score; that must not
            // win over the higher, direct similarity of the node.
            let seed_known = index
                .get(&seed.node_id)
                .map(|&at| results[at].score >= seed.score)
                .unwrap_or(false);
            if !seed_known {
                upsert(
                    &mut results,
                    &mut index,
                    SemanticSearchResult {
                        node_id: seed.node_id.clone(),
                        score: seed.score,
                        hop_count: 0,
                        path: vec![seed.node_id.clone()],
                    },
                );
            }

            // Breadth-first traversal. Each node keeps its best decay factor
            // and its fewest hops *separately*: with edge weights the route
            // with the best factor is not always the shortest one, and a
            // shorter route must still be expanded because it has hops left.
            // Without weights the first (shortest) arrival is both, so this
            // reduces to the plain visited-set BFS.
            let mut best_factor: HashMap<NodeId, f32> = HashMap::new();
            let mut best_hops: HashMap<NodeId, usize> = HashMap::new();
            best_factor.insert(seed.node_id.clone(), 1.0);
            best_hops.insert(seed.node_id.clone(), 0);

            let mut queue: VecDeque<(NodeId, Vec<NodeId>, f32, usize)> = VecDeque::new();
            queue.push_back((seed.node_id.clone(), vec![seed.node_id.clone()], 1.0, 0));

            while let Some((node_id, path, factor, hop)) = queue.pop_front() {
                if hop >= hops.max_hops {
                    continue;
                }

                for (neighbor, edge) in adjacent(&node_id) {
                    if path.contains(&neighbor) {
                        continue;
                    }
                    if strict && gate.is_some_and(|g| !g.passes(&neighbor)) {
                        continue;
                    }

                    let new_hop = hop + 1;
                    // Unweighted: decay ** hops exactly, not a running
                    // product, so scores are bit-identical to the releases
                    // before edge weights.
                    let new_factor = match (weight_field, edge) {
                        (Some(field), Some(edge)) => factor * hops.decay * edge_weight(edge, field),
                        _ => hops.decay.powi(new_hop as i32),
                    };

                    let known_factor = best_factor.get(&neighbor).copied();
                    let known_hops = best_hops.get(&neighbor).copied();
                    if new_factor <= known_factor.unwrap_or(-1.0)
                        && new_hop >= known_hops.unwrap_or(new_hop + 1)
                    {
                        continue;
                    }
                    best_factor.insert(
                        neighbor.clone(),
                        new_factor.max(known_factor.unwrap_or(-1.0)),
                    );
                    best_hops.insert(neighbor.clone(), new_hop.min(known_hops.unwrap_or(new_hop)));

                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());

                    if gate.is_none_or(|g| g.passes(&neighbor)) {
                        let neighbor_key = format!("{}:{}", neighbor.node_type, neighbor.id);
                        let new_score = combined(traversal_base * new_factor, &neighbor_key);
                        let better = index
                            .get(&neighbor)
                            .map(|&at| results[at].score < new_score)
                            .unwrap_or(true);
                        if better {
                            upsert(
                                &mut results,
                                &mut index,
                                SemanticSearchResult {
                                    node_id: neighbor.clone(),
                                    score: new_score,
                                    hop_count: new_hop,
                                    path: new_path.clone(),
                                },
                            );
                        }
                    }

                    queue.push_back((neighbor, new_path, new_factor, new_hop));
                }
            }
        }

        let mut sorted = results;
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(hops.top_k_results);

        Ok(sorted)
    }

    /// Semantic multi-hop search, original positional form.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_multi_hop(
        &self,
        query: &str,
        rel_type: Option<&str>,
        max_hops: usize,
        top_k_seeds: usize,
        top_k_results: usize,
        direction: Direction,
        decay: f32,
        threshold: f32,
    ) -> Result<Vec<SemanticSearchResult>, EmbeddingError> {
        self.semantic_multi_hop_with(
            query,
            &MultiHopOptions {
                rel_type,
                max_hops,
                top_k_seeds,
                top_k_results,
                direction,
                decay,
                threshold,
                blend: 0.0,
            },
        )
    }

    /// Fewest-hops path from `start` to `end` whose every node passes `gate`.
    fn gated_shortest_path(
        &self,
        start: &NodeId,
        end: &NodeId,
        rel_type: Option<&str>,
        max_hops: usize,
        direction: Direction,
        gate: &Gate,
    ) -> Option<Vec<NodeId>> {
        if !gate.passes(start) || !gate.passes(end) {
            return None;
        }
        if start == end {
            return Some(vec![start.clone()]);
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(start.clone());
        let mut queue: VecDeque<Vec<NodeId>> = VecDeque::new();
        queue.push_back(vec![start.clone()]);
        while let Some(path) = queue.pop_front() {
            if path.len() > max_hops {
                continue;
            }
            let last = &path[path.len() - 1];
            for neighbor in self.graph.neighbors(&last.as_ref(), rel_type, direction) {
                if visited.contains(&neighbor) || !gate.passes(&neighbor) {
                    continue;
                }
                let mut next = path.clone();
                next.push(neighbor.clone());
                if &neighbor == end {
                    return Some(next);
                }
                visited.insert(neighbor);
                queue.push_back(next);
            }
        }
        None
    }

    /// Find a semantically-rooted path to a target node.
    ///
    /// Every seed that reaches the target is a candidate and the best score
    /// wins: a slightly less similar seed one hop from the target beats a
    /// better seed five hops away. Seeds arrive best-first, so a tie goes to
    /// the more similar one. The score is `max(seed_similarity, 0) *
    /// decay^hops`, as in `semantic_multi_hop`: a negative seed passes no
    /// relevance along rather than "decaying" toward zero and outscoring itself.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_path(
        &self,
        query: &str,
        target: &NodeId,
        rel_type: Option<&str>,
        max_hops: usize,
        top_k_seeds: usize,
        direction: Direction,
        decay: f32,
        threshold: f32,
    ) -> Result<Option<SemanticSearchResult>, EmbeddingError> {
        self.semantic_path_with(
            query,
            target,
            &PathOptions {
                rel_type,
                max_hops,
                top_k_seeds,
                direction,
                decay,
                threshold,
                filter: NodeFilter::default(),
            },
        )
    }

    /// `semantic_path` for a text or vector query, with node filtering.
    ///
    /// Seeds are chosen among passing nodes, and the *target must pass* or the
    /// result is `None`. By default the path may run through failing nodes (it
    /// is the fewest-hops path, as without a filter); with `strict` it is the
    /// fewest-hops path that uses passing nodes only, which can be longer or
    /// not exist.
    pub fn semantic_path_with<'q>(
        &self,
        query: impl Into<Query<'q>>,
        target: &NodeId,
        options: &PathOptions,
    ) -> Result<Option<SemanticSearchResult>, EmbeddingError> {
        let gate = options.filter.gate(&self.graph);
        if gate.as_ref().is_some_and(|g| !g.passes(target)) {
            return Ok(None);
        }

        let query_vector = self.embedding_store.resolve_query(query.into())?;
        let seeds = self.seed_search(
            &query_vector,
            Some(options.top_k_seeds),
            None,
            options.threshold,
            gate.as_ref(),
            false,
        )?;

        let mut best: Option<SemanticSearchResult> = None;
        for seed in seeds {
            let nodes = match gate.as_ref() {
                Some(gate) if options.filter.strict => match self.gated_shortest_path(
                    &seed.node_id,
                    target,
                    options.rel_type,
                    options.max_hops,
                    options.direction,
                    gate,
                ) {
                    Some(nodes) => nodes,
                    None => continue,
                },
                _ => match self.graph.shortest_path(
                    &seed.node_id.as_ref(),
                    &target.as_ref(),
                    options.rel_type,
                    options.max_hops,
                    options.direction,
                ) {
                    Some(path) => path.nodes,
                    None => continue,
                },
            };
            let hops = nodes.len() - 1;
            let score = seed.score.max(0.0) * options.decay.powi(hops as i32);
            if best.as_ref().is_none_or(|b| score > b.score) {
                best = Some(SemanticSearchResult {
                    node_id: target.clone(),
                    score,
                    hop_count: hops,
                    path: nodes,
                });
            }
        }

        Ok(best)
    }

    /// Shortest route from any of `members` to a kept node outside them.
    fn shortest_link(
        &self,
        members: &[NodeId],
        keep: &HashSet<NodeId>,
        rel_type: Option<&str>,
        direction: Direction,
        max_hops: usize,
        gate: Option<&Gate>,
    ) -> Option<Vec<NodeId>> {
        let mut visited: HashSet<NodeId> = members.iter().cloned().collect();
        let mut queue: VecDeque<Vec<NodeId>> = members.iter().map(|m| vec![m.clone()]).collect();
        while let Some(path) = queue.pop_front() {
            if path.len() > max_hops {
                continue;
            }
            let last = &path[path.len() - 1];
            for neighbor in self.graph.neighbors(&last.as_ref(), rel_type, direction) {
                if visited.contains(&neighbor) {
                    continue;
                }
                let mut next = path.clone();
                next.push(neighbor.clone());
                if keep.contains(&neighbor) {
                    return Some(next);
                }
                if gate.is_some_and(|g| !g.passes(&neighbor)) {
                    continue;
                }
                if !self.graph.has_node(&neighbor.node_type, &neighbor.id) {
                    continue;
                }
                visited.insert(neighbor);
                queue.push_back(next);
            }
        }
        None
    }

    /// Grow `keep` with connector nodes until it is as connected as the hop
    /// budget allows. Returns the connectors added, in order.
    ///
    /// Greedy: each round finds, over all pairs of components, the shortest
    /// route (at most `max_hops` edges, honouring `rel_type`/`direction`)
    /// whose interior nodes lie outside `keep`, adds its interior, and
    /// recomputes components. Each round merges at least two, so it
    /// terminates; it stops when one component remains or no route fits the
    /// budget. `gate` (set for strict filtering) bars failing nodes from
    /// serving as connectors.
    fn connect_components(
        &self,
        keep: &mut HashSet<NodeId>,
        rel_type: Option<&str>,
        direction: Direction,
        max_hops: usize,
        gate: Option<&Gate>,
    ) -> Vec<NodeId> {
        let mut added: Vec<NodeId> = Vec::new();
        let order = |id: &NodeId| (id.node_type.clone(), id.id.clone());

        let mut rel_types: Vec<&String> = self.graph.edge_types();
        rel_types.sort();
        let rel_types: Vec<&String> = rel_types
            .into_iter()
            .filter(|rel| rel_type.is_none_or(|wanted| wanted == rel.as_str()))
            .collect();

        while keep.len() > 1 {
            // Weakly connected components of the induced subgraph.
            let mut parent: HashMap<NodeId, NodeId> =
                keep.iter().map(|id| (id.clone(), id.clone())).collect();
            fn find(parent: &mut HashMap<NodeId, NodeId>, id: &NodeId) -> NodeId {
                let mut root = id.clone();
                while parent[&root] != root {
                    root = parent[&root].clone();
                }
                let mut node = id.clone();
                while parent[&node] != root {
                    let next = parent[&node].clone();
                    parent.insert(node, root.clone());
                    node = next;
                }
                root
            }
            for rel in &rel_types {
                for edge in self.graph.edges_of_type(rel) {
                    if keep.contains(&edge.source) && keep.contains(&edge.target) {
                        let a = find(&mut parent, &edge.source);
                        let b = find(&mut parent, &edge.target);
                        parent.insert(a, b);
                    }
                }
            }
            let mut groups: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let members_of: Vec<NodeId> = keep.iter().cloned().collect();
            for id in members_of {
                let root = find(&mut parent, &id);
                groups.entry(root).or_default().push(id);
            }
            if groups.len() <= 1 {
                break;
            }

            let mut ordered: Vec<Vec<NodeId>> = groups
                .into_values()
                .map(|mut members| {
                    members.sort_by_key(order);
                    members
                })
                .collect();
            ordered.sort_by_key(|members| order(&members[0]));

            let mut best_route: Option<Vec<NodeId>> = None;
            for members in &ordered {
                if let Some(route) =
                    self.shortest_link(members, keep, rel_type, direction, max_hops, gate)
                {
                    if best_route.as_ref().is_none_or(|b| route.len() < b.len()) {
                        best_route = Some(route);
                    }
                }
            }
            let route = match best_route {
                Some(route) => route,
                None => break,
            };
            for id in &route[1..route.len() - 1] {
                if keep.insert(id.clone()) {
                    added.push(id.clone());
                }
            }
        }
        added
    }

    /// Extract the slice of the graph a query is actually about.
    ///
    /// Runs `semantic_multi_hop_with`, then returns a new SemanticGraph holding
    /// exactly those nodes, every edge induced between them, and their
    /// embeddings - so the result can be traversed, searched or serialized on
    /// its own. This is what makes an ISON graph useful as LLM context: a
    /// small, on-topic graph to inject rather than the whole store.
    pub fn semantic_subgraph(
        &self,
        query: &str,
        options: &MultiHopOptions,
        name: &str,
        encoder: Box<dyn EmbeddingEncoder>,
    ) -> Result<SemanticGraph, EmbeddingError> {
        self.semantic_subgraph_with(
            query,
            &SubgraphOptions {
                search: options.clone().into(),
                name: Some(name),
                ..SubgraphOptions::default()
            },
            encoder,
        )
    }

    /// `semantic_subgraph` for a text or vector query, with edge weighting,
    /// node filtering and optional connector nodes (see `SubgraphOptions`).
    /// Only edges of `rel_type` (every type when `None`) are induced.
    pub fn semantic_subgraph_with<'q>(
        &self,
        query: impl Into<Query<'q>>,
        options: &SubgraphOptions,
        encoder: Box<dyn EmbeddingEncoder>,
    ) -> Result<SemanticGraph, EmbeddingError> {
        if options.connect_max_hops < 1 {
            return Err(EmbeddingError::InvalidOption(format!(
                "connect_max_hops must be >= 1, got {}",
                options.connect_max_hops
            )));
        }
        let hops = &options.search.hops;
        let results = self.semantic_multi_hop_search(query, &options.search)?;

        let mut ordered: Vec<NodeId> = results.iter().map(|r| r.node_id.clone()).collect();
        let mut keep: HashSet<NodeId> = ordered.iter().cloned().collect();
        if options.connect && !keep.is_empty() {
            let gate = if options.search.filter.strict {
                options.search.filter.gate(&self.graph)
            } else {
                None
            };
            let connectors = self.connect_components(
                &mut keep,
                hops.rel_type,
                hops.direction,
                options.connect_max_hops,
                gate.as_ref(),
            );
            ordered.extend(connectors);
        }

        let name = options
            .name
            .map(String::from)
            .unwrap_or_else(|| format!("{}_subgraph", self.graph.name));
        let mut sub = SemanticGraph::with_options(&name, encoder, false, self.embed_fields.clone());

        for id in &ordered {
            let node = match self.graph.get_node(&id.node_type, &id.id) {
                Ok(node) => node,
                Err(_) => continue,
            };
            let props: Vec<(&str, Value)> = node
                .properties
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            sub.graph
                .add_node(&node.node_type, &node.id, props)
                .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;

            if let Some(record) = self.embedding_store.get(id) {
                sub.embedding_store.add_with_model(
                    id.clone(),
                    &record.text,
                    record.vector,
                    &record.model,
                )?;
            }
        }

        let mut rel_types: Vec<String> = self.graph.edge_types().into_iter().cloned().collect();
        rel_types.sort();
        for rel_type in rel_types {
            if hops.rel_type.is_some_and(|wanted| wanted != rel_type) {
                continue;
            }
            for edge in self.graph.edges_of_type(&rel_type) {
                if keep.contains(&edge.source) && keep.contains(&edge.target) {
                    let props: Vec<(&str, Value)> = edge
                        .properties
                        .iter()
                        .map(|(k, v)| (k.as_str(), v.clone()))
                        .collect();
                    sub.graph
                        .add_edge(
                            &edge.rel_type,
                            (&edge.source.node_type, &edge.source.id),
                            (&edge.target.node_type, &edge.target.id),
                            props,
                        )
                        .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;
                }
            }
        }

        Ok(sub)
    }

    // -- model lifecycle ------------------------------------------------------

    /// Nodes whose stored vector was not made by the current encoder.
    ///
    /// Stale means the vector's recorded model differs from the encoder's
    /// model name, or its length differs from the encoder's dimension. Vectors
    /// recorded as `"unknown"` / `"provided"` (from ports that store no model,
    /// or supplied without an encoder) cannot be judged by model and are left
    /// out unless `include_unknown` is true; a dimension mismatch always
    /// counts. Needs an encoder.
    pub fn stale_embeddings(&self, include_unknown: bool) -> Result<Vec<NodeId>, EmbeddingError> {
        if self.embedding_store.encoder().is_none() {
            return Err(EmbeddingError::NoEncoder);
        }
        Ok(self.embedding_store.stale(None, None, include_unknown))
    }

    /// Re-encode every stale embedding with the current encoder.
    ///
    /// Each vector is rebuilt from the text it was originally embedded from
    /// (so a custom embed text survives); a vector stored with no text falls
    /// back to the node's embed fields. If the encoder's dimension differs from
    /// the store's, every stored vector is stale, so the store is emptied, its
    /// pinned dimension reset, and everything re-embedded.
    ///
    /// `include_unknown` also re-embeds vectors of unknown provenance;
    /// `batch_size` is the number of texts per encoder call. Returns the number
    /// of embeddings rewritten.
    pub fn reembed_stale(
        &mut self,
        include_unknown: bool,
        batch_size: usize,
    ) -> Result<usize, EmbeddingError> {
        let stale = self.stale_embeddings(include_unknown)?;
        if stale.is_empty() {
            return Ok(0);
        }

        let mut pairs: Vec<(NodeId, String)> = Vec::new();
        for id in stale {
            let record = match self.embedding_store.get(&id) {
                Some(record) => record,
                None => continue,
            };
            let mut text = record.text.clone();
            if text.is_empty() {
                text = self
                    .graph
                    .get_node(&id.node_type, &id.id)
                    .map(|node| self.embed_text_for(&node.properties))
                    .unwrap_or_default();
                if text.is_empty() {
                    text = format!("{}:{}", id.node_type, id.id);
                }
            }
            pairs.push((id, text));
        }

        let encoder_dimension = self.embedding_store.encoder().map(|e| e.dimension());
        if self.embedding_store.dimension() != encoder_dimension {
            self.embedding_store.clear_with(true);
        }

        let mut count = 0;
        for chunk in pairs.chunks(batch_size.max(1)) {
            let items: Vec<(NodeId, &str)> = chunk
                .iter()
                .map(|(id, text)| (id.clone(), text.as_str()))
                .collect();
            count += self.embed_nodes(&items)?;
        }
        Ok(count)
    }

    // -- persistence ------------------------------------------------------

    /// Sidecar path holding the embeddings for a saved graph:
    /// `<path>.embeddings.json`.
    pub fn embeddings_path_for(path: impl AsRef<Path>) -> PathBuf {
        let mut name = path.as_ref().as_os_str().to_os_string();
        name.push(EMBEDDING_SIDECAR_SUFFIX);
        PathBuf::from(name)
    }

    fn is_isonl(path: &Path, format: Option<&str>) -> bool {
        match format {
            Some(format) => format == "isonl",
            None => path.extension().is_some_and(|e| e == "isonl"),
        }
    }

    fn graph_name(path: &Path) -> &str {
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("graph")
    }

    /// Save the graph to `path`, and its embeddings alongside it.
    ///
    /// The ISON graph format carries nodes and edges only, so embeddings go to
    /// a JSON sidecar at `<path>.embeddings.json` (the `ison-embeddings`
    /// payload, readable by every port) that `load` picks up automatically.
    /// `format` is `"ison"`, `"isonl"`, or `None` to choose by extension.
    /// Pass `embeddings = false` for a graph-only save.
    ///
    /// Both files are written atomically (temp file, flush, rename), so a
    /// crash never leaves a truncated file. The pair cannot be replaced as one
    /// unit, so the sidecar also records a `graph_digest` of the graph it
    /// belongs to; `load` checks it and catches a crash between the two writes,
    /// a sidecar copied from another save, and a graph edited afterwards. The
    /// digest is taken of the graph *as it reads back from the text just
    /// written*, so a round trip that normalises a value does not cause a false
    /// mismatch.
    ///
    /// With `embeddings = false` an existing sidecar is left in place (and will
    /// fail the digest check on load, as it no longer matches).
    pub fn save(
        &self,
        path: impl AsRef<Path>,
        format: Option<&str>,
        embeddings: bool,
    ) -> Result<(), EmbeddingError> {
        let path = path.as_ref();
        let isonl = Self::is_isonl(path, format);
        let text = if isonl {
            self.graph.to_isonl()
        } else {
            self.graph.to_ison()
        };
        atomic_write(path, text.as_bytes())?;

        if embeddings {
            let (reread, int_ids) = digest::parse_graph_text(&text, isonl, Self::graph_name(path))?;
            let digest = graph_digest_with_int_ids(&reread, &int_ids);
            self.embedding_store
                .save_embeddings_with(Self::embeddings_path_for(path), Some(&digest))?;
        }
        Ok(())
    }

    /// Load a graph saved by `save` (or by any port), restoring its embeddings
    /// from the sidecar when present.
    ///
    /// `encoder` is attached to the loaded graph; pass the one you saved with
    /// so later text searches live in the same vector space as the restored
    /// vectors. A sidecar whose graph digest does not match the graph file is
    /// an error (`LoadOptions::on_mismatch` changes that).
    pub fn load(
        path: impl AsRef<Path>,
        encoder: Box<dyn EmbeddingEncoder>,
    ) -> Result<Self, EmbeddingError> {
        Self::load_with(path, encoder, &LoadOptions::default())
    }

    /// `load` with explicit options.
    pub fn load_with(
        path: impl AsRef<Path>,
        encoder: Box<dyn EmbeddingEncoder>,
        options: &LoadOptions,
    ) -> Result<Self, EmbeddingError> {
        let path = path.as_ref();
        let isonl = Self::is_isonl(path, options.format.as_deref());
        let text = std::fs::read_to_string(path)
            .map_err(|e| EmbeddingError::GraphError(format!("IO error: {e}")))?;
        let name = Self::graph_name(path);
        let graph = if isonl {
            ISONGraph::from_isonl(&text, Some(name))
        } else {
            ISONGraph::from_ison(&text, Some(name))
        }
        .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;

        // The loaded graph is adopted as is, so loading never auto-embeds:
        // the vectors come from the sidecar, already computed, and re-deriving
        // them would need the encoder and overwrite what was saved.
        let mut store = EmbeddingStore::new(Some(encoder));
        store.set_version(options.version.clone());
        store.set_on_model_mismatch(options.on_model_mismatch);
        let mut loaded = Self {
            graph,
            embedding_store: store,
            auto_embed: options.auto_embed,
            embed_fields: options.embed_fields.clone().unwrap_or_else(|| {
                ["name", "description", "content", "text"]
                    .iter()
                    .map(|f| f.to_string())
                    .collect()
            }),
        };

        if !options.embeddings {
            return Ok(loaded);
        }
        let sidecar = Self::embeddings_path_for(path);
        if !sidecar.is_file() {
            return Ok(loaded);
        }

        let json = std::fs::read_to_string(&sidecar)
            .map_err(|e| EmbeddingError::Io(format!("reading {}: {}", sidecar.display(), e)))?;
        let payload: ExportedEmbeddings = serde_json::from_str(&json).map_err(|e| {
            EmbeddingError::Payload(format!(
                "unreadable embeddings sidecar {}: {}",
                sidecar.display(),
                e
            ))
        })?;

        let mut problem: Option<String> = None;
        let mut unverified: Option<String> = None;
        if let Some(recorded) = &payload.graph_digest {
            let basis = payload
                .graph_digest_basis
                .as_deref()
                .unwrap_or(GRAPH_DIGEST_BASIS);
            if basis == GRAPH_DIGEST_BASIS {
                let int_ids = digest::int_ids_of_text(&text, isonl)?;
                let actual = graph_digest_with_int_ids(&loaded.graph, &int_ids);
                // The Python port hashes the graph as ITS parser reads it back,
                // and that parser reads an exponent-form float (`1e-05`, which
                // is how Python writes one) as a string. Accept that reading
                // of the same file before calling anything a mismatch.
                let python_view = digest::graph_digest_python_view(&loaded.graph, &int_ids);
                if *recorded != actual && *recorded != python_view {
                    let message = format!(
                        "embeddings sidecar {} does not belong to graph {}: it was written \
                         for graph digest {}, but the graph reads as {} (the sidecar is from \
                         a different save, or the graph was edited since).",
                        sidecar.display(),
                        path.display(),
                        recorded,
                        actual
                    );
                    if digest::has_exponent_floats(&loaded.graph) {
                        // Ports can disagree on how such a float reads back, in
                        // ways this build cannot enumerate, so a mismatch here
                        // may be false: warn instead of failing.
                        unverified = Some(format!(
                            "{message} The graph holds floats in exponent form (such as \
                             1e-05), which ports parse differently, so the digest could not be \
                             verified exactly; loading the sidecar's vectors anyway."
                        ));
                    } else {
                        problem = Some(format!(
                            "{message} Load with GraphMismatch::Reembed to rebuild the vectors \
                             or GraphMismatch::Warn to use them anyway."
                        ));
                    }
                }
            }
        }

        if let Some(notice) = unverified {
            loaded.embedding_store.warn(notice);
        }
        match (problem, options.on_mismatch) {
            (Some(problem), GraphMismatch::Error) => {
                return Err(EmbeddingError::GraphEmbeddingMismatch(problem));
            }
            (Some(_), GraphMismatch::Reembed) => {
                loaded.embed_all_nodes(None::<fn(&Node) -> String>)?;
            }
            (problem, _) => {
                if let Some(problem) = problem {
                    loaded.embedding_store.warn(problem);
                }
                loaded.embedding_store.import_embeddings(&payload, false)?;
                let stored_version = loaded
                    .embedding_store
                    .last_import_meta()
                    .encoder_version
                    .clone();
                let current_version = loaded.embedding_store.version();
                if let (Some(stored), Some(current)) = (stored_version, current_version) {
                    if stored != current {
                        loaded.embedding_store.warn(format!(
                            "embeddings were made by encoder version '{stored}', but the \
                             encoder in use is version '{current}'"
                        ));
                    }
                }
            }
        }
        Ok(loaded)
    }

    /// Embed all nodes in the graph, in one batched encoder call.
    pub fn embed_all_nodes<F>(&mut self, text_fn: Option<F>) -> Result<usize, EmbeddingError>
    where
        F: Fn(&Node) -> String,
    {
        let embed_fields = self.embed_fields.clone();

        let default_fn = |node: &Node| -> String {
            let parts: Vec<String> = embed_fields
                .iter()
                .filter_map(|f| node.properties.get(f))
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| v.to_string())
                .filter(|text| !text.is_empty())
                .collect();

            if parts.is_empty() {
                format!("{}:{}", node.node_type, node.id)
            } else {
                parts.join(" ")
            }
        };

        // Collect node info first to avoid borrow issues
        let node_info: Vec<(NodeId, String)> = self
            .graph
            .nodes()
            .map(|node| {
                let text = match &text_fn {
                    Some(f) => f(node),
                    None => default_fn(node),
                };
                (NodeId::new(&node.node_type, &node.id), text)
            })
            .filter(|(_, text)| !text.is_empty())
            .collect();

        let batch: Vec<(NodeId, &str)> = node_info
            .iter()
            .map(|(node_id, text)| (node_id.clone(), text.as_str()))
            .collect();

        self.embedding_store.add_batch(&batch)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserted identically in the Python, TypeScript, JavaScript and C++
    /// suites - keeping the ports bit-identical is what lets an embedding
    /// store written by one be searched by another. Written at the same
    /// precision as the other ports' goldens so the two can be compared by
    /// eye, which is why clippy's excessive_precision is off here.
    #[allow(clippy::excessive_precision)]
    const GOLDEN_HELLO: [f32; 4] = [0.8381705284, -0.8255031109, 0.5899617672, 0.4615051746];
    #[allow(clippy::excessive_precision)]
    const GOLDEN_ISON: [f32; 4] = [-0.3658730984, -0.2765417099, 0.2944871187, -0.6502785683];
    #[allow(clippy::excessive_precision)]
    const GOLDEN_EMPTY: [f32; 4] = [-0.4520676136, -0.3797093630, 0.3339765072, 0.9023951292];

    fn seeded_graph() -> SemanticGraph {
        let mut graph = SemanticGraph::with_options(
            "seeds",
            Box::new(MockEncoder::new(4)),
            false,
            vec!["name".to_string()],
        );
        graph.add_node("person", "1", vec![], None).unwrap();
        graph.add_node("person", "2", vec![], None).unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        // person:1 matches the query perfectly, person:2 slightly less well.
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "a",
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "2"),
                "b",
                Some(vec![0.9, 0.19f32.sqrt(), 0.0, 0.0]),
            )
            .unwrap();
        graph
    }

    /// Encoder returning hand-picked vectors so scores are exact.
    struct FixedEncoder;
    impl EmbeddingEncoder for FixedEncoder {
        fn dimension(&self) -> usize {
            4
        }
        fn encode(&self, _text: &str) -> Vec<f32> {
            vec![1.0, 0.0, 0.0, 0.0]
        }
    }

    #[test]
    fn test_mock_encoder_dimension() {
        let encoder = MockEncoder::new(384);
        assert_eq!(encoder.dimension(), 384);
    }

    #[test]
    fn test_mock_encoder_deterministic() {
        let encoder = MockEncoder::new(384);
        let v1 = encoder.encode("hello");
        let v2 = encoder.encode("hello");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_mock_encoder_different() {
        let encoder = MockEncoder::new(384);
        let v1 = encoder.encode("hello");
        let v2 = encoder.encode("world");
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_mock_encoder_matches_other_ports() {
        let encoder = MockEncoder::new(4);
        for (text, want) in [
            ("hello", GOLDEN_HELLO),
            ("ison", GOLDEN_ISON),
            ("", GOLDEN_EMPTY),
        ] {
            let got = encoder.encode(text);
            for (i, expected) in want.iter().enumerate() {
                assert!(
                    (got[i] - expected).abs() < 1e-6,
                    "{text:?}[{i}] = {}, expected {expected}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn test_mock_encoder_has_no_collisions() {
        // The previous LCG scheme emitted the low 16 bits of the state, so
        // every vector was determined by seed mod 65536: 280 of these 5,000
        // texts collided outright.
        let encoder = MockEncoder::new(8);
        let mut seen: HashSet<Vec<u32>> = HashSet::new();
        for i in 0..5000 {
            let bits: Vec<u32> = encoder
                .encode(&format!("text{i}"))
                .iter()
                .map(|v| v.to_bits())
                .collect();
            seen.insert(bits);
        }
        assert_eq!(seen.len(), 5000);
    }

    #[test]
    fn test_mock_encoder_values_in_range() {
        for value in MockEncoder::new(128).encode("range check") {
            assert!((-1.0..1.0).contains(&value), "{value} out of range");
        }
    }

    #[test]
    fn test_embedding_store_add_get() {
        let encoder = MockEncoder::new(384);
        let mut store = EmbeddingStore::new(Some(Box::new(encoder)));

        let node_id = NodeId::new("person", "1");
        store
            .add(node_id.clone(), "Alice is an engineer", None)
            .unwrap();

        let record = store.get(&node_id).unwrap();
        assert_eq!(record.text, "Alice is an engineer");
    }

    #[test]
    fn test_embedding_store_remove() {
        let encoder = MockEncoder::new(384);
        let mut store = EmbeddingStore::new(Some(Box::new(encoder)));

        let node_id = NodeId::new("person", "1");
        store.add(node_id.clone(), "test", None).unwrap();
        assert!(store.remove(&node_id));
        assert!(store.get(&node_id).is_none());
    }

    #[test]
    fn test_similarity_search() {
        let encoder = MockEncoder::new(384);
        let mut store = EmbeddingStore::new(Some(Box::new(encoder)));

        store
            .add(NodeId::new("person", "1"), "software engineer", None)
            .unwrap();
        store
            .add(NodeId::new("person", "2"), "data scientist", None)
            .unwrap();

        // Use negative threshold to get all results with mock encoder
        let results = store
            .similarity_search("software engineer", 2, None, -1.0)
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch_is_rejected() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        store
            .add(NodeId::new("a", "1"), "full width", None)
            .unwrap();

        let err = store.add(NodeId::new("a", "2"), "too short", Some(vec![1.0, 0.0]));
        assert!(matches!(err, Err(EmbeddingError::DimensionMismatch { .. })));

        let err = store.add(NodeId::new("a", "3"), "empty", Some(vec![]));
        assert!(matches!(err, Err(EmbeddingError::EmptyVector(_))));

        let err = store.similarity_search_vector(&[1.0, 0.0], 1, None, -1.0);
        assert!(matches!(err, Err(EmbeddingError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_top_k_selection_matches_a_full_sort() {
        // Selecting the best k must return what ordering everything would.
        let encoder = MockEncoder::new(64);
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(64))));
        let texts: Vec<String> = (0..500).map(|i| format!("text {i}")).collect();
        let entries: Vec<(NodeId, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (NodeId::new("n", i.to_string()), t.as_str()))
            .collect();
        store.add_batch(&entries).unwrap();

        for k in [1usize, 5, 10, 50, 200] {
            for qi in 0..5 {
                let query = encoder.encode(&format!("text {}", qi * 17));
                let got = store
                    .similarity_search_vector(&query, k, None, -1.0)
                    .unwrap();
                let all = store
                    .similarity_search_vector(&query, usize::MAX, None, -1.0)
                    .unwrap();
                let want = &all[..k.min(all.len())];
                assert_eq!(got.len(), want.len(), "k={k}");
                for (a, b) in got.iter().zip(want.iter()) {
                    assert_eq!(a.node_id, b.node_id, "k={k}");
                    assert_eq!(a.score, b.score, "k={k}");
                }
            }
        }
    }

    #[test]
    fn test_top_k_edges() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(16))));
        store
            .add_batch(&[
                (NodeId::new("n", "1"), "one"),
                (NodeId::new("n", "2"), "two"),
            ])
            .unwrap();
        let q = MockEncoder::new(16).encode("anything");
        assert_eq!(
            store
                .similarity_search_vector(&q, 100, None, -1.0)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            store
                .similarity_search_vector(&q, 0, None, -1.0)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn test_dimension_pinned_by_first_vector() {
        let mut store = EmbeddingStore::new(None);
        assert_eq!(store.dimension(), None);
        store
            .add(NodeId::new("a", "1"), "x", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(store.dimension(), Some(3));
    }

    #[test]
    fn test_text_without_encoder_is_rejected() {
        let mut store = EmbeddingStore::new(None);
        assert!(matches!(
            store.add(NodeId::new("a", "1"), "no encoder", None),
            Err(EmbeddingError::NoEncoder)
        ));
    }

    #[test]
    fn test_semantic_graph_add_node() {
        let encoder = MockEncoder::new(384);
        let mut graph = SemanticGraph::new("test", Box::new(encoder));

        graph
            .add_node(
                "person",
                "1",
                vec![("name", Value::String("Alice".to_string()))],
                Some("Alice is an engineer"),
            )
            .unwrap();

        assert_eq!(graph.graph().node_count(), 1);
        assert_eq!(graph.embedding_store().count(), 1);
    }

    #[test]
    fn test_add_node_auto_embeds_from_properties() {
        let mut graph = SemanticGraph::new("test", Box::new(MockEncoder::new(16)));
        graph
            .add_node(
                "person",
                "1",
                vec![
                    ("name", Value::String("Alice".to_string())),
                    ("description", Value::String("engineer".to_string())),
                ],
                None,
            )
            .unwrap();

        assert_eq!(graph.embedding_store().count(), 1);
        let record = graph.get_embedding(&NodeId::new("person", "1")).unwrap();
        assert_eq!(record.text, "Alice engineer");
    }

    #[test]
    fn test_semantic_graph_similarity() {
        let encoder = MockEncoder::new(384);
        let mut graph = SemanticGraph::new("test", Box::new(encoder));

        graph
            .add_node("person", "1", vec![], Some("software engineer"))
            .unwrap();
        graph
            .add_node("person", "2", vec![], Some("data scientist"))
            .unwrap();

        // Use negative threshold to get all results with mock encoder
        let results = graph
            .similarity_search("software engineer", 5, None, -1.0)
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_multi_hop() {
        let encoder = MockEncoder::new(384);
        let mut graph = SemanticGraph::new("test", Box::new(encoder));

        graph
            .add_node("person", "1", vec![], Some("software engineer"))
            .unwrap();
        graph
            .add_node("person", "2", vec![], Some("manager"))
            .unwrap();

        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();

        let results = graph
            .semantic_multi_hop(
                "software",
                Some("KNOWS"),
                2,
                5,
                10,
                Direction::Out,
                0.8,
                -1.0,
            )
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_seed_keeps_its_own_score() {
        // Seeds were inserted only when absent from the result map, so an
        // entry written by an earlier traversal won even when it scored
        // lower: person:2 is itself a 0.9 match at hop 0, but came back at
        // seed(1.0) * decay(0.8) = 0.8, at hop 1, down a path through
        // person:1 that it never needed.
        let mut graph = seeded_graph();
        graph
            .embedding_store_mut()
            .set_encoder(Box::new(FixedEncoder));

        let results = graph
            .semantic_multi_hop("query", Some("KNOWS"), 2, 5, 10, Direction::Out, 0.8, 0.3)
            .unwrap();

        let second = results.iter().find(|r| r.node_id.id == "2").unwrap();
        assert!(
            (second.score - 0.9).abs() < 1e-6,
            "score was {}",
            second.score
        );
        assert_eq!(second.hop_count, 0);
        assert_eq!(second.path, vec![NodeId::new("person", "2")]);
    }

    #[test]
    fn test_traversal_still_reaches_non_seeds() {
        let mut graph = seeded_graph();
        graph
            .embedding_store_mut()
            .set_encoder(Box::new(FixedEncoder));
        graph.add_node("person", "3", vec![], None).unwrap();
        graph
            .add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![])
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "3"),
                "c",
                Some(vec![0.0, 0.0, 1.0, 0.0]),
            )
            .unwrap();

        let results = graph
            .semantic_multi_hop("query", Some("KNOWS"), 2, 5, 10, Direction::Out, 0.8, 0.3)
            .unwrap();

        let third = results.iter().find(|r| r.node_id.id == "3").unwrap();
        assert!(third.hop_count > 0);
        assert!(third.score > 0.0);
    }

    #[test]
    fn test_negative_seed_does_not_grow_with_distance() {
        let mut graph = SemanticGraph::with_options("neg", Box::new(FixedEncoder), false, vec![]);
        graph.add_node("person", "1", vec![], None).unwrap();
        graph.add_node("person", "2", vec![], None).unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "a",
                Some(vec![-1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "2"),
                "b",
                Some(vec![0.0, 0.0, 1.0, 0.0]),
            )
            .unwrap();

        let results = graph
            .semantic_multi_hop("query", Some("KNOWS"), 1, 5, 10, Direction::Out, 0.8, -1.0)
            .unwrap();

        let first = results.iter().find(|r| r.node_id.id == "1").unwrap();
        let second = results.iter().find(|r| r.node_id.id == "2").unwrap();
        assert!((first.score + 1.0).abs() < 1e-6);
        // 0.0, not -1.0 * 0.8 = -0.8, which would have outranked the seed.
        assert!(second.score.abs() < 1e-6, "score was {}", second.score);
    }

    #[test]
    fn test_blend_lifts_a_relevant_distant_node() {
        struct QueryEncoder;
        impl EmbeddingEncoder for QueryEncoder {
            fn dimension(&self) -> usize {
                4
            }
            fn encode(&self, _text: &str) -> Vec<f32> {
                vec![1.0, 0.0, 0.0, 0.0]
            }
        }

        let mut graph = SemanticGraph::with_options("blend", Box::new(QueryEncoder), false, vec![]);
        for id in ["1", "2", "3"] {
            graph.add_node("person", id, vec![], None).unwrap();
        }
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "3"), vec![])
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "seed",
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "2"),
                "off",
                Some(vec![0.0, 1.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "3"),
                "on",
                Some(vec![0.8, 0.6, 0.0, 0.0]),
            )
            .unwrap();

        let base_options = MultiHopOptions {
            rel_type: Some("KNOWS"),
            max_hops: 1,
            top_k_seeds: 1,
            threshold: 0.5,
            ..Default::default()
        };

        let plain = graph
            .semantic_multi_hop_with("query", &base_options)
            .unwrap();
        let plain2 = plain.iter().find(|r| r.node_id.id == "2").unwrap().score;
        let plain3 = plain.iter().find(|r| r.node_id.id == "3").unwrap().score;
        assert!((plain2 - plain3).abs() < 1e-6);

        let blended = graph
            .semantic_multi_hop_with(
                "query",
                &MultiHopOptions {
                    blend: 0.5,
                    ..base_options
                },
            )
            .unwrap();
        let blended2 = blended.iter().find(|r| r.node_id.id == "2").unwrap().score;
        let blended3 = blended.iter().find(|r| r.node_id.id == "3").unwrap().score;
        assert!(blended3 > blended2);
    }

    #[test]
    fn test_blend_out_of_range_is_rejected() {
        let graph = seeded_graph();
        let result = graph.semantic_multi_hop_with(
            "query",
            &MultiHopOptions {
                blend: 1.5,
                ..Default::default()
            },
        );
        assert!(matches!(result, Err(EmbeddingError::InvalidBlend(_))));
    }

    #[test]
    fn test_semantic_path_finds_a_route() {
        let mut graph = SemanticGraph::with_options("paths", Box::new(FixedEncoder), false, vec![]);
        for id in ["1", "2", "3"] {
            graph.add_node("person", id, vec![], None).unwrap();
        }
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![])
            .unwrap();
        // Distinct vectors, and a threshold that admits only person:1, so the
        // seed is deterministic. Giving every node the same vector leaves the
        // seed order to HashMap iteration, and a run that seeds on the target
        // finds the zero-hop path from the target to itself.
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "a",
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "2"),
                "b",
                Some(vec![0.0, 1.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "3"),
                "c",
                Some(vec![0.0, 0.0, 1.0, 0.0]),
            )
            .unwrap();

        let result = graph
            .semantic_path(
                "query",
                &NodeId::new("person", "3"),
                Some("KNOWS"),
                5,
                1,
                Direction::Out,
                0.9,
                0.5,
            )
            .unwrap();

        let result = result.expect("a path should exist");
        assert_eq!(result.node_id, NodeId::new("person", "3"));
        assert_eq!(result.hop_count, 2);
        assert!(
            (result.score - 0.81).abs() < 1e-5,
            "score was {}",
            result.score
        );
        assert_eq!(result.path.last().unwrap(), &NodeId::new("person", "3"));
    }

    #[test]
    fn test_semantic_path_returns_none_without_a_route() {
        let mut graph = SemanticGraph::with_options("paths", Box::new(FixedEncoder), false, vec![]);
        graph.add_node("person", "1", vec![], None).unwrap();
        graph.add_node("person", "2", vec![], None).unwrap();
        // Only person:1 clears the threshold, so it is the sole seed - and it
        // has no edge to the target. (Were person:2 a seed too, the path from
        // the target to itself would legitimately be found at zero hops.)
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "a",
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "2"),
                "b",
                Some(vec![0.0, 1.0, 0.0, 0.0]),
            )
            .unwrap();

        let result = graph
            .semantic_path(
                "query",
                &NodeId::new("person", "2"),
                Some("KNOWS"),
                5,
                3,
                Direction::Out,
                0.9,
                0.5,
            )
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_similar_to_node_excludes_itself() {
        let mut graph = SemanticGraph::new("similar", Box::new(MockEncoder::new(16)));
        graph
            .add_node(
                "person",
                "1",
                vec![("name", Value::String("Alice".to_string()))],
                None,
            )
            .unwrap();
        graph
            .add_node(
                "person",
                "2",
                vec![("name", Value::String("Bob".to_string()))],
                None,
            )
            .unwrap();
        graph
            .add_node(
                "person",
                "3",
                vec![("name", Value::String("Carol".to_string()))],
                None,
            )
            .unwrap();

        let results = graph
            .similar_to_node(&NodeId::new("person", "1"), 2, None, -1.0)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.node_id.id != "1"));
    }

    #[test]
    fn test_similar_to_node_without_embedding() {
        let mut graph =
            SemanticGraph::with_options("similar", Box::new(MockEncoder::new(16)), false, vec![]);
        graph.add_node("person", "9", vec![], None).unwrap();
        let results = graph
            .similar_to_node(&NodeId::new("person", "9"), 5, None, -1.0)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_semantic_subgraph_is_a_working_graph() {
        let mut graph = SemanticGraph::new("full", Box::new(MockEncoder::new(16)));
        graph
            .add_node(
                "person",
                "1",
                vec![("name", Value::String("Alice".to_string()))],
                None,
            )
            .unwrap();
        graph
            .add_node(
                "person",
                "2",
                vec![("name", Value::String("Bob".to_string()))],
                None,
            )
            .unwrap();
        graph
            .add_node(
                "person",
                "3",
                vec![("name", Value::String("Carol".to_string()))],
                None,
            )
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![])
            .unwrap();

        let sub = graph
            .semantic_subgraph(
                "Alice",
                &MultiHopOptions {
                    rel_type: Some("KNOWS"),
                    max_hops: 1,
                    top_k_seeds: 1,
                    top_k_results: 2,
                    threshold: -1.0,
                    ..Default::default()
                },
                "slice",
                Box::new(MockEncoder::new(16)),
            )
            .unwrap();

        assert!(sub.graph().node_count() > 0);
        assert!(sub.graph().node_count() <= 2);
        // every kept node brought its embedding along
        assert_eq!(sub.embedding_store().count(), sub.graph().node_count());
    }

    #[test]
    fn test_embed_text_is_identical_across_ports() {
        // The same golden string is asserted in all six suites. ISONGraph
        // 1.4.0 made property values typed, which is where the ports could
        // drift: Python would otherwise render None as "None" and True as
        // "True", while JavaScript renders "null" and "true". Different text
        // means a different vector, which would break the cross-port
        // guarantee.
        let mut graph = SemanticGraph::new("props", Box::new(MockEncoder::new(8)));
        graph
            .add_node(
                "doc",
                "1",
                vec![
                    ("name", Value::String("Report".to_string())),
                    ("description", Value::Null),
                    ("content", Value::Float(1.5)),
                    ("text", Value::Bool(true)),
                ],
                None,
            )
            .unwrap();

        let record = graph.get_embedding(&NodeId::new("doc", "1")).unwrap();
        assert_eq!(record.text, "Report 1.5 true");
    }

    #[test]
    fn test_floats_render_the_same_way_in_every_port() {
        // The same values are asserted in all six suites. Python used to
        // write an integral float as "1.0" where the others write "1", and
        // C++ used six significant figures, rendering 1/3 as "0.333333".
        // Either one means a different embedding text, and so a different
        // vector, for the same graph.
        let graph = SemanticGraph::new("floats", Box::new(MockEncoder::new(8)));
        let render = |v: f64| {
            let mut props = HashMap::new();
            props.insert("name".to_string(), Value::Float(v));
            graph.embed_text_for(&props)
        };
        assert_eq!(render(1.0), "1");
        assert_eq!(render(100.0), "100");
        assert_eq!(render(1.5), "1.5");
        assert_eq!(render(0.1), "0.1");
        assert_eq!(render(1.0 / 3.0), "0.3333333333333333");
    }

    #[test]
    fn test_null_contributes_nothing_to_embed_text() {
        let graph = SemanticGraph::new("props", Box::new(MockEncoder::new(8)));
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::Null);
        props.insert("description".to_string(), Value::String("kept".to_string()));
        assert_eq!(graph.embed_text_for(&props), "kept");
    }

    #[test]
    fn test_embed_all_nodes_batches() {
        let mut graph = SemanticGraph::with_options(
            "batch",
            Box::new(MockEncoder::new(16)),
            false,
            vec!["name".to_string()],
        );
        for i in 0..10 {
            graph
                .add_node(
                    "doc",
                    &i.to_string(),
                    vec![("name", Value::String("Document".to_string()))],
                    None,
                )
                .unwrap();
        }
        let count = graph.embed_all_nodes(None::<fn(&Node) -> String>).unwrap();
        assert_eq!(count, 10);
        assert_eq!(graph.embedding_store().count(), 10);
    }

    #[test]
    fn test_embeddings_round_trip_through_json() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        store
            .add(NodeId::new("person", "1"), "Alice", None)
            .unwrap();
        store
            .add(NodeId::new("account", "007"), "Bond", None)
            .unwrap();

        let json = store.to_json().unwrap();
        assert!(json.contains(EMBEDDING_FORMAT));

        let mut other = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        assert_eq!(other.from_json(&json, false).unwrap(), 2);
        assert_eq!(
            other.get(&NodeId::new("person", "1")).unwrap().text,
            "Alice"
        );
        // a numeric-looking string id keeps its exact form
        assert_eq!(
            other.get(&NodeId::new("account", "007")).unwrap().text,
            "Bond"
        );
    }

    #[test]
    fn test_imports_a_payload_written_by_the_python_port() {
        // Captured verbatim from the Python port. Its node ids are typed, so
        // an int id arrives unquoted ("id":1) where this port always writes a
        // string - both have to land on the same NodeId here.
        let python_payload = r#"{"format":"ison-embeddings","version":1,"dimension":4,"count":2,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","model":"mock-4","vector":[-0.8610728979110718,-0.33433854579925537,0.8408418893814087,-0.2771846055984497]},{"type":"account","id":"007","id_type":"str","text":"Bond","model":"mock-4","vector":[-0.9531009197235107,0.9167747497558594,0.5073114633560181,-0.0420149564743042]}]}"#;

        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(4))));
        assert_eq!(store.from_json(python_payload, false).unwrap(), 2);
        assert_eq!(
            store.get(&NodeId::new("person", "1")).unwrap().text,
            "Alice"
        );
        assert_eq!(
            store.get(&NodeId::new("account", "007")).unwrap().text,
            "Bond"
        );

        // and the vectors match what this port encodes for the same text
        let encoder = MockEncoder::new(4);
        let expected = encoder.encode("Alice");
        let stored = &store.get(&NodeId::new("person", "1")).unwrap().vector;
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (stored[i] - want).abs() < 1e-6,
                "component {i}: {} vs {want}",
                stored[i]
            );
        }
    }

    #[test]
    fn test_import_rejects_a_foreign_payload() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        let payload = ExportedEmbeddings {
            format: "something-else".to_string(),
            version: EMBEDDING_FORMAT_VERSION,
            dimension: None,
            count: 0,
            embeddings: vec![],
            encoder_model: None,
            encoder_version: None,
            graph_digest: None,
            graph_digest_basis: None,
        };
        assert!(matches!(
            store.import_embeddings(&payload, false),
            Err(EmbeddingError::Payload(_))
        ));
    }

    // -- update_node keeps the embedding in step ---------------------------

    fn text_prop(text: &str) -> Value {
        Value::String(text.to_string())
    }

    #[test]
    fn test_update_node_reembeds_when_an_embed_field_changes() {
        let mut graph = SemanticGraph::new("u", Box::new(MockEncoder::new(8)));
        graph
            .add_node(
                "person",
                "1",
                vec![("name", text_prop("Alice engineer"))],
                None,
            )
            .unwrap();
        let id = NodeId::new("person", "1");
        assert_eq!(graph.get_embedding(&id).unwrap().text, "Alice engineer");

        graph
            .update_node("person", "1", vec![("name", text_prop("Zed painter"))])
            .unwrap();

        let record = graph.get_embedding(&id).unwrap();
        assert_eq!(record.text, "Zed painter");
        assert_eq!(record.vector, MockEncoder::new(8).encode("Zed painter"));
    }

    #[test]
    fn test_update_node_embeds_from_the_merged_properties() {
        let mut graph = SemanticGraph::new("u", Box::new(MockEncoder::new(8)));
        graph
            .add_node(
                "doc",
                "1",
                vec![("name", text_prop("Title")), ("text", text_prop("body"))],
                None,
            )
            .unwrap();
        // Only `text` changes; `name` must still contribute.
        graph
            .update_node("doc", "1", vec![("text", text_prop("new body"))])
            .unwrap();
        assert_eq!(
            graph.get_embedding(&NodeId::new("doc", "1")).unwrap().text,
            "Title new body"
        );
    }

    #[test]
    fn test_update_node_drops_the_vector_when_no_text_remains() {
        let mut graph = SemanticGraph::new("u", Box::new(MockEncoder::new(8)));
        graph
            .add_node("person", "1", vec![("name", text_prop("Alice"))], None)
            .unwrap();
        let id = NodeId::new("person", "1");
        assert!(graph.get_embedding(&id).is_some());

        graph
            .update_node("person", "1", vec![("name", text_prop(""))])
            .unwrap();
        assert!(graph.get_embedding(&id).is_none());
        assert!(graph
            .similarity_search("Alice", 5, None, -1.0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_update_node_leaves_the_embedding_alone_without_an_embed_field() {
        let mut graph = SemanticGraph::new("u", Box::new(MockEncoder::new(8)));
        graph
            .add_node(
                "person",
                "1",
                vec![("name", text_prop("Alice"))],
                Some("custom text"),
            )
            .unwrap();
        let id = NodeId::new("person", "1");

        graph
            .update_node("person", "1", vec![("age", Value::Int(30))])
            .unwrap();
        // Not re-derived from properties: the caller-supplied text survives.
        assert_eq!(graph.get_embedding(&id).unwrap().text, "custom text");
    }

    #[test]
    fn test_update_node_leaves_the_embedding_alone_when_auto_embed_is_off() {
        let mut graph = SemanticGraph::with_options(
            "u",
            Box::new(MockEncoder::new(8)),
            false,
            vec!["name".to_string()],
        );
        graph
            .add_node(
                "person",
                "1",
                vec![("name", text_prop("Alice"))],
                Some("managed"),
            )
            .unwrap();
        let id = NodeId::new("person", "1");

        graph
            .update_node("person", "1", vec![("name", text_prop("Bob"))])
            .unwrap();
        assert_eq!(graph.get_embedding(&id).unwrap().text, "managed");
        // ...but the property itself did change.
        assert_eq!(
            graph.graph().get_node("person", "1").unwrap().properties["name"],
            text_prop("Bob")
        );
    }

    #[test]
    fn test_update_node_on_a_missing_node_is_an_error() {
        let mut graph = SemanticGraph::new("u", Box::new(MockEncoder::new(8)));
        assert!(matches!(
            graph.update_node("person", "nope", vec![("name", text_prop("x"))]),
            Err(EmbeddingError::GraphError(_))
        ));
    }

    #[test]
    fn test_edges_are_added_and_removed_without_touching_embeddings() {
        let mut graph = SemanticGraph::new("e", Box::new(MockEncoder::new(8)));
        for id in ["1", "2"] {
            graph
                .add_node("person", id, vec![("name", text_prop(id))], None)
                .unwrap();
        }
        let before = graph.embedding_store().count();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        assert!(graph
            .graph()
            .has_edge("KNOWS", ("person", "1"), ("person", "2")));
        assert!(matches!(
            graph.add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![]),
            Err(EmbeddingError::GraphError(_))
        ));
        assert!(matches!(
            graph.add_edge("KNOWS", ("person", "1"), ("person", "9"), vec![]),
            Err(EmbeddingError::GraphError(_))
        ));
        graph
            .remove_edge("KNOWS", ("person", "1"), ("person", "2"))
            .unwrap();
        assert!(!graph
            .graph()
            .has_edge("KNOWS", ("person", "1"), ("person", "2")));
        assert!(matches!(
            graph.remove_edge("KNOWS", ("person", "1"), ("person", "2")),
            Err(EmbeddingError::GraphError(_))
        ));
        assert_eq!(graph.embedding_store().count(), before);
    }

    // -- semantic_path picks the best seed ---------------------------------

    /// Seeds a (score 1.0) and s (score 0.9) both reach the target; a is three
    /// hops away and s is one.
    fn two_seed_graph() -> SemanticGraph {
        let mut graph = SemanticGraph::with_options("p", Box::new(FixedEncoder), false, vec![]);
        for id in ["a", "b", "c", "t", "s"] {
            graph.add_node("n", id, vec![], None).unwrap();
        }
        for (from, to) in [("a", "b"), ("b", "c"), ("c", "t"), ("s", "t")] {
            graph.add_edge("E", ("n", from), ("n", to), vec![]).unwrap();
        }
        let vectors: [(&str, Vec<f32>); 3] = [
            ("a", vec![1.0, 0.0, 0.0, 0.0]),
            ("s", vec![0.9, 0.19f32.sqrt(), 0.0, 0.0]),
            ("b", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        for (id, vector) in vectors {
            graph
                .embedding_store_mut()
                .add(NodeId::new("n", id), id, Some(vector))
                .unwrap();
        }
        graph
    }

    #[test]
    fn test_semantic_path_prefers_the_best_scoring_seed_over_the_first() {
        let graph = two_seed_graph();
        let result = graph
            .semantic_path(
                "q",
                &NodeId::new("n", "t"),
                Some("E"),
                5,
                2,
                Direction::Out,
                0.5,
                0.5,
            )
            .unwrap()
            .expect("a path should exist");

        // a: 1.0 * 0.5^3 = 0.125 ; s: 0.9 * 0.5^1 = 0.45
        assert_eq!(result.hop_count, 1);
        assert_eq!(result.path.first().unwrap(), &NodeId::new("n", "s"));
        assert!((result.score - 0.45).abs() < 1e-5, "score {}", result.score);
    }

    #[test]
    fn test_semantic_path_keeps_the_closer_seed_when_it_is_also_the_best() {
        let graph = two_seed_graph();
        // With no decay the more similar seed (a, 1.0) wins despite its length.
        let result = graph
            .semantic_path(
                "q",
                &NodeId::new("n", "t"),
                Some("E"),
                5,
                2,
                Direction::Out,
                1.0,
                0.5,
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.path.first().unwrap(), &NodeId::new("n", "a"));
        assert_eq!(result.hop_count, 3);
        assert!((result.score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_semantic_path_clamps_a_negative_seed_to_zero() {
        let mut graph = SemanticGraph::with_options("p", Box::new(FixedEncoder), false, vec![]);
        graph.add_node("n", "neg", vec![], None).unwrap();
        graph.add_node("n", "t", vec![], None).unwrap();
        graph
            .add_edge("E", ("n", "neg"), ("n", "t"), vec![])
            .unwrap();
        // Opposite to the query: cosine -1. The target has no embedding, so it
        // is not itself a seed.
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("n", "neg"),
                "neg",
                Some(vec![-1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();

        let result = graph
            .semantic_path(
                "q",
                &NodeId::new("n", "t"),
                Some("E"),
                5,
                3,
                Direction::Out,
                0.9,
                -1.0,
            )
            .unwrap()
            .expect("a path should exist");
        assert_eq!(result.hop_count, 1);
        // Previously -1.0 * 0.9 = -0.9, i.e. below a zero-relevance result.
        assert_eq!(result.score, 0.0);
    }

    // -- save / load with the embeddings sidecar ----------------------------

    fn persisted_graph() -> SemanticGraph {
        let mut graph = SemanticGraph::new("g", Box::new(MockEncoder::new(8)));
        graph
            .add_node(
                "person",
                "1",
                vec![("name", text_prop("Alice engineer"))],
                None,
            )
            .unwrap();
        graph
            .add_node(
                "person",
                "2",
                vec![("name", text_prop("Bob analyst"))],
                None,
            )
            .unwrap();
        graph
            .add_node("doc", "007", vec![("text", text_prop("Bond file"))], None)
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
    }

    #[test]
    fn test_save_and_load_round_trip_with_embeddings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        let graph = persisted_graph();
        graph.save(&path, None, true).unwrap();

        let sidecar = dir.path().join("g.ison.embeddings.json");
        assert_eq!(SemanticGraph::embeddings_path_for(&path), sidecar);
        assert!(sidecar.is_file());

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.graph().node_count(), 3);
        assert_eq!(loaded.embedding_store().count(), 3);
        assert_eq!(
            loaded
                .graph()
                .neighbors(&("person", "1"), Some("KNOWS"), Direction::Out),
            vec![NodeId::new("person", "2")]
        );
        for id in [("person", "1"), ("person", "2"), ("doc", "007")] {
            let node_id = NodeId::new(id.0, id.1);
            let a = graph.get_embedding(&node_id).unwrap();
            let b = loaded.get_embedding(&node_id).unwrap();
            assert_eq!(a.text, b.text);
            assert_eq!(a.vector, b.vector);
        }
        // searches work on the restored vectors
        let hits = loaded
            .similarity_search("Alice engineer", 1, None, -1.0)
            .unwrap();
        assert_eq!(hits[0].node_id, NodeId::new("person", "1"));
    }

    #[test]
    fn test_save_without_embeddings_writes_no_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        persisted_graph().save(&path, None, false).unwrap();
        assert!(!SemanticGraph::embeddings_path_for(&path).exists());

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.graph().node_count(), 3);
        assert_eq!(loaded.embedding_store().count(), 0);
    }

    #[test]
    fn test_load_can_skip_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        persisted_graph().save(&path, None, true).unwrap();

        let options = LoadOptions {
            embeddings: false,
            ..LoadOptions::default()
        };
        let loaded =
            SemanticGraph::load_with(&path, Box::new(MockEncoder::new(8)), &options).unwrap();
        assert_eq!(loaded.embedding_store().count(), 0);
    }

    #[test]
    fn test_load_does_not_reembed_over_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        let mut graph = persisted_graph();
        // A caller-managed vector that differs from what the encoder would make.
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("person", "1"),
                "Alice engineer",
                Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        graph.save(&path, None, true).unwrap();

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(
            loaded
                .get_embedding(&NodeId::new("person", "1"))
                .unwrap()
                .vector,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_load_of_a_missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let result =
            SemanticGraph::load(dir.path().join("nope.ison"), Box::new(MockEncoder::new(8)));
        assert!(matches!(result, Err(EmbeddingError::GraphError(_))));
    }

    #[test]
    fn test_loads_a_graph_and_sidecar_written_by_the_python_port() {
        // Captured verbatim from SemanticGraph.save() in the Python port
        // (MockEncoder(8)). Python's int node id 2 is written unquoted in both
        // files; it has to land on the same string id here.
        let graph_file = "edges.KNOWS\nsource target\n:person:1 :person:2\n\nnodes.doc\nid text\n\"007\" \"Bond file\"\n\nnodes.person\nid name\n\"1\" \"Alice engineer\"\n2 \"Bob analyst\"\n";
        let sidecar = r#"{"format": "ison-embeddings", "version": 1, "dimension": 8, "count": 3, "embeddings": [{"type": "person", "id": "1", "id_type": "str", "text": "Alice engineer", "model": "mock-8", "vector": [-0.2115306854248047, -0.6689056158065796, -0.053275227546691895, 0.08514654636383057, -0.49772799015045166, 0.5832090377807617, -0.6525833606719971, -0.5084015130996704]}, {"type": "person", "id": 2, "id_type": "int", "text": "Bob analyst", "model": "mock-8", "vector": [0.963505744934082, 0.406618595123291, 0.7846142053604126, -0.2902500629425049, -0.055470943450927734, 0.9724726676940918, 0.4998619556427002, 0.5487697124481201]}, {"type": "doc", "id": "007", "id_type": "str", "text": "Bond file", "model": "mock-8", "vector": [0.5228255987167358, -0.6460075378417969, 0.875827431678772, 0.7919962406158447, 0.5467689037322998, 0.9042309522628784, -0.22582519054412842, 0.2529853582382202]}]}"#;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("py.ison");
        std::fs::write(&path, graph_file).unwrap();
        std::fs::write(SemanticGraph::embeddings_path_for(&path), sidecar).unwrap();

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.graph().node_count(), 3);
        assert_eq!(loaded.embedding_store().count(), 3);

        // Every restored vector is exactly what this port encodes for the same
        // text (MockEncoder is bit-identical across ports; the f64 -> f32
        // narrowing on import is lossless for f32-origin values).
        let encoder = MockEncoder::new(8);
        for id in [("person", "1"), ("person", "2"), ("doc", "007")] {
            let record = loaded.get_embedding(&NodeId::new(id.0, id.1)).unwrap();
            assert_eq!(record.vector, encoder.encode(&record.text), "{id:?}");
        }
        assert_eq!(
            loaded
                .graph()
                .neighbors(&("person", "1"), Some("KNOWS"), Direction::Out),
            vec![NodeId::new("person", "2")]
        );
    }

    // =========================================================================
    // Graph / vector consistency, model lifecycle, connectivity, weights,
    // filters. Mirrors the Python port's TestSaveLoadConsistency,
    // TestModelLifecycle, TestSubgraphConnectivity, TestEdgeWeighting and
    // TestNodeFiltering.
    // =========================================================================

    const Q: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

    fn named_graph_with(encoder: MockEncoder, names: [&str; 3]) -> SemanticGraph {
        let mut graph = SemanticGraph::new("g", Box::new(encoder));
        for (i, name) in names.iter().enumerate() {
            graph
                .add_node(
                    "person",
                    &(i + 1).to_string(),
                    vec![("name", text_prop(name))],
                    None,
                )
                .unwrap();
        }
        graph
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
    }

    fn named_graph(encoder: MockEncoder) -> SemanticGraph {
        named_graph_with(encoder, ["Alice", "Bob", "Carol"])
    }

    /// Hand-vectored graph for traversal tests (query [1,0,0,0] is the target).
    ///
    /// ```text
    /// person 1 --KNOWS--> company 100 --KNOWS--> person 3
    /// person 2 (isolated)                         person 3 has no vector
    /// ```
    /// company 100 is the single best match for the query, person 1 next.
    fn vector_graph() -> SemanticGraph {
        let mut graph = SemanticGraph::with_options(
            "v",
            Box::new(MockEncoder::new(4)),
            false,
            vec!["name".to_string()],
        );
        for (id, name, active) in [("1", "P1", true), ("2", "P2", false), ("3", "P3", true)] {
            graph
                .add_node(
                    "person",
                    id,
                    vec![("name", text_prop(name)), ("active", Value::Bool(active))],
                    None,
                )
                .unwrap();
        }
        graph
            .add_node("company", "100", vec![("name", text_prop("C"))], None)
            .unwrap();
        graph
            .add_edge("KNOWS", ("person", "1"), ("company", "100"), vec![])
            .unwrap();
        graph
            .add_edge("KNOWS", ("company", "100"), ("person", "3"), vec![])
            .unwrap();
        let store = graph.embedding_store_mut();
        store
            .add(
                NodeId::new("company", "100"),
                "c",
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        store
            .add(
                NodeId::new("person", "1"),
                "p1",
                Some(vec![0.9, 0.1, 0.0, 0.0]),
            )
            .unwrap();
        store
            .add(
                NodeId::new("person", "2"),
                "p2",
                Some(vec![0.8, 0.2, 0.0, 0.0]),
            )
            .unwrap();
        graph
    }

    fn ids(results: &[SimilarityResult]) -> Vec<NodeId> {
        results.iter().map(|r| r.node_id.clone()).collect()
    }

    fn nid(node_type: &str, id: &str) -> NodeId {
        NodeId::new(node_type, id)
    }

    fn is_active(node: &Node) -> bool {
        matches!(node.properties.get("active"), Some(Value::Bool(true)))
    }

    fn read_sidecar(path: &Path) -> ExportedEmbeddings {
        let json = std::fs::read_to_string(SemanticGraph::embeddings_path_for(path)).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    // -- save / load consistency ----------------------------------------------

    #[test]
    fn test_sidecar_records_digest_and_stays_version_1() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        let encoder = MockEncoder::new(8);
        let model = encoder.model_name().to_string();
        named_graph(encoder).save(&path, None, true).unwrap();

        let payload = read_sidecar(&path);
        assert_eq!(payload.format, EMBEDDING_FORMAT);
        assert_eq!(payload.version, 1);
        assert!(payload.graph_digest.unwrap().starts_with("sha256:"));
        assert_eq!(
            payload.graph_digest_basis.as_deref(),
            Some(GRAPH_DIGEST_BASIS)
        );
        assert_eq!(payload.encoder_model, Some(model));

        // Optional metadata trails the array so naive readers (the C++ one
        // scans for keys) meet "format"/"version"/"embeddings" first.
        let raw = std::fs::read_to_string(SemanticGraph::embeddings_path_for(&path)).unwrap();
        assert!(raw.starts_with(
            r#"{"format":"ison-embeddings","version":1,"dimension":8,"count":3,"embeddings":["#
        ));
        let at = |key: &str| raw.find(&format!("\"{key}\":")).unwrap();
        assert!(at("embeddings") < at("encoder_model"));
        assert!(at("encoder_model") < at("graph_digest"));
        assert!(at("graph_digest") < at("graph_digest_basis"));
    }

    #[test]
    fn test_saves_leave_no_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        named_graph(MockEncoder::new(8))
            .save(dir.path().join("g.ison"), None, true)
            .unwrap();
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, vec!["g.ison", "g.ison.embeddings.json"]);
    }

    #[test]
    fn test_failed_write_keeps_the_previous_file_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("keep.txt");
        atomic_write(&target, b"before").unwrap();
        atomic_write(&target, b"after").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "after");

        // A rename onto a directory fails; the target must be untouched and
        // the temp file removed.
        let blocked = dir.path().join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("inside"), "x").unwrap();
        assert!(matches!(
            atomic_write(&blocked, b"nope"),
            Err(EmbeddingError::Io(_))
        ));
        assert!(blocked.join("inside").is_file());
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn test_matching_pair_loads_clean_with_typed_values() {
        // Typed values must not make a healthy save look mismatched.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ison");
        let mut graph = SemanticGraph::with_options(
            "t",
            Box::new(MockEncoder::new(8)),
            false,
            vec!["name".to_string()],
        );
        graph
            .add_node(
                "person",
                "1",
                vec![
                    ("name", text_prop("A b")),
                    ("age", Value::Int(30)),
                    ("ratio", Value::Float(1.0)),
                    ("tiny", Value::Float(0.00001)),
                    ("ok", Value::Bool(true)),
                    ("note", Value::Null),
                    (
                        "accent",
                        text_prop("r\u{e9}sum\u{e9} \u{65e5}\u{672c} \"q\""),
                    ),
                ],
                None,
            )
            .unwrap();
        graph.add_node("account", "007", vec![], None).unwrap();
        graph
            .add_edge(
                "KNOWS",
                ("person", "1"),
                ("account", "007"),
                vec![("w", Value::Float(0.5)), ("n", Value::Int(2))],
            )
            .unwrap();
        graph.embed_node(&nid("person", "1"), "a").unwrap();
        graph.save(&path, None, true).unwrap();

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 1);
        assert!(loaded.embedding_store().warnings().is_empty());
    }

    #[test]
    fn test_isonl_round_trip_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.isonl");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
    }

    fn edit_bob(path: &Path) {
        let text = std::fs::read_to_string(path).unwrap();
        std::fs::write(path, text.replace("Bob", "Robert")).unwrap();
    }

    #[test]
    fn test_sidecar_from_another_save_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (pa, pb) = (dir.path().join("a.ison"), dir.path().join("b.ison"));
        named_graph_with(MockEncoder::new(8), ["Alice", "Bob", "Carol"])
            .save(&pa, None, true)
            .unwrap();
        named_graph_with(MockEncoder::new(8), ["Xavier", "Yolanda", "Zed"])
            .save(&pb, None, true)
            .unwrap();
        // b's vectors dropped next to a's graph
        std::fs::copy(
            SemanticGraph::embeddings_path_for(&pb),
            SemanticGraph::embeddings_path_for(&pa),
        )
        .unwrap();
        match SemanticGraph::load(&pa, Box::new(MockEncoder::new(8))) {
            Err(EmbeddingError::GraphEmbeddingMismatch(message)) => {
                assert!(message.contains("digest"), "{message}")
            }
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("a foreign sidecar loaded"),
        }
    }

    #[test]
    fn test_edited_graph_is_rejected_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        edit_bob(&path);
        assert!(matches!(
            SemanticGraph::load(&path, Box::new(MockEncoder::new(8))),
            Err(EmbeddingError::GraphEmbeddingMismatch(_))
        ));
    }

    #[test]
    fn test_warn_mode_loads_the_old_vectors_anyway() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        edit_bob(&path);
        let options = LoadOptions {
            on_mismatch: GraphMismatch::Warn,
            ..LoadOptions::default()
        };
        let loaded =
            SemanticGraph::load_with(&path, Box::new(MockEncoder::new(8)), &options).unwrap();
        let warnings = loaded.embedding_store().warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("digest"), "{}", warnings[0]);
        // the stale text, from the old sidecar
        assert_eq!(
            loaded.get_embedding(&nid("person", "2")).unwrap().text,
            "Bob"
        );
    }

    #[test]
    fn test_reembed_mode_rebuilds_from_the_edited_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        edit_bob(&path);
        let options = LoadOptions {
            on_mismatch: GraphMismatch::Reembed,
            ..LoadOptions::default()
        };
        let loaded =
            SemanticGraph::load_with(&path, Box::new(MockEncoder::new(8)), &options).unwrap();
        let record = loaded.get_embedding(&nid("person", "2")).unwrap();
        assert_eq!(record.text, "Robert");
        assert_eq!(record.vector, MockEncoder::new(8).encode("Robert"));
        assert_eq!(loaded.embedding_store().count(), 3);
    }

    #[test]
    fn test_sidecar_without_digest_loads_as_before() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        let mut payload = read_sidecar(&path);
        payload.graph_digest = None;
        payload.graph_digest_basis = None;
        payload.encoder_model = None;
        std::fs::write(
            SemanticGraph::embeddings_path_for(&path),
            serde_json::to_string(&payload).unwrap(),
        )
        .unwrap();
        // even an edited graph is accepted: there is nothing to compare with
        edit_bob(&path);
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
        assert!(loaded.embedding_store().warnings().is_empty());
    }

    #[test]
    fn test_digest_of_unknown_basis_is_not_compared() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8))
            .save(&path, None, true)
            .unwrap();
        let mut payload = read_sidecar(&path);
        payload.graph_digest = Some(format!("sha256:{}", "0".repeat(64)));
        payload.graph_digest_basis = Some("some-other-port-v9".to_string());
        std::fs::write(
            SemanticGraph::embeddings_path_for(&path),
            serde_json::to_string(&payload).unwrap(),
        )
        .unwrap();
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
    }

    #[test]
    fn test_legacy_payload_with_unknown_fields_imports() {
        // Version-1 readers must ignore fields they do not know, and keep the
        // ones they do in last_import_meta.
        let mut source = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        source.add(nid("person", "1"), "Alice", None).unwrap();
        let mut json: serde_json::Value =
            serde_json::to_value(source.export_embeddings_with(Some("sha256:abc"))).unwrap();
        json["some_future_field"] = serde_json::json!({"x": 1});
        json["encoder_version"] = serde_json::json!(2); // a number, not text

        let mut target = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        assert_eq!(target.from_json(&json.to_string(), false).unwrap(), 1);
        let meta = target.last_import_meta();
        assert_eq!(meta.graph_digest.as_deref(), Some("sha256:abc"));
        assert_eq!(meta.graph_digest_basis.as_deref(), Some(GRAPH_DIGEST_BASIS));
        assert_eq!(meta.encoder_version.as_deref(), Some("2"));
        assert_eq!(meta.encoder_model.as_deref(), Some("mock-8"));
    }

    #[test]
    fn test_digest_ignores_insertion_order_but_sees_content() {
        let build = |order: [&str; 2]| {
            let mut g = ISONGraph::new("a");
            for id in order {
                g.add_node("t", id, vec![("x", Value::Int(id.parse().unwrap()))])
                    .unwrap();
            }
            g.add_edge("R", ("t", "1"), ("t", "2"), vec![]).unwrap();
            g
        };
        let a = build(["1", "2"]);
        let mut b = build(["2", "1"]);
        assert_eq!(graph_digest(&a), graph_digest(&b));
        b.update_node("t", "1", vec![("x", Value::Int(99))])
            .unwrap();
        assert_ne!(graph_digest(&a), graph_digest(&b));

        // an int id and a str id that print alike are different nodes
        let mut c = ISONGraph::new("c");
        c.add_node("t", "7", vec![]).unwrap();
        let ints: HashSet<NodeId> = [nid("t", "7")].into_iter().collect();
        assert_ne!(graph_digest(&c), graph_digest_with_int_ids(&c, &ints));
        // the graph name does not matter
        let mut d = ISONGraph::new("other name");
        d.add_node("t", "7", vec![]).unwrap();
        assert_eq!(graph_digest(&c), graph_digest(&d));
    }

    #[test]
    fn test_digest_reads_integer_ids_from_the_text() {
        // Python writes an int id unquoted and a str id quoted; both load as
        // the same string here, and the digest has to keep them apart.
        let int_text = "nodes.t\nid\n7\n";
        let str_text = "nodes.t\nid\n\"7\"\n";
        let as_int = graph_digest_of_text(int_text, false).unwrap();
        let as_str = graph_digest_of_text(str_text, false).unwrap();
        assert_ne!(as_int, as_str);
        // and a string-id graph built here digests like the quoted text
        let mut g = ISONGraph::new("g");
        g.add_node("t", "7", vec![]).unwrap();
        assert_eq!(graph_digest(&g), as_str);
    }

    #[test]
    fn test_loads_a_sidecar_whose_digest_names_integer_ids() {
        // A graph file with an unquoted integer id, and a sidecar carrying the
        // digest of the graph as another port reads it. The digest only
        // matches if the integer id is told from a string one.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ints.ison");
        let text = "edges.KNOWS\nsource target\n:person:1 :person:2\n\nnodes.person\nid name\n1 \"Alice\"\n\"2\" \"Bob\"\n";
        std::fs::write(&path, text).unwrap();

        let digest = graph_digest_of_text(text, false).unwrap();
        let parsed = ISONGraph::from_ison(text, Some("ints")).unwrap();
        assert_ne!(
            digest,
            graph_digest(&parsed),
            "the int id must change the digest"
        );

        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        store.add(nid("person", "1"), "Alice", None).unwrap();
        store
            .save_embeddings_with(SemanticGraph::embeddings_path_for(&path), Some(&digest))
            .unwrap();

        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 1);

        // a sidecar recorded for the all-string reading is a mismatch
        store
            .save_embeddings_with(
                SemanticGraph::embeddings_path_for(&path),
                Some(&graph_digest(&parsed)),
            )
            .unwrap();
        assert!(matches!(
            SemanticGraph::load(&path, Box::new(MockEncoder::new(8))),
            Err(EmbeddingError::GraphEmbeddingMismatch(_))
        ));
    }

    #[test]
    fn test_graph_digest_matches_the_shared_golden_fixture() {
        // The fixture is shared by every port; the Python port generated it.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/graph_digest_golden.json"
        );
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("golden fixture")).unwrap();
        assert_eq!(fixture["basis"], GRAPH_DIGEST_BASIS);

        fn value_of(v: &serde_json::Value) -> Value {
            match v {
                serde_json::Value::Null => Value::Null,
                serde_json::Value::Bool(b) => Value::Bool(*b),
                serde_json::Value::Number(n) if n.is_i64() => Value::Int(n.as_i64().unwrap()),
                serde_json::Value::Number(n) => Value::Float(n.as_f64().unwrap()),
                serde_json::Value::String(s) => Value::String(s.clone()),
                other => panic!("unsupported fixture value {other}"),
            }
        }
        fn props_of(v: &serde_json::Value) -> Vec<(String, Value)> {
            v.as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), value_of(v)))
                .collect()
        }
        // (id text, whether the fixture wrote it as an integer)
        fn id_of(v: &serde_json::Value) -> (String, bool) {
            match v {
                serde_json::Value::Number(n) => (n.to_string(), true),
                serde_json::Value::String(s) => (s.clone(), false),
                other => panic!("unsupported id {other}"),
            }
        }

        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 5);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let mut graph = if case["directed"].as_bool().unwrap() {
                ISONGraph::new("golden")
            } else {
                ISONGraph::undirected("golden")
            };
            let mut int_ids = HashSet::new();
            for node in case["nodes"].as_array().unwrap() {
                let node_type = node["type"].as_str().unwrap();
                let (id, is_int) = id_of(&node["id"]);
                if is_int {
                    int_ids.insert(nid(node_type, &id));
                }
                let props = props_of(&node["properties"]);
                let borrowed: Vec<(&str, Value)> =
                    props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
                graph.add_node(node_type, &id, borrowed).unwrap();
            }
            for edge in case["edges"].as_array().unwrap() {
                let (source, target) = (&edge["source"], &edge["target"]);
                let (st, (sid, _)) = (source[0].as_str().unwrap(), id_of(&source[1]));
                let (tt, (tid, _)) = (target[0].as_str().unwrap(), id_of(&target[1]));
                let props = props_of(&edge["properties"]);
                let borrowed: Vec<(&str, Value)> =
                    props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
                graph
                    .add_edge(
                        edge["rel_type"].as_str().unwrap(),
                        (st, &sid),
                        (tt, &tid),
                        borrowed,
                    )
                    .unwrap();
            }
            assert_eq!(
                graph_digest_with_int_ids(&graph, &int_ids),
                case["digest"].as_str().unwrap(),
                "golden case {name:?}"
            );
        }
    }

    // -- model lifecycle -------------------------------------------------------

    fn retagged(dimension: usize, name: &str) -> MockEncoder {
        MockEncoder::new(dimension).with_model_name(name)
    }

    #[test]
    fn test_models_counts_vectors_per_model() {
        let graph = named_graph(MockEncoder::new(8));
        let expected: HashMap<String, usize> = [("mock-8".to_string(), 3)].into_iter().collect();
        assert_eq!(graph.embedding_store().models(), expected);
    }

    #[test]
    fn test_models_tracks_overwrites_and_removals() {
        let mut graph = named_graph(MockEncoder::new(8));
        graph.set_encoder(Box::new(retagged(8, "v2")));
        graph.embed_node(&nid("person", "1"), "again").unwrap();
        let models = graph.embedding_store().models();
        assert_eq!(models.get("mock-8"), Some(&2));
        assert_eq!(models.get("v2"), Some(&1));
        graph.remove_node("person", "2").unwrap();
        assert_eq!(graph.embedding_store().models().get("mock-8"), Some(&1));
        graph.embedding_store_mut().clear();
        assert!(graph.embedding_store().models().is_empty());
    }

    #[test]
    fn test_stale_by_model_name_then_reembed() {
        let mut graph = named_graph(MockEncoder::new(16));
        graph.set_encoder(Box::new(retagged(16, "next-gen-16")));

        let stale = graph.stale_embeddings(false).unwrap();
        assert_eq!(
            stale,
            vec![nid("person", "1"), nid("person", "2"), nid("person", "3")]
        );
        let before: HashMap<String, usize> = [("mock-16".to_string(), 3)].into_iter().collect();
        assert_eq!(graph.embedding_store().models(), before);

        assert_eq!(graph.reembed_stale(false, 256).unwrap(), 3);
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
        let after: HashMap<String, usize> = [("next-gen-16".to_string(), 3)].into_iter().collect();
        assert_eq!(graph.embedding_store().models(), after);
    }

    #[test]
    fn test_nothing_is_stale_under_the_same_encoder() {
        let mut graph = named_graph(MockEncoder::new(8));
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
        assert_eq!(graph.reembed_stale(false, 256).unwrap(), 0);
    }

    #[test]
    fn test_stale_by_dimension_resets_the_store() {
        let mut graph = named_graph(MockEncoder::new(8));
        graph.set_encoder(Box::new(MockEncoder::new(16)));
        assert_eq!(graph.stale_embeddings(false).unwrap().len(), 3);

        assert_eq!(graph.reembed_stale(false, 256).unwrap(), 3);
        assert_eq!(graph.embedding_store().dimension(), Some(16));
        assert_eq!(
            graph
                .get_embedding(&nid("person", "1"))
                .unwrap()
                .vector
                .len(),
            16
        );
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
    }

    #[test]
    fn test_reembed_batches_by_batch_size() {
        let mut graph = named_graph(MockEncoder::new(8));
        graph.set_encoder(Box::new(retagged(8, "v2")));
        assert_eq!(graph.reembed_stale(false, 2).unwrap(), 3);
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
        // a zero batch size cannot stall: it is treated as one
        graph.set_encoder(Box::new(retagged(8, "v3")));
        assert_eq!(graph.reembed_stale(false, 0).unwrap(), 3);
    }

    #[test]
    fn test_reembed_keeps_custom_embed_text() {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(8)),
            false,
            vec!["name".to_string()],
        );
        graph
            .add_node(
                "person",
                "1",
                vec![("name", text_prop("A"))],
                Some("a very specific blurb"),
            )
            .unwrap();
        graph.set_encoder(Box::new(retagged(8, "v2")));
        graph.reembed_stale(false, 256).unwrap();
        let record = graph.get_embedding(&nid("person", "1")).unwrap();
        assert_eq!(record.text, "a very specific blurb");
        assert_eq!(record.model, "v2");
    }

    #[test]
    fn test_reembed_falls_back_to_the_embed_fields_for_a_vector_with_no_text() {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(8)),
            false,
            vec!["name".to_string()],
        );
        graph
            .add_node("person", "1", vec![("name", text_prop("Alice"))], None)
            .unwrap();
        graph
            .embedding_store_mut()
            .add(nid("person", "1"), "", Some(vec![0.5; 8]))
            .unwrap();
        graph.set_encoder(Box::new(retagged(8, "v2")));
        // the vector was stamped mock-8 by the old encoder: stale
        assert_eq!(graph.reembed_stale(false, 256).unwrap(), 1);
        assert_eq!(
            graph.get_embedding(&nid("person", "1")).unwrap().text,
            "Alice"
        );
    }

    #[test]
    fn test_unknown_model_is_not_stale_unless_asked() {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(8)),
            false,
            vec!["name".to_string()],
        );
        graph.add_node("person", "1", vec![], None).unwrap();
        // a vector handed in while the store has no encoder -> "provided"
        let encoder = graph.embedding_store_mut().take_encoder().unwrap();
        graph
            .embedding_store_mut()
            .add(nid("person", "1"), "x", Some(vec![0.1; 8]))
            .unwrap();
        graph.embedding_store_mut().set_encoder(encoder);

        let expected: HashMap<String, usize> = [("provided".to_string(), 1)].into_iter().collect();
        assert_eq!(graph.embedding_store().models(), expected);
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
        assert_eq!(
            graph.stale_embeddings(true).unwrap(),
            vec![nid("person", "1")]
        );
    }

    #[test]
    fn test_imported_vectors_without_a_model_are_unknown() {
        let python_payload = r#"{"format":"ison-embeddings","version":1,"dimension":4,"count":1,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","vector":[1.0,0.0,0.0,0.0]}]}"#;
        let mut graph = SemanticGraph::new("g", Box::new(MockEncoder::new(4)));
        graph
            .embedding_store_mut()
            .from_json(python_payload, false)
            .unwrap();
        assert_eq!(graph.embedding_store().models().get("unknown"), Some(&1));
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
        assert_eq!(graph.stale_embeddings(true).unwrap().len(), 1);
    }

    #[test]
    fn test_the_model_survives_an_export_import_round_trip() {
        let mut source = EmbeddingStore::new(Some(Box::new(retagged(4, "m-x"))));
        source.add(nid("a", "1"), "t", None).unwrap();
        let mut target = EmbeddingStore::new(None);
        target.from_json(&source.to_json().unwrap(), false).unwrap();
        assert_eq!(target.get(&nid("a", "1")).unwrap().model, "m-x");
    }

    #[test]
    fn test_text_search_with_a_different_model_warns_once() {
        let mut graph = named_graph(MockEncoder::new(16));
        graph.set_encoder(Box::new(retagged(16, "other-16")));
        graph.similarity_search("hello", 1, None, -1.0).unwrap();
        let warnings = graph.embedding_store().warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("other-16") && warnings[0].contains("mock-16"));
        graph
            .similarity_search("hello again", 1, None, -1.0)
            .unwrap();
        graph
            .semantic_multi_hop_search("hello", &SearchOptions::default())
            .unwrap();
        assert_eq!(graph.embedding_store().warnings().len(), 1);
    }

    #[test]
    fn test_text_search_with_a_different_model_can_raise() {
        let mut graph = SemanticGraph::new("g", Box::new(MockEncoder::new(16)));
        graph.set_on_model_mismatch(ModelMismatch::Error);
        graph
            .add_node("person", "1", vec![("name", text_prop("Alice"))], None)
            .unwrap();
        graph.set_encoder(Box::new(retagged(16, "other-16")));
        match graph.similarity_search("hello", 1, None, 0.0) {
            Err(EmbeddingError::ModelMismatch(message)) => {
                assert!(message.contains("mock-16"), "{message}")
            }
            other => panic!("expected a model mismatch, got {other:?}"),
        }
        assert!(matches!(
            graph.semantic_multi_hop_search("hello", &SearchOptions::default()),
            Err(EmbeddingError::ModelMismatch(_))
        ));
        // re-embedding is the cure
        graph.reembed_stale(false, 256).unwrap();
        assert!(!graph
            .similarity_search("hello", 1, None, -1.0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_vector_queries_and_ignore_mode_are_never_checked() {
        let mut graph = named_graph(MockEncoder::new(16));
        graph.set_encoder(Box::new(retagged(16, "other-16")));
        let probe = MockEncoder::new(16).encode("x");
        graph
            .similarity_search_with(
                &probe,
                &SimilarityOptions {
                    top_k: Some(1),
                    threshold: -1.0,
                    ..SimilarityOptions::default()
                },
            )
            .unwrap();
        assert!(graph.embedding_store().warnings().is_empty());

        let mut quiet = SemanticGraph::new("g", Box::new(MockEncoder::new(16)));
        quiet.set_on_model_mismatch(ModelMismatch::Ignore);
        quiet
            .add_node("person", "1", vec![("name", text_prop("Alice"))], None)
            .unwrap();
        quiet.set_encoder(Box::new(retagged(16, "other-16")));
        quiet.similarity_search("x", 1, None, -1.0).unwrap();
        assert!(quiet.embedding_store().warnings().is_empty());
    }

    #[test]
    fn test_a_model_that_says_nothing_never_triggers_a_mismatch() {
        // An encoder with no model name ("unknown") makes no claim either way.
        struct Anonymous;
        impl EmbeddingEncoder for Anonymous {
            fn dimension(&self) -> usize {
                4
            }
            fn encode(&self, _: &str) -> Vec<f32> {
                vec![1.0, 0.0, 0.0, 0.0]
            }
        }
        let mut graph = named_graph(MockEncoder::new(4));
        graph.set_on_model_mismatch(ModelMismatch::Error);
        graph.set_encoder(Box::new(Anonymous));
        graph.similarity_search("x", 1, None, -1.0).unwrap();
        // ...and judges nothing stale by model, only by dimension
        assert!(graph.stale_embeddings(false).unwrap().is_empty());
    }

    #[test]
    fn test_version_is_recorded_in_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        let graph = named_graph(MockEncoder::new(8).with_version("2026.1"));
        graph.save(&path, None, true).unwrap();
        assert_eq!(
            read_sidecar(&path).encoder_version.as_deref(),
            Some("2026.1")
        );

        // an explicit version wins over the encoder's
        let mut store =
            EmbeddingStore::new(Some(Box::new(MockEncoder::new(8).with_version("2026.1"))));
        store.set_version(Some("explicit".to_string()));
        store.add(nid("a", "1"), "x", None).unwrap();
        assert_eq!(
            store.export_embeddings().encoder_version.as_deref(),
            Some("explicit")
        );
        // and nothing is written when no version is known
        let plain = named_graph(MockEncoder::new(8));
        let json = plain.embedding_store().to_json().unwrap();
        assert!(!json.contains("encoder_version"));
    }

    #[test]
    fn test_loading_with_a_different_version_warns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.ison");
        named_graph(MockEncoder::new(8).with_version("1"))
            .save(&path, None, true)
            .unwrap();

        let loaded =
            SemanticGraph::load(&path, Box::new(MockEncoder::new(8).with_version("2"))).unwrap();
        let warnings = loaded.embedding_store().warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("version '1'"), "{}", warnings[0]);

        let same =
            SemanticGraph::load(&path, Box::new(MockEncoder::new(8).with_version("1"))).unwrap();
        assert!(same.embedding_store().warnings().is_empty());
        // unversioned: no claim
        let unversioned = SemanticGraph::load(&path, Box::new(MockEncoder::new(8))).unwrap();
        assert!(unversioned.embedding_store().warnings().is_empty());
        // an explicit version in the options is compared too
        let options = LoadOptions {
            version: Some("3".to_string()),
            ..LoadOptions::default()
        };
        let explicit =
            SemanticGraph::load_with(&path, Box::new(MockEncoder::new(8)), &options).unwrap();
        assert_eq!(explicit.embedding_store().warnings().len(), 1);
    }

    #[test]
    fn test_clear_can_keep_the_pinned_dimension() {
        let mut store = EmbeddingStore::new(None);
        store
            .add(nid("a", "1"), "x", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();
        store.clear_with(false);
        assert_eq!(store.count(), 0);
        assert_eq!(store.dimension(), Some(3));
        assert!(matches!(
            store.add(nid("a", "2"), "x", Some(vec![1.0, 0.0])),
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
        store.clear_with(true);
        assert_eq!(store.dimension(), None);
        store.add(nid("a", "2"), "x", Some(vec![1.0, 0.0])).unwrap();
    }

    // -- subgraph connectivity --------------------------------------------------

    /// A -> X -> B (or the reversed shape, per the flags), with embeddings on
    /// A and B only.
    fn bridge(rel_ab: &str, rel_xb: &str, a_to_x: bool, b_to_x: bool) -> SemanticGraph {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        for id in ["A", "B", "X"] {
            let kind = if id == "X" { "hub" } else { "leaf" };
            graph
                .add_node("n", id, vec![("kind", text_prop(kind))], None)
                .unwrap();
        }
        if a_to_x {
            graph
                .add_edge(rel_ab, ("n", "A"), ("n", "X"), vec![])
                .unwrap();
        } else {
            graph
                .add_edge(rel_ab, ("n", "X"), ("n", "A"), vec![])
                .unwrap();
        }
        if b_to_x {
            graph
                .add_edge(rel_xb, ("n", "B"), ("n", "X"), vec![])
                .unwrap();
        } else {
            graph
                .add_edge(rel_xb, ("n", "X"), ("n", "B"), vec![])
                .unwrap();
        }
        let store = graph.embedding_store_mut();
        store.add(nid("n", "A"), "a", Some(vec![1.0, 0.0])).unwrap();
        store.add(nid("n", "B"), "b", Some(vec![0.9, 0.1])).unwrap();
        graph
    }

    fn sub_options<'a>() -> SubgraphOptions<'a> {
        SubgraphOptions {
            search: SearchOptions {
                hops: MultiHopOptions {
                    max_hops: 0,
                    top_k_seeds: 2,
                    top_k_results: 2,
                    threshold: -1.0,
                    ..MultiHopOptions::default()
                },
                ..SearchOptions::default()
            },
            ..SubgraphOptions::default()
        }
    }

    fn sub_of(
        graph: &SemanticGraph,
        options: &SubgraphOptions,
    ) -> Result<SemanticGraph, EmbeddingError> {
        graph.semantic_subgraph_with(&[1.0f32, 0.0], options, Box::new(MockEncoder::new(2)))
    }

    fn refs(graph: &SemanticGraph) -> HashSet<(String, String)> {
        graph
            .graph()
            .nodes()
            .map(|n| (n.node_type.clone(), n.id.clone()))
            .collect()
    }

    fn set(items: &[(&str, &str)]) -> HashSet<(String, String)> {
        items
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn test_default_leaves_the_selection_disconnected() {
        let sub = sub_of(&bridge("R", "R", true, false), &sub_options()).unwrap();
        assert_eq!(refs(&sub), set(&[("n", "A"), ("n", "B")]));
        assert_eq!(sub.graph().edge_count(), 0);
    }

    #[test]
    fn test_connect_adds_the_bridging_node_and_its_edges() {
        let options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        let sub = sub_of(&bridge("R", "R", true, false), &options).unwrap();
        assert_eq!(refs(&sub), set(&[("n", "A"), ("n", "B"), ("n", "X")]));
        assert_eq!(sub.graph().edge_count(), 2);
        // the connector carries its own properties
        let x = sub.graph().get_node("n", "X").unwrap();
        assert_eq!(x.properties.len(), 1);
        assert!(matches!(x.properties.get("kind"), Some(Value::String(s)) if s == "hub"));
    }

    #[test]
    fn test_connector_nodes_may_exceed_top_k_results() {
        let options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        let sub = sub_of(&bridge("R", "R", true, false), &options).unwrap();
        assert_eq!(sub.graph().node_count(), 3);
        assert_eq!(options.search.hops.top_k_results, 2);
    }

    #[test]
    fn test_hop_budget_limits_the_search() {
        let graph = bridge("R", "R", true, false);
        let with_budget = |hops| SubgraphOptions {
            connect: true,
            connect_max_hops: hops,
            ..sub_options()
        };
        assert_eq!(
            refs(&sub_of(&graph, &with_budget(1)).unwrap()),
            set(&[("n", "A"), ("n", "B")])
        );
        assert!(refs(&sub_of(&graph, &with_budget(2)).unwrap()).contains(&("n".into(), "X".into())));
        assert!(matches!(
            sub_of(&graph, &with_budget(0)),
            Err(EmbeddingError::InvalidOption(_))
        ));
    }

    #[test]
    fn test_longer_budget_adds_a_chain_of_connectors() {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        for id in ["A", "M1", "M2", "B"] {
            graph.add_node("n", id, vec![], None).unwrap();
        }
        for (s, t) in [("A", "M1"), ("M1", "M2"), ("M2", "B")] {
            graph.add_edge("R", ("n", s), ("n", t), vec![]).unwrap();
        }
        let store = graph.embedding_store_mut();
        store.add(nid("n", "A"), "a", Some(vec![1.0, 0.0])).unwrap();
        store.add(nid("n", "B"), "b", Some(vec![0.9, 0.1])).unwrap();
        let with_budget = |hops| SubgraphOptions {
            connect: true,
            connect_max_hops: hops,
            ..sub_options()
        };
        assert_eq!(
            sub_of(&graph, &with_budget(2))
                .unwrap()
                .graph()
                .node_count(),
            2
        );
        assert_eq!(
            sub_of(&graph, &with_budget(3))
                .unwrap()
                .graph()
                .node_count(),
            4
        );
    }

    #[test]
    fn test_connect_direction_is_respected() {
        let graph = bridge("R", "R", true, true); // A -> X <- B
        let mut options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        assert_eq!(
            refs(&sub_of(&graph, &options).unwrap()),
            set(&[("n", "A"), ("n", "B")])
        );
        options.search.hops.direction = Direction::Both;
        assert!(refs(&sub_of(&graph, &options).unwrap()).contains(&("n".into(), "X".into())));
    }

    #[test]
    fn test_connect_rel_type_is_respected() {
        let graph = bridge("KNOWS", "LIKES", true, false);
        let mut options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        options.search.hops.rel_type = Some("KNOWS");
        assert_eq!(
            refs(&sub_of(&graph, &options).unwrap()),
            set(&[("n", "A"), ("n", "B")])
        );
        options.search.hops.rel_type = None;
        assert!(refs(&sub_of(&graph, &options).unwrap()).contains(&("n".into(), "X".into())));
    }

    #[test]
    fn test_one_hub_joins_three_components() {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        graph.add_node("n", "HUB", vec![], None).unwrap();
        for (i, id) in ["A", "B", "C"].iter().enumerate() {
            graph.add_node("n", id, vec![], None).unwrap();
            graph
                .add_edge("R", ("n", id), ("n", "HUB"), vec![])
                .unwrap();
            graph
                .embedding_store_mut()
                .add(
                    nid("n", id),
                    id,
                    Some(vec![1.0 - i as f32 * 0.05, 0.1 * i as f32]),
                )
                .unwrap();
        }
        let mut options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        options.search.hops.top_k_seeds = 3;
        options.search.hops.top_k_results = 3;
        options.search.hops.direction = Direction::Both;
        let sub = sub_of(&graph, &options).unwrap();
        assert_eq!(
            refs(&sub),
            set(&[("n", "A"), ("n", "B"), ("n", "C"), ("n", "HUB")])
        );
    }

    #[test]
    fn test_strict_filter_bars_failing_nodes_as_connectors() {
        let graph = bridge("R", "R", true, false);
        let leaf =
            |n: &Node| matches!(n.properties.get("kind"), Some(Value::String(s)) if s == "leaf");
        let mut options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        options.search.filter = NodeFilter::new().predicate(&leaf);
        // lax: X fails the filter but may still bridge
        assert!(refs(&sub_of(&graph, &options).unwrap()).contains(&("n".into(), "X".into())));
        options.search.filter = NodeFilter::new().predicate(&leaf).strict();
        assert_eq!(
            refs(&sub_of(&graph, &options).unwrap()),
            set(&[("n", "A"), ("n", "B")])
        );
    }

    #[test]
    fn test_subgraph_only_induces_edges_of_the_requested_type() {
        let graph = bridge("KNOWS", "LIKES", true, false);
        let mut options = SubgraphOptions {
            connect: true,
            ..sub_options()
        };
        options.search.hops.max_hops = 0;
        let all = sub_of(&graph, &options).unwrap();
        assert_eq!(all.graph().edge_count(), 2);
        options.search.hops.rel_type = Some("KNOWS");
        options.connect = false;
        // A and B are selected; neither KNOWS nor LIKES joins them directly
        assert_eq!(sub_of(&graph, &options).unwrap().graph().edge_count(), 0);
    }

    // -- edge weighting ------------------------------------------------------

    /// S -> P (weight `p_weight`) and S -> Q (weight `q_weight`, or none).
    fn fan(p_weight: Value, q_weight: Option<Value>) -> SemanticGraph {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        for id in ["S", "P", "Q"] {
            graph.add_node("n", id, vec![], None).unwrap();
        }
        graph
            .add_edge("R", ("n", "S"), ("n", "P"), vec![("w", p_weight)])
            .unwrap();
        let q_props = q_weight.map(|w| vec![("w", w)]).unwrap_or_default();
        graph
            .add_edge("R", ("n", "S"), ("n", "Q"), q_props)
            .unwrap();
        graph
            .embedding_store_mut()
            .add(nid("n", "S"), "s", Some(vec![1.0, 0.0]))
            .unwrap();
        graph
    }

    fn weighted_options<'a>(weight_field: Option<&'a str>, max_hops: usize) -> SearchOptions<'a> {
        SearchOptions {
            hops: MultiHopOptions {
                max_hops,
                top_k_seeds: 1,
                threshold: -1.0,
                ..MultiHopOptions::default()
            },
            weight_field,
            ..SearchOptions::default()
        }
    }

    fn scores(graph: &SemanticGraph, weight_field: Option<&str>) -> HashMap<String, f32> {
        graph
            .semantic_multi_hop_search(&[1.0f32, 0.0], &weighted_options(weight_field, 1))
            .unwrap()
            .into_iter()
            .map(|r| (r.node_id.id, r.score))
            .collect()
    }

    fn near(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn test_default_ignores_edge_properties() {
        let scores = scores(&fan(Value::Float(0.1), None), None);
        near(scores["P"], 0.8);
        near(scores["Q"], 0.8);
    }

    #[test]
    fn test_weight_multiplies_the_decay() {
        let scores = scores(&fan(Value::Float(0.5), None), Some("w"));
        near(scores["S"], 1.0);
        near(scores["P"], 0.8 * 0.5);
        near(scores["Q"], 0.8); // missing -> 1.0
    }

    #[test]
    fn test_weights_are_clamped_and_junk_counts_as_one() {
        let p = |w: Value| scores(&fan(w, None), Some("w"))["P"];
        near(p(Value::Float(5.0)), 0.8);
        near(p(Value::Int(5)), 0.8);
        near(p(Value::Float(-3.0)), 0.0);
        near(p(Value::String("heavy".into())), 0.8);
        near(p(Value::Bool(true)), 0.8);
        near(p(Value::Bool(false)), 0.8);
        near(p(Value::Null), 0.8);
        near(p(Value::Float(f64::NAN)), 0.8);
        near(p(Value::Int(0)), 0.0);
    }

    #[test]
    fn test_a_zero_weight_scores_zero_but_stays_reachable() {
        let found = fan(Value::Int(0), None)
            .semantic_multi_hop_search(&[1.0f32, 0.0], &weighted_options(Some("w"), 1))
            .unwrap();
        let p = found.iter().find(|r| r.node_id.id == "P").unwrap();
        assert_eq!(p.score, 0.0);
        assert_eq!(p.hop_count, 1);
    }

    fn triangle() -> SemanticGraph {
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        for id in ["S", "M", "T"] {
            graph.add_node("n", id, vec![], None).unwrap();
        }
        graph
            .add_edge("R", ("n", "S"), ("n", "T"), vec![("w", Value::Float(0.1))])
            .unwrap(); // direct but weak
        graph
            .add_edge("R", ("n", "S"), ("n", "M"), vec![("w", Value::Float(1.0))])
            .unwrap();
        graph
            .add_edge("R", ("n", "M"), ("n", "T"), vec![("w", Value::Float(1.0))])
            .unwrap(); // 2 hops, strong
        graph
            .embedding_store_mut()
            .add(nid("n", "S"), "s", Some(vec![1.0, 0.0]))
            .unwrap();
        graph
    }

    #[test]
    fn test_best_route_can_be_longer_than_the_shortest() {
        let graph = triangle();
        let q = [1.0f32, 0.0];
        let by_id = |results: Vec<SemanticSearchResult>| -> HashMap<String, SemanticSearchResult> {
            results
                .into_iter()
                .map(|r| (r.node_id.id.clone(), r))
                .collect()
        };

        let plain = by_id(
            graph
                .semantic_multi_hop_search(&q, &weighted_options(None, 2))
                .unwrap(),
        );
        assert_eq!(plain["T"].hop_count, 1);
        near(plain["T"].score, 0.8);

        let weighted = by_id(
            graph
                .semantic_multi_hop_search(&q, &weighted_options(Some("w"), 2))
                .unwrap(),
        );
        assert_eq!(weighted["T"].hop_count, 2);
        assert_eq!(
            weighted["T"].path,
            vec![nid("n", "S"), nid("n", "M"), nid("n", "T")]
        );
        near(weighted["T"].score, 0.64);
    }

    #[test]
    fn test_hop_budget_still_applies_to_the_better_route() {
        let found = triangle()
            .semantic_multi_hop_search(&[1.0f32, 0.0], &weighted_options(Some("w"), 1))
            .unwrap();
        let t = found.iter().find(|r| r.node_id.id == "T").unwrap();
        near(t.score, 0.8 * 0.1);
    }

    #[test]
    fn test_subgraph_selection_follows_the_weights() {
        let graph = fan(Value::Float(0.1), None); // P weak, Q unweighted
        let mut options = SubgraphOptions {
            search: weighted_options(Some("w"), 1),
            ..SubgraphOptions::default()
        };
        options.search.hops.top_k_results = 2;
        let sub = sub_of(&graph, &options).unwrap();
        assert_eq!(refs(&sub), set(&[("n", "S"), ("n", "Q")]));
    }

    #[test]
    fn test_unweighted_scores_are_unchanged_by_the_traversal_rewrite() {
        // weight_field None must give exactly decay ** hops (not a running
        // product), as before edge weights existed.
        let mut graph = SemanticGraph::with_options(
            "g",
            Box::new(MockEncoder::new(2)),
            false,
            vec!["name".to_string()],
        );
        for id in ["A", "B", "C", "D"] {
            graph.add_node("n", id, vec![], None).unwrap();
        }
        for (s, t) in [("A", "B"), ("B", "C"), ("C", "D")] {
            graph.add_edge("R", ("n", s), ("n", t), vec![]).unwrap();
        }
        graph
            .embedding_store_mut()
            .add(nid("n", "A"), "a", Some(vec![0.6, 0.8]))
            .unwrap();
        let found = graph
            .semantic_multi_hop_search(
                &[0.6f32, 0.8],
                &SearchOptions {
                    hops: MultiHopOptions {
                        max_hops: 3,
                        top_k_seeds: 1,
                        threshold: -1.0,
                        decay: 0.7,
                        ..MultiHopOptions::default()
                    },
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        let base = found[0].score;
        for r in &found {
            assert_eq!(
                r.score,
                base * 0.7f32.powi(r.hop_count as i32),
                "{:?}",
                r.node_id
            );
        }
        assert_eq!(found.len(), 4);
    }

    // -- node filtering --------------------------------------------------------

    fn search(
        graph: &SemanticGraph,
        filter: NodeFilter,
        top_k: Option<usize>,
        threshold: f32,
    ) -> Vec<SimilarityResult> {
        graph
            .similarity_search_with(
                &Q,
                &SimilarityOptions {
                    top_k,
                    threshold,
                    filter,
                    ..SimilarityOptions::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn test_similarity_search_filters_before_taking_top_k() {
        let graph = vector_graph();
        // the single best match is the company; filtering must look past it
        assert_eq!(
            ids(&search(&graph, NodeFilter::new(), Some(1), 0.0)),
            vec![nid("company", "100")]
        );
        let top = search(&graph, NodeFilter::new().types(["person"]), Some(1), 0.0);
        assert_eq!(ids(&top), vec![nid("person", "1")]);
    }

    #[test]
    fn test_node_filter_and_types_combine_with_and() {
        let graph = vector_graph();
        let active = search(&graph, NodeFilter::new().predicate(&is_active), None, -1.0);
        assert_eq!(ids(&active), vec![nid("person", "1")]);
        let none = search(
            &graph,
            NodeFilter::new().types(["company"]).predicate(&is_active),
            None,
            -1.0,
        );
        assert!(none.is_empty());
    }

    #[test]
    fn test_one_type_and_an_empty_list_matches_nothing() {
        let graph = vector_graph();
        assert_eq!(
            ids(&search(
                &graph,
                NodeFilter::new().types(["company"]),
                None,
                0.0
            )),
            vec![nid("company", "100")]
        );
        assert!(search(
            &graph,
            NodeFilter::new().types(Vec::<String>::new()),
            None,
            0.0
        )
        .is_empty());
    }

    #[test]
    fn test_embeddings_without_a_node_fail_node_filter_only() {
        let mut graph = vector_graph();
        graph
            .embedding_store_mut()
            .add(nid("ghost", "1"), "g", Some(vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let by_type = search(&graph, NodeFilter::new().types(["ghost"]), None, -1.0);
        assert!(ids(&by_type).contains(&nid("ghost", "1")));
        let always = |_: &Node| true;
        let by_predicate = search(&graph, NodeFilter::new().predicate(&always), None, -1.0);
        assert!(!ids(&by_predicate).contains(&nid("ghost", "1")));
    }

    fn filtered_search<'a>(
        top_k_seeds: usize,
        max_hops: usize,
        filter: NodeFilter<'a>,
    ) -> SearchOptions<'a> {
        SearchOptions {
            hops: MultiHopOptions {
                max_hops,
                top_k_seeds,
                threshold: -1.0,
                ..MultiHopOptions::default()
            },
            filter,
            ..SearchOptions::default()
        }
    }

    fn by_id(results: Vec<SemanticSearchResult>) -> HashMap<NodeId, SemanticSearchResult> {
        results
            .into_iter()
            .map(|r| (r.node_id.clone(), r))
            .collect()
    }

    #[test]
    fn test_multi_hop_seeds_and_results_are_filtered() {
        let graph = vector_graph();
        let found = graph
            .semantic_multi_hop_search(
                &Q,
                &filtered_search(1, 2, NodeFilter::new().types(["person"])),
            )
            .unwrap();
        let refs: Vec<NodeId> = found.iter().map(|r| r.node_id.clone()).collect();
        // the best match (company) did not take the seed slot, and is not returned
        assert_eq!(refs[0], nid("person", "1"));
        assert!(!refs.contains(&nid("company", "100")));
    }

    #[test]
    fn test_filtered_nodes_are_traversed_but_not_returned_by_default() {
        let graph = vector_graph();
        let found = by_id(
            graph
                .semantic_multi_hop_search(
                    &Q,
                    &filtered_search(1, 2, NodeFilter::new().types(["person"])),
                )
                .unwrap(),
        );
        let reach = &found[&nid("person", "3")];
        assert_eq!(
            reach.path,
            vec![
                nid("person", "1"),
                nid("company", "100"),
                nid("person", "3")
            ]
        );
        assert_eq!(reach.hop_count, 2); // the company still cost a hop
        near(reach.score, found[&nid("person", "1")].score * 0.8 * 0.8);
    }

    #[test]
    fn test_strict_filter_walls_off_the_traversal() {
        let graph = vector_graph();
        let found = by_id(
            graph
                .semantic_multi_hop_search(
                    &Q,
                    &filtered_search(1, 2, NodeFilter::new().types(["person"]).strict()),
                )
                .unwrap(),
        );
        assert!(!found.contains_key(&nid("person", "3")));
        assert!(!found.contains_key(&nid("company", "100")));
    }

    #[test]
    fn test_blend_path_respects_the_filter_too() {
        let graph = vector_graph();
        let mut options = filtered_search(2, 1, NodeFilter::new().types(["person"]));
        options.hops.blend = 0.5;
        let found = graph.semantic_multi_hop_search(&Q, &options).unwrap();
        assert!(!found.is_empty());
        assert!(found.iter().all(|r| r.node_id.node_type == "person"));
    }

    #[test]
    fn test_semantic_path_requires_a_passing_target() {
        let graph = vector_graph();
        let target = nid("company", "100");
        let options = PathOptions {
            threshold: -1.0,
            filter: NodeFilter::new().types(["person"]),
            ..PathOptions::default()
        };
        assert!(graph
            .semantic_path_with(&Q, &target, &options)
            .unwrap()
            .is_none());
        let plain = PathOptions {
            threshold: -1.0,
            ..PathOptions::default()
        };
        assert!(graph
            .semantic_path_with(&Q, &target, &plain)
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_semantic_path_may_cross_failing_nodes_unless_strict() {
        let graph = vector_graph();
        let target = nid("person", "3");
        let lax = graph
            .semantic_path_with(
                &Q,
                &target,
                &PathOptions {
                    threshold: -1.0,
                    top_k_seeds: 1,
                    filter: NodeFilter::new().types(["person"]),
                    ..PathOptions::default()
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            lax.path,
            vec![
                nid("person", "1"),
                nid("company", "100"),
                nid("person", "3")
            ]
        );
        let strict = graph
            .semantic_path_with(
                &Q,
                &target,
                &PathOptions {
                    threshold: -1.0,
                    top_k_seeds: 1,
                    filter: NodeFilter::new().types(["person"]).strict(),
                    ..PathOptions::default()
                },
            )
            .unwrap();
        assert!(strict.is_none());
    }

    #[test]
    fn test_semantic_path_seeds_come_from_passing_nodes() {
        let graph = vector_graph();
        let people = |n: &Node| n.node_type == "person";
        let result = graph
            .semantic_path_with(
                &Q,
                &nid("person", "3"),
                &PathOptions {
                    threshold: -1.0,
                    top_k_seeds: 1,
                    filter: NodeFilter::new().predicate(&people),
                    ..PathOptions::default()
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.path[0], nid("person", "1"));
    }

    #[test]
    fn test_subgraph_holds_only_passing_nodes() {
        let graph = vector_graph();
        let mut options = SubgraphOptions {
            search: filtered_search(2, 2, NodeFilter::new().types(["person"])),
            ..SubgraphOptions::default()
        };
        options.search.hops.top_k_results = 10;
        let sub = graph
            .semantic_subgraph_with(&Q, &options, Box::new(MockEncoder::new(4)))
            .unwrap();
        assert!(sub.graph().nodes().all(|n| n.node_type == "person"));
        assert!(sub.graph().has_node("person", "3"));
    }

    #[test]
    fn test_defaults_are_unchanged() {
        let graph = vector_graph();
        let plain = graph
            .semantic_multi_hop_search(&Q, &filtered_search(1, 2, NodeFilter::new()))
            .unwrap();
        assert_eq!(plain[0].node_id, nid("company", "100"));
        assert!(plain.iter().any(|r| r.node_id == nid("person", "3")));

        // and the original positional / options-struct entry points agree
        let positional = graph
            .semantic_multi_hop_with(
                "irrelevant text",
                &MultiHopOptions {
                    top_k_seeds: 1,
                    threshold: -1.0,
                    ..MultiHopOptions::default()
                },
            )
            .unwrap();
        assert!(!positional.is_empty());
    }

    #[test]
    fn test_a_text_query_goes_through_the_filter_too() {
        let mut graph = SemanticGraph::new("g", Box::new(MockEncoder::new(16)));
        graph
            .add_node("person", "1", vec![("name", text_prop("Alice"))], None)
            .unwrap();
        graph
            .add_node(
                "company",
                "9",
                vec![("name", text_prop("Alice Corp"))],
                None,
            )
            .unwrap();
        let hits = graph
            .similarity_search_with(
                "Alice",
                &SimilarityOptions {
                    top_k: Some(5),
                    threshold: -1.0,
                    filter: NodeFilter::new().types(["company"]),
                    ..SimilarityOptions::default()
                },
            )
            .unwrap();
        assert_eq!(ids(&hits), vec![nid("company", "9")]);
    }

    // -- graphs saved by the Python port, verbatim -------------------------------

    const PY_PLAIN_GRAPH: &str = "edges.KNOWS\nsource target w\n:person:1 :person:2 0.5\n\nnodes.account\nid name\n\"007\" Bond\n\nnodes.person\nid age name ratio\n1 30 \"Alice \u{e9}\" 0.5\n2 null Bob null";
    const PY_PLAIN_SIDECAR: &str = r#"{"format": "ison-embeddings", "version": 1, "dimension": 4, "count": 3, "embeddings": [{"type": "person", "id": 1, "id_type": "int", "text": "Alice", "model": "mock-4", "vector": [-0.8610728979110718, -0.33433854579925537, 0.8408418893814087, -0.2771846055984497]}, {"type": "person", "id": 2, "id_type": "int", "text": "Bob", "model": "mock-4", "vector": [-0.22434258460998535, 0.785423755645752, -0.014674782752990723, 0.6692934036254883]}, {"type": "account", "id": "007", "id_type": "str", "text": "Bond", "model": "mock-4", "vector": [-0.9531009197235107, 0.9167747497558594, 0.5073114633560181, -0.0420149564743042]}], "encoder_model": "mock-4", "graph_digest": "sha256:b4fbdb21f517cb99dbbea381c2ea15b19e3029be02a33b2c224b1126e9f7d7ba", "graph_digest_basis": "nodes-edges-json-v1"}"#;

    // Python writes floats with repr, so these two are in exponent form, which
    // its own parser reads back as strings.
    const PY_EXP_GRAPH: &str = "edges.KNOWS\nsource target w\n:person:1 :person:2 0.5\n\nnodes.account\nid name\n\"007\" Bond\n\nnodes.person\nid huge name tiny\n1 1.5e+20 \"Alice \u{e9}\" 1e-05\n2 null Bob null";
    const PY_EXP_SIDECAR: &str = r#"{"format": "ison-embeddings", "version": 1, "dimension": 4, "count": 3, "embeddings": [{"type": "person", "id": 1, "id_type": "int", "text": "Alice", "model": "mock-4", "vector": [-0.8610728979110718, -0.33433854579925537, 0.8408418893814087, -0.2771846055984497]}, {"type": "person", "id": 2, "id_type": "int", "text": "Bob", "model": "mock-4", "vector": [-0.22434258460998535, 0.785423755645752, -0.014674782752990723, 0.6692934036254883]}, {"type": "account", "id": "007", "id_type": "str", "text": "Bond", "model": "mock-4", "vector": [-0.9531009197235107, 0.9167747497558594, 0.5073114633560181, -0.0420149564743042]}], "encoder_model": "mock-4", "graph_digest": "sha256:9e43319824a5c174ec4eb0bc85ae90f04c327a7eb38a3d3fcf1133b421560739", "graph_digest_basis": "nodes-edges-json-v1"}"#;

    fn write_pair(dir: &Path, name: &str, graph: &str, sidecar: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, graph).unwrap();
        std::fs::write(SemanticGraph::embeddings_path_for(&path), sidecar).unwrap();
        path
    }

    #[test]
    fn test_a_python_saved_pair_verifies_against_its_digest() {
        // Python's int ids (1, 2) are unquoted and its str id ("007") quoted;
        // the digest only matches if this port tells them apart.
        let dir = tempfile::tempdir().unwrap();
        let path = write_pair(dir.path(), "py.ison", PY_PLAIN_GRAPH, PY_PLAIN_SIDECAR);
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(4))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
        assert!(loaded.embedding_store().warnings().is_empty());
        let meta = loaded.embedding_store().last_import_meta();
        assert_eq!(meta.encoder_model.as_deref(), Some("mock-4"));

        // an edit made afterwards is caught
        let edited = PY_PLAIN_GRAPH.replace("Bob", "Robert");
        let path = write_pair(dir.path(), "edited.ison", &edited, PY_PLAIN_SIDECAR);
        assert!(matches!(
            SemanticGraph::load(&path, Box::new(MockEncoder::new(4))),
            Err(EmbeddingError::GraphEmbeddingMismatch(_))
        ));
    }

    #[test]
    fn test_a_python_saved_pair_with_exponent_floats_verifies_too() {
        // Python reads `1e-05` back as a string and hashes it as one; this
        // port reads a float. The digest is accepted on Python's reading.
        let dir = tempfile::tempdir().unwrap();
        let path = write_pair(dir.path(), "exp.ison", PY_EXP_GRAPH, PY_EXP_SIDECAR);
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(4))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
        assert!(
            loaded.embedding_store().warnings().is_empty(),
            "{:?}",
            loaded.embedding_store().warnings()
        );
        let tiny = loaded.graph().get_node("person", "1").unwrap().properties["tiny"].clone();
        assert!(matches!(tiny, Value::Float(f) if f == 0.00001));
    }

    #[test]
    fn test_a_digest_that_cannot_be_verified_exactly_only_warns() {
        // Ports disagree about exponent-form floats, so for a graph holding
        // one a digest that matches neither reading degrades to a warning
        // rather than a hard, possibly false, error.
        let dir = tempfile::tempdir().unwrap();
        let edited = PY_EXP_GRAPH.replace("Bob", "Robert");
        let path = write_pair(dir.path(), "exp.ison", &edited, PY_EXP_SIDECAR);
        let loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(4))).unwrap();
        assert_eq!(loaded.embedding_store().count(), 3);
        let warnings = loaded.embedding_store().warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("could not be verified exactly"));
    }
}
