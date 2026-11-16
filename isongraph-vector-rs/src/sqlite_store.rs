//! SQLite-backed embedding storage, optionally indexed with sqlite-vec.
//!
//! Enabled by the `sqlite` feature. The schema is identical to the one the
//! Python, C#, TypeScript, JavaScript and C++ ports write, so a database
//! created by any of them can be opened by any other:
//!
//! ```sql
//! CREATE TABLE embeddings (
//!     id TEXT PRIMARY KEY, node_type TEXT, node_id TEXT,
//!     node_id_type TEXT, vector BLOB, text TEXT, model TEXT, created_at TIMESTAMP,
//!     UNIQUE(node_type, node_id));
//! CREATE VIRTUAL TABLE vec_embeddings USING vec0(
//!     node_type text, embedding float[N] distance_metric=cosine);
//! ```
//!
//! Vectors are little-endian float32, four bytes per dimension.
//!
//! A database written by an older release of any port may lack the `model` (or
//! `node_id_type`) column; opening it adds the column, with `'unknown'` as the
//! model of the vectors already there, rather than failing.
//!
//! Beyond storage the store has the same model lifecycle as the in-memory
//! `EmbeddingStore`: every vector is stamped with the encoder's `model_name()`,
//! `models()` reports what produced the stored vectors, `stale()` lists the
//! ones that no longer fit the encoder, and a text query against vectors from
//! another model warns (or errors, per `on_model_mismatch`).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Mutex, Once};

use md5::{Digest, Md5};
use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    atomic_write, is_unknown_model, mismatch_report, EmbeddingEncoder, EmbeddingError, EmbeddingId,
    EmbeddingRecord, ExportedEmbedding, ExportedEmbeddings, ImportMeta, ModelMismatch, Query,
    SimilarityResult, EMBEDDING_FORMAT, EMBEDDING_FORMAT_VERSION, GRAPH_DIGEST_BASIS,
};
use ison_graph_rs::NodeId;

/// Virtual table holding the vec0 index, alongside the `embeddings` table.
pub const VEC_TABLE: &str = "vec_embeddings";

static REGISTER_VEC: Once = Once::new();

/// Register sqlite-vec so every connection opened afterwards has `vec0`.
///
/// sqlite3_auto_extension is process-wide and must run before the connection
/// is opened, so this is done once per process rather than per store.
fn register_sqlite_vec() {
    REGISTER_VEC.call_once(|| unsafe {
        // The signature is spelled out rather than inferred: sqlite3_auto_extension
        // takes an entry point whose arguments are all raw pointers, so an
        // inferred transmute would silently accept the wrong shape.
        type EntryPoint = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int;
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), EntryPoint>(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

fn sql_err(e: rusqlite::Error) -> EmbeddingError {
    EmbeddingError::Storage(e.to_string())
}

/// SQLite-backed embedding storage.
pub struct SqliteEmbeddingStore {
    conn: Connection,
    encoder: Option<Box<dyn EmbeddingEncoder>>,
    dimension: Option<usize>,
    sqlite_vec: bool,
    vec_ready: bool,
    version: Option<String>,
    on_model_mismatch: ModelMismatch,
    /// `{model: count}` over the table, filled on first use and dropped by any
    /// write, so a text query does not group the whole table every time.
    model_cache: Mutex<Option<HashMap<String, usize>>>,
    /// Model pairs already warned about (one warning per pair).
    warned: Mutex<HashSet<String>>,
    /// Every warning this store has issued.
    warnings: Mutex<Vec<String>>,
    last_import_meta: ImportMeta,
}

impl SqliteEmbeddingStore {
    /// Open (or create) a store at `path`. Use ":memory:" for a scratch store.
    ///
    /// `sqlite_vec` indexes vectors in a `vec0` virtual table and answers
    /// top-k searches from it instead of scanning every row. Results are the
    /// same either way - vec0 runs an exact brute-force KNN in C.
    pub fn open<P: AsRef<Path>>(
        path: P,
        encoder: Option<Box<dyn EmbeddingEncoder>>,
        sqlite_vec: bool,
    ) -> Result<Self, EmbeddingError> {
        if sqlite_vec {
            register_sqlite_vec();
        }
        let path = path.as_ref();
        let conn = if path.as_os_str() == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(path)
        }
        .map_err(sql_err)?;

        let mut store = Self {
            conn,
            encoder,
            dimension: None,
            sqlite_vec,
            vec_ready: false,
            version: None,
            on_model_mismatch: ModelMismatch::Warn,
            model_cache: Mutex::new(None),
            warned: Mutex::new(HashSet::new()),
            warnings: Mutex::new(Vec::new()),
            last_import_meta: ImportMeta::default(),
        };
        store.create_schema()?;
        store.adopt_stored_dimension()?;
        if let (true, Some(dim)) = (store.sqlite_vec, store.dimension) {
            store.ensure_vec_table(dim)?;
        }
        Ok(store)
    }

    /// Open an in-memory store.
    pub fn open_in_memory(
        encoder: Option<Box<dyn EmbeddingEncoder>>,
        sqlite_vec: bool,
    ) -> Result<Self, EmbeddingError> {
        Self::open(":memory:", encoder, sqlite_vec)
    }

    fn create_schema(&self) -> Result<(), EmbeddingError> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS embeddings (
                     id TEXT PRIMARY KEY,
                     node_type TEXT NOT NULL,
                     node_id TEXT NOT NULL,
                     node_id_type TEXT NOT NULL DEFAULT 'str',
                     vector BLOB NOT NULL,
                     text TEXT NOT NULL,
                     model TEXT DEFAULT 'all-MiniLM-L6-v2',
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     UNIQUE(node_type, node_id)
                 );
                 CREATE INDEX IF NOT EXISTS idx_embeddings_node
                     ON embeddings(node_type, node_id);",
            )
            .map_err(sql_err)?;

        // CREATE TABLE IF NOT EXISTS does nothing to a table an older release
        // made, so add what it lacks. A vector already there has no recorded
        // model, which is what 'unknown' says; the `model` default of a table
        // created now is left as the other ports write it.
        let columns: HashSet<String> = {
            let mut stmt = self
                .conn
                .prepare("PRAGMA table_info(embeddings)")
                .map_err(sql_err)?;
            let names = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .map_err(sql_err)?;
            names.collect::<Result<_, _>>().map_err(sql_err)?
        };
        if !columns.contains("node_id_type") {
            self.conn
                .execute_batch(
                    "ALTER TABLE embeddings ADD COLUMN node_id_type TEXT NOT NULL DEFAULT 'str'",
                )
                .map_err(sql_err)?;
        }
        if !columns.contains("model") {
            self.conn
                .execute_batch("ALTER TABLE embeddings ADD COLUMN model TEXT DEFAULT 'unknown'")
                .map_err(sql_err)?;
        }
        Ok(())
    }

    fn adopt_stored_dimension(&mut self) -> Result<(), EmbeddingError> {
        let bytes: Option<i64> = self
            .conn
            .query_row("SELECT LENGTH(vector) FROM embeddings LIMIT 1", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(sql_err)?;
        if let Some(n) = bytes {
            if n > 0 {
                self.dimension = Some(n as usize / 4);
            }
        }
        Ok(())
    }

    /// Create the vec0 index once the vector width is known, and backfill it.
    fn ensure_vec_table(&mut self, dimension: usize) -> Result<(), EmbeddingError> {
        if !self.sqlite_vec || self.vec_ready {
            return Ok(());
        }
        // node_type is a vec0 metadata column so a type-filtered search stays
        // exact - filtering after a top-k fetch would silently drop matches.
        self.conn
            .execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(
                     node_type text,
                     embedding float[{dimension}] distance_metric=cosine
                 );"
            ))
            .map_err(sql_err)?;

        let indexed: i64 = self
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM {VEC_TABLE}"), [], |r| {
                r.get(0)
            })
            .map_err(sql_err)?;
        let stored: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .map_err(sql_err)?;

        if indexed != stored {
            // Opening a database that was written without the index, or with
            // it switched off.
            self.conn
                .execute(&format!("DELETE FROM {VEC_TABLE}"), [])
                .map_err(sql_err)?;
            let rows: Vec<(i64, String, Vec<u8>)> = {
                let mut stmt = self
                    .conn
                    .prepare("SELECT rowid, node_type, vector FROM embeddings")
                    .map_err(sql_err)?;
                let mapped = stmt
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                    .map_err(sql_err)?;
                mapped.collect::<Result<_, _>>().map_err(sql_err)?
            };
            for (rowid, node_type, vector) in rows {
                self.conn
                    .execute(
                        &format!(
                            "INSERT INTO {VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)"
                        ),
                        params![rowid, node_type, vector],
                    )
                    .map_err(sql_err)?;
            }
        }

        self.vec_ready = true;
        Ok(())
    }

    /// Whether top-k search is answered from the sqlite-vec index.
    pub fn sqlite_vec(&self) -> bool {
        self.sqlite_vec
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

    /// Every warning this store has issued (model mismatch). Each is also
    /// printed to stderr.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.lock().map(|w| w.clone()).unwrap_or_default()
    }

    pub fn clear_warnings(&self) {
        if let Ok(mut w) = self.warnings.lock() {
            w.clear();
        }
    }

    fn warn(&self, message: String) {
        eprintln!("warning: {message}");
        if let Ok(mut w) = self.warnings.lock() {
            w.push(message);
        }
    }

    /// Metadata of the most recent imported payload (empty before any).
    pub fn last_import_meta(&self) -> &ImportMeta {
        &self.last_import_meta
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

    fn invalidate_models(&self) {
        if let Ok(mut cache) = self.model_cache.lock() {
            *cache = None;
        }
    }

    /// Which models produced the stored vectors: `{model_name: count}`.
    ///
    /// `"unknown"` / `"provided"` appear as-is (vectors from a port that
    /// records no model, or handed in with no encoder attached); a row with no
    /// model at all counts as `"unknown"`. Unlike the in-memory store this can
    /// fail, because it reads the database.
    pub fn models(&self) -> Result<HashMap<String, usize>, EmbeddingError> {
        let mut cache = self
            .model_cache
            .lock()
            .map_err(|_| EmbeddingError::Storage("model cache poisoned".to_string()))?;
        if let Some(counts) = cache.as_ref() {
            return Ok(counts.clone());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT COALESCE(model, 'unknown'), COUNT(*) FROM embeddings GROUP BY 1")
            .map_err(sql_err)?;
        let counts: HashMap<String, usize> = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
            })
            .map_err(sql_err)?
            .collect::<Result<_, _>>()
            .map_err(sql_err)?;
        *cache = Some(counts.clone());
        Ok(counts)
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
        let counts = self.models()?;
        let mut foreign: Vec<&String> = counts
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
    ) -> Result<Vec<NodeId>, EmbeddingError> {
        let model = model.map(String::from).or_else(|| self.current_model());
        let dimension = dimension.or_else(|| self.encoder.as_ref().map(|e| e.dimension()));
        if model.is_none() && dimension.is_none() {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT node_type, node_id, COALESCE(model, 'unknown'), LENGTH(vector)
                 FROM embeddings ORDER BY node_type, node_id",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })
            .map_err(sql_err)?;
        let mut out = Vec::new();
        for row in rows {
            let (node_type, id, row_model, bytes) = row.map_err(sql_err)?;
            let by_model = model.as_ref().is_some_and(|m| {
                row_model != *m && (include_unknown || !is_unknown_model(&row_model))
            });
            let by_dim = dimension.is_some_and(|d| bytes as usize / 4 != d);
            if by_model || by_dim {
                out.push(NodeId::new(&node_type, &id));
            }
        }
        Ok(out)
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

    /// Surrogate primary key, matching what the other ports compute.
    fn embedding_id(node_id: &NodeId) -> String {
        let mut hasher = Md5::new();
        hasher.update(format!("{}:{}", node_id.node_type, node_id.id).as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    fn pack(vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn unpack(bytes: &[u8]) -> Vec<f32> {
        // as_chunks rather than chunks_exact(4): the chunk width is a constant,
        // so this hands back [u8; 4] arrays and from_le_bytes takes them
        // directly, with no per-element indexing and no bounds checks.
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
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

    fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let (mut dot, mut n1, mut n2) = (0.0f32, 0.0f32, 0.0f32);
        for (a, b) in v1.iter().zip(v2.iter()) {
            dot += a * b;
            n1 += a * a;
            n2 += b * b;
        }
        let (n1, n2) = (n1.sqrt(), n2.sqrt());
        if n1 == 0.0 || n2 == 0.0 {
            0.0
        } else {
            dot / (n1 * n2)
        }
    }

    /// Insert or update one row, keeping the vec0 index in step.
    ///
    /// An UPSERT rather than INSERT OR REPLACE: REPLACE assigns a new rowid,
    /// which would orphan the matching vec0 entry and reset created_at. `id`
    /// is a pure function of (node_type, node_id), so a conflict on one
    /// implies a conflict on the other.
    fn write_row(
        &mut self,
        node_id: &NodeId,
        vector: &[f32],
        text: &str,
        model: &str,
        node_id_type: &str,
    ) -> Result<String, EmbeddingError> {
        let emb_id = Self::embedding_id(node_id);
        let blob = Self::pack(vector);
        self.invalidate_models();

        self.conn
            .execute(
                "INSERT INTO embeddings
                 (id, node_type, node_id, node_id_type, vector, text, model)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(node_type, node_id) DO UPDATE SET
                     node_id_type = excluded.node_id_type,
                     vector = excluded.vector,
                     text = excluded.text,
                     model = excluded.model",
                params![
                    emb_id,
                    node_id.node_type,
                    node_id.id,
                    node_id_type,
                    blob,
                    text,
                    model
                ],
            )
            .map_err(sql_err)?;

        if self.sqlite_vec {
            let dim = vector.len();
            self.ensure_vec_table(dim)?;
            let rowid: i64 = self
                .conn
                .query_row(
                    "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?",
                    params![node_id.node_type, node_id.id],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            // vec0 has no upsert; replace the entry at this rowid.
            self.conn
                .execute(
                    &format!("DELETE FROM {VEC_TABLE} WHERE rowid = ?"),
                    params![rowid],
                )
                .map_err(sql_err)?;
            self.conn
                .execute(
                    &format!(
                        "INSERT INTO {VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)"
                    ),
                    params![rowid, node_id.node_type, blob],
                )
                .map_err(sql_err)?;
        }

        Ok(emb_id)
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
        self.write_row(&node_id, &vector, text, &model, "str")
    }

    /// Add many embeddings in one transaction, encoding their texts together.
    pub fn add_batch(&mut self, entries: &[(NodeId, &str)]) -> Result<usize, EmbeddingError> {
        if entries.is_empty() {
            return Ok(0);
        }
        let texts: Vec<&str> = entries.iter().map(|(_, t)| *t).collect();
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
        for v in &vectors {
            self.check_dimension(v, "vector")?;
        }

        let model = self.write_model();
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        for ((node_id, text), vector) in entries.iter().zip(vectors.iter()) {
            if let Err(e) = self.write_row(node_id, vector, text, &model, "str") {
                let _ = self.conn.execute_batch("ROLLBACK");
                self.invalidate_models();
                return Err(e);
            }
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        Ok(entries.len())
    }

    pub fn get(&self, node_id: &NodeId) -> Result<Option<EmbeddingRecord>, EmbeddingError> {
        self.conn
            .query_row(
                "SELECT id, node_type, node_id, vector, text, model FROM embeddings
                 WHERE node_type = ? AND node_id = ?",
                params![node_id.node_type, node_id.id],
                |r| {
                    let blob: Vec<u8> = r.get(3)?;
                    Ok(EmbeddingRecord {
                        id: r.get(0)?,
                        node_id: NodeId::new(&r.get::<_, String>(1)?, &r.get::<_, String>(2)?),
                        vector: Self::unpack(&blob),
                        text: r.get(4)?,
                        model: r
                            .get::<_, Option<String>>(5)?
                            .unwrap_or_else(|| "unknown".to_string()),
                    })
                },
            )
            .optional()
            .map_err(sql_err)
    }

    pub fn remove(&mut self, node_id: &NodeId) -> Result<bool, EmbeddingError> {
        self.invalidate_models();
        if self.sqlite_vec && self.vec_ready {
            let rowid: Option<i64> = self
                .conn
                .query_row(
                    "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?",
                    params![node_id.node_type, node_id.id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(sql_err)?;
            if let Some(rowid) = rowid {
                self.conn
                    .execute(
                        &format!("DELETE FROM {VEC_TABLE} WHERE rowid = ?"),
                        params![rowid],
                    )
                    .map_err(sql_err)?;
            }
        }
        let changed = self
            .conn
            .execute(
                "DELETE FROM embeddings WHERE node_type = ? AND node_id = ?",
                params![node_id.node_type, node_id.id],
            )
            .map_err(sql_err)?;
        Ok(changed > 0)
    }

    pub fn count(&self) -> Result<usize, EmbeddingError> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .map_err(sql_err)?;
        Ok(n as usize)
    }

    /// Remove every embedding and forget the pinned dimension, so the next
    /// vector may have a different length. Use `clear_with(false)` to keep the
    /// store refusing vectors of the old length.
    pub fn clear(&mut self) -> Result<(), EmbeddingError> {
        self.clear_with(true)
    }

    /// Remove every embedding. With `reset_dimension` the pinned vector length
    /// is forgotten too (switching to an encoder of a different dimension), and
    /// the vec0 index, whose width is part of its definition, is dropped to be
    /// rebuilt at the new width; without it the store keeps refusing vectors of
    /// the wrong length even while empty.
    pub fn clear_with(&mut self, reset_dimension: bool) -> Result<(), EmbeddingError> {
        self.invalidate_models();
        self.conn
            .execute("DELETE FROM embeddings", [])
            .map_err(sql_err)?;
        if self.sqlite_vec && self.vec_ready {
            self.conn
                .execute(&format!("DELETE FROM {VEC_TABLE}"), [])
                .map_err(sql_err)?;
        }
        if reset_dimension {
            if self.sqlite_vec {
                self.conn
                    .execute_batch(&format!("DROP TABLE IF EXISTS {VEC_TABLE}"))
                    .map_err(sql_err)?;
                self.vec_ready = false;
            }
            self.dimension = None;
        }
        Ok(())
    }

    /// Score every stored vector against a query. Always an exact scan.
    fn scan(
        &self,
        query_vector: &[f32],
        node_type: Option<&str>,
    ) -> Result<Vec<(NodeId, f32, String)>, EmbeddingError> {
        let mut stmt = self
            .conn
            .prepare("SELECT node_type, node_id, vector, text FROM embeddings")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| {
                let blob: Vec<u8> = r.get(2)?;
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    blob,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(sql_err)?;

        let mut out = Vec::new();
        for row in rows {
            let (ntype, nid, blob, text) = row.map_err(sql_err)?;
            if let Some(want) = node_type {
                if ntype != want {
                    continue;
                }
            }
            let score = Self::cosine_similarity(query_vector, &Self::unpack(&blob));
            out.push((NodeId::new(&ntype, &nid), score, text));
        }
        Ok(out)
    }

    /// Top-k from the vec0 index.
    fn vec_search(
        &mut self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.ensure_vec_table(query_vector.len())?;
        let blob = Self::pack(query_vector);

        let sql = if node_type.is_some() {
            format!(
                "SELECT e.node_type, e.node_id, e.text, v.distance
                 FROM {VEC_TABLE} v JOIN embeddings e ON e.rowid = v.rowid
                 WHERE v.node_type = ? AND v.embedding MATCH ? AND k = ?"
            )
        } else {
            format!(
                "SELECT e.node_type, e.node_id, e.text, v.distance
                 FROM {VEC_TABLE} v JOIN embeddings e ON e.rowid = v.rowid
                 WHERE v.embedding MATCH ? AND k = ?"
            )
        };

        let mut stmt = self.conn.prepare(&sql).map_err(sql_err)?;
        let map = |r: &rusqlite::Row| -> rusqlite::Result<(String, String, String, f64)> {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        };
        let rows: Vec<(String, String, String, f64)> = match node_type {
            Some(t) => stmt
                .query_map(params![t, blob, top_k as i64], map)
                .map_err(sql_err)?
                .collect::<Result<_, _>>()
                .map_err(sql_err)?,
            None => stmt
                .query_map(params![blob, top_k as i64], map)
                .map_err(sql_err)?
                .collect::<Result<_, _>>()
                .map_err(sql_err)?,
        };

        // Rows arrive ordered by distance ascending, i.e. score descending.
        Ok(rows
            .into_iter()
            .filter_map(|(ntype, nid, text, distance)| {
                let score = 1.0 - distance as f32;
                if score >= threshold {
                    Some(SimilarityResult {
                        node_id: NodeId::new(&ntype, &nid),
                        score,
                        text,
                    })
                } else {
                    None
                }
            })
            .collect())
    }

    pub fn similarity_search(
        &mut self,
        query: &str,
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let query_vector = self.resolve_query(Query::Text(query))?;
        self.similarity_search_vector(&query_vector, top_k, node_type, threshold)
    }

    pub fn similarity_search_vector(
        &mut self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.check_query(query_vector)?;

        // A zero vector has no direction for cosine to measure; vec0 returns
        // NaN there, while the scan defines it as 0.0.
        if self.sqlite_vec && top_k > 0 && query_vector.iter().any(|v| *v != 0.0) {
            return self.vec_search(query_vector, top_k, node_type, threshold);
        }

        let mut results: Vec<SimilarityResult> = self
            .scan(query_vector, node_type)?
            .into_iter()
            .filter(|(_, score, _)| *score >= threshold)
            .map(|(node_id, score, text)| SimilarityResult {
                node_id,
                score,
                text,
            })
            .collect();
        // Selecting the top k rather than ordering every candidate: O(n log k)
        // against O(n log n), same rows in the same order.
        let by_score = |a: &SimilarityResult, b: &SimilarityResult| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if results.len() > top_k {
            results.select_nth_unstable_by(top_k - 1, by_score);
            results.truncate(top_k);
        }
        results.sort_by(by_score);
        Ok(results)
    }

    /// Cosine score for every embedded node, keyed by `type:id`.
    ///
    /// Always scans, even with the index on: the index answers top-k queries,
    /// and blending needs a score for every node.
    pub fn score_map(&self, query_vector: &[f32]) -> Result<HashMap<String, f32>, EmbeddingError> {
        self.check_query(query_vector)?;
        Ok(self
            .scan(query_vector, None)?
            .into_iter()
            .map(|(node_id, score, _)| (format!("{}:{}", node_id.node_type, node_id.id), score))
            .collect())
    }

    /// Serialize every embedding into the portable cross-language payload.
    pub fn export_embeddings(&self) -> Result<ExportedEmbeddings, EmbeddingError> {
        self.export_embeddings_with(None)
    }

    /// `export_embeddings`, tying the payload to one graph by recording its
    /// `graph_digest` (see `graph_digest`) and the basis it was taken on.
    ///
    /// Version 1 of the payload is unchanged. The optional metadata trails the
    /// `embeddings` array: `encoder_model` and `encoder_version` when known,
    /// and `graph_digest` / `graph_digest_basis` when a digest is given.
    pub fn export_embeddings_with(
        &self,
        graph_digest: Option<&str>,
    ) -> Result<ExportedEmbeddings, EmbeddingError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT node_type, node_id, node_id_type, vector, text,
                        COALESCE(model, 'unknown')
                 FROM embeddings ORDER BY node_type, node_id",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| {
                let blob: Vec<u8> = r.get(3)?;
                Ok(ExportedEmbedding {
                    node_type: r.get(0)?,
                    id: EmbeddingId::Str(r.get(1)?),
                    id_type: r.get(2)?,
                    text: r.get(4)?,
                    model: Some(r.get(5)?),
                    vector: Self::unpack(&blob),
                })
            })
            .map_err(sql_err)?;

        let embeddings: Vec<ExportedEmbedding> = rows.collect::<Result<_, _>>().map_err(sql_err)?;
        Ok(ExportedEmbeddings {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_FORMAT_VERSION,
            dimension: self.dimension,
            count: embeddings.len(),
            embeddings,
            encoder_model: self.current_model(),
            encoder_version: self.version(),
            graph_digest: graph_digest.map(String::from),
            graph_digest_basis: graph_digest.map(|_| GRAPH_DIGEST_BASIS.to_string()),
        })
    }

    /// Load a payload produced by `export_embeddings` in any language port.
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
            self.clear_with(true)?;
            self.dimension = payload.dimension;
        }

        let mut written = 0;
        for item in &payload.embeddings {
            let node_id = NodeId::new(&item.node_type, item.id.as_string());
            self.check_dimension(&item.vector, "vector")?;
            let model = item.model.as_deref().unwrap_or("unknown");
            self.write_row(&node_id, &item.vector, &item.text, model, &item.id_type)?;
            written += 1;
        }
        Ok(written)
    }

    /// Serialize the portable payload to a JSON string.
    pub fn to_json(&self) -> Result<String, EmbeddingError> {
        serde_json::to_string(&self.export_embeddings()?)
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
        let payload = self.export_embeddings_with(graph_digest)?;
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
