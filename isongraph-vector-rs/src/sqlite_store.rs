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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;

use md5::{Digest, Md5};
use rusqlite::{params, Connection, OptionalExtension};

use crate::{
    EmbeddingEncoder, EmbeddingError, EmbeddingId, EmbeddingRecord, ExportedEmbedding,
    ExportedEmbeddings, SimilarityResult, EMBEDDING_FORMAT, EMBEDDING_FORMAT_VERSION,
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
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
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
            .map_err(sql_err)
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
            .query_row(&format!("SELECT COUNT(*) FROM {VEC_TABLE}"), [], |r| r.get(0))
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
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
                    &format!("INSERT INTO {VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)"),
                    params![rowid, node_id.node_type, blob],
                )
                .map_err(sql_err)?;
        }

        Ok(emb_id)
    }

    fn model_name(&self) -> String {
        if self.encoder.is_some() {
            "unknown".to_string()
        } else {
            "provided".to_string()
        }
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
        let model = self.model_name();
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

        let model = self.model_name();
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        for ((node_id, text), vector) in entries.iter().zip(vectors.iter()) {
            if let Err(e) = self.write_row(node_id, vector, text, &model, "str") {
                let _ = self.conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        Ok(entries.len())
    }

    pub fn get(&self, node_id: &NodeId) -> Result<Option<EmbeddingRecord>, EmbeddingError> {
        self.conn
            .query_row(
                "SELECT id, node_type, node_id, vector, text FROM embeddings
                 WHERE node_type = ? AND node_id = ?",
                params![node_id.node_type, node_id.id],
                |r| {
                    let blob: Vec<u8> = r.get(3)?;
                    Ok(EmbeddingRecord {
                        id: r.get(0)?,
                        node_id: NodeId::new(&r.get::<_, String>(1)?, &r.get::<_, String>(2)?),
                        vector: Self::unpack(&blob),
                        text: r.get(4)?,
                    })
                },
            )
            .optional()
            .map_err(sql_err)
    }

    pub fn remove(&mut self, node_id: &NodeId) -> Result<bool, EmbeddingError> {
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

    pub fn clear(&mut self) -> Result<(), EmbeddingError> {
        self.conn
            .execute("DELETE FROM embeddings", [])
            .map_err(sql_err)?;
        if self.sqlite_vec && self.vec_ready {
            self.conn
                .execute(&format!("DELETE FROM {VEC_TABLE}"), [])
                .map_err(sql_err)?;
        }
        self.dimension = None;
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
        let query_vector = match &self.encoder {
            Some(enc) => enc.encode(query),
            None => return Err(EmbeddingError::NoEncoder),
        };
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
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
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
        let mut stmt = self
            .conn
            .prepare(
                "SELECT node_type, node_id, node_id_type, vector, text
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
                    vector: Self::unpack(&blob),
                })
            })
            .map_err(sql_err)?;

        let embeddings: Vec<ExportedEmbedding> =
            rows.collect::<Result<_, _>>().map_err(sql_err)?;
        Ok(ExportedEmbeddings {
            format: EMBEDDING_FORMAT.to_string(),
            version: EMBEDDING_FORMAT_VERSION,
            dimension: self.dimension,
            count: embeddings.len(),
            embeddings,
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
        if replace {
            self.clear()?;
        }

        let model = "unknown".to_string();
        let mut written = 0;
        for item in &payload.embeddings {
            let node_id = NodeId::new(&item.node_type, &item.id.as_string());
            self.check_dimension(&item.vector, "vector")?;
            self.write_row(&node_id, &item.vector, &item.text, &model, &item.id_type)?;
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
}
