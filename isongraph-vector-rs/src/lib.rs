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

use std::collections::{HashMap, HashSet, VecDeque};
use ison_graph_rs::{ISONGraph, Node, NodeId, Direction, Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Property values are ISON values, re-exported so callers do not need a
/// direct dependency on `ison-graph` to build a property list.
pub use ison_graph_rs::Value as PropertyValue;

/// SQLite-backed storage, behind the `sqlite` feature.
#[cfg(feature = "sqlite")]
pub mod sqlite_store;
#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteEmbeddingStore;

pub const VERSION: &str = "1.0.0";

/// Portable embedding payload shared by every language port, so a store built
/// in one can be searched by another.
pub const EMBEDDING_FORMAT: &str = "ison-embeddings";
pub const EMBEDDING_FORMAT_VERSION: u32 = 1;

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
    DimensionMismatch { what: String, got: usize, expected: usize },
    #[error("{0} is empty")]
    EmptyVector(String),
    #[error("blend must be in [0, 1], got {0}")]
    InvalidBlend(f32),
    #[error("embedding payload error: {0}")]
    Payload(String),
    /// The SQLite-backed store failed at the database layer.
    #[error("storage error: {0}")]
    Storage(String),
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
}

impl MockEncoder {
    const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;

    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        Self { dimension }
    }
}

impl EmbeddingEncoder for MockEncoder {
    fn dimension(&self) -> usize {
        self.dimension
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
}

// =============================================================================
// Embedding Store
// =============================================================================

pub struct EmbeddingStore {
    embeddings: HashMap<String, EmbeddingRecord>,
    encoder: Option<Box<dyn EmbeddingEncoder>>,
    dimension: Option<usize>,
}

impl EmbeddingStore {
    pub fn new(encoder: Option<Box<dyn EmbeddingEncoder>>) -> Self {
        Self {
            embeddings: HashMap::new(),
            encoder,
            dimension: None,
        }
    }

    /// Vector length this store holds, or `None` while it is still empty.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    pub fn set_encoder(&mut self, encoder: Box<dyn EmbeddingEncoder>) {
        self.encoder = Some(encoder);
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

    fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for (a, b) in v1.iter().zip(v2.iter()) {
            dot += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        norm1 = norm1.sqrt();
        norm2 = norm2.sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
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

        let key = Self::node_id_to_key(&node_id);
        self.embeddings.insert(
            key.clone(),
            EmbeddingRecord {
                id: key.clone(),
                node_id,
                vector,
                text: text.to_string(),
            },
        );

        Ok(key)
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
        for ((node_id, text), vector) in entries.iter().zip(vectors) {
            self.add(node_id.clone(), text, Some(vector))?;
        }
        Ok(entries.len())
    }

    pub fn get(&self, node_id: &NodeId) -> Option<&EmbeddingRecord> {
        let key = Self::node_id_to_key(node_id);
        self.embeddings.get(&key)
    }

    pub fn remove(&mut self, node_id: &NodeId) -> bool {
        let key = Self::node_id_to_key(node_id);
        self.embeddings.remove(&key).is_some()
    }

    pub fn similarity_search(
        &self,
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
        &self,
        query_vector: &[f32],
        top_k: usize,
        node_type: Option<&str>,
        threshold: f32,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        self.check_query(query_vector)?;

        let mut results: Vec<SimilarityResult> = self
            .embeddings
            .values()
            .filter(|record| node_type.map_or(true, |t| record.node_id.node_type == t))
            .filter_map(|record| {
                let score = Self::cosine_similarity(query_vector, &record.vector);
                if score >= threshold {
                    Some(SimilarityResult {
                        node_id: record.node_id.clone(),
                        score,
                        text: record.text.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Cosine score for every embedded node, keyed by `type:id`.
    ///
    /// One scan serving both seed selection and the per-node relevance that
    /// `semantic_multi_hop` blends in, instead of two.
    pub fn score_map(&self, query_vector: &[f32]) -> Result<HashMap<String, f32>, EmbeddingError> {
        self.check_query(query_vector)?;
        Ok(self
            .embeddings
            .iter()
            .map(|(key, record)| (key.clone(), Self::cosine_similarity(query_vector, &record.vector)))
            .collect())
    }

    pub fn count(&self) -> usize {
        self.embeddings.len()
    }

    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.dimension = None;
    }

    /// Serialize every embedding into the portable cross-language payload.
    pub fn export_embeddings(&self) -> ExportedEmbeddings {
        let mut embeddings: Vec<ExportedEmbedding> = self
            .embeddings
            .values()
            .map(|record| ExportedEmbedding {
                node_type: record.node_id.node_type.clone(),
                id: EmbeddingId::Str(record.node_id.id.clone()),
                // Rust node ids are always strings; the other ports use this
                // to tell a numeric id from a numeric-looking string one.
                id_type: "str".to_string(),
                text: record.text.clone(),
                vector: record.vector.clone(),
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
        }
    }

    /// Load a payload produced by `export_embeddings` (in any language port).
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
            self.clear();
        }

        let mut written = 0;
        for item in &payload.embeddings {
            let node_id = NodeId::new(&item.node_type, &item.id.as_string());
            self.add(node_id, &item.text, Some(item.vector.clone()))?;
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

// =============================================================================
// Semantic Graph
// =============================================================================

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

    pub fn graph_mut(&mut self) -> &mut ISONGraph {
        &mut self.graph
    }

    pub fn embedding_store(&self) -> &EmbeddingStore {
        &self.embedding_store
    }

    pub fn embedding_store_mut(&mut self) -> &mut EmbeddingStore {
        &mut self.embedding_store
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

    pub fn get_embedding(&self, node_id: &NodeId) -> Option<&EmbeddingRecord> {
        self.embedding_store.get(node_id)
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
        let record = match self.embedding_store.get(node_id) {
            Some(record) => record,
            None => return Ok(vec![]),
        };
        let mut results = self.embedding_store.similarity_search_vector(
            &record.vector,
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
        if !(0.0..=1.0).contains(&options.blend) {
            return Err(EmbeddingError::InvalidBlend(options.blend));
        }

        let query_vector = match &self.embedding_store.encoder {
            Some(enc) => enc.encode(query),
            None => return Err(EmbeddingError::NoEncoder),
        };

        // One scan serves both seed selection and per-node relevance.
        let scores = self.embedding_store.score_map(&query_vector)?;
        if scores.is_empty() {
            return Ok(vec![]);
        }

        let seeds = self.embedding_store.similarity_search_vector(
            &query_vector,
            options.top_k_seeds,
            None,
            options.threshold,
        )?;
        if seeds.is_empty() {
            return Ok(vec![]);
        }

        let combined = |base: f32, key: &str| -> f32 {
            if options.blend <= 0.0 {
                base
            } else {
                (1.0 - options.blend) * base
                    + options.blend * scores.get(key).copied().unwrap_or(0.0)
            }
        };

        let mut results: HashMap<String, SemanticSearchResult> = HashMap::new();

        for seed in &seeds {
            let seed_key = format!("{}:{}", seed.node_id.node_type, seed.node_id.id);

            // Cosine similarity is signed, and `base * decay ** hops` only
            // decays a non-negative base: multiplying a negative seed score by
            // 0.8 moves it toward zero, so everything it reached outranked the
            // seed itself and relevance grew with distance.
            let traversal_base = seed.score.max(0.0);

            // A seed is a hop-0 match on its own merit. An earlier traversal
            // may already have recorded it with a decayed score; that must not
            // win over the higher, direct similarity of the node. (Only the
            // neighbour branch below used to compare scores, so a direct match
            // reachable from a better seed came back at seed.score * decay, at
            // hop 1, down a path it never needed.)
            let replace_seed = results
                .get(&seed_key)
                .map_or(true, |existing| existing.score < seed.score);
            if replace_seed {
                results.insert(
                    seed_key.clone(),
                    SemanticSearchResult {
                        node_id: seed.node_id.clone(),
                        score: seed.score,
                        hop_count: 0,
                        path: vec![seed.node_id.clone()],
                    },
                );
            }

            // BFS traversal
            let mut queue: VecDeque<(NodeId, Vec<NodeId>, usize)> = VecDeque::new();
            queue.push_back((seed.node_id.clone(), vec![seed.node_id.clone()], 0));

            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(seed_key);

            while let Some((node_id, path, hop)) = queue.pop_front() {
                if hop >= options.max_hops {
                    continue;
                }

                let node_ref = (&node_id.node_type as &str, &node_id.id as &str);
                let neighbors = self.graph.neighbors(&node_ref, options.rel_type, options.direction);

                for neighbor in neighbors {
                    let neighbor_key = format!("{}:{}", neighbor.node_type, neighbor.id);

                    if visited.contains(&neighbor_key) {
                        continue;
                    }
                    visited.insert(neighbor_key.clone());

                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    let new_hop = hop + 1;
                    let new_score = combined(
                        traversal_base * options.decay.powi(new_hop as i32),
                        &neighbor_key,
                    );

                    let should_update = results
                        .get(&neighbor_key)
                        .map_or(true, |existing| existing.score < new_score);

                    if should_update {
                        results.insert(
                            neighbor_key,
                            SemanticSearchResult {
                                node_id: neighbor.clone(),
                                score: new_score,
                                hop_count: new_hop,
                                path: new_path.clone(),
                            },
                        );
                    }

                    queue.push_back((neighbor, new_path, new_hop));
                }
            }
        }

        let mut sorted: Vec<SemanticSearchResult> = results.into_values().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(options.top_k_results);

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

    /// Find a semantically-rooted path to a target node.
    ///
    /// Starts from the most similar nodes and returns the first path found to
    /// the target, scored by the similarity of the seed decayed per hop.
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
        let seeds = self
            .embedding_store
            .similarity_search(query, top_k_seeds, None, threshold)?;

        for seed in seeds {
            let start = (&seed.node_id.node_type as &str, &seed.node_id.id as &str);
            let end = (&target.node_type as &str, &target.id as &str);
            if let Some(path) = self.graph.shortest_path(&start, &end, rel_type, max_hops, direction) {
                let hops = path.length();
                return Ok(Some(SemanticSearchResult {
                    node_id: target.clone(),
                    score: seed.score * decay.powi(hops as i32),
                    hop_count: hops,
                    path: path.nodes,
                }));
            }
        }

        Ok(None)
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
        let results = self.semantic_multi_hop_with(query, options)?;
        let keep: HashSet<String> = results
            .iter()
            .map(|r| format!("{}:{}", r.node_id.node_type, r.node_id.id))
            .collect();

        let mut sub = SemanticGraph::with_options(
            name,
            encoder,
            false,
            self.embed_fields.clone(),
        );

        for result in &results {
            let node = match self.graph.get_node(&result.node_id.node_type, &result.node_id.id) {
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

            if let Some(record) = self.embedding_store.get(&result.node_id) {
                sub.embedding_store.add(
                    result.node_id.clone(),
                    &record.text,
                    Some(record.vector.clone()),
                )?;
            }
        }

        for rel_type in self.graph.edge_types().into_iter().cloned().collect::<Vec<String>>() {
            for edge in self.graph.edges_of_type(&rel_type) {
                let source_key = format!("{}:{}", edge.source.node_type, edge.source.id);
                let target_key = format!("{}:{}", edge.target.node_type, edge.target.id);
                if keep.contains(&source_key) && keep.contains(&target_key) {
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
    /// store written by one be searched by another.
    const GOLDEN_HELLO: [f32; 4] = [0.8381705284, -0.8255031109, 0.5899617672, 0.4615051746];
    const GOLDEN_ISON: [f32; 4] = [-0.3658730984, -0.2765417099, 0.2944871187, -0.6502785683];
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
            .graph_mut()
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        // person:1 matches the query perfectly, person:2 slightly less well.
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "1"), "a", Some(vec![1.0, 0.0, 0.0, 0.0]))
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
        for (text, want) in [("hello", GOLDEN_HELLO), ("ison", GOLDEN_ISON), ("", GOLDEN_EMPTY)] {
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
            let bits: Vec<u32> = encoder.encode(&format!("text{i}")).iter().map(|v| v.to_bits()).collect();
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
        store.add(node_id.clone(), "Alice is an engineer", None).unwrap();

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

        store.add(NodeId::new("person", "1"), "software engineer", None).unwrap();
        store.add(NodeId::new("person", "2"), "data scientist", None).unwrap();

        // Use negative threshold to get all results with mock encoder
        let results = store.similarity_search("software engineer", 2, None, -1.0).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch_is_rejected() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        store.add(NodeId::new("a", "1"), "full width", None).unwrap();

        let err = store.add(NodeId::new("a", "2"), "too short", Some(vec![1.0, 0.0]));
        assert!(matches!(err, Err(EmbeddingError::DimensionMismatch { .. })));

        let err = store.add(NodeId::new("a", "3"), "empty", Some(vec![]));
        assert!(matches!(err, Err(EmbeddingError::EmptyVector(_))));

        let err = store.similarity_search_vector(&[1.0, 0.0], 1, None, -1.0);
        assert!(matches!(err, Err(EmbeddingError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_dimension_pinned_by_first_vector() {
        let mut store = EmbeddingStore::new(None);
        assert_eq!(store.dimension(), None);
        store.add(NodeId::new("a", "1"), "x", Some(vec![1.0, 0.0, 0.0])).unwrap();
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

        graph.add_node("person", "1", vec![("name", Value::String("Alice".to_string()))], Some("Alice is an engineer")).unwrap();

        assert_eq!(graph.graph().node_count(), 1);
        assert_eq!(graph.embedding_store().count(), 1);
    }

    #[test]
    fn test_add_node_auto_embeds_from_properties() {
        let mut graph = SemanticGraph::new("test", Box::new(MockEncoder::new(16)));
        graph.add_node("person", "1", vec![("name", Value::String("Alice".to_string())), ("description", Value::String("engineer".to_string()))], None).unwrap();

        assert_eq!(graph.embedding_store().count(), 1);
        let record = graph.get_embedding(&NodeId::new("person", "1")).unwrap();
        assert_eq!(record.text, "Alice engineer");
    }

    #[test]
    fn test_semantic_graph_similarity() {
        let encoder = MockEncoder::new(384);
        let mut graph = SemanticGraph::new("test", Box::new(encoder));

        graph.add_node("person", "1", vec![], Some("software engineer")).unwrap();
        graph.add_node("person", "2", vec![], Some("data scientist")).unwrap();

        // Use negative threshold to get all results with mock encoder
        let results = graph.similarity_search("software engineer", 5, None, -1.0).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_multi_hop() {
        let encoder = MockEncoder::new(384);
        let mut graph = SemanticGraph::new("test", Box::new(encoder));

        graph.add_node("person", "1", vec![], Some("software engineer")).unwrap();
        graph.add_node("person", "2", vec![], Some("manager")).unwrap();

        graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![]).unwrap();

        let results = graph
            .semantic_multi_hop("software", Some("KNOWS"), 2, 5, 10, Direction::Out, 0.8, -1.0)
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
        graph.embedding_store_mut().set_encoder(Box::new(FixedEncoder));

        let results = graph
            .semantic_multi_hop("query", Some("KNOWS"), 2, 5, 10, Direction::Out, 0.8, 0.3)
            .unwrap();

        let second = results.iter().find(|r| r.node_id.id == "2").unwrap();
        assert!((second.score - 0.9).abs() < 1e-6, "score was {}", second.score);
        assert_eq!(second.hop_count, 0);
        assert_eq!(second.path, vec![NodeId::new("person", "2")]);
    }

    #[test]
    fn test_traversal_still_reaches_non_seeds() {
        let mut graph = seeded_graph();
        graph.embedding_store_mut().set_encoder(Box::new(FixedEncoder));
        graph.add_node("person", "3", vec![], None).unwrap();
        graph
            .graph_mut()
            .add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![])
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "3"), "c", Some(vec![0.0, 0.0, 1.0, 0.0]))
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
        let mut graph = SemanticGraph::with_options(
            "neg",
            Box::new(FixedEncoder),
            false,
            vec![],
        );
        graph.add_node("person", "1", vec![], None).unwrap();
        graph.add_node("person", "2", vec![], None).unwrap();
        graph
            .graph_mut()
            .add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "1"), "a", Some(vec![-1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "2"), "b", Some(vec![0.0, 0.0, 1.0, 0.0]))
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
        graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![]).unwrap();
        graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "3"), vec![]).unwrap();
        graph.embedding_store_mut().add(NodeId::new("person", "1"), "seed", Some(vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        graph.embedding_store_mut().add(NodeId::new("person", "2"), "off", Some(vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        graph.embedding_store_mut().add(NodeId::new("person", "3"), "on", Some(vec![0.8, 0.6, 0.0, 0.0])).unwrap();

        let base_options = MultiHopOptions {
            rel_type: Some("KNOWS"),
            max_hops: 1,
            top_k_seeds: 1,
            threshold: 0.5,
            ..Default::default()
        };

        let plain = graph.semantic_multi_hop_with("query", &base_options).unwrap();
        let plain2 = plain.iter().find(|r| r.node_id.id == "2").unwrap().score;
        let plain3 = plain.iter().find(|r| r.node_id.id == "3").unwrap().score;
        assert!((plain2 - plain3).abs() < 1e-6);

        let blended = graph
            .semantic_multi_hop_with(
                "query",
                &MultiHopOptions { blend: 0.5, ..base_options },
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
            &MultiHopOptions { blend: 1.5, ..Default::default() },
        );
        assert!(matches!(result, Err(EmbeddingError::InvalidBlend(_))));
    }

    #[test]
    fn test_semantic_path_finds_a_route() {
        let mut graph = SemanticGraph::with_options("paths", Box::new(FixedEncoder), false, vec![]);
        for id in ["1", "2", "3"] {
            graph.add_node("person", id, vec![], None).unwrap();
        }
        graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![]).unwrap();
        graph.graph_mut().add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![]).unwrap();
        // Distinct vectors, and a threshold that admits only person:1, so the
        // seed is deterministic. Giving every node the same vector leaves the
        // seed order to HashMap iteration, and a run that seeds on the target
        // finds the zero-hop path from the target to itself.
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "1"), "a", Some(vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "2"), "b", Some(vec![0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "3"), "c", Some(vec![0.0, 0.0, 1.0, 0.0]))
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
        assert!((result.score - 0.81).abs() < 1e-5, "score was {}", result.score);
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
            .add(NodeId::new("person", "1"), "a", Some(vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        graph
            .embedding_store_mut()
            .add(NodeId::new("person", "2"), "b", Some(vec![0.0, 1.0, 0.0, 0.0]))
            .unwrap();

        let result = graph
            .semantic_path("query", &NodeId::new("person", "2"), Some("KNOWS"), 5, 3, Direction::Out, 0.9, 0.5)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_similar_to_node_excludes_itself() {
        let mut graph = SemanticGraph::new("similar", Box::new(MockEncoder::new(16)));
        graph.add_node("person", "1", vec![("name", Value::String("Alice".to_string()))], None).unwrap();
        graph.add_node("person", "2", vec![("name", Value::String("Bob".to_string()))], None).unwrap();
        graph.add_node("person", "3", vec![("name", Value::String("Carol".to_string()))], None).unwrap();

        let results = graph
            .similar_to_node(&NodeId::new("person", "1"), 2, None, -1.0)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.node_id.id != "1"));
    }

    #[test]
    fn test_similar_to_node_without_embedding() {
        let mut graph = SemanticGraph::with_options("similar", Box::new(MockEncoder::new(16)), false, vec![]);
        graph.add_node("person", "9", vec![], None).unwrap();
        let results = graph.similar_to_node(&NodeId::new("person", "9"), 5, None, -1.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_semantic_subgraph_is_a_working_graph() {
        let mut graph = SemanticGraph::new("full", Box::new(MockEncoder::new(16)));
        graph.add_node("person", "1", vec![("name", Value::String("Alice".to_string()))], None).unwrap();
        graph.add_node("person", "2", vec![("name", Value::String("Bob".to_string()))], None).unwrap();
        graph.add_node("person", "3", vec![("name", Value::String("Carol".to_string()))], None).unwrap();
        graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![]).unwrap();
        graph.graph_mut().add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![]).unwrap();

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
            graph.add_node("doc", &i.to_string(), vec![("name", Value::String("Document".to_string()))], None).unwrap();
        }
        let count = graph.embed_all_nodes(None::<fn(&Node) -> String>).unwrap();
        assert_eq!(count, 10);
        assert_eq!(graph.embedding_store().count(), 10);
    }

    #[test]
    fn test_embeddings_round_trip_through_json() {
        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        store.add(NodeId::new("person", "1"), "Alice", None).unwrap();
        store.add(NodeId::new("account", "007"), "Bond", None).unwrap();

        let json = store.to_json().unwrap();
        assert!(json.contains(EMBEDDING_FORMAT));

        let mut other = EmbeddingStore::new(Some(Box::new(MockEncoder::new(8))));
        assert_eq!(other.from_json(&json, false).unwrap(), 2);
        assert_eq!(other.get(&NodeId::new("person", "1")).unwrap().text, "Alice");
        // a numeric-looking string id keeps its exact form
        assert_eq!(other.get(&NodeId::new("account", "007")).unwrap().text, "Bond");
    }

    #[test]
    fn test_imports_a_payload_written_by_the_python_port() {
        // Captured verbatim from the Python port. Its node ids are typed, so
        // an int id arrives unquoted ("id":1) where this port always writes a
        // string - both have to land on the same NodeId here.
        let python_payload = r#"{"format":"ison-embeddings","version":1,"dimension":4,"count":2,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","model":"mock-4","vector":[-0.8610728979110718,-0.33433854579925537,0.8408418893814087,-0.2771846055984497]},{"type":"account","id":"007","id_type":"str","text":"Bond","model":"mock-4","vector":[-0.9531009197235107,0.9167747497558594,0.5073114633560181,-0.0420149564743042]}]}"#;

        let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(4))));
        assert_eq!(store.from_json(python_payload, false).unwrap(), 2);
        assert_eq!(store.get(&NodeId::new("person", "1")).unwrap().text, "Alice");
        assert_eq!(store.get(&NodeId::new("account", "007")).unwrap().text, "Bond");

        // and the vectors match what this port encodes for the same text
        let encoder = MockEncoder::new(4);
        let expected = encoder.encode("Alice");
        let stored = &store.get(&NodeId::new("person", "1")).unwrap().vector;
        for (i, want) in expected.iter().enumerate() {
            assert!((stored[i] - want).abs() < 1e-6, "component {i}: {} vs {want}", stored[i]);
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
        };
        assert!(matches!(
            store.import_embeddings(&payload, false),
            Err(EmbeddingError::Payload(_))
        ));
    }
}
