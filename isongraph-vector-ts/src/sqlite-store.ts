/**
 * SQLite-backed embedding storage, optionally indexed with sqlite-vec.
 *
 * A separate entry point (`isongraph-vector-ts/sqlite`) rather than part of the
 * main module: it imports `node:sqlite`, so importing it from a browser bundle
 * would break. The main entry stays runtime-agnostic.
 *
 * The schema is identical to the one the Python, C#, Rust, JavaScript and C++
 * ports write, so a database created by any of them can be opened by any
 * other. Vectors are little-endian float32, four bytes per dimension.
 *
 * @example
 * ```typescript
 * import { SqliteEmbeddingStore } from 'isongraph-vector-ts/sqlite';
 * import { MockEncoder } from 'isongraph-vector-ts';
 *
 * const store = new SqliteEmbeddingStore('embeddings.db', new MockEncoder(384), true);
 * store.add(['person', 1], 'Alice is a software engineer');
 * const hits = store.similaritySearch('engineering', 5);
 * ```
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { DatabaseSync } from 'node:sqlite';
import type { NodeRef } from 'ison-graph-ts';
import {
  DimensionMismatchError,
  EmbeddingError,
  ModelMismatchError,
  NoEncoderError,
  EMBEDDING_FORMAT,
  EMBEDDING_FORMAT_VERSION,
  GRAPH_DIGEST_BASIS,
  UNKNOWN_MODELS,
  type EmbeddingEncoder,
  type EmbeddingRecord,
  type EmbeddingStoreOptions,
  type ExportedEmbedding,
  type ExportedEmbeddings,
  type ImportMeta,
  type ModelMismatchMode,
  type SimilarityResult,
  type StaleOptions,
  selectTopK,
} from './index.js';

/** Virtual table holding the vec0 index, alongside the `embeddings` table. */
export const VEC_TABLE = 'vec_embeddings';

/** SQLite-backed embedding storage. */
export class SqliteEmbeddingStore {
  private _db: DatabaseSync;
  private _encoder: EmbeddingEncoder | null;
  private _dimension: number | null = null;
  private _sqliteVec: boolean;
  private _vecReady = false;
  private _version: string | null;
  private _onModelMismatch: ModelMismatchMode = 'warn';
  private _warnedMismatch: Set<string> = new Set();
  private _modelsCache: { dataVersion: number; counts: Record<string, number> } | null = null;

  /** Metadata of the most recent imported sidecar (empty before any). */
  lastImportMeta: ImportMeta = {};

  /**
   * Open (or create) a store. Use ':memory:' for a scratch store.
   *
   * @param path      Database path, or ':memory:'
   * @param encoder   Encoder used when a text is added without a vector
   * @param sqliteVec Answer top-k searches through a sqlite-vec vec0 virtual
   *                  table instead of scanning every row. Results are the same
   *                  either way - vec0 runs an exact brute-force KNN in C.
   * @param options   `version` (recorded in the sidecar) and `onModelMismatch`
   *                  ('warn' | 'error' | 'ignore'), as for EmbeddingStore
   */
  constructor(
    path: string = ':memory:',
    encoder?: EmbeddingEncoder,
    sqliteVec: boolean = false,
    options: EmbeddingStoreOptions = {}
  ) {
    this._encoder = encoder ?? null;
    this._sqliteVec = sqliteVec;
    this._version = options.version ?? null;
    this.onModelMismatch = options.onModelMismatch ?? 'warn';
    this._db = new DatabaseSync(path, { allowExtension: sqliteVec });

    if (sqliteVec) this._loadVecExtension();
    this._createSchema();
    this._adoptStoredDimension();
    if (this._sqliteVec && this._dimension !== null) this._ensureVecTable(this._dimension);
  }

  private _loadVecExtension(): void {
    let sqliteVec: { load(db: unknown): void };
    try {
      // Optional dependency, and CommonJS - reached through createRequire
      // because this module is ESM. Only needed when the index is on.
      sqliteVec = createRequire(import.meta.url)('sqlite-vec');
    } catch {
      throw new EmbeddingError(
        'sqliteVec: true needs the sqlite-vec package: npm install sqlite-vec'
      );
    }
    try {
      this._db.enableLoadExtension(true);
      sqliteVec.load(this._db);
    } catch (err) {
      throw new EmbeddingError(
        `could not load the sqlite-vec extension: ${(err as Error).message}. ` +
        'Construct the store with sqliteVec: false for the exact scan.'
      );
    }
  }

  private _createSchema(): void {
    this._db.exec(`
      CREATE TABLE IF NOT EXISTS embeddings (
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
          ON embeddings(node_type, node_id);
    `);

    // The shared schema has always had a `model` column, but a database made by
    // a very early build may not. Add it with the "no information" default, so
    // such rows read as 'unknown' rather than as some real model. ADD COLUMN
    // with a default is metadata-only in SQLite (no rewrite) and every other
    // port ignores columns it does not know.
    const columns = this._db.prepare('PRAGMA table_info(embeddings)').all() as Array<{ name: string }>;
    if (!columns.some(c => c.name === 'model')) {
      this._db.exec("ALTER TABLE embeddings ADD COLUMN model TEXT DEFAULT 'unknown'");
    }
  }

  private _adoptStoredDimension(): void {
    const row = this._db
      .prepare('SELECT LENGTH(vector) AS n FROM embeddings LIMIT 1')
      .get() as { n?: number } | undefined;
    if (row && row.n) this._dimension = Number(row.n) / 4;
  }

  /** Create the vec0 index once the vector width is known, and backfill it. */
  private _ensureVecTable(dimension: number): void {
    if (!this._sqliteVec || this._vecReady) return;

    // node_type is a vec0 metadata column so a type-filtered search stays
    // exact - filtering after a top-k fetch would silently drop matches.
    this._db.exec(
      `CREATE VIRTUAL TABLE IF NOT EXISTS ${VEC_TABLE} USING vec0(
           node_type text,
           embedding float[${dimension}] distance_metric=cosine
       )`
    );

    const indexed = Number((this._db.prepare(`SELECT COUNT(*) AS c FROM ${VEC_TABLE}`).get() as { c: number }).c);
    const stored = Number((this._db.prepare('SELECT COUNT(*) AS c FROM embeddings').get() as { c: number }).c);

    if (indexed !== stored) {
      // Opening a database written without the index, or with it off.
      this._db.exec(`DELETE FROM ${VEC_TABLE}`);
      const rows = this._db
        .prepare('SELECT rowid AS rid, node_type, vector FROM embeddings')
        .all() as Array<{ rid: number | bigint; node_type: string; vector: Uint8Array }>;
      const insert = this._db.prepare(
        `INSERT INTO ${VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)`
      );
      for (const row of rows) insert.run(BigInt(row.rid), row.node_type, row.vector);
    }

    this._vecReady = true;
  }

  /** Whether top-k search is answered from the sqlite-vec index. */
  get sqliteVec(): boolean {
    return this._sqliteVec;
  }

  /** Vector length this store holds, or null while it is still empty. */
  get dimension(): number | null {
    return this._dimension;
  }

  get encoder(): EmbeddingEncoder | null {
    return this._encoder;
  }

  set encoder(encoder: EmbeddingEncoder | null) {
    this._encoder = encoder;
  }

  /** Encoder/model version recorded in the sidecar, or null if unknown. */
  get version(): string | null {
    if (this._version !== null) return this._version;
    const fromEncoder = this._encoder?.version;
    return fromEncoder === undefined || fromEncoder === null ? null : fromEncoder;
  }

  set version(value: string | null) {
    this._version = value;
  }

  /** How a text query reacts to vectors from another model (see EmbeddingStoreOptions). */
  get onModelMismatch(): ModelMismatchMode {
    return this._onModelMismatch;
  }

  set onModelMismatch(mode: ModelMismatchMode) {
    if (mode !== 'warn' && mode !== 'error' && mode !== 'ignore') {
      throw new EmbeddingError(
        `onModelMismatch must be 'warn', 'error' or 'ignore', got ${JSON.stringify(mode)}`
      );
    }
    this._onModelMismatch = mode;
  }

  /** Surrogate primary key, matching what the other ports compute. */
  private static _embeddingId(nodeRef: NodeRef): string {
    return createHash('md5').update(`${nodeRef[0]}:${nodeRef[1]}`).digest('hex').slice(0, 16);
  }

  private static _pack(vector: number[]): Uint8Array {
    return new Uint8Array(new Float32Array(vector).buffer);
  }

  private static _unpack(bytes: Uint8Array): number[] {
    const copy = new Uint8Array(bytes);   // ensure the buffer is aligned
    return Array.from(new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4));
  }

  private _checkDimension(vector: number[], what: string): void {
    if (vector.length === 0) throw new DimensionMismatchError(`${what} is empty`);
    if (this._dimension === null) this._dimension = vector.length;
    else if (vector.length !== this._dimension) {
      throw new DimensionMismatchError(
        `${what} has ${vector.length} dimensions, store holds ${this._dimension}`
      );
    }
  }

  private _checkQuery(vector: number[]): void {
    if (vector.length === 0) throw new DimensionMismatchError('query vector is empty');
    if (this._dimension !== null && vector.length !== this._dimension) {
      throw new DimensionMismatchError(
        `query vector has ${vector.length} dimensions, store holds ${this._dimension}`
      );
    }
  }

  private static _cosineSimilarity(v1: number[], v2: number[]): number {
    let dot = 0, n1 = 0, n2 = 0;
    const n = Math.min(v1.length, v2.length);
    for (let i = 0; i < n; i++) {
      dot += v1[i] * v2[i];
      n1 += v1[i] * v1[i];
      n2 += v2[i] * v2[i];
    }
    n1 = Math.sqrt(n1);
    n2 = Math.sqrt(n2);
    return n1 === 0 || n2 === 0 ? 0 : dot / (n1 * n2);
  }

  /** Model recorded on vectors written now: the encoder's, or "provided"/"unknown". */
  private _modelName(): string {
    if (!this._encoder) return 'provided';
    return this._encoder.modelName || 'unknown';
  }

  /** The encoder's model, or null when it has none (or an uninformative one). */
  private _currentModel(): string | null {
    const name = this._encoder?.modelName;
    return name && !UNKNOWN_MODELS.has(name) ? name : null;
  }

  /**
   * Which models produced the stored vectors: `{ modelName: count }`.
   *
   * "unknown" / "provided" appear as-is (vectors imported from a port that
   * records no model, or handed in with no encoder attached). The counts are
   * cached, and the cache is dropped on every write through this store and
   * whenever another connection commits a change (`PRAGMA data_version`).
   */
  models(): Record<string, number> {
    const dataVersion = Number(
      (this._db.prepare('PRAGMA data_version').get() as { data_version: number | bigint }).data_version
    );
    if (this._modelsCache === null || this._modelsCache.dataVersion !== dataVersion) {
      const rows = this._db
        .prepare("SELECT COALESCE(model, 'unknown') AS m, COUNT(*) AS n FROM embeddings GROUP BY m")
        .all() as Array<{ m: string; n: number | bigint }>;
      const counts: Record<string, number> = {};
      for (const row of rows) counts[row.m] = Number(row.n);
      this._modelsCache = { dataVersion, counts };
    }
    return { ...this._modelsCache.counts };
  }

  /** Complain when a text query's encoder is not the stored vectors' model. */
  private _checkModelCompat(): void {
    if (this._onModelMismatch === 'ignore') return;
    const current = this._currentModel();
    if (current === null) return;
    const foreign = Object.keys(this.models())
      .filter(m => !UNKNOWN_MODELS.has(m) && m !== current)
      .sort();
    if (foreign.length === 0) return;

    const message =
      `encoder model '${current}' differs from the model that produced the ` +
      `stored vectors (${foreign.map(m => `'${m}'`).join(', ')}); ` +
      `scores compare vectors from different spaces. Re-embed the stale ` +
      `vectors (see stale()) or search with the original encoder.`;
    if (this._onModelMismatch === 'error') throw new ModelMismatchError(message);

    const key = JSON.stringify([current, foreign]);
    if (!this._warnedMismatch.has(key)) {
      this._warnedMismatch.add(key);
      if (typeof console !== 'undefined') console.warn(message);
    }
  }

  /**
   * Node refs whose stored vector does not belong to the current encoder.
   *
   * A vector is stale when its recorded model differs from `model` (default:
   * the encoder's `modelName`) or its length differs from `dimension`
   * (default: the encoder's `dimension`). Vectors whose model is
   * "unknown"/"provided" cannot be judged by model and are skipped unless
   * `includeUnknown` is true (a dimension mismatch still counts). With no
   * encoder and no arguments nothing can be judged and the result is empty.
   *
   * Takes either the positional form or an options object.
   */
  stale(
    modelOrOptions?: string | null | StaleOptions,
    dimensionArg?: number | null,
    includeUnknownArg: boolean = false
  ): NodeRef[] {
    const options: StaleOptions =
      modelOrOptions !== null && typeof modelOrOptions === 'object'
        ? modelOrOptions
        : { model: modelOrOptions, dimension: dimensionArg, includeUnknown: includeUnknownArg };

    const model = options.model ?? this._currentModel();
    let dimension = options.dimension ?? null;
    if (dimension === null && this._encoder) {
      const d = Number(this._encoder.dimension);
      dimension = Number.isFinite(d) ? Math.trunc(d) : null;
    }
    const includeUnknown = options.includeUnknown ?? false;

    const rows = this._db
      .prepare(
        "SELECT node_type, node_id, node_id_type, COALESCE(model, 'unknown') AS model, " +
        'LENGTH(vector) AS n FROM embeddings ORDER BY node_type, node_id'
      )
      .all() as Array<{
        node_type: string; node_id: string; node_id_type: string; model: string; n: number | bigint;
      }>;

    const out: NodeRef[] = [];
    for (const row of rows) {
      const byModel =
        model !== null &&
        row.model !== model &&
        (includeUnknown || !UNKNOWN_MODELS.has(row.model));
      const byDim = dimension !== null && Number(row.n) / 4 !== dimension;
      if (byModel || byDim) {
        out.push([row.node_type, row.node_id_type === 'int' ? Number(row.node_id) : row.node_id] as NodeRef);
      }
    }
    return out;
  }

  /** Resolve a query to a vector, checking model compatibility for text. */
  private _resolveQuery(query: string | number[]): number[] {
    let queryVector: number[];
    if (typeof query === 'string') {
      if (!this._encoder) throw new NoEncoderError('No encoder provided for text query');
      this._checkModelCompat();
      queryVector = this._encoder.encode(query);
    } else {
      queryVector = query;
    }
    this._checkQuery(queryVector);
    return queryVector;
  }

  /**
   * Insert or update one row, keeping the vec0 index in step.
   *
   * An UPSERT rather than INSERT OR REPLACE: REPLACE assigns a new rowid,
   * which would orphan the matching vec0 entry and reset created_at.
   */
  private _writeRow(
    nodeRef: NodeRef,
    vector: number[],
    text: string,
    model: string,
    nodeIdType: string = 'str'
  ): string {
    const embId = SqliteEmbeddingStore._embeddingId(nodeRef);
    const blob = SqliteEmbeddingStore._pack(vector);
    this._modelsCache = null;

    this._db.prepare(`
      INSERT INTO embeddings
      (id, node_type, node_id, node_id_type, vector, text, model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(node_type, node_id) DO UPDATE SET
          node_id_type = excluded.node_id_type,
          vector = excluded.vector,
          text = excluded.text,
          model = excluded.model
    `).run(embId, String(nodeRef[0]), String(nodeRef[1]), nodeIdType, blob, text, model);

    if (this._sqliteVec) {
      this._ensureVecTable(vector.length);
      const row = this._db
        .prepare('SELECT rowid AS rid FROM embeddings WHERE node_type = ? AND node_id = ?')
        .get(String(nodeRef[0]), String(nodeRef[1])) as { rid: number | bigint };
      const rowid = BigInt(row.rid);
      // vec0 has no upsert; replace the entry at this rowid.
      this._db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(rowid);
      this._db
        .prepare(`INSERT INTO ${VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)`)
        .run(rowid, String(nodeRef[0]), blob);
    }

    return embId;
  }

  add(nodeRef: NodeRef, text: string, vector?: number[]): string {
    if (vector === undefined) {
      if (!this._encoder) throw new NoEncoderError('No encoder provided and no vector given');
      vector = this._encoder.encode(text);
    }
    this._checkDimension(vector, 'vector');
    return this._writeRow(nodeRef, vector, text, this._modelName());
  }

  /** Add many embeddings in one transaction, encoding their texts together. */
  addBatch(entries: Array<[NodeRef, string]>): number {
    if (entries.length === 0) return 0;
    if (!this._encoder) throw new NoEncoderError('No encoder provided and no vectors given');

    const vectors = this._encoder.encodeBatch(entries.map(([, text]) => text));
    if (vectors.length !== entries.length) {
      throw new EmbeddingError(
        `encoder returned ${vectors.length} vectors for ${entries.length} texts`
      );
    }
    for (const vector of vectors) this._checkDimension(vector, 'vector');

    const model = this._modelName();
    this._db.exec('BEGIN');
    try {
      entries.forEach(([nodeRef, text], i) => this._writeRow(nodeRef, vectors[i], text, model));
      this._db.exec('COMMIT');
    } catch (err) {
      this._db.exec('ROLLBACK');
      throw err;
    }
    return entries.length;
  }

  get(nodeRef: NodeRef): EmbeddingRecord | null {
    const row = this._db
      .prepare(
        'SELECT id, node_type, node_id, vector, text, model FROM embeddings ' +
        'WHERE node_type = ? AND node_id = ?'
      )
      .get(String(nodeRef[0]), String(nodeRef[1])) as
      | { id: string; node_type: string; node_id: string; vector: Uint8Array; text: string; model: string }
      | undefined;

    if (!row) return null;
    return {
      id: row.id,
      nodeRef: [row.node_type, row.node_id] as NodeRef,
      vector: SqliteEmbeddingStore._unpack(row.vector),
      text: row.text,
      model: row.model ?? 'unknown',
    };
  }

  remove(nodeRef: NodeRef): boolean {
    this._modelsCache = null;
    if (this._sqliteVec && this._vecReady) {
      const row = this._db
        .prepare('SELECT rowid AS rid FROM embeddings WHERE node_type = ? AND node_id = ?')
        .get(String(nodeRef[0]), String(nodeRef[1])) as { rid: number | bigint } | undefined;
      if (row) this._db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(BigInt(row.rid));
    }
    const result = this._db
      .prepare('DELETE FROM embeddings WHERE node_type = ? AND node_id = ?')
      .run(String(nodeRef[0]), String(nodeRef[1]));
    return Number(result.changes) > 0;
  }

  count(): number {
    return Number((this._db.prepare('SELECT COUNT(*) AS c FROM embeddings').get() as { c: number }).c);
  }

  /**
   * Clear all embeddings.
   *
   * @param resetDimension Also forget the pinned vector length, so the next
   *   vector may have a different one (switching to an encoder of a different
   *   dimension); with the sqlite-vec index on, the vec0 table - fixed to the
   *   old width - is dropped and rebuilt at the new one. Off by default: a
   *   store normally refuses vectors of the wrong length even when empty.
   */
  clear(resetDimension: boolean = false): void {
    this._modelsCache = null;
    this._db.exec('DELETE FROM embeddings');
    if (this._sqliteVec && this._vecReady) {
      if (resetDimension) {
        this._db.exec(`DROP TABLE IF EXISTS ${VEC_TABLE}`);
        this._vecReady = false;
      } else {
        this._db.exec(`DELETE FROM ${VEC_TABLE}`);
      }
    }
    if (resetDimension) this._dimension = null;
  }

  /**
   * Cosine similarity against a stored blob, without materialising it.
   *
   * The obvious route - unpack the blob into a number[] and reuse the array
   * helper - allocates a 384-element JS array per row. Over 20,000 rows that
   * measured 535 ms per query against 17 ms for the same search through the
   * vec0 index. Reading through a Float32Array view instead keeps the scan
   * usable when the index is off.
   */
  private static _cosineAgainstBlob(query: number[], bytes: Uint8Array): number {
    // Copied so the buffer is 4-byte aligned; node:sqlite gives no guarantee.
    const copy = new Uint8Array(bytes);
    const stored = new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);

    let dot = 0, n1 = 0, n2 = 0;
    const n = Math.min(query.length, stored.length);
    for (let i = 0; i < n; i++) {
      dot += query[i] * stored[i];
      n1 += query[i] * query[i];
      n2 += stored[i] * stored[i];
    }
    n1 = Math.sqrt(n1);
    n2 = Math.sqrt(n2);
    return n1 === 0 || n2 === 0 ? 0 : dot / (n1 * n2);
  }

  /** Score every stored vector against a query. Always an exact scan. */
  private _scan(queryVector: number[], nodeType?: string): SimilarityResult[] {
    const rows = this._db
      .prepare('SELECT node_type, node_id, vector, text FROM embeddings')
      .all() as Array<{ node_type: string; node_id: string; vector: Uint8Array; text: string }>;

    const results: SimilarityResult[] = [];
    for (const row of rows) {
      if (nodeType && row.node_type !== nodeType) continue;
      results.push({
        nodeRef: [row.node_type, row.node_id] as NodeRef,
        score: SqliteEmbeddingStore._cosineAgainstBlob(queryVector, row.vector),
        text: row.text,
      });
    }
    return results;
  }

  /** Top-k from the vec0 index. */
  private _vecSearch(
    queryVector: number[],
    topK: number,
    nodeType: string | undefined,
    threshold: number
  ): SimilarityResult[] {
    this._ensureVecTable(queryVector.length);
    const blob = SqliteEmbeddingStore._pack(queryVector);

    const select =
      `SELECT e.node_type AS node_type, e.node_id AS node_id, e.text AS text, ` +
      `v.distance AS distance FROM ${VEC_TABLE} v JOIN embeddings e ON e.rowid = v.rowid `;

    const rows = (nodeType
      ? this._db
          .prepare(select + 'WHERE v.node_type = ? AND v.embedding MATCH ? AND k = ?')
          .all(nodeType, blob, topK)
      : this._db
          .prepare(select + 'WHERE v.embedding MATCH ? AND k = ?')
          .all(blob, topK)) as Array<{
      node_type: string; node_id: string; text: string; distance: number;
    }>;

    // Rows arrive ordered by distance ascending, i.e. score descending.
    const results: SimilarityResult[] = [];
    for (const row of rows) {
      const score = 1 - Number(row.distance);
      if (score >= threshold) {
        results.push({ nodeRef: [row.node_type, row.node_id] as NodeRef, score, text: row.text });
      }
    }
    return results;
  }

  similaritySearch(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): SimilarityResult[] {
    const queryVector = this._resolveQuery(query);

    // A zero vector has no direction for cosine to measure; vec0 returns NaN
    // there, while the scan defines it as 0.0.
    if (this._sqliteVec && topK !== null && topK > 0 && queryVector.some(v => v !== 0)) {
      return this._vecSearch(queryVector, topK, nodeType, threshold);
    }

    const results = this._scan(queryVector, nodeType).filter(r => r.score >= threshold);
    if (topK === null) return results.sort((a, b) => b.score - a.score);
    return selectTopK(results, topK);
  }

  /**
   * Cosine score for every embedded node, keyed by `type:id`.
   *
   * Always scans, even with the index on: the index answers top-k queries, and
   * blending needs a score for every node.
   */
  scoreMap(query: string | number[], nodeType?: string): Map<string, number> {
    const queryVector = this._resolveQuery(query);

    const scores = new Map<string, number>();
    for (const result of this._scan(queryVector, nodeType)) {
      scores.set(`${result.nodeRef[0]}:${result.nodeRef[1]}`, result.score);
    }
    return scores;
  }

  /**
   * Serialize every embedding into the portable cross-language payload.
   *
   * Beyond the version-1 fields it appends, when known: `encoder_model` (the
   * encoder's modelName), `encoder_version` (this store's `version`) and, when
   * `graphDigest` is given, `graph_digest` plus `graph_digest_basis`. Absent
   * values are omitted, not written as null. Each embedding carries the
   * `model` that produced it.
   */
  exportEmbeddings(graphDigest?: string | null): ExportedEmbeddings {
    const rows = this._db
      .prepare(
        "SELECT node_type, node_id, node_id_type, vector, text, COALESCE(model, 'unknown') AS model " +
        'FROM embeddings ORDER BY node_type, node_id'
      )
      .all() as Array<{
        node_type: string; node_id: string; node_id_type: string;
        vector: Uint8Array; text: string; model: string;
      }>;

    const embeddings: ExportedEmbedding[] = rows.map(row => ({
      type: row.node_type,
      id: row.node_id,
      id_type: (row.node_id_type === 'int' ? 'int' : 'str') as 'int' | 'str',
      text: row.text,
      model: row.model,
      vector: SqliteEmbeddingStore._unpack(row.vector),
    }));

    const payload: ExportedEmbeddings = {
      format: EMBEDDING_FORMAT,
      version: EMBEDDING_FORMAT_VERSION,
      dimension: this._dimension,
      count: embeddings.length,
      embeddings,
    };
    const encoderModel = this._currentModel();
    if (encoderModel !== null) payload.encoder_model = encoderModel;
    const version = this.version;
    if (version !== null) payload.encoder_version = String(version);
    if (graphDigest !== undefined && graphDigest !== null) {
      payload.graph_digest = graphDigest;
      payload.graph_digest_basis = GRAPH_DIGEST_BASIS;
    }
    return payload;
  }

  /** Load a payload produced by exportEmbeddings (in any language port). */
  importEmbeddings(payload: ExportedEmbeddings, replace: boolean = false): number {
    if (!payload || payload.format !== EMBEDDING_FORMAT) {
      throw new EmbeddingError(
        `not an ${EMBEDDING_FORMAT} payload: format=${payload && payload.format}`
      );
    }
    if (payload.version !== EMBEDDING_FORMAT_VERSION) {
      throw new EmbeddingError(
        `unsupported ${EMBEDDING_FORMAT} version ${payload.version} ` +
        `(this build reads version ${EMBEDDING_FORMAT_VERSION})`
      );
    }

    const meta: ImportMeta = {};
    for (const key of ['encoder_model', 'encoder_version', 'graph_digest', 'graph_digest_basis'] as const) {
      if (payload[key] !== undefined && payload[key] !== null) meta[key] = payload[key];
    }
    this.lastImportMeta = meta;

    if (replace) {
      this.clear(true);
      this._dimension = payload.dimension ?? null;
    }

    let written = 0;
    this._db.exec('BEGIN');
    try {
      for (const item of payload.embeddings ?? []) {
        this._checkDimension(item.vector, 'vector');
        this._writeRow(
          [item.type, String(item.id)] as NodeRef,
          item.vector, item.text ?? '', item.model ?? 'unknown', item.id_type ?? 'str'
        );
        written++;
      }
      this._db.exec('COMMIT');
    } catch (err) {
      this._db.exec('ROLLBACK');
      throw err;
    }
    return written;
  }

  /** Release the database file. */
  close(): void {
    this._db.close();
  }
}
