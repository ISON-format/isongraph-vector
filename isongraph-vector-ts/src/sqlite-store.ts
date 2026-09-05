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
  NoEncoderError,
  EMBEDDING_FORMAT,
  EMBEDDING_FORMAT_VERSION,
  type EmbeddingEncoder,
  type EmbeddingRecord,
  type ExportedEmbedding,
  type ExportedEmbeddings,
  type SimilarityResult,
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

  /**
   * Open (or create) a store. Use ':memory:' for a scratch store.
   *
   * @param path      Database path, or ':memory:'
   * @param encoder   Encoder used when a text is added without a vector
   * @param sqliteVec Answer top-k searches through a sqlite-vec vec0 virtual
   *                  table instead of scanning every row. Results are the same
   *                  either way - vec0 runs an exact brute-force KNN in C.
   */
  constructor(path: string = ':memory:', encoder?: EmbeddingEncoder, sqliteVec: boolean = false) {
    this._encoder = encoder ?? null;
    this._sqliteVec = sqliteVec;
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

  private _modelName(): string {
    return this._encoder ? 'unknown' : 'provided';
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
        'SELECT id, node_type, node_id, vector, text FROM embeddings ' +
        'WHERE node_type = ? AND node_id = ?'
      )
      .get(String(nodeRef[0]), String(nodeRef[1])) as
      | { id: string; node_type: string; node_id: string; vector: Uint8Array; text: string }
      | undefined;

    if (!row) return null;
    return {
      id: row.id,
      nodeRef: [row.node_type, row.node_id] as NodeRef,
      vector: SqliteEmbeddingStore._unpack(row.vector),
      text: row.text,
    };
  }

  remove(nodeRef: NodeRef): boolean {
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

  clear(): void {
    this._db.exec('DELETE FROM embeddings');
    if (this._sqliteVec && this._vecReady) this._db.exec(`DELETE FROM ${VEC_TABLE}`);
    this._dimension = null;
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
        score: SqliteEmbeddingStore._cosineSimilarity(
          queryVector, SqliteEmbeddingStore._unpack(row.vector)),
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
    let queryVector: number[];
    if (typeof query === 'string') {
      if (!this._encoder) throw new NoEncoderError('No encoder provided for text query');
      queryVector = this._encoder.encode(query);
    } else {
      queryVector = query;
    }
    this._checkQuery(queryVector);

    // A zero vector has no direction for cosine to measure; vec0 returns NaN
    // there, while the scan defines it as 0.0.
    if (this._sqliteVec && topK !== null && topK > 0 && queryVector.some(v => v !== 0)) {
      return this._vecSearch(queryVector, topK, nodeType, threshold);
    }

    const results = this._scan(queryVector, nodeType)
      .filter(r => r.score >= threshold)
      .sort((a, b) => b.score - a.score);
    return topK === null ? results : results.slice(0, topK);
  }

  /**
   * Cosine score for every embedded node, keyed by `type:id`.
   *
   * Always scans, even with the index on: the index answers top-k queries, and
   * blending needs a score for every node.
   */
  scoreMap(query: string | number[], nodeType?: string): Map<string, number> {
    let queryVector: number[];
    if (typeof query === 'string') {
      if (!this._encoder) throw new NoEncoderError('No encoder provided for text query');
      queryVector = this._encoder.encode(query);
    } else {
      queryVector = query;
    }
    this._checkQuery(queryVector);

    const scores = new Map<string, number>();
    for (const result of this._scan(queryVector, nodeType)) {
      scores.set(`${result.nodeRef[0]}:${result.nodeRef[1]}`, result.score);
    }
    return scores;
  }

  /** Serialize every embedding into the portable cross-language payload. */
  exportEmbeddings(): ExportedEmbeddings {
    const rows = this._db
      .prepare(
        'SELECT node_type, node_id, node_id_type, vector, text FROM embeddings ' +
        'ORDER BY node_type, node_id'
      )
      .all() as Array<{
        node_type: string; node_id: string; node_id_type: string;
        vector: Uint8Array; text: string;
      }>;

    const embeddings: ExportedEmbedding[] = rows.map(row => ({
      type: row.node_type,
      id: row.node_id,
      id_type: (row.node_id_type === 'int' ? 'int' : 'str') as 'int' | 'str',
      text: row.text,
      vector: SqliteEmbeddingStore._unpack(row.vector),
    }));

    return {
      format: EMBEDDING_FORMAT,
      version: EMBEDDING_FORMAT_VERSION,
      dimension: this._dimension,
      count: embeddings.length,
      embeddings,
    };
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
    if (replace) this.clear();

    let written = 0;
    this._db.exec('BEGIN');
    try {
      for (const item of payload.embeddings ?? []) {
        this._checkDimension(item.vector, 'vector');
        this._writeRow(
          [item.type, String(item.id)] as NodeRef,
          item.vector, item.text ?? '', 'unknown', item.id_type ?? 'str'
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
