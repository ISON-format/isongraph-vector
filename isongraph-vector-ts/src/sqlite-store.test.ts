/**
 * SQLite-backed store. Node-only: the main entry point stays runtime-agnostic,
 * so these exercise the separate `/sqlite` entry.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SqliteEmbeddingStore } from './sqlite-store.js';
import {
  MockEncoder, EmbeddingStore, DimensionMismatchError, EmbeddingError, ModelMismatchError,
  NoEncoderError, GRAPH_DIGEST_BASIS,
} from './index.js';
import type { NodeRef } from 'ison-graph-ts';

const open = (sqliteVec = false, dim = 32, path = ':memory:') =>
  new SqliteEmbeddingStore(path, new MockEncoder(dim), sqliteVec);

const tempDirs: string[] = [];
function tempDb(name: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'igv-'));
  tempDirs.push(dir);
  return join(dir, name);
}
afterEach(() => {
  while (tempDirs.length) rmSync(tempDirs.pop()!, { recursive: true, force: true });
});

describe('SqliteEmbeddingStore', () => {
  it('adds and reads back', () => {
    const store = open();
    store.add(['person', '1'], 'Alice is an engineer');

    const record = store.get(['person', '1'])!;
    expect(record.text).toBe('Alice is an engineer');
    expect(record.vector.length).toBe(32);
    expect(store.count()).toBe(1);
    store.close();
  });

  it('removes and clears', () => {
    const store = open();
    store.add(['a', '1'], 'one');
    store.add(['a', '2'], 'two');

    expect(store.remove(['a', '1'])).toBe(true);
    expect(store.remove(['a', '1'])).toBe(false);
    expect(store.count()).toBe(1);

    store.clear();
    expect(store.count()).toBe(0);
    store.close();
  });

  it('overwrites rather than duplicating', () => {
    const store = open();
    store.add(['a', '1'], 'first');
    store.add(['a', '1'], 'second');
    expect(store.count()).toBe(1);
    expect(store.get(['a', '1'] as NodeRef)!.text).toBe('second');
    store.close();
  });

  it('validates dimensions', () => {
    const store = open();
    store.add(['a', '1'], 'full width');

    expect(() => store.add(['a', '2'], 'short', [1, 0])).toThrow(DimensionMismatchError);
    expect(() => store.add(['a', '3'], 'empty', [])).toThrow(DimensionMismatchError);
    expect(() => store.similaritySearch([1, 0], 1)).toThrow(DimensionMismatchError);
    store.close();
  });

  it('rejects text with no encoder', () => {
    const store = new SqliteEmbeddingStore(':memory:');
    expect(() => store.add(['a', '1'], 'no encoder here')).toThrow(NoEncoderError);
    store.close();
  });

  it('batches in one transaction', () => {
    const store = open();
    const entries: Array<[NodeRef, string]> = Array.from({ length: 10 },
      (_, i) => [['doc', String(i)] as NodeRef, `document ${i}`]);
    expect(store.addBatch(entries)).toBe(10);
    expect(store.count()).toBe(10);
    store.close();
  });

  it('survives close and reopen', () => {
    const path = tempDb('store.db');

    const first = open(false, 32, path);
    first.add(['person', '1'], 'Alice');
    const probe = first.get(['person', '1'] as NodeRef)!.vector;
    first.close();
    expect(existsSync(path)).toBe(true);

    const reopened = open(false, 32, path);
    expect(reopened.count()).toBe(1);
    expect(reopened.get(['person', '1'] as NodeRef)!.vector).toEqual(probe);
    // dimension is recovered from the stored rows
    expect(reopened.dimension).toBe(32);
    reopened.close();
  });

  it('round-trips the portable payload', () => {
    const source = open();
    source.add(['person', '1'], 'Alice');
    source.add(['account', '007'], 'Bond');

    const target = open(true);
    expect(target.importEmbeddings(source.exportEmbeddings())).toBe(2);
    expect(target.get(['account', '007'] as NodeRef)!.text).toBe('Bond');
    source.close();
    target.close();
  });
});

describe('sqlite-vec index', () => {
  it('reports the backend', () => {
    const plain = open();
    const indexed = open(true);
    expect(plain.sqliteVec).toBe(false);
    expect(indexed.sqliteVec).toBe(true);
    plain.close();
    indexed.close();
  });

  it('matches the scan exactly', () => {
    const encoder = new MockEncoder(32);
    const entries: Array<[NodeRef, string]> = Array.from({ length: 200 },
      (_, i) => [['n', String(i)] as NodeRef, `text ${i}`]);

    const scan = open();
    const indexed = open(true);
    scan.addBatch(entries);
    indexed.addBatch(entries);

    for (const probe of ['text 0', 'text 57', 'text 199']) {
      const query = encoder.encode(probe);
      const want = scan.similaritySearch(query, 10, undefined, -1.0);
      const got = indexed.similaritySearch(query, 10, undefined, -1.0);

      expect(got.length).toBe(want.length);
      want.forEach((r, i) => {
        expect(got[i].nodeRef).toEqual(r.nodeRef);
        expect(got[i].score).toBeCloseTo(r.score, 5);
      });
    }
    scan.close();
    indexed.close();
  });

  it('keeps a type filter exact', () => {
    // Filtering after a top-k fetch would drop the rare node entirely; the
    // type is a vec0 metadata column so the search itself is narrowed.
    const store = open(true);
    for (let i = 0; i < 50; i++) store.add(['crowd', String(i)], `crowd ${i}`);
    store.add(['rare', '1'], 'the only rare node');

    const results = store.similaritySearch('anything', 5, 'rare', -1.0);
    expect(results.length).toBe(1);
    expect(results[0].nodeRef).toEqual(['rare', '1']);
    store.close();
  });

  it('reaches the index on update and delete', () => {
    const store = open(true);
    store.add(['a', '1'], 'first');
    store.add(['a', '2'], 'second');

    const second = store.get(['a', '2'] as NodeRef)!.vector;
    store.add(['a', '1'], 'first', second);

    const top = store.similaritySearch(second, 2, undefined, -1.0);
    expect(top.length).toBe(2);
    top.forEach(r => expect(r.score).toBeCloseTo(1.0, 5));

    store.remove(['a', '1']);
    const after = store.similaritySearch(second, 10, undefined, -1.0);
    expect(after.map(r => r.nodeRef)).toEqual([['a', '2']]);
    store.close();
  });

  it('indexes a database written without it', () => {
    const path = tempDb('backfill.db');

    const plain = open(false, 32, path);
    for (let i = 0; i < 20; i++) plain.add(['n', String(i)], `text ${i}`);
    const probe = plain.get(['n', '7'] as NodeRef)!.vector;
    plain.close();

    const indexed = open(true, 32, path);
    expect(indexed.count()).toBe(20);
    const results = indexed.similaritySearch(probe, 3, undefined, -1.0);
    expect(results[0].nodeRef).toEqual(['n', '7']);
    expect(results[0].score).toBeCloseTo(1.0, 5);
    indexed.close();
  });

  it('keeps scoreMap covering everything', () => {
    // Blending needs every score, so scoreMap must not take the top-k path.
    const store = open(true);
    for (let i = 0; i < 30; i++) store.add(['n', String(i)], `text ${i}`);
    expect(store.scoreMap('anything').size).toBe(30);
    store.close();
  });

  it('falls back to the scan for a zero query vector', () => {
    const store = new SqliteEmbeddingStore(':memory:', new MockEncoder(4), true);
    store.add(['a', '1'], 'x', [1, 0, 0, 0]);

    const results = store.similaritySearch([0, 0, 0, 0], 1, undefined, -1.0);
    expect(results.length).toBe(1);
    expect(results[0].score).toBe(0);
    store.close();
  });

  it('writes the schema every port shares', async () => {
    // A database written here must be readable by the Python, C#, Rust and
    // C++ ports: same table, same columns, little-endian float32 vectors.
    const path = tempDb('schema.db');
    const store = new SqliteEmbeddingStore(path, new MockEncoder(4), false);
    store.add(['person', '1'], 'Alice');
    store.close();

    const { DatabaseSync } = await import('node:sqlite');
    const db = new DatabaseSync(path);
    const row = db
      .prepare('SELECT id, node_type, node_id, node_id_type, vector, text FROM embeddings')
      .get() as any;
    expect(row.id.length).toBe(16);            // md5 prefix
    expect(row.node_type).toBe('person');
    expect(row.node_id).toBe('1');
    expect(row.node_id_type).toBe('str');
    expect(row.vector.length).toBe(16);        // 4 dims x 4 bytes
    expect(row.text).toBe('Alice');
    db.close();
  });
});

// -----------------------------------------------------------------------------
// Model lifecycle (mirrors the in-memory EmbeddingStore)
// -----------------------------------------------------------------------------

function retagged(dimension: number, name: string, version?: string): MockEncoder {
  const encoder = new MockEncoder(dimension, version);
  encoder.modelName = name;
  return encoder;
}

describe('SqliteEmbeddingStore model lifecycle', () => {
  it('stamps vectors with the encoder model and counts them', () => {
    const store = open(false, 8);
    store.add(['a', '1'], 'one');
    store.addBatch([[['a', '2'], 'two'], [['a', '3'], 'three']]);
    expect(store.models()).toEqual({ 'mock-8': 3 });
    expect(store.get(['a', '1'])!.model).toBe('mock-8');

    store.encoder = null;                        // a vector handed in -> "provided"
    store.add(['a', '4'], 'four', new Array(8).fill(0.5));
    expect(store.models()).toEqual({ 'mock-8': 3, provided: 1 });
    store.close();
  });

  it('is stale by model name and the cure is to re-add', () => {
    const store = open(false, 8);
    store.addBatch([[['a', '1'], 'one'], [['a', '2'], 'two']]);
    store.encoder = retagged(8, 'next-gen-8');

    expect(store.stale()).toEqual([['a', '1'], ['a', '2']]);
    for (const ref of store.stale()) store.add(ref, store.get(ref)!.text);
    expect(store.stale()).toEqual([]);
    expect(store.models()).toEqual({ 'next-gen-8': 2 });
    store.close();
  });

  it('judges staleness by dimension and by explicit arguments', () => {
    const store = open(false, 8);
    store.addBatch([[['a', '1'], 'one']]);
    expect(store.stale()).toEqual([]);
    expect(store.stale('other-model').length).toBe(1);
    expect(store.stale({ model: 'other-model' }).length).toBe(1);
    expect(store.stale(null, 99).length).toBe(1);
    expect(store.stale({ dimension: 8 })).toEqual([]);

    store.encoder = new MockEncoder(16);         // same family, new width
    expect(store.stale().length).toBe(1);
    store.close();
  });

  it('does not judge unknown vectors by model unless asked', () => {
    const store = new SqliteEmbeddingStore(':memory:');
    store.add(['a', '1'], 'x', new Array(8).fill(0.1));
    store.encoder = new MockEncoder(8);
    expect(store.models()).toEqual({ provided: 1 });
    expect(store.stale()).toEqual([]);
    expect(store.stale({ includeUnknown: true })).toEqual([['a', '1']]);
    expect(store.stale(null, undefined, true)).toEqual([['a', '1']]);
    store.close();
  });

  it('with no encoder and no arguments nothing can be judged', () => {
    const store = new SqliteEmbeddingStore(':memory:');
    store.add(['a', '1'], 'x', [1, 2]);
    expect(store.stale()).toEqual([]);
    store.close();
  });

  it('a text search with a different model warns once per pair', () => {
    const store = open(false, 16);
    store.add(['a', '1'], 'one');
    store.encoder = retagged(16, 'other-16');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    store.similaritySearch('hello', 1, undefined, -1.0);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain('mock-16');
    store.similaritySearch('again', 1, undefined, -1.0);
    store.scoreMap('and again');
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
    store.close();
  });

  it('a text search with a different model can throw', () => {
    const store = new SqliteEmbeddingStore(':memory:', new MockEncoder(16), false, { onModelMismatch: 'error' });
    store.add(['a', '1'], 'one');
    store.encoder = retagged(16, 'other-16');
    expect(() => store.similaritySearch('hello', 1)).toThrow(ModelMismatchError);
    expect(() => store.scoreMap('hello')).toThrow(/mock-16/);
    store.add(['a', '1'], 'one');                // re-embedded under the new model
    expect(store.similaritySearch('hello', 1, undefined, -1.0).length).toBe(1);
    store.close();
  });

  it('vector queries and ignore mode are never checked', () => {
    const store = open(false, 16);
    store.add(['a', '1'], 'one');
    store.encoder = retagged(16, 'other-16');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    store.similaritySearch(new MockEncoder(16).encode('x'), 1, undefined, -1.0);
    store.onModelMismatch = 'ignore';
    store.similaritySearch('x', 1, undefined, -1.0);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
    store.close();
  });

  it('rejects an unknown mismatch mode', () => {
    expect(() => new SqliteEmbeddingStore(':memory:', undefined, false, { onModelMismatch: 'loud' as any }))
      .toThrow(EmbeddingError);
  });

  it('exports and imports the per-record model and the sidecar metadata', () => {
    const source = new SqliteEmbeddingStore(':memory:', new MockEncoder(8, '2026.1'));
    source.addBatch([[['a', '1'], 'one'], [['a', '2'], 'two']]);
    source.encoder = retagged(8, 'second-8');
    source.add(['a', '3'], 'three');

    const payload = source.exportEmbeddings('abc123');
    expect(payload.version).toBe(1);
    expect(payload.encoder_model).toBe('second-8');
    expect(payload.graph_digest).toBe('abc123');
    expect(payload.graph_digest_basis).toBe(GRAPH_DIGEST_BASIS);
    expect(payload.embeddings.map(e => e.model)).toEqual(['mock-8', 'mock-8', 'second-8']);
    expect('encoder_version' in source.exportEmbeddings()).toBe(false);   // the new encoder has none

    const target = open(false, 8);
    expect(target.importEmbeddings(payload)).toBe(3);
    expect(target.models()).toEqual({ 'mock-8': 2, 'second-8': 1 });
    expect(target.lastImportMeta).toEqual({
      encoder_model: 'second-8', graph_digest: 'abc123', graph_digest_basis: GRAPH_DIGEST_BASIS,
    });

    // the in-memory store reads the same payload, and back again
    const memory = new EmbeddingStore();
    memory.importEmbeddings(payload);
    expect(memory.models()).toEqual({ 'mock-8': 2, 'second-8': 1 });
    const roundTrip = new SqliteEmbeddingStore(':memory:');
    roundTrip.importEmbeddings(memory.exportEmbeddings());
    expect(roundTrip.models()).toEqual({ 'mock-8': 2, 'second-8': 1 });
    source.close();
    target.close();
    roundTrip.close();
  });

  it('records the version, explicit options winning over the encoder', () => {
    const store = new SqliteEmbeddingStore(':memory:', new MockEncoder(8, '2026.1'));
    store.add(['a', '1'], 'x');
    expect(store.version).toBe('2026.1');
    expect(store.exportEmbeddings().encoder_version).toBe('2026.1');
    store.version = 'explicit';
    expect(store.exportEmbeddings().encoder_version).toBe('explicit');
    const viaOptions = new SqliteEmbeddingStore(':memory:', new MockEncoder(8), false, { version: 'opt' });
    expect(viaOptions.version).toBe('opt');
    expect(new SqliteEmbeddingStore(':memory:', new MockEncoder(8)).version).toBeNull();
    store.close();
    viaOptions.close();
  });

  it('an imported payload with no per-record model reads as unknown', () => {
    const store = open(false, 2);
    store.importEmbeddings({
      format: 'ison-embeddings', version: 1, dimension: 2, count: 1,
      embeddings: [{ type: 'a', id: '1', id_type: 'str', text: 'x', vector: [1, 0] }],
    });
    expect(store.models()).toEqual({ unknown: 1 });
    expect(store.stale()).toEqual([]);                          // unknown is not judged
    expect(store.stale({ includeUnknown: true }).length).toBe(1);
    store.close();
  });

  it('clear keeps the dimension unless asked to forget it', () => {
    for (const vec of [false, true]) {
      const store = new SqliteEmbeddingStore(':memory:', new MockEncoder(4), vec);
      store.add(['a', '1'], 'x');
      store.clear();
      expect(store.count()).toBe(0);
      expect(store.dimension).toBe(4);
      expect(() => store.add(['a', '1'], 'y', [1, 2, 3])).toThrow(DimensionMismatchError);

      store.clear(true);
      expect(store.dimension).toBeNull();
      store.add(['a', '1'], 'y', [1, 2, 3]);                     // a new width is now fine
      expect(store.dimension).toBe(3);
      expect(store.similaritySearch([1, 2, 3], 1, undefined, -1.0).length).toBe(1);
      store.close();
    }
  });

  it('import with replace adopts the payload dimension', () => {
    const wide = new SqliteEmbeddingStore(':memory:', new MockEncoder(8), true);
    wide.add(['a', '1'], 'x');
    const narrow = new SqliteEmbeddingStore(':memory:', new MockEncoder(4), true);
    narrow.add(['b', '1'], 'y');
    expect(wide.importEmbeddings(narrow.exportEmbeddings(), true)).toBe(1);
    expect(wide.dimension).toBe(4);
    expect(wide.count()).toBe(1);
    expect(wide.similaritySearch([1, 0, 0, 0], 1, undefined, -1.0).length).toBe(1);
    wide.close();
    narrow.close();
  });

  it('opens a database whose table lacks the model column', async () => {
    const path = tempDb('legacy.db');
    const { DatabaseSync } = await import('node:sqlite');
    const raw = new DatabaseSync(path);
    raw.exec(`CREATE TABLE embeddings (
        id TEXT PRIMARY KEY, node_type TEXT NOT NULL, node_id TEXT NOT NULL,
        node_id_type TEXT NOT NULL DEFAULT 'str', vector BLOB NOT NULL, text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(node_type, node_id))`);
    raw.prepare('INSERT INTO embeddings (id, node_type, node_id, vector, text) VALUES (?, ?, ?, ?, ?)')
      .run('abc', 'old', '1', new Uint8Array(new Float32Array([1, 0, 0, 0]).buffer), 'legacy row');
    raw.close();

    const store = new SqliteEmbeddingStore(path, new MockEncoder(4));
    expect(store.models()).toEqual({ unknown: 1 });
    expect(store.get(['old', '1'])!.model).toBe('unknown');
    store.add(['new', '1'], 'fresh');
    expect(store.models()).toEqual({ unknown: 1, 'mock-4': 1 });
    store.close();

    // the ALTER is idempotent
    const again = new SqliteEmbeddingStore(path, new MockEncoder(4));
    expect(again.count()).toBe(2);
    again.close();
  });

  it('sees models written through another connection', () => {
    const path = tempDb('shared.db');
    const a = new SqliteEmbeddingStore(path, new MockEncoder(4));
    const b = new SqliteEmbeddingStore(path, retagged(4, 'writer-b'));
    a.add(['a', '1'], 'x');
    expect(a.models()).toEqual({ 'mock-4': 1 });
    b.add(['b', '1'], 'y');
    expect(a.models()).toEqual({ 'mock-4': 1, 'writer-b': 1 });
    a.close();
    b.close();
  });

  it('writes the model column other ports read', async () => {
    const path = tempDb('model-column.db');
    const store = new SqliteEmbeddingStore(path, new MockEncoder(4));
    store.add(['person', '1'], 'Alice');
    store.close();

    const { DatabaseSync } = await import('node:sqlite');
    const db = new DatabaseSync(path);
    expect((db.prepare('SELECT model FROM embeddings').get() as any).model).toBe('mock-4');
    db.close();
  });

  it('works with saveEmbeddings and loadEmbeddings from ./node', async () => {
    const { saveEmbeddings, loadEmbeddings } = await import('./node.js');
    const path = join(tempDb('x'), '..', 'side.embeddings.json');
    const source = open(false, 8);
    source.addBatch([[['a', '1'], 'one'], [['a', '2'], 'two']]);
    expect(saveEmbeddings(source, path, 'digest-1')).toBe(2);
    expect(JSON.parse(readFileSync(path, 'utf8')).graph_digest).toBe('digest-1');

    const target = open(false, 8);
    expect(loadEmbeddings(target, path)).toBe(2);
    expect(target.models()).toEqual({ 'mock-8': 2 });
    expect(target.lastImportMeta.graph_digest).toBe('digest-1');
    source.close();
    target.close();
  });
});
