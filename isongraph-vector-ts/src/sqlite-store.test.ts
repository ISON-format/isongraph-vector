/**
 * SQLite-backed store. Node-only: the main entry point stays runtime-agnostic,
 * so these exercise the separate `/sqlite` entry.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SqliteEmbeddingStore } from './sqlite-store.js';
import { MockEncoder, DimensionMismatchError, NoEncoderError } from './index.js';
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
