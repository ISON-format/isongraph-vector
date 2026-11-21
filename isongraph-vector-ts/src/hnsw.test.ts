/**
 * The opt-in HNSW graph index (`ann: 'hnsw'`, DESIGN.md section 12 of the Rust
 * crate). Mirrors the C# and C++ HnswTests: recall and exactness, scores equal
 * to the exact scan's, filters and the exact fallbacks, upkeep under writes,
 * deletes, updates and compaction, the rebuild rule, determinism, that nothing
 * about the graph reaches a saved file, and the Rust reference answers.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  EmbeddingStore, SemanticGraph, MockEncoder, EmbeddingError, graphDigest,
  SplitMix64, hnswLevelSequence,
  type NodeRef,
  type SimilarityResult,
} from './index.js';
import { saveGraph, loadGraph, embeddingsPathFor, snapshotPathFor } from './node.js';

// A build of a few thousand rows in plain JS takes seconds, and several stores are built per test.
vi.setConfig({ testTimeout: 180_000 });

/** Small seeded generator, so a failure reproduces. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(next: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
}

/** Tight clusters (like real embeddings) or uniform noise, and queries from the same distribution. */
function makeData(n: number, dim: number, clusteredData: boolean, seed = 1) {
  const next = rng(seed);
  const clusters = 30;
  const centers = Array.from({ length: clusters }, () => {
    const v = Array.from({ length: dim }, () => gaussian(next));
    const norm = Math.hypot(...v);
    return v.map(x => x / norm);
  });
  const draw = (count: number): number[][] =>
    Array.from({ length: count }, () => {
      if (!clusteredData) return Array.from({ length: dim }, () => gaussian(next));
      const center = centers[Math.floor(next() * clusters)];
      return center.map(x => x + (0.6 * gaussian(next)) / Math.sqrt(dim));
    });
  return { data: draw(n), draw };
}

type Opts = Record<string, any>;

function hnswStore(
  data: number[][],
  { types, minSize = 0, seed = 0, ann = 'hnsw', annHnsw, ...rest }:
    { types?: (i: number) => string; minSize?: number; seed?: number; ann?: 'hnsw' | 'ivf' | null; annHnsw?: Opts } & Opts = {}
): EmbeddingStore {
  const store = new EmbeddingStore(undefined, { ann, annMinSize: minSize, annSeed: seed, annHnsw, ...rest });
  data.forEach((v, i) => store.add([types ? types(i) : 'doc', i] as NodeRef, `text ${i}`, v));
  return store;
}

const key = (r: { nodeRef: NodeRef }) => `${r.nodeRef[0]}:${r.nodeRef[1]}`;
const shape = (hits: SimilarityResult[]) => hits.map(h => [key(h), h.score]);
const ids = (hits: SimilarityResult[]) => hits.map(h => Number(h.nodeRef[1]));

/** The graph and its table (null before the first build). */
function inspect(store: EmbeddingStore) {
  const table = store.snapshotState().table;
  return { table, graph: table === null ? null : table.hnsw };
}

/** DESIGN.md 12.6 invariants against the table; null when they hold. */
function verify(store: EmbeddingStore): string | null {
  const { table, graph } = inspect(store);
  if (table === null) return graph === null ? null : 'graph without a table';
  return graph === null ? null : graph.verify(table);
}

/** A returned hit's score is the exact scan's score for that id, bit for bit (a double here). */
function expectScoresExact(store: EmbeddingStore, q: number[], hits: SimilarityResult[]): void {
  const map = store.scoreMap(q);
  for (const h of hits) expect(Object.is(h.score, map.get(key(h)))).toBe(true);
}

/** Best first, equal scores in slot (here: insertion = numeric id) order. */
function expectOrdered(hits: SimilarityResult[]): void {
  for (let i = 1; i < hits.length; i++) {
    expect(hits[i - 1].score).toBeGreaterThanOrEqual(hits[i].score);
    if (hits[i - 1].score === hits[i].score) expect(Number(hits[i - 1].nodeRef[1])).toBeLessThan(Number(hits[i].nodeRef[1]));
  }
}

const recall = (store: EmbeddingStore, queries: number[][], k = 10) => store.recallCheck(queries, k).recall;

// -- the portable pieces -------------------------------------------------------------------------

/** SplitMix64 and the high-64-bit `below`, as BigInt arithmetic: an implementation independent of the limb code. */
function splitMixRef(seed: bigint) {
  const M = (1n << 64n) - 1n;
  let s = seed & M;
  const next = (): bigint => {
    s = (s + 0x9e3779b97f4a7c15n) & M;
    let z = s;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M;
    return z ^ (z >> 31n);
  };
  return { next, below: (n: number): number => Number((next() * BigInt(n)) >> 64n) };
}

describe('hnsw: portable rules', () => {
  it('SplitMix64 reproduces the published reference values', () => {
    // seed 0: the sequence every SplitMix64 implementation gives
    const g = new SplitMix64(0);
    const out: bigint[] = [];
    for (let i = 0; i < 4; i++) {
      g.next();
      out.push(g.value());
    }
    expect(out).toEqual([0xe220a8397b1dcdafn, 0x6e789e6aa1b965f4n, 0x06c45d188009454fn, 0xf88bb8a8724c81ecn]);
    // seed 12345 ^ the level generator's constant (computed with Python big integers)
    const h = new SplitMix64(0x4857535748575357n ^ 12345n);
    h.next();
    expect(h.value()).toBe(0xdcb512bcc80864d8n);
    h.next();
    expect(h.value()).toBe(0xef16fbaf1b89a77fn);
  });

  it('SplitMix64 and below(n) agree with BigInt arithmetic over many draws, seeds and n', () => {
    for (const seed of [0n, 1n, 0xffffffffn, 0x100000000n, 0x4857535748575357n, 0xffffffffffffffffn, 0x8000000000000000n]) {
      const a = new SplitMix64(seed);
      const ref = splitMixRef(seed);
      for (let i = 0; i < 4000; i++) {
        a.next();
        expect(a.value()).toBe(ref.next());
      }
      const b = new SplitMix64(seed);
      const refB = splitMixRef(seed);
      for (let i = 0; i < 3000; i++) {
        const n = [1, 2, 3, 16, 64, 2001, 3001, 1 << 20, (1 << 21) - 1, 1 << 21, 1 << 30, 4294967295][i % 12];
        expect(b.below(n)).toBe(refB.below(n));
      }
    }
    // a negative or fractional number seed wraps like an unsigned cast of the truncated value
    expect(new SplitMix64(-1).value()).toBe(new SplitMix64(0xffffffffffffffffn).value());
  });

  it('the level draw matches an independent SplitMix64 reference', () => {
    // Expected values: a separate implementation (Python big integers), the same ones the C# and C++ tests carry.
    const m16 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    expect(hnswLevelSequence(0, 16, 64)).toEqual(m16);
    const m4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0];
    expect(hnswLevelSequence(0, 4, 64)).toEqual(m4);
    const m8 = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
    expect(hnswLevelSequence(12345, 8, 64)).toEqual(m8);
  });

  it('levels follow the geometric law P(level >= l) = m^-l, and the seed matters', () => {
    const levels = hnswLevelSequence(0, 16, 160_000);
    const p1 = levels.filter(l => l >= 1).length / 160_000;
    const p2 = levels.filter(l => l >= 2).length / 160_000;
    expect(Math.abs(p1 - 1 / 16)).toBeLessThan(0.004);
    expect(Math.abs(p2 - 1 / 256)).toBeLessThan(0.001);
    expect(levels.reduce((a, b) => Math.max(a, b), 0)).toBeLessThanOrEqual(24);
    expect(hnswLevelSequence(3, 16, 500)).toEqual(hnswLevelSequence(3, 16, 500));
    expect(hnswLevelSequence(3, 16, 500)).not.toEqual(hnswLevelSequence(4, 16, 500));
    expect(hnswLevelSequence(-3, 16, 500)).not.toEqual(hnswLevelSequence(3, 16, 500));
  });
});

// -- configuration ---------------------------------------------------------------------------------

describe('hnsw: configuration', () => {
  it('validates its options, and ef is adjustable without a rebuild', () => {
    expect(() => new EmbeddingStore(undefined, { ann: 'nope' as any })).toThrow(EmbeddingError);
    expect(() => new EmbeddingStore(undefined, { ann: 'hnsw', annHnsw: { m: 1 } })).toThrow(/annHnsw\.m/);
    expect(() => new EmbeddingStore(undefined, { ann: 'hnsw', annHnsw: { m: 65 } })).toThrow(/annHnsw\.m/);
    expect(() => new EmbeddingStore(undefined, { ann: 'hnsw', annHnsw: { m: 2.5 } })).toThrow(/annHnsw\.m/);
    expect(() => new EmbeddingStore(undefined, { ann: 'hnsw', annHnsw: { efConstruction: 0 } })).toThrow(/efConstruction/);
    expect(() => new EmbeddingStore(undefined, { ann: 'hnsw', annHnsw: { efSearch: 0 } })).toThrow(/efSearch/);

    const store = new EmbeddingStore(undefined, { ann: 'hnsw', annMinSize: 0, annHnsw: { m: 8, efSearch: 33 } });
    expect(store.ann).toBe('hnsw');
    expect(store.annEfSearch).toBe(33);
    expect(store.annStats().efSearch).toBe(33);
    store.setAnnEfSearch(null);
    expect(store.annEfSearch).toBeNull();
    expect(store.annStats().efSearch).toBe(64);                      // automatic, by store size
    expect(() => store.setAnnEfSearch(0)).toThrow(EmbeddingError);
    expect(() => { store.annEfSearch = 1.5; }).toThrow(EmbeddingError);
    store.annEfSearch = 70;
    expect(store.annStats().efSearch).toBe(70);

    // an options object given to another mode is ignored
    expect(new EmbeddingStore(undefined, { ann: 'ivf', annHnsw: { m: 8 } }).ann).toBe('ivf');
    expect(() => new EmbeddingStore().buildAnnIndex()).toThrow(/ivf' or 'hnsw/);
    const { data } = makeData(50, 8, true);
    const small = hnswStore(data);
    expect(() => small.similaritySearchEf(data[0], 5, undefined, 0, 0)).toThrow(EmbeddingError);
  });

  it('chooses ef by store size: 64 under 10k, 96 under 200k, 128 under 600k, 192 above', () => {
    const store = new EmbeddingStore(undefined, { ann: 'hnsw', annMinSize: 0 });
    const at = (n: number) => {
      for (let i = store.count(); i < n; i++) store.add(['d', i], '', [1, i % 7, i % 5]);
      return store.annStats().efSearch;
    };
    expect(at(9_999)).toBe(64);
    expect(at(10_000)).toBe(96);
    expect(at(199_999)).toBe(96);
    expect(at(200_000)).toBe(128);
    expect(at(599_999)).toBe(128);
    expect(at(600_000)).toBe(192);
    expect(store.annStats().active).toBe(false);                     // adding rows built nothing
  });

  it('works from the browser-safe entry: index imports nothing from node', () => {
    const ext = import.meta.url.endsWith('.ts') ? 'ts' : 'js';          // this file's own flavour of index
    const source = readFileSync(new URL(`./index.${ext}`, import.meta.url), 'utf8');
    expect(source).not.toMatch(/from\s+['"]node:/);
    expect(source).not.toMatch(/require\(|process\.|Buffer\b/);
  });
});

// -- recall and exactness ----------------------------------------------------------------------------

describe('hnsw: recall and exactness', () => {
  for (const clusteredData of [true, false]) {
    it(`recall against exact is high and scores equal the exact scan's (${clusteredData ? 'clustered' : 'random'})`, () => {
      const n = 3000;
      const dim = 24;
      const { data, draw } = makeData(n, dim, clusteredData);
      const store = hnswStore(data);
      const queries = draw(80);
      const report = store.recallCheck(queries, 10);
      expect(report.approximate).toBe(true);
      expect(report.nlist).toBe(0);
      expect(report.efSearch).toBe(64);
      expect(report.recall).toBeGreaterThanOrEqual(clusteredData ? 0.97 : 0.9);

      for (const q of queries.slice(0, 40)) {
        const hits = store.similaritySearch(q, 10);
        expect(hits).toHaveLength(10);
        expectScoresExact(store, q, hits);
        expectOrdered(hits);
        expect(hits.every(h => typeof h.text === 'string')).toBe(true);
      }
      const stats = store.annStats();
      expect(stats.mode).toBe('hnsw');
      expect(stats.active).toBe(true);
      expect(stats.graphNodes).toBe(n);
      expect(stats.trainedN).toBe(n);
      expect(stats.nlist).toBeNull();
      expect(stats.nprobe).toBeNull();
      expect(stats.pending).toBe(0);
      expect(stats.searches.builds).toBe(1);
      expect(stats.maxLevel).toBeGreaterThanOrEqual(1);
      expect(verify(store)).toBeNull();
    });
  }

  it('a higher ef does not lower recall, and the per-call ef equals the configured one', () => {
    const { data, draw } = makeData(2500, 24, false);
    const store = hnswStore(data, { annHnsw: { m: 8, efConstruction: 60 } });
    const queries = draw(60);
    store.annEfSearch = 12;
    const low = recall(store, queries);
    store.annEfSearch = 400;
    const high = recall(store, queries);
    expect(high).toBeGreaterThanOrEqual(low);
    expect(high).toBeGreaterThan(0.95);
    expect(low).toBeLessThan(high);                                // a tiny ef is visibly worse on noise

    store.annEfSearch = 12;
    for (const q of queries.slice(0, 20)) {
      const viaCall = store.similaritySearchEf(q, 10, undefined, 0, 400);
      store.annEfSearch = 400;
      const viaConfig = store.similaritySearch(q, 10);
      store.annEfSearch = 12;
      expect(shape(viaCall)).toEqual(shape(viaConfig));
    }
  });

  it('answers exactly below minSize and when topK covers the store', () => {
    const { data, draw } = makeData(500, 16, true);
    const queries = draw(10);

    const belowMin = hnswStore(data, { minSize: 1000 });             // 500 live < 1000: no graph
    for (const q of queries) {
      expect(shape(belowMin.similaritySearch(q, 10))).toEqual(shape(belowMin.similaritySearch(q, 10, undefined, 0, true)));
    }
    expect(belowMin.annStats().active).toBe(false);
    expect(belowMin.recallCheck(queries).approximate).toBe(false);

    const store = hnswStore(data, { minSize: 0 });
    for (const q of queries) {
      for (const k of [500, 501, 100000]) {
        expect(shape(store.similaritySearch(q, k))).toEqual(shape(store.similaritySearch(q, k, undefined, 0, true)));
      }
    }
    expect(store.annStats().searches.approximate).toBe(0);
    expect(shape(store.similaritySearch(queries[0], null))).toEqual(shape(store.similaritySearch(queries[0], null, undefined, 0, true)));

    const before = store.annStats().searches.approximate;
    store.similaritySearch(queries[0], 10);
    expect(store.annStats().searches.approximate).toBe(before + 1);
    store.similaritySearch(queries[0], 10, undefined, 0, true);
    expect(store.annStats().searches.approximate).toBe(before + 1);
    // the default minimum is 50000
    expect(new EmbeddingStore(undefined, { ann: 'hnsw' }).annStats().minSize).toBe(50000);
  });

  it('duplicates, ties and zero vectors keep the scan order and the exact scores', () => {
    const next = rng(9);
    const dim = 16;
    const base = makeData(300, dim, true, 3).data;
    // every row appears three times (equal scores), plus zero rows (norm 0: live and finite)
    const rows: number[][] = [];
    for (let i = 0; i < 700; i++) rows.push(i % 50 === 49 ? new Array(dim).fill(0) : base[i % 300]);
    const store = hnswStore(rows);
    for (let t = 0; t < 40; t++) {
      const q = Array.from({ length: dim }, () => gaussian(next));
      const hits = store.similaritySearch(q, 25);
      expectScoresExact(store, q, hits);
      expectOrdered(hits);
    }
    // a duplicate of a stored row: all copies come back with the same score, lower slot first
    const top = store.similaritySearch(rows[0], 5);
    expect(top[0].score).toBeCloseTo(1, 4);
    expect(ids(top.slice(0, 2))).toEqual([0, 300]);
    expect(verify(store)).toBeNull();
    // a zero query has no direction: the scan answers it
    const approx = store.annStats().searches.approximate;
    const zero = store.similaritySearch(new Array(dim).fill(0), 5);
    expect(shape(zero)).toEqual(shape(store.similaritySearch(new Array(dim).fill(0), 5, undefined, 0, true)));
    expect(store.annStats().searches.approximate).toBe(approx);
  });
});

// -- filters --------------------------------------------------------------------------------------------

describe('hnsw: filters', () => {
  it('type filter and threshold, including selective ones that fall back to the scan', () => {
    const n = 2500;
    const dim = 24;
    const { data, draw } = makeData(n, dim, true, 4);
    // 1 in 100 is "rare" (25 rows), 5 rows are "tiny" (fewer than k), the rest "doc"
    const typeOf = (i: number) => (i % 500 === 5 ? 'tiny' : i % 100 === 0 ? 'rare' : 'doc');
    const store = hnswStore(data, { types: typeOf });
    const queries = draw(30);

    for (const q of queries) {
      const docs = store.similaritySearch(q, 10, 'doc', -1);
      expect(docs).toHaveLength(10);
      expect(docs.every(r => r.nodeRef[0] === 'doc')).toBe(true);
      expectScoresExact(store, q, docs);

      // selective: ef widens, possibly all the way to the scan; always k results of that type
      const rare = store.similaritySearch(q, 10, 'rare', -1);
      expect(rare).toHaveLength(10);
      expect(rare.every(r => r.nodeRef[0] === 'rare')).toBe(true);
      expectScoresExact(store, q, rare);
      expectOrdered(rare);

      // fewer matches than k: nothing to widen to but the scan, which answers exactly
      const tiny = store.similaritySearch(q, 10, 'tiny', -1);
      expect(shape(tiny)).toEqual(shape(store.similaritySearch(q, 10, 'tiny', -1, true)));
      expect(tiny).toHaveLength(5);

      // a type nobody has
      expect(store.similaritySearch(q, 10, 'nope')).toEqual([]);
    }
    const recallRare = queries.map(q => {
      const a = new Set(store.similaritySearch(q, 10, 'rare', -1).map(key));
      const e = store.similaritySearch(q, 10, 'rare', -1, true).map(key);
      return e.filter(x => a.has(x)).length / 10;
    });
    expect(recallRare.reduce((x, y) => x + y, 0) / recallRare.length).toBeGreaterThanOrEqual(0.8);
    // "tiny" has fewer than k rows: the search can never fill k, so every one ended in the scan
    expect(store.annStats().searches.exact).toBeGreaterThanOrEqual(queries.length);

    // threshold: every answer passes it, scores exact; one nobody reaches is empty (the scan says so)
    for (const q of queries.slice(0, 10)) {
      const hits = store.similaritySearch(q, 10, undefined, 0.6);
      expect(hits.every(r => r.score >= 0.6)).toBe(true);
      expectScoresExact(store, q, hits);
      expect(store.similaritySearch(q, 10, undefined, 1.5)).toEqual([]);
    }
    // type and threshold together
    const both = store.similaritySearch(queries[0], 5, 'rare', 0.1);
    expect(both.every(r => r.nodeRef[0] === 'rare' && r.score >= 0.1)).toBe(true);
  });

  describe('in a SemanticGraph', () => {
    function graph(n = 1500, dim = 24, extra: Opts = {}) {
      const { data, draw } = makeData(n, dim, true, 5);
      const g = new SemanticGraph({
        name: 'hnsw', encoder: new MockEncoder(dim), autoEmbed: false, ann: 'hnsw', annMinSize: 0, ...extra,
      });
      for (let i = 0; i < n; i++) g.addNode('item', i, { group: i % 50 });
      g.embeddingStore.addBatch(data.map((v, i) => [['item', i] as NodeRef, `item ${i}`, v] as [NodeRef, string, number[]]));
      return { g, data, draw };
    }

    it('passes the options to its store, and a selective nodeFilter still gets topK results', () => {
      const { g, draw } = graph(1500, 24, { annHnsw: { m: 12, efConstruction: 80, efSearch: 50 } });
      const store = g.embeddingStore;
      expect(store.ann).toBe('hnsw');
      expect(store.annEfSearch).toBe(50);
      const q = draw(1)[0];
      expect(g.similaritySearch(q, 10)).toHaveLength(10);
      expect(store.annStats().active).toBe(true);
      expect(g.recallCheck(draw(20)).recall).toBeGreaterThanOrEqual(0.9);

      const group7 = (n: any) => n.properties.group === 7;          // 1 node in 50
      const res = g.similaritySearch(q, 10, undefined, 0, group7);
      expect(res).toHaveLength(10);
      expect(res.every(r => g.getNodeByRef(r.nodeRef).properties.group === 7)).toBe(true);
      const want = g.similaritySearch(q, 10, undefined, 0, group7, undefined, true);
      expect(res.filter(r => want.some(w => key(w) === key(r))).length).toBeGreaterThanOrEqual(7);
      const scores = res.map(r => r.score);
      expect(scores).toEqual([...scores].sort((a, b) => b - a));
      expect(store.annStats().searches.widened).toBeGreaterThan(0);          // a gate passing 1 in 50 widens ef
      expectScoresExact(store, q, res);

      // nobody passes: the widening ends in the scan and the answer is empty
      expect(g.similaritySearch(q, 10, undefined, 0, () => false)).toEqual([]);
      // few pass (3 nodes): fewer than k exist, the exact scan answers
      const few = g.similaritySearch(q, 10, undefined, -1, (n: any) => n.properties.group === 7 && Number(n.id) < 150);
      expect(few).toHaveLength(3);
      const even = (n: any) => n.properties.group % 2 === 0;
      const got = g.similaritySearch(q, 10, undefined, 0, even);
      const ref = g.similaritySearch(q, 10, undefined, 0, even, undefined, true);
      expect(got.filter(r => ref.some(w => key(w) === key(r))).length).toBeGreaterThanOrEqual(8);
      // nodeTypes goes through the same gate
      expect(g.similaritySearch(q, 5, undefined, 0, undefined, ['item'])).toHaveLength(5);
      expect(g.similaritySearch(q, 5, undefined, 0, undefined, ['other'])).toEqual([]);

      const before = store.annStats().searches.approximate;
      g.similaritySearch(q, 5, undefined, 0, undefined, undefined, true);
      g.similaritySearch(q, 5, undefined, 0, () => true, undefined, true);
      expect(store.annStats().searches.approximate).toBe(before);
      const near = g.similarToNode(['item', 3], 5);
      expect(near).toHaveLength(5);
      expect(near.map(key)).not.toContain('item:3');
    });

    it('semanticMultiHop takes its seeds from the graph, and async text search honours exact', async () => {
      const { g, draw } = graph(1200);
      for (let i = 0; i < 1199; i++) g.addEdge('NEXT', ['item', i], ['item', i + 1]);
      const out = g.semanticMultiHop(draw(1)[0], { relType: 'NEXT', maxHops: 2, topKSeeds: 3, threshold: -1 });
      expect(out.length).toBeGreaterThan(0);
      expect(out[0].hopCount).toBe(0);
      expect(g.embeddingStore.annStats().searches.approximate).toBeGreaterThanOrEqual(1);

      const encoder = new MockEncoder(32);
      const t = new SemanticGraph({ encoder, ann: 'hnsw', annMinSize: 0 });
      for (let i = 0; i < 400; i++) t.addNode('doc', i, { name: `entry number ${i} about topic ${i % 9}` });
      const before = t.embeddingStore.annStats().searches.approximate;
      const hits = await t.similaritySearchAsync('entry number 12 about topic 3', 3);
      expect(hits).toHaveLength(3);
      const used = t.embeddingStore.annStats().searches.approximate;
      expect(used).toBe(before + 1);
      await t.similaritySearchAsync('entry number 12 about topic 3', 3, undefined, 0, undefined, undefined, true);
      expect(t.embeddingStore.annStats().searches.approximate).toBe(used);
    });
  });
});

// -- writes, deletes, updates, compaction -------------------------------------------------------------------

describe('hnsw: writes, deletes, updates and compaction', () => {
  it('inserts are linked by the write and found at once', () => {
    const dim = 24;
    const { data, draw } = makeData(2000, dim, true, 6);
    const store = hnswStore(data);
    store.buildAnnIndex();
    expect(store.annStats().searches.builds).toBe(1);

    const extra = draw(250);
    extra.forEach((v, i) => {
      store.add(['doc', 2000 + i], 'new', v);
      if (i % 50 === 49) expect(verify(store)).toBeNull();
    });
    const stats = store.annStats();
    expect(stats.searches.builds).toBe(1);                         // inserts never rebuild
    expect(stats.graphNodes).toBe(2250);
    expect(stats.inserts).toBe(250);
    expect(stats.churn).toBe(0);                                   // inserts are not churn
    let self = 0;
    extra.forEach((v, i) => {
      if (store.similaritySearch(v, 1)[0].nodeRef[1] === 2000 + i) self++;
    });
    expect(self).toBeGreaterThanOrEqual(240);
    expect(verify(store)).toBeNull();
  });

  it('deletes tombstone, compaction renumbers and bridges, and no dead id is ever returned', () => {
    const n = 2400;
    const { data, draw } = makeData(n, 24, true, 8);
    const store = hnswStore(data);
    const queries = draw(50);
    store.buildAnnIndex();
    expect(recall(store, queries)).toBeGreaterThanOrEqual(0.95);

    const dead = new Set<number>();
    const next = rng(2);
    const kill = (until: number) => {
      while (dead.size < until) {
        const id = Math.floor(next() * n);
        if (!dead.has(id)) {
          dead.add(id);
          expect(store.remove(['doc', id])).toBe(true);
        }
      }
    };
    // below the 25% compaction line the node stays as a tombstone
    kill(300);
    let stats = store.annStats();
    expect(stats.graphNodes).toBe(n);
    expect(stats.graphDead).toBe(300);
    expect(verify(store)).toBeNull();
    for (const q of queries) for (const r of store.similaritySearch(q, 10)) expect(dead.has(Number(r.nodeRef[1]))).toBe(false);
    expect(recall(store, queries)).toBeGreaterThanOrEqual(0.93);

    // past 25% dead the table compacts and the graph follows (tombstones leave, links are renumbered)
    kill(750);
    expect(verify(store)).toBeNull();
    stats = store.annStats();
    // the 601st removal crossed 25% dead (601 * 4 > 2400): the table compacted and the graph followed
    expect(stats.graphNodes).toBe(n - 601);
    expect(stats.graphDead).toBe(dead.size - 601);
    expect(store.remove(['doc', [...dead][0]])).toBe(false);
    for (const q of queries) {
      const hits = store.similaritySearch(q, 10);
      expect(hits).toHaveLength(10);
      expect(hits.every(r => !dead.has(Number(r.nodeRef[1])))).toBe(true);
      expectScoresExact(store, q, hits);
    }
    expect(recall(store, queries)).toBeGreaterThanOrEqual(0.93);
    expect(store.annStats().searches.builds).toBe(1);               // bridging, not a rebuild
  });

  it('compaction bridges over tombstones so the graph stays connected and dense', () => {
    const n = 3000;
    const { data } = makeData(n, 12, true, 11);
    const opts = { annHnsw: { m: 12, efConstruction: 100 } };
    const store = hnswStore(data, opts);
    store.buildAnnIndex();
    for (let i = 0; i < n; i += 3) store.remove(['doc', i]);       // one in three: compacts on the way
    expect(verify(store)).toBeNull();
    expect(store.count()).toBe(2000);

    const measure = (s: EmbeddingStore) => {
      const { table, graph } = inspect(s);
      const t = table!;
      const g = graph!;
      const seen = new Uint8Array(g.slots);
      const stack = [g.entry];
      seen[g.entry] = 1;
      let reach = 0;
      let degree = 0;
      while (stack.length > 0) {
        const cur = stack.pop()!;
        if (cur < t.n && t.keys[cur] !== null) reach++;
        const links = g.linksOf(cur, 0);
        degree += links.length;
        for (const nb of links) if (!seen[nb]) { seen[nb] = 1; stack.push(nb); }
      }
      return { reach, degree: degree / t.n };
    };
    const { reach, degree } = measure(store);
    expect(reach * 100).toBeGreaterThanOrEqual(2000 * 99);          // 99% of the live rows reachable after compaction

    // lists were refilled from the tombstones' links: degree stays near that of a graph built from scratch
    const fresh = hnswStore(data.filter((_, i) => i % 3 !== 0), opts);
    fresh.buildAnnIndex();
    expect(degree).toBeGreaterThanOrEqual(0.85 * measure(fresh).degree);
  });

  it('deleting the newest rows leaves nothing dead behind when later writes reuse their slots', () => {
    // The table pops trailing tombstones, so their graph nodes outlive their slots (as nodes with a NaN norm,
    // routed through but never returned) until a new row takes the slot, which relinks the node as an update.
    const { data, draw } = makeData(1500, 16, true, 22);
    const store = hnswStore(data);
    store.buildAnnIndex();
    const gone = new Set<number>();
    for (let i = 1499; i >= 1400; i--) {
      expect(store.remove(['doc', i])).toBe(true);
      gone.add(i);
    }
    expect(store.count()).toBe(1400);
    expect(verify(store)).toBeNull();
    expect(inspect(store).graph!.slots).toBe(1500);                // the popped slots still have nodes
    // enough deletes at the front to compact the table: the graph follows, and the popped slots' nodes leave with it
    for (let i = 0; i < 450; i++) {
      expect(store.remove(['doc', i])).toBe(true);
      gone.add(i);
    }
    expect(verify(store)).toBeNull();
    expect(inspect(store).graph!.slots).toBe(inspect(store).table!.n);
    expect(store.annStats().graphDead).toBeLessThan(150);
    const more = draw(1700);
    more.slice(0, 1000).forEach((v, i) => store.add(['doc', 5000 + i], 'new', v));
    for (const q of draw(30)) for (const r of store.similaritySearch(q, 10)) expect(gone.has(Number(r.nodeRef[1]))).toBe(false);
    expect(verify(store)).toBeNull();
    more.slice(1000).forEach((v, i) => store.add(['doc', 6000 + i], 'new', v));
    expect(verify(store)).toBeNull();
    for (const q of draw(30)) {
      const hits = store.similaritySearch(q, 10);
      expect(hits).toHaveLength(10);
      expect(hits.every(r => !gone.has(Number(r.nodeRef[1])))).toBe(true);
      expectScoresExact(store, q, hits);
    }
    // a reused slot is an update of the old node: it is found at its new vector
    const probe = store.similaritySearch(more[1699], 1);
    expect(probe[0].nodeRef).toEqual(['doc', 6000 + 699]);
  });

  it('updates relink the node and find it at its new position', () => {
    const n = 2000;
    const dim = 24;
    const { data } = makeData(n, dim, true, 12);
    const store = hnswStore(data);
    store.buildAnnIndex();
    const moved = makeData(300, dim, false, 99).data;
    moved.forEach((v, i) => store.add(['doc', i * 5], 'moved', v));   // same key: update in place
    const stats = store.annStats();
    expect(stats.churn).toBe(300);
    expect(stats.graphNodes).toBe(n);
    expect(stats.searches.builds).toBe(1);
    expect(verify(store)).toBeNull();
    let found = 0;
    moved.forEach((v, i) => {
      const top = store.similaritySearch(v, 1)[0];
      if (top.nodeRef[1] === i * 5 && top.score > 0.9999) found++;
    });
    expect(found).toBeGreaterThanOrEqual(285);
    // scores are of the current vector, never the old position
    const q = makeData(1, dim, true, 12).data[0];
    expectScoresExact(store, q, store.similaritySearch(q, 10));
  });

  it('heavy churn marks the graph stale and the next search rebuilds it', () => {
    const dim = 12;
    const { data } = makeData(800, dim, true, 13);
    const other = makeData(800, dim, false, 14).data;
    const store = hnswStore(data);
    store.buildAnnIndex();
    for (let i = 0; i < 799; i++) store.add(['doc', i], 'u', other[i]);   // 799 updates: not yet more than the node count
    expect(store.annStats().searches.builds).toBe(1);
    store.similaritySearch(other[0], 5);
    expect(store.annStats().searches.builds).toBe(1);
    for (let i = 0; i < 5; i++) store.add(['doc', i], 'u', data[i]);      // 804 > 800
    expect(store.annStats().searches.builds).toBe(1);                      // lazily: nothing happens at write time
    const hits = store.similaritySearch(data[0], 5);
    const stats = store.annStats();
    expect(stats.searches.builds).toBe(2);
    expect(stats.churn).toBe(0);
    expect(stats.graphNodes).toBe(800);
    expect(hits).toHaveLength(5);
    expect(verify(store)).toBeNull();

    // deletes count too: remove most of the rows, compaction keeps the churn, a rebuild follows
    for (let i = 0; i < 560; i++) store.remove(['doc', i]);
    expect(store.annStats().churn).toBeGreaterThanOrEqual(240);
    expect(store.similaritySearch(other[799], 5)).toHaveLength(5);
    expect(store.annStats().searches.builds).toBe(3);
    expect(verify(store)).toBeNull();
    // buildAnnIndex(true) rebuilds on demand
    store.buildAnnIndex(true);
    expect(store.annStats().searches.builds).toBe(4);
  });

  it('is dropped below minSize and by clear, and rebuilt later', () => {
    const { data } = makeData(500, 12, true, 15);
    const store = hnswStore(data, { minSize: 400 });
    store.buildAnnIndex();
    expect(store.annStats().active).toBe(true);
    for (let i = 0; i < 150; i++) store.remove(['doc', i]);                // 350 live < 400
    expect(store.annStats().active).toBe(false);
    const q = data[300];
    expect(shape(store.similaritySearch(q, 10))).toEqual(shape(store.similaritySearch(q, 10, undefined, 0, true)));
    expect(verify(store)).toBeNull();

    for (let i = 0; i < 100; i++) store.add(['doc', 1000 + i], 'new', data[i]);
    store.similaritySearch(q, 10);
    expect(store.annStats().active).toBe(true);
    store.clear();
    expect(store.annStats().active).toBe(false);
    expect(store.similaritySearch(q, 10)).toEqual([]);
    expect(verify(store)).toBeNull();
  });

  it('keeps every invariant through random writes, updates, deletes and compaction', () => {
    const store = new EmbeddingStore(undefined, { ann: 'hnsw', annMinSize: 200, annSeed: 2, annHnsw: { m: 6, efConstruction: 40 } });
    const next = rng(5);
    const live = new Map<string, number[]>();
    const dim = 16;
    const vec = () => Array.from({ length: dim }, () => gaussian(next));
    for (let step = 0; step < 2500; step++) {
      const id = Math.floor(next() * 900);
      const ref: NodeRef = [id % 2 === 0 ? 'x' : 'y', id];
      const k = `${ref[0]}:${ref[1]}`;
      if (next() < 0.35) {
        expect(store.remove(ref)).toBe(live.delete(k));
      } else {
        const v = vec();
        store.add(ref, 't', v);
        live.set(k, v);
      }
      if (step % 9 === 0) {
        const q = vec();
        const hits = store.similaritySearch(q, 8, undefined, -1);
        expect(hits).toHaveLength(Math.min(8, live.size));
        expect(hits.every(h => live.has(key(h)))).toBe(true);
        expectScoresExact(store, q, hits);
      }
      if (step % 100 === 0) expect(verify(store)).toBeNull();
    }
    expect(verify(store)).toBeNull();
    expect(store.count()).toBe(live.size);
    expect(store.annStats().searches.approximate).toBeGreaterThan(0);
  });
});

// -- determinism -----------------------------------------------------------------------------------------------

describe('hnsw: determinism', () => {
  it('the graph depends on the seed, the data and the insertion order, and nothing else', () => {
    const { data } = makeData(1500, 12, true, 16);
    const hash = (seed = 0) => {
      const s = hnswStore(data, { seed });
      s.buildAnnIndex();
      expect(verify(s)).toBeNull();
      return inspect(s).graph!.structureHash();
    };
    const one = hash();
    expect(hash()).toBe(one);
    expect(hash(1)).not.toBe(one);

    // writes after the build use the same routine and are deterministic too
    const more = makeData(200, 12, false, 17).data;
    const withWrites = () => {
      const s = hnswStore(data);
      s.buildAnnIndex();
      more.forEach((v, i) => s.add(['doc', i % 2 === 0 ? 5000 + i : i], 'w', v));       // inserts and updates
      return inspect(s).graph!.structureHash();
    };
    expect(withWrites()).toBe(withWrites());
    // a graph built by the lazy first search is the one built by buildAnnIndex
    const lazy = hnswStore(data);
    lazy.similaritySearch(data[0], 5);
    expect(inspect(lazy).graph!.structureHash()).toBe(one);
  });

  it('repeated searches, which reuse the scratch buffers, give identical answers', () => {
    const { data, draw } = makeData(1500, 16, true, 18);
    const store = hnswStore(data);
    const queries = draw(20);
    const first = queries.map(q => shape(store.similaritySearch(q, 10)));
    for (let round = 0; round < 3; round++) {
      queries.forEach((q, i) => expect(shape(store.similaritySearch(q, 10))).toEqual(first[i]));
    }
  });
});

// -- persistence: nothing changes ---------------------------------------------------------------------------------

describe('hnsw: files', () => {
  const dirs: string[] = [];
  afterEach(() => { while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true }); });
  const tempDir = () => {
    const dir = mkdtempSync(join(tmpdir(), 'isongraph-vector-hnsw-'));
    dirs.push(dir);
    return dir;
  };

  function build(ann: 'hnsw' | null, n = 1200, dim = 32) {
    const { data } = makeData(n, dim, true, 31);
    const g = new SemanticGraph({ name: 'hnsw', encoder: new MockEncoder(dim), autoEmbed: false, ann, annMinSize: 0 });
    for (let i = 0; i < n; i++) g.addNode('item', i, { group: i % 50 });
    g.embeddingStore.addBatch(data.map((v, i) => [['item', i] as NodeRef, `item ${i}`, v] as [NodeRef, string, number[]]));
    return g;
  }

  it('the graph file, sidecar, digest and binary snapshot are byte-identical with hnsw on or off', () => {
    const dir = tempDir();
    const plain = build(null);
    const hnsw = build('hnsw');
    hnsw.embeddingStore.buildAnnIndex();
    expect(hnsw.embeddingStore.annStats().active).toBe(true);
    hnsw.embeddingStore.similaritySearch(Array.from({ length: 32 }, (_, i) => i + 1), 5);

    const pa = join(dir, 'plain.ison');
    const pb = join(dir, 'hnsw.ison');
    saveGraph(plain, pa, { snapshot: true });
    saveGraph(hnsw, pb, { snapshot: true });
    expect(readFileSync(pb).equals(readFileSync(pa))).toBe(true);
    expect(readFileSync(embeddingsPathFor(pb)).equals(readFileSync(embeddingsPathFor(pa)))).toBe(true);
    expect(existsSync(snapshotPathFor(pb))).toBe(true);
    expect(readFileSync(snapshotPathFor(pb)).equals(readFileSync(snapshotPathFor(pa)))).toBe(true);
    expect(readFileSync(embeddingsPathFor(pb), 'utf8')).not.toMatch(/hnsw|graph_nodes|level/i);
    expect(graphDigest(hnsw)).toBe(graphDigest(plain));
    expect(JSON.stringify(hnsw.embeddingStore.exportEmbeddings('d'))).toBe(JSON.stringify(plain.embeddingStore.exportEmbeddings('d')));

    // a JSON-only save is identical too
    const pc = join(dir, 'json.ison');
    saveGraph(hnsw, pc);
    expect(readFileSync(embeddingsPathFor(pc)).equals(readFileSync(embeddingsPathFor(pa)))).toBe(true);
    expect(existsSync(snapshotPathFor(pc))).toBe(false);
  });

  for (const snapshot of [false, true]) {
    it(`a loaded store builds the graph on its first search (${snapshot ? 'snapshot' : 'JSON sidecar'})`, () => {
      const dir = tempDir();
      const original = build('hnsw', 1200);
      const path = join(dir, 'g.ison');
      saveGraph(original, path, { snapshot });
      expect(existsSync(snapshotPathFor(path))).toBe(snapshot);

      const loaded = loadGraph(path, {
        encoder: new MockEncoder(32), ann: 'hnsw', annMinSize: 0, annHnsw: { m: 12, efConstruction: 80 },
      });
      const store = loaded.embeddingStore;
      expect(store.ann).toBe('hnsw');
      expect(store.annStats().active).toBe(false);                   // loading builds nothing
      const q = Array.from({ length: 32 }, (_, i) => i + 1);
      const hits = store.similaritySearch(q, 10);
      expect(hits).toHaveLength(10);
      expect(store.annStats().active).toBe(true);
      expect(store.annStats().graphNodes).toBe(1200);
      expect(store.annStats().searches.builds).toBe(1);
      expectScoresExact(store, q, hits);
      expect(verify(store)).toBeNull();
      expect(store.recallCheck([q], 10).approximate).toBe(true);
    });
  }
});

// -- the Rust reference ---------------------------------------------------------------------------------------

/**
 * The answers `isongraph-vector-rs/examples/hnsw_xcheck.rs` gives for
 * `3000 16 12 10 8 50 1 16 600` (n, dim, queries, k, m, efConstruction, seed,
 * ef, noise), ids only: the graph the Rust code builds, grows by inserts,
 * updates and deletes, and compacts is the one this port builds. Scores are left
 * out on purpose: this port sums in float64, as its exact scan always has, so a
 * score differs from Rust's float32 one in the last bits. (The same text sits in
 * the C# and C++ tests.)
 */
const RUST_ANSWERS = `
built q0 2382 1486 1380 325 72 1207 1695 756 43 2652
built q1 555 104 2357 1292 428 2482 1537 2045 1347 1561
built q2 1538 973 1027 2796 1559 1075 1893 2210 1101 1510
built q3 118 2338 2694 333 2548 613 1588 1067 2740 1253
built q4 1178 2487 1580 1076 757 2645 2493 1133 1209 170
built q5 726 302 2709 1592 1866 1724 2364 2620 2813 2695
built q6 1411 1315 268 2262 1033 1375 1002 1867 2573 2239
built q7 819 2888 179 337 2856 908 2025 1594 1675 2044
built q8 701 892 1798 791 2748 727 224 1611 2007 2903
built q9 35 935 1673 2096 789 964 1100 2021 443 541
built q10 631 2196 875 2302 969 1918 1571 50 1575 820
built q11 238 1196 2079 929 2420 1661 2637 2252 1294 322
built stats nodes=3000 dead=0 level=Some(3) approximate=12
writes q0 2382 1486 1380 72 1207 1695 3191 756 43 2652
writes q1 3213 555 2357 1292 356 428 2482 1537 2045 1347
writes q2 1538 973 2796 1559 1075 1893 2210 1101 1510 330
writes q3 3170 118 2338 2694 333 2548 613 1588 1067 2740
writes q4 2487 1580 757 2645 3255 1133 170 110 1040 2558
writes q5 726 302 2709 1592 1866 1724 2364 2620 2813 2695
writes q6 1411 1315 268 2262 1326 3005 3085 1033 1375 1002
writes q7 2888 179 337 2856 908 2025 1594 1675 2044 262
writes q8 701 892 1798 791 2748 727 3219 224 1742 1611
writes q9 35 935 1673 2096 104 789 964 1100 2021 443
writes q10 631 2196 875 2302 455 969 1918 1118 1571 50
writes q11 238 2079 929 2420 1661 3223 2637 2252 1294 3218
writes stats nodes=3300 dead=120 level=Some(3) approximate=24
compacted q0 2382 1380 72 1695 3191 756 2652 776 840 1350
compacted q1 3213 555 2357 1292 356 428 2045 2864 1347 2733
compacted q2 1538 2796 1559 1893 2210 1101 330 2015 3187 810
compacted q3 3170 2694 333 1067 1253 735 1643 1461 3190 2841
compacted q4 2487 1580 2645 2493 3255 1133 110 1040 2558 128
compacted q5 726 302 2709 1592 1866 1724 2364 2813 2756 3088
compacted q6 2262 1326 3005 3085 1002 728 2573 1487 2289 2673
compacted q7 2888 179 2856 908 2025 2262 507 2054 1428 1034
compacted q8 701 791 2748 3219 224 1742 1611 2007 909 2903
compacted q9 35 935 1673 2096 104 789 1100 2021 443 1539
compacted q10 2196 875 455 969 1571 50 1575 831 329 3046
compacted q11 2079 929 2420 1661 3223 1520 2637 2252 3218 2639
compacted stats nodes=2474 dead=254 level=Some(3) approximate=36
`;

describe('hnsw: the Rust reference', () => {
  it('answers equal the Rust crate through build, writes and compaction', () => {
    const n = 3000;
    const dim = 16;
    const nq = 12;
    const k = 10;
    const noise = 600;
    const clusters = 30;
    const centersRng = new SplitMix64(1n ^ 0x51n);
    const centers: number[][] = [];
    for (let c = 0; c < clusters; c++) centers.push(Array.from({ length: dim }, () => centersRng.below(2001) - 1000));
    const draw = (r: SplitMix64): number[] => {
      const c = centers[r.below(clusters)];
      return c.map(x => (x + r.below(2 * noise + 1) - noise) / 1024);
    };
    const data = new SplitMix64(1n ^ 0x52n);
    const qrng = new SplitMix64(1n ^ 0x53n);
    const queries = Array.from({ length: nq }, () => draw(qrng));
    const store = new EmbeddingStore(undefined, { ann: 'hnsw', annMinSize: 0, annHnsw: { m: 8, efConstruction: 50, efSearch: 16 } });
    for (let i = 0; i < n; i++) store.add(['doc', i], '', draw(data));
    store.buildAnnIndex();

    const lines: string[] = [];
    const dump = (phase: string) => {
      queries.forEach((q, i) => lines.push(`${phase} q${i} ` + store.similaritySearch(q, k, undefined, -2).map(r => r.nodeRef[1]).join(' ')));
      const s = store.annStats();
      lines.push(`${phase} stats nodes=${s.graphNodes} dead=${s.graphDead} level=Some(${s.maxLevel}) approximate=${s.searches.approximate}`);
    };
    dump('built');
    const ops = new SplitMix64(1n ^ 0x54n);
    for (let i = 0; i < n / 10; i++) store.add(['doc', n + i], '', draw(ops));
    for (let i = 0; i < n / 20; i++) store.add(['doc', (i * 13) % n], '', draw(ops));
    for (let i = 0; i < n / 25; i++) store.remove(['doc', (i * 17 + 5) % n]);
    dump('writes');
    for (let i = 0; i < n; i++) if (i % 3 === 1) store.remove(['doc', i]);
    expect(verify(store)).toBeNull();
    dump('compacted');

    expect(lines).toEqual(RUST_ANSWERS.split('\n').map(l => l.trim()).filter(l => l.length > 0));
  });
});

// -- memory ------------------------------------------------------------------------------------------------------

describe('hnsw: memory', () => {
  it('the graph holds links only: about 146 bytes a node at m = 16, and no copy of any vector', () => {
    const n = 3072;
    const dim = 32;
    const { data } = makeData(n, dim, true, 21);
    const store = hnswStore(data);
    const stats = store.buildAnnIndex();
    const perNode = stats.indexBytes / n;
    expect(perNode).toBeGreaterThan(130);
    expect(perNode).toBeLessThan(175);
    expect(perNode).toBeLessThan(dim * 4 * 2);                     // not even one more copy of a 32-float row
    expect(stats.vectorBytes).toBeGreaterThanOrEqual(n * dim * 4);  // the store's own rows, once
    expect(stats.cacheBytes).toBe(0);
  });
});
