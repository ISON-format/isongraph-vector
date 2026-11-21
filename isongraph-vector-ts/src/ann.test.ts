/**
 * The opt-in IVF approximate index. Mirrors the Python port's
 * TestApproximateIndex / TestApproximateIndexInGraph: recall on clustered
 * data, exactness when every cell is probed, upkeep under writes and deletes,
 * filters, and that nothing about the index reaches a saved file.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  EmbeddingStore, SemanticGraph, MockEncoder, EmbeddingError, graphDigest,
  type NodeRef,
} from './index.js';
import { saveGraph, loadGraph, embeddingsPathFor } from './node.js';

/** Recall@10 demanded on the clustered data below (4000 vectors, 40 clusters, 64 cells, nprobe 8). */
const RECALL_FLOOR = 0.95;

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

/** Clustered, roughly unit vectors and a function drawing queries from the same clusters. */
function clustered(n: number, dim = 32, clusters = 40, seed = 1, noise = 1.0) {
  const next = rng(seed);
  const centers: number[][] = [];
  for (let c = 0; c < clusters; c++) {
    const v = Array.from({ length: dim }, () => gaussian(next));
    const norm = Math.hypot(...v);
    centers.push(v.map(x => x / norm));
  }
  const draw = (count: number): number[][] =>
    Array.from({ length: count }, () => {
      const center = centers[Math.floor(next() * clusters)];
      return center.map(x => x + (noise * gaussian(next)) / Math.sqrt(dim));
    });
  return { data: draw(n), draw };
}

function annStore(
  { n = 4000, dim = 32, seed = 1, types, ...options }:
    { n?: number; dim?: number; seed?: number; types?: (i: number) => string } & Record<string, any> = {}
) {
  const { data, draw } = clustered(n, dim, 40, seed);
  const store = new EmbeddingStore(undefined, {
    ann: 'ivf', annMinSize: 0, annNlist: 64, annNprobe: 8, ...options,
  });
  store.addBatch(data.map((v, i) => [[types ? types(i) : 'doc', i] as NodeRef, `text ${i}`, v] as [NodeRef, string, number[]]));
  return { store, data, draw };
}

const refs = (results: Array<{ nodeRef: NodeRef }>) => results.map(r => `${r.nodeRef[0]}:${r.nodeRef[1]}`);

describe('approximate index: store', () => {
  it('is off by default and exact search is untouched', () => {
    const { store, data, draw } = annStore({ ann: null });
    expect(store.ann).toBeNull();
    expect(store.annStats().enabled).toBe(false);
    expect(() => store.buildAnnIndex()).toThrow(/ann/);

    // identical (not merely close) to the same store with ann on and exact: true
    const { store: other } = annStore();
    for (const q of draw(5)) {
      const a = store.similaritySearch(q, 10);
      const b = other.similaritySearch(q, 10, undefined, 0, true);
      expect(a.map(r => [r.nodeRef, r.score])).toEqual(b.map(r => [r.nodeRef, r.score]));
    }
    expect(data.length).toBe(4000);
  });

  it('validates its arguments', () => {
    expect(() => new EmbeddingStore(undefined, { ann: 'nope' as any })).toThrow(EmbeddingError);
    expect(() => new EmbeddingStore(undefined, { annNprobe: 0 })).toThrow(/annNprobe/);
    expect(() => new EmbeddingStore(undefined, { ann: 'ivf', annNlist: 0 })).toThrow(/annNlist/);
    expect(() => new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: -1 })).toThrow(/annMinSize/);
    expect(() => new EmbeddingStore(undefined, { ann: 'ivf', annNlist: 1.5 })).toThrow(/annNlist/);
    const store = new EmbeddingStore(undefined, { ann: 'ivf' });
    expect(() => { store.annNprobe = 0; }).toThrow(EmbeddingError);
    store.annNprobe = 5;
    expect(store.annNprobe).toBe(5);
    store.annNprobe = null;
    expect(store.annNprobe).toBeNull();
  });

  it('reaches the recall floor on clustered data and reports it', () => {
    const { store, draw } = annStore();
    const report = store.recallCheck(draw(60), 10);
    expect(report.approximate).toBe(true);
    expect(report.nlist).toBe(64);
    expect(report.nprobe).toBe(8);
    expect(report.queries).toBe(60);
    expect(report.k).toBe(10);
    expect(report.recall).toBeGreaterThanOrEqual(RECALL_FLOOR);
    expect(report.minRecall).toBeLessThanOrEqual(report.recall);
    expect(report.minRecall).toBeGreaterThanOrEqual(0);
    expect(report.annMs).toBeGreaterThanOrEqual(0);
    expect(report.exactMs).toBeGreaterThanOrEqual(0);
    const stats = store.annStats();
    expect(stats.active).toBe(true);
    expect(stats.nlist).toBe(64);
    expect(stats.trainedN).toBe(4000);
    expect(stats.indexBytes).toBeGreaterThan(0);
    expect(stats.searches.approximate).toBeGreaterThanOrEqual(60);
  });

  it('returns the exact cosine score of each node, best first', () => {
    const { store, draw } = annStore();
    for (const q of draw(10)) {
      const truth = new Map(store.similaritySearch(q, 100, undefined, 0, true).map(r => [`${r.nodeRef}`, r.score]));
      const got = store.similaritySearch(q, 10);
      for (const r of got) if (truth.has(`${r.nodeRef}`)) expect(r.score).toBe(truth.get(`${r.nodeRef}`));
      const scores = got.map(r => r.score);
      expect(scores).toEqual([...scores].sort((a, b) => b - a));
      expect(got.every(r => typeof r.text === 'string')).toBe(true);
    }
    expect(store.annStats().searches.approximate).toBe(10);          // the index answered, not the scan
  });

  it('probing every cell is the exact scan', () => {
    const { store, data, draw } = annStore({ annNprobe: 64 });          // nprobe == nlist
    const reference = new EmbeddingStore();
    reference.addBatch(data.map((v, i) => [['doc', i] as NodeRef, `text ${i}`, v] as [NodeRef, string, number[]]));
    for (const q of draw(8)) {
      expect(store.similaritySearch(q, 10)).toEqual(reference.similaritySearch(q, 10));
    }
    expect(store.recallCheck(draw(5)).approximate).toBe(false);
    expect(store.annStats().searches.approximate).toBe(0);
  });

  it('stays exact under annMinSize, and the default minimum is 50000', () => {
    const { store, draw } = annStore({ n: 500, annMinSize: 1000 });
    const q = draw(1)[0];
    expect(refs(store.similaritySearch(q, 10))).toEqual(refs(store.similaritySearch(q, 10, undefined, 0, true)));
    expect(store.annStats().active).toBe(false);
    expect(store.recallCheck([q]).approximate).toBe(false);

    const dflt = new EmbeddingStore(undefined, { ann: 'ivf' });
    expect(dflt.annStats().minSize).toBe(50000);
    expect(dflt.annStats().mode).toBe('ivf');
    dflt.addBatch(clustered(2000).data.map((v, i) => [['doc', i] as NodeRef, '', v] as [NodeRef, string, number[]]));
    dflt.similaritySearch(clustered(1).data[0], 5);
    expect(dflt.annStats().active).toBe(false);
  });

  it('chooses nlist and nprobe automatically', () => {
    const { store, draw } = annStore({ n: 2500, annNlist: undefined, annNprobe: undefined });
    store.similaritySearch(draw(1)[0], 5);
    const stats = store.annStats();
    expect(stats.nlist).toBe(200);            // 4 * sqrt(2500)
    expect(stats.nprobe).toBe(10);            // 5% of the cells
    expect(stats.active).toBe(true);

    const mid = annStore({ n: 100, annNlist: undefined, annNprobe: undefined });
    mid.store.similaritySearch(mid.draw(1)[0], 5);
    expect(mid.store.annStats().nlist).toBe(40);    // 4 * sqrt(100)
    expect(mid.store.annStats().nprobe).toBe(2);

    const tiny = annStore({ n: 9, annNlist: undefined, annNprobe: undefined });
    tiny.store.similaritySearch(tiny.draw(1)[0], 5);
    expect(tiny.store.annStats().nlist).toBe(9);    // at least 16, but never more cells than vectors
    expect(tiny.store.annStats().nprobe).toBe(1);
  });

  it('topK null and exact: true scan everything; topK 0 is empty', () => {
    const { store, draw } = annStore();
    const q = draw(1)[0];
    const before = store.annStats().searches.approximate;
    expect(store.similaritySearch(q, null, undefined, -1).length).toBe(4000);
    store.similaritySearch(q, 5, undefined, 0, true);
    expect(store.annStats().searches.approximate).toBe(before);
    expect(store.similaritySearch(q, 0)).toEqual([]);
  });

  it('a zero query falls back to the exact scan', () => {
    const { store } = annStore();
    const res = store.similaritySearch(new Array(32).fill(0), 5, undefined, 0);
    expect(res.map(r => r.score)).toEqual([0, 0, 0, 0, 0]);
    expect(store.annStats().searches).toMatchObject({ approximate: 0, exact: 1 });
  });

  it('shows writes immediately, and an update moves the row', () => {
    const { store, draw } = annStore();
    store.buildAnnIndex();
    const fresh = draw(1)[0];
    store.add(['new', 1], 'fresh', fresh);
    const top = store.similaritySearch(fresh, 3);
    expect(top[0].nodeRef).toEqual(['new', 1]);
    expect(top[0].score).toBeCloseTo(1, 6);
    expect(store.annStats().pending).toBe(1);

    const target = draw(1)[0];
    store.add(['doc', 7], 'moved', target);
    const hit = store.similaritySearch(target, 1)[0];
    expect(hit.nodeRef).toEqual(['doc', 7]);
    expect(hit.score).toBeCloseTo(1, 6);
    for (const q of [target, fresh]) {
      const res = store.similaritySearch(q, 400, undefined, -1);
      expect(new Set(refs(res)).size).toBe(res.length);              // no stale second entry
      const truth = new Map(store.similaritySearch(q, null, undefined, -1).map(r => [`${r.nodeRef}`, r.score]));
      for (const r of res) expect(r.score).toBeCloseTo(truth.get(`${r.nodeRef}`)!, 9);
    }
  });

  it('never returns a deleted node, follows compaction without retraining', () => {
    const { store, draw } = annStore();
    store.buildAnnIndex();
    const gone = new Set<string>();
    for (let i = 0; i < 4000; i += 3) {                              // a third: forces compaction
      store.remove(['doc', i]);
      gone.add(`doc:${i}`);
    }
    const builds = store.annStats().searches.builds;
    for (const q of draw(15)) {
      expect(refs(store.similaritySearch(q, 10)).filter(r => gone.has(r))).toEqual([]);
      // a tombstone would score 0, which beats any negative cosine
      expect(refs(store.similaritySearch(q, 250, undefined, -1)).filter(r => gone.has(r))).toEqual([]);
    }
    expect(store.recallCheck(draw(40)).recall).toBeGreaterThanOrEqual(RECALL_FLOOR);
    expect(store.annStats().searches.builds).toBe(builds);           // followed, not retrained
    expect(store.count()).toBe(4000 - gone.size);
  });

  it('folds pending rows into cells and retrains after heavy churn', () => {
    const { store, draw } = annStore({ n: 3000 });
    store.buildAnnIndex();
    const builds = store.annStats().searches.builds;
    const extra = draw(700);
    store.addBatch(extra.map((v, i) => [['late', i] as NodeRef, '', v] as [NodeRef, string, number[]]));
    expect(store.annStats().pending).toBe(700);
    store.similaritySearch(extra[0], 5);                             // >= 512 pending: folded
    let stats = store.annStats();
    expect(stats.pending).toBe(0);
    expect(stats.searches.builds).toBe(builds);

    const more = draw(1200);
    store.addBatch(more.map((v, i) => [['later', i] as NodeRef, '', v] as [NodeRef, string, number[]]));
    store.similaritySearch(more[0], 5);                              // churn 1900 > 50% of 3000
    stats = store.annStats();
    expect(stats.searches.builds).toBe(builds + 1);
    expect(stats.trainedN).toBe(4900);
    expect(store.recallCheck(draw(30)).recall).toBeGreaterThanOrEqual(RECALL_FLOOR);
  });

  it('drops the index with a cleared store, and rebuilds for a new dimension', () => {
    const { store, data } = annStore({ n: 1500, annNlist: 16, annNprobe: 4 });
    store.buildAnnIndex();
    store.clear();
    expect(store.similaritySearch(data[0], 5)).toEqual([]);
    expect(store.annStats().active).toBe(false);
    store.clear(true);
    const small = clustered(1200, 8, 12, 3);
    store.addBatch(small.data.map((v, i) => [['doc', i] as NodeRef, '', v] as [NodeRef, string, number[]]));
    const res = store.similaritySearch(small.data[5], 5);
    expect(res[0].nodeRef).toEqual(['doc', 5]);
    expect(store.annStats().active).toBe(true);
    expect(store.dimension).toBe(8);
  });

  it('applies nodeType, widening the probe for a rare type', () => {
    const { store, draw } = annStore({ types: i => (i % 200 === 0 ? 'rare' : 'common'), annNprobe: 2 });
    const q = draw(1)[0];
    const common = store.similaritySearch(q, 10, 'common');
    expect(common).toHaveLength(10);
    expect(new Set(common.map(r => r.nodeRef[0]))).toEqual(new Set(['common']));
    // 20 rare rows in the whole store and a topK of 15: two cells cannot hold them
    const widenedBefore = store.annStats().searches.widened;
    const rare = store.similaritySearch(q, 15, 'rare', -1);
    expect(rare).toHaveLength(15);
    expect(new Set(rare.map(r => r.nodeRef[0]))).toEqual(new Set(['rare']));
    const counts = store.annStats().searches;
    expect(counts.widened > widenedBefore || counts.exact > 0).toBe(true);
    expect(store.similaritySearch(q, 5, 'nope')).toEqual([]);
  });

  it('respects the threshold', () => {
    const { store, draw } = annStore();
    const q = draw(1)[0];
    const best = store.similaritySearch(q, 1, undefined, 0, true)[0].score;
    const res = store.similaritySearch(q, 10, undefined, best - 1e-3);
    expect(res.every(r => r.score >= best - 1e-3)).toBe(true);
    expect(store.similaritySearch(q, 10, undefined, 2)).toEqual([]);
    const cut = store.similaritySearch(q, 10, undefined, 0, true)[4].score;
    const got = store.similaritySearch(q, 10, undefined, cut);
    const want = store.similaritySearch(q, 10, undefined, cut, true);
    expect(new Set(refs(got))).toEqual(new Set(refs(want)));
    // a threshold that plenty clear is applied inside the probed cells: no scan needed
    const before = store.annStats().searches.approximate;
    const floor = store.similaritySearch(q, 30, undefined, 0, true)[29].score;
    const above = store.similaritySearch(q, 10, undefined, floor);
    expect(above).toHaveLength(10);
    expect(above.every(r => r.score >= floor)).toBe(true);
    expect(store.annStats().searches.approximate).toBe(before + 1);
  });

  it('breaks ties in insertion order, like the exact scan', () => {
    const store = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annNlist: 4, annNprobe: 2 });
    const same = [1, 0.5, 0.25, 0];
    store.addBatch([...Array.from({ length: 40 }, (_, i) => [['t', i] as NodeRef, '', same] as [NodeRef, string, number[]]),
      [['x', 1], '', [0, 1, 0, 0]], [['x', 2], '', [0, 0, 1, 0]], [['x', 3], '', [0, 0, 0, 1]], [['x', 4], '', [-1, 0, 0, 0]]]);
    expect(refs(store.similaritySearch(same, 5))).toEqual(['t:0', 't:1', 't:2', 't:3', 't:4']);
    expect(store.annStats().searches.approximate).toBe(1);
  });

  it('recallCheck takes one text-free query, a list, or a typed array', () => {
    const { store, data, draw } = annStore();
    const q = draw(3);
    expect(store.recallCheck(q[0]).queries).toBe(1);                 // one flat vector
    expect(store.recallCheck(q).queries).toBe(3);                    // a list of them
    expect(store.recallCheck(new Float32Array(q[0])).queries).toBe(1);
    const off = new EmbeddingStore();
    off.addBatch(data.slice(0, 100).map((v, i) => [['doc', i] as NodeRef, '', v] as [NodeRef, string, number[]]));
    const r = off.recallCheck(q);
    expect(r.recall).toBe(1);
    expect(r.approximate).toBe(false);
    expect(() => store.recallCheck(q, 0)).toThrow(/k/);
  });

  it('recallCheck takes text queries through the encoder', async () => {
    const encoder = new MockEncoder(32);
    const store = new EmbeddingStore(encoder, { ann: 'ivf', annMinSize: 0, annNlist: 16, annNprobe: 16 });
    store.addBatch(Array.from({ length: 300 }, (_, i) => [['doc', i] as NodeRef, `word${i} other${i % 7}`] as [NodeRef, string]));
    expect(store.recallCheck('word5 other5').queries).toBe(1);
    expect((await store.recallCheckAsync(['word5', 'word9'])).queries).toBe(2);
  });

  it('is deterministic for a seed', () => {
    const run = (annSeed: number) => {
      const { store, draw } = annStore({ annSeed });
      const out = draw(20).map(q => refs(store.similaritySearch(q, 10)));
      expect(store.annStats().searches.approximate).toBe(20);
      return out;
    };
    expect(run(0)).toEqual(run(0));
    expect(run(7)).toEqual(run(7));
  });

  it('survives an import and keeps every row findable', () => {
    const { store, draw } = annStore({ n: 1200, annNlist: 16, annNprobe: 4 });
    store.buildAnnIndex();
    const q = draw(1)[0];
    const payload = JSON.parse(JSON.stringify(store.exportEmbeddings()));
    payload.embeddings.push({ type: 'imp', id: 1, id_type: 'int', text: 'imported', vector: q });
    payload.count += 1;
    store.importEmbeddings(payload);
    expect(store.similaritySearch(q, 1)[0].nodeRef).toEqual(['imp', 1]);
    const other = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annNlist: 16, annNprobe: 4 });
    other.similaritySearch(q, 1);                                    // empty: nothing built, no error
    other.importEmbeddings(payload, true);
    expect(other.similaritySearch(q, 1)[0].nodeRef).toEqual(['imp', 1]);
    expect(other.annStats().active).toBe(true);
  });

  it('never changes what is exported', () => {
    const { store, data } = annStore({ n: 800, annNlist: 16, annNprobe: 4 });
    const plain = new EmbeddingStore();
    plain.addBatch(data.slice(0, 800).map((v, i) => [['doc', i] as NodeRef, `text ${i}`, v] as [NodeRef, string, number[]]));
    store.buildAnnIndex();
    store.similaritySearch(data[0], 5);
    expect(JSON.stringify(store.exportEmbeddings())).toBe(JSON.stringify(plain.exportEmbeddings()));
  });

  it('survives writes and searches interleaved (single thread, async)', async () => {
    const { store, draw } = annStore({ n: 3000, annNlist: 32, annNprobe: 4 });
    const extra = draw(600);
    const queries = draw(40);
    await Promise.all([
      (async () => {
        for (let i = 0; i < 100; i++) {
          store.add(['w', i], '', extra[i % 600]);
          if (i % 4 === 0) store.remove(['doc', i]);
          if (i % 20 === 0) store.buildAnnIndex();
          await Promise.resolve();
        }
      })(),
      (async () => {
        for (let i = 0; i < 120; i++) {
          const res = await store.similaritySearchAsync(queries[i % 40], 8, undefined, -1);
          expect(res).toHaveLength(8);
          expect(new Set(refs(res)).size).toBe(8);
          const scores = res.map(r => r.score);
          expect(scores).toEqual([...scores].sort((a, b) => b - a));
        }
      })(),
    ]);
    expect(store.recallCheck(queries.slice(0, 20)).recall).toBeGreaterThanOrEqual(RECALL_FLOOR);
  });
});

describe('approximate index: in a SemanticGraph', () => {
  function graph(n = 3000, dim = 32, extra: Record<string, any> = {}) {
    const { data, draw } = clustered(n, dim);
    const g = new SemanticGraph({
      name: 'ann', encoder: new MockEncoder(dim), autoEmbed: false,
      ann: 'ivf', annMinSize: 0, annNlist: 48, annNprobe: 6, ...extra,
    });
    for (let i = 0; i < n; i++) g.addNode('item', i, { group: i % 50 });
    g.embeddingStore.addBatch(data.map((v, i) => [['item', i] as NodeRef, `item ${i}`, v] as [NodeRef, string, number[]]));
    return { g, data, draw };
  }

  it('passes the options to its store, and recallCheck through', () => {
    const { g, draw } = graph();
    const store = g.embeddingStore;
    expect(store.ann).toBe('ivf');
    expect(store.annNprobe).toBe(6);
    expect(g.similaritySearch(draw(1)[0], 5)).toHaveLength(5);
    expect(store.annStats().active).toBe(true);
    expect(g.recallCheck(draw(20)).recall).toBeGreaterThanOrEqual(RECALL_FLOOR);
    expect(new SemanticGraph().embeddingStore.ann).toBeNull();
  });

  it('exact: true skips the index, and similarToNode works', () => {
    const { g, draw } = graph();
    const q = draw(1)[0];
    const before = g.embeddingStore.annStats().searches.approximate;
    g.similaritySearch(q, 5, undefined, 0, undefined, undefined, true);
    g.similaritySearch(q, 5, undefined, 0, () => true, undefined, true);
    expect(g.embeddingStore.annStats().searches.approximate).toBe(before);
    const near = g.similarToNode(['item', 3], 5);
    expect(near).toHaveLength(5);
    expect(refs(near)).not.toContain('item:3');
  });

  it('a selective nodeFilter still gets topK results', () => {
    const { g, draw } = graph();
    const q = draw(1)[0];
    const group7 = (n: any) => n.properties.group === 7;        // 1 node in 50
    const res = g.similaritySearch(q, 10, undefined, 0, group7);
    expect(res).toHaveLength(10);
    expect(res.every(r => g.getNodeByRef(r.nodeRef).properties.group === 7)).toBe(true);
    const want = g.similaritySearch(q, 10, undefined, 0, group7, undefined, true);
    const overlap = refs(res).filter(r => refs(want).includes(r)).length;
    expect(overlap).toBeGreaterThanOrEqual(7);
    const scores = res.map(r => r.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(g.embeddingStore.annStats().searches.approximate).toBeGreaterThanOrEqual(1);   // probed cells, not a scan
    // a filter nothing passes terminates with nothing
    expect(g.similaritySearch(q, 10, undefined, 0, () => false)).toEqual([]);
    // a lenient one agrees closely with the scan
    const even = (n: any) => n.properties.group % 2 === 0;
    const got = g.similaritySearch(q, 10, undefined, 0, even);
    const ref = g.similaritySearch(q, 10, undefined, 0, even, undefined, true);
    expect(refs(got).filter(r => refs(ref).includes(r)).length).toBeGreaterThanOrEqual(8);
    // nodeTypes goes through the same gate
    expect(g.similaritySearch(q, 5, undefined, 0, undefined, ['item'])).toHaveLength(5);
    expect(g.similaritySearch(q, 5, undefined, 0, undefined, ['other'])).toEqual([]);
  });

  it('semanticMultiHop takes its seeds from the index, blend > 0 does not', () => {
    const { g, draw } = graph(2000);
    for (let i = 0; i < 1999; i++) g.addEdge('NEXT', ['item', i], ['item', i + 1]);
    const q = draw(1)[0];
    const out = g.semanticMultiHop(q, { relType: 'NEXT', maxHops: 2, topKSeeds: 3, threshold: -1 });
    expect(out.length).toBeGreaterThan(0);
    expect(out[0].hopCount).toBe(0);
    const used = g.embeddingStore.annStats().searches.approximate;
    expect(used).toBeGreaterThanOrEqual(1);
    g.semanticMultiHop(q, { relType: 'NEXT', maxHops: 1, blend: 0.3, threshold: -1 });
    expect(g.embeddingStore.annStats().searches.approximate).toBe(used + 1);   // the seed search only; scoreMap is a full scan
  });

  it('async text search honours exact', async () => {
    const encoder = new MockEncoder(32);
    const g = new SemanticGraph({ encoder, ann: 'ivf', annMinSize: 0, annNlist: 16, annNprobe: 4 });
    for (let i = 0; i < 400; i++) g.addNode('doc', i, { name: `entry number ${i} about topic ${i % 9}` });
    const before = g.embeddingStore.annStats().searches.approximate;
    const hits = await g.similaritySearchAsync('entry number 12 about topic 3', 3);
    expect(hits).toHaveLength(3);
    const used = g.embeddingStore.annStats().searches.approximate;
    expect(used).toBe(before + 1);
    await g.similaritySearchAsync('entry number 12 about topic 3', 3, undefined, 0, undefined, undefined, true);
    expect(g.embeddingStore.annStats().searches.approximate).toBe(used);
  });

  describe('files', () => {
    const dirs: string[] = [];
    afterEach(() => { while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true }); });

    it('loadGraph passes ann along and rebuilds lazily; nothing about it is saved', () => {
      const dir = mkdtempSync(join(tmpdir(), 'isongraph-vector-ann-'));
      dirs.push(dir);
      const { g, draw } = graph(1500);
      const path = join(dir, 'g.ison');
      saveGraph(g, path);

      const plain = new SemanticGraph({ name: 'ann', encoder: new MockEncoder(32), autoEmbed: false });
      for (let i = 0; i < 1500; i++) plain.addNode('item', i, { group: i % 50 });
      plain.embeddingStore.importEmbeddings(g.embeddingStore.exportEmbeddings());
      const plainPath = join(dir, 'plain.ison');
      saveGraph(plain, plainPath);
      // same graph, same vectors: the ann graph's files are exactly the plain graph's
      expect(readFileSync(path, 'utf8')).toBe(readFileSync(plainPath, 'utf8'));
      expect(readFileSync(embeddingsPathFor(path), 'utf8')).toBe(readFileSync(embeddingsPathFor(plainPath), 'utf8'));
      expect(readFileSync(embeddingsPathFor(path), 'utf8')).not.toMatch(/ivf|nlist|centroid/i);
      expect(graphDigest(g)).toBe(graphDigest(plain));

      const loaded = loadGraph(path, {
        encoder: g.encoder!, ann: 'ivf', annMinSize: 0, annNlist: 32, annNprobe: 4,
      });
      expect(loaded.embeddingStore.ann).toBe('ivf');
      expect(loaded.embeddingStore.annStats().active).toBe(false);   // nothing persisted
      expect(loaded.similaritySearch(draw(1)[0], 5)).toHaveLength(5);
      expect(loaded.embeddingStore.annStats().active).toBe(true);
    });
  });
});
