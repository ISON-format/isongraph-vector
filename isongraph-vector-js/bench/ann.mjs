#!/usr/bin/env node
/**
 * Exact vs IVF approximate search: latency and recall@k on synthetic clustered
 * vectors (default 384-d, 10k / 100k / 1M).
 *
 *   node --max-old-space-size=16384 bench/ann.mjs                    # all three sizes
 *   node bench/ann.mjs --sizes 10000,100000                         # smaller, default heap
 *   node bench/ann.mjs --sizes 100000 --queries 200 --nprobe 1,8,32  # custom
 *
 * The vectors are CLUSTERED (n/100 Gaussian clusters, 100 points each, unit
 * length), the same regime as benchmarks/ann.py "clustered": an IVF index's
 * recall depends entirely on how data clusters, and uniform random vectors
 * (where it degenerates to a coin flip) would say nothing. Real embeddings sit
 * somewhere else - run store.recallCheck() on your own queries before relying
 * on a setting.
 *
 * Stored vectors are plain JS arrays of doubles (8 bytes per component) plus
 * the index's Float32Array copy (4 bytes), so 1M x 384 needs about 12 GB of
 * heap: run it with --max-old-space-size. A size that will not fit is skipped
 * with a message instead of crashing.
 *
 * To measure the TypeScript build instead, pass --port ts (needs `npm run
 * build` in isongraph-vector-ts first); --entry <file> imports any other build.
 */
import { getHeapStatistics } from 'node:v8';
import { cpus, totalmem } from 'node:os';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

function args() {
  const out = {
    sizes: [10000, 100000, 1000000], dim: 384, queries: 100, k: 10, nprobe: null, port: 'js', seed: 4242,
  };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const [flag, inline] = argv[i].split('=');
    const value = () => inline ?? argv[++i];
    if (flag === '--sizes') out.sizes = value().split(',').map(Number);
    else if (flag === '--dim') out.dim = Number(value());
    else if (flag === '--queries') out.queries = Number(value());
    else if (flag === '--k') out.k = Number(value());
    else if (flag === '--nprobe') out.nprobe = value().split(',').map(Number);
    else if (flag === '--port') out.port = value();
    else if (flag === '--entry') out.entry = value();
    else if (flag === '--seed') out.seed = Number(value());
    else if (flag === '--help' || flag === '-h') {
      console.log('usage: node bench/ann.mjs [--sizes 10000,100000,1000000] [--dim 384] [--queries 100] [--k 10] [--nprobe 1,8,32] [--port js|ts] [--entry file]');
      process.exit(0);
    } else throw new Error(`unknown argument ${argv[i]}`);
  }
  return out;
}

const opts = args();
const entry = opts.entry ? resolve(opts.entry) : opts.port === 'ts'
  ? resolve(here, '../../isongraph-vector-ts/dist/index.js')
  : resolve(here, '../src/index.js');
const { EmbeddingStore } = await import(pathToFileURL(entry).href);

// -- data ----------------------------------------------------------------------

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Clustered {
  constructor(n, dim, seed) {
    this.dim = dim;
    this.next = rng(seed ^ n);
    const clusters = Math.max(8, Math.floor(n / 100));
    this.centers = Array.from({ length: clusters }, () => {
      const v = Float64Array.from({ length: dim }, () => this.gauss());
      const norm = Math.hypot(...v);
      return v.map(x => x / norm);
    });
  }

  gauss() {
    return Math.sqrt(-2 * Math.log(1 - this.next())) * Math.cos(2 * Math.PI * this.next());
  }

  /** One unit vector near a random cluster centre (noise norm 1 across the vector). */
  draw() {
    const { dim } = this;
    const center = this.centers[Math.floor(this.next() * this.centers.length)];
    const v = new Array(dim);
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      v[j] = center[j] + this.gauss() / Math.sqrt(dim);
      sum += v[j] * v[j];
    }
    const norm = Math.sqrt(sum);
    for (let j = 0; j < dim; j++) v[j] /= norm;
    return v;
  }
}

// -- measuring -------------------------------------------------------------------

const median = values => {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const percentile = (values, q) => {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.round(q * (s.length - 1)))];
};
const fmt = ms => (ms >= 100 ? ms.toFixed(0) : ms >= 10 ? ms.toFixed(1) : ms.toFixed(2));
const keyOf = r => `${r.nodeRef[0]}:${r.nodeRef[1]}`;

function timed(store, queries, k, exact) {
  for (const q of queries.slice(0, 3)) store.similaritySearch(q, k, undefined, 0, exact);   // warm up
  const times = [];
  const found = [];
  for (const q of queries) {
    const t0 = performance.now();
    const res = store.similaritySearch(q, k, undefined, 0, exact);
    times.push(performance.now() - t0);
    found.push(new Set(res.map(keyOf)));
  }
  return { times, found };
}

function recall(found, truth) {
  const per = found.map((f, i) => {
    let hit = 0;
    for (const key of truth[i]) if (f.has(key)) hit++;
    return truth[i].size ? hit / truth[i].size : 1;
  });
  return { mean: per.reduce((a, b) => a + b, 0) / per.length, min: Math.min(...per) };
}

// -- run -------------------------------------------------------------------------

const heapLimit = getHeapStatistics().heap_size_limit;
console.log(`# exact vs IVF (${opts.port}), dim ${opts.dim}, k ${opts.k}, ${opts.queries} queries, warm single-thread`);
console.log(`node ${process.version}, ${cpus().length} logical CPUs, ${(totalmem() / 2 ** 30).toFixed(1)} GB RAM, heap limit ${(heapLimit / 2 ** 30).toFixed(1)} GB\n`);

for (const n of opts.sizes) {
  const need = n * opts.dim * 13 * 1.2;          // doubles + float32 copy + slack
  if (need > heapLimit * 0.9) {
    console.log(`## N = ${n.toLocaleString('en-US')}: skipped (needs about ${(need / 2 ** 30).toFixed(1)} GB of heap, limit ${(heapLimit / 2 ** 30).toFixed(1)} GB; rerun with --max-old-space-size)\n`);
    continue;
  }
  console.log(`## N = ${n.toLocaleString('en-US')}`);
  const data = new Clustered(n, opts.dim, opts.seed);
  const store = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annSeed: 0 });

  let t0 = performance.now();
  const CHUNK = 5000;
  for (let start = 0; start < n; start += CHUNK) {
    const entries = [];
    for (let i = start; i < Math.min(n, start + CHUNK); i++) entries.push([['doc', i], '', data.draw()]);
    store.addBatch(entries);
  }
  console.log(`ingest ${(((n) / (performance.now() - t0)) * 1000).toFixed(0)} vectors/s`);

  const queries = Array.from({ length: opts.queries }, () => data.draw());
  const exact = timed(store, queries, opts.k, true);
  console.log(`exact: median ${fmt(median(exact.times))} ms, p95 ${fmt(percentile(exact.times, 0.95))} ms`);

  t0 = performance.now();
  const stats = store.buildAnnIndex();
  const buildSeconds = (performance.now() - t0) / 1000;
  console.log(`index build ${buildSeconds.toFixed(1)} s: nlist ${stats.nlist}, automatic nprobe ${stats.nprobe}, ` +
    `index ${(stats.indexBytes / 2 ** 20).toFixed(1)} MB + float32 copy ${(stats.cacheBytes / 2 ** 20).toFixed(0)} MB\n`);

  const sweep = opts.nprobe ?? [...new Set([1, 4, 16, 64, stats.nprobe])].sort((a, b) => a - b);
  console.log('| nprobe | % of cells | recall@k mean | worst | median ms | p95 ms | speed-up |');
  console.log('|---|---|---|---|---|---|---|');
  const exactMedian = median(exact.times);
  for (const nprobe of sweep) {
    store.annNprobe = nprobe;
    const run = timed(store, queries, opts.k, false);
    const r = recall(run.found, exact.found);
    const label = nprobe === stats.nprobe ? `${nprobe} (auto)` : `${nprobe}`;
    console.log(`| ${label} | ${(100 * nprobe / stats.nlist).toFixed(1)} | ${r.mean.toFixed(3)} | ${r.min.toFixed(2)} | ` +
      `${fmt(median(run.times))} | ${fmt(percentile(run.times, 0.95))} | ${(exactMedian / median(run.times)).toFixed(1)}x |`);
  }
  console.log('');
}
