#!/usr/bin/env node
/**
 * HNSW (opt-in graph index) checks and measurements for the JavaScript and TypeScript ports.
 *
 *   node bench/hnsw.mjs xdump [n dim queries k m efc seed ef noise]
 *       Prints what the Rust crate's examples/hnsw_xcheck.rs prints (same arguments, same
 *       integer-generated data, same phases: built / writes / compacted), so the two can be
 *       compared line by line: `diff` shows only the score bits, which differ by design
 *       (this port scores in float64, as its exact scan does; the bits printed here are the float32
 *       rounding of that score).
 *   node bench/hnsw.mjs real  [--n 100000] [--queries 1000] [--ef 64,96,128,192] [--ivf] [--writes 300]
 *       Measures the real text-384 vectors (build time, recall@10 by ef, single-query QPS, insert latency,
 *       memory); needs the files named by --data (default D:/Goldmine/ISON/bench_data/prep/text384).
 *   node bench/hnsw.mjs synth [--n 50000] [--dim 384] ...   the same on clustered synthetic unit vectors.
 *
 * Add --port ts to run the compiled TypeScript build (npm run build in isongraph-vector-ts first) instead
 * of the JavaScript one; --entry <file> imports any other build.
 */
import { readFileSync, existsSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { resolve, dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const mode = argv[0] && !argv[0].startsWith('--') ? argv[0] : 'xdump';
const flags = {};
const positional = [];
for (let i = mode === argv[0] ? 1 : 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) {
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) flags[argv[i].slice(2)] = true;
    else { flags[argv[i].slice(2)] = next; i++; }
  } else positional.push(argv[i]);
}
const entry = flags.entry
  ? resolve(flags.entry)
  : flags.port === 'ts'
    ? resolve(here, '../../isongraph-vector-ts/dist/index.js')
    : resolve(here, '../src/index.js');
const { EmbeddingStore, SplitMix64 } = await import(pathToFileURL(entry).href);

const num = (name, d) => (flags[name] === undefined ? d : Number(flags[name]));
const nowS = () => Number(process.hrtime.bigint()) / 1e9;

// ---------------------------------------------------------------------------------------------
// xdump: the Rust example's output for deterministic integer data
// ---------------------------------------------------------------------------------------------

function xdump() {
  const p = (i, d) => (positional[i] === undefined ? d : Number(positional[i]));
  const [n, dim, nq, k, m, efc, seed, ef, noise] =
    [p(0, 5000), p(1, 32), p(2, 100), p(3, 10), p(4, 16), p(5, 200), p(6, 1), p(7, 0), p(8, 500)];
  const CLUSTERS = 30;
  const csRng = new SplitMix64(BigInt(seed) ^ 0x51n);
  const centers = [];
  for (let c = 0; c < CLUSTERS; c++) {
    centers.push(Array.from({ length: dim }, () => csRng.below(2001) - 1000));
  }
  const draw = rng => {
    const c = centers[rng.below(CLUSTERS)];
    return c.map(x => (x + rng.below(2 * noise + 1) - noise) / 1024);
  };
  const data = new SplitMix64(BigInt(seed) ^ 0x52n);
  const qrng = new SplitMix64(BigInt(seed) ^ 0x53n);
  const queries = Array.from({ length: nq }, () => draw(qrng));

  const store = new EmbeddingStore(undefined, {
    ann: 'hnsw', annMinSize: 0, annHnsw: { m, efConstruction: efc, efSearch: ef === 0 ? undefined : ef },
  });
  const ref = id => ['doc', String(id)];
  for (let i = 0; i < n; i++) store.add(ref(i), '', draw(data));
  store.buildAnnIndex();

  const bits = f => {
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, f);
    return b.getUint32(0).toString(16).padStart(8, '0');
  };
  const line = (phase, tag, i, hits) =>
    `${phase} ${tag}${i}` + hits.map(h => ` ${h.nodeRef[1]}:${bits(h.score)}`).join('');
  const dump = phase => {
    queries.forEach((q, i) => console.log(line(phase, 'q', i, store.similaritySearch(q, k, undefined, -2))));
    queries.forEach((q, i) => console.log(line(phase, 'x', i, store.similaritySearch(q, k, undefined, -2, true))));
    const s = store.annStats();
    console.log(`${phase} stats nodes=${s.graphNodes} dead=${s.graphDead} level=Some(${s.maxLevel}) approximate=${s.searches.approximate}`);
  };
  console.log(`params n=${n} dim=${dim} queries=${nq} k=${k} m=${m} efc=${efc} seed=${seed} ef=${ef} noise=${noise}`);
  dump('built');

  const ops = new SplitMix64(BigInt(seed) ^ 0x54n);
  for (let i = 0; i < Math.floor(n / 10); i++) store.add(ref(n + i), '', draw(ops));
  for (let i = 0; i < Math.floor(n / 20); i++) store.add(ref((i * 13) % n), '', draw(ops));
  for (let i = 0; i < Math.floor(n / 25); i++) store.remove(ref((i * 17 + 5) % n));
  dump('writes');
  for (let i = 0; i < n; i++) if (i % 3 === 1) store.remove(ref(i));
  dump('compacted');
}

// ---------------------------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------------------------

function median(values) {
  const v = [...values].sort((a, b) => a - b);
  return v.length % 2 ? v[v.length >> 1] : 0.5 * (v[(v.length >> 1) - 1] + v[v.length >> 1]);
}

function readF32(path) {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2);
}

function unit(v) {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return v.map(x => x / s);
}

/** Clustered synthetic unit vectors, in the spirit of the Rust crate's examples/hnsw_bench.rs (n/100 clusters). */
function synthetic(n, dim, nq, seed) {
  const rng = new SplitMix64(seed);
  const uniform = () => ((rng.next(), rng.hi >>> 8) + 0.5) / 16777216;
  const gauss = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const clusters = Math.max(1, Math.floor(n / 100));
  const centers = Array.from({ length: clusters }, () => unit(Array.from({ length: dim }, gauss)));
  const point = () => {
    const c = centers[rng.below(clusters)];
    return unit(c.map(x => x + 0.3 * gauss() / Math.sqrt(dim)));
  };
  const base = Array.from({ length: n }, point);
  const queries = Array.from({ length: nq }, point);
  return { base, queries };
}

/** Bytes the process holds on the JS heap and in typed arrays, after a full collection (needs --expose-gc). */
function held() {
  if (typeof globalThis.gc === 'function') globalThis.gc();
  const m = process.memoryUsage();
  return m.heapUsed + m.external;
}

async function measure() {
  const n = num('n', 100000);
  const nq = num('queries', 1000);
  const k = 10;
  const efs = String(flags.ef ?? '64,96,128,192').split(',').map(Number);
  const writes = num('writes', 300);
  const runs = num('runs', 3);
  const only = flags.only ?? 'hnsw';                    // hnsw | ivf | both
  let dim;
  let base;
  let queries;
  let truth = null;
  if (mode === 'real') {
    const dir = flags.data ?? 'D:/Goldmine/ISON/bench_data/prep/text384';
    dim = num('dim', 384);
    const all = readF32(join(dir, 'base.f32'));
    const q = readF32(join(dir, 'queries.f32'));
    base = Array.from({ length: n }, (_, i) => all.subarray(i * dim, (i + 1) * dim));     // float32 views: no 300 MB of doubles
    queries = Array.from({ length: Math.min(nq, q.length / dim) }, (_, i) => Array.from(q.subarray(i * dim, (i + 1) * dim)));
    const gtPath = join(dir, `gt_${n}.i32`);
    if (existsSync(gtPath)) {
      const buf = readFileSync(gtPath);
      const gt = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2);
      const width = gt.length / (q.length / dim);
      truth = queries.map((_, i) => new Set(Array.from(gt.subarray(i * width, i * width + k))));
    }
  } else {
    dim = num('dim', 384);
    ({ base, queries } = synthetic(n, dim, nq, 4242));
  }
  const rawBytes = n * dim * 4;
  const report = { port: flags.port ?? 'js', data: mode, n, dim, queries: queries.length, node: process.version };
  const baseline = held();

  const make = ann => {
    const s = new EmbeddingStore(undefined, { ann, annMinSize: 0 });
    for (let i = 0; i < n; i++) s.add(['doc', i], '', base[i]);
    return s;
  };
  // recall@k over the queries; `lastApproximate` is the share of them the index answered (the rest ended in the exact scan)
  let lastApproximate = 1;
  const recallOf = (store, ef) => {
    let sum = 0;
    const before = store.annStats().searches.approximate;
    for (let i = 0; i < queries.length; i++) {
      const got = ef === undefined ? store.similaritySearch(queries[i], k) : store.similaritySearchEf(queries[i], k, undefined, 0, ef);
      let hit = 0;
      for (const r of got) if (truth[i].has(r.nodeRef[1])) hit++;
      sum += hit / k;
    }
    lastApproximate = (store.annStats().searches.approximate - before) / queries.length;
    return sum / queries.length;
  };
  const qps = (store, ef) => {
    const t0 = nowS();
    for (let i = 0; i < queries.length; i++) {
      if (ef === undefined) store.similaritySearch(queries[i], k); else store.similaritySearchEf(queries[i], k, undefined, 0, ef);
    }
    return queries.length / (nowS() - t0);
  };
  const spread = values => ({ median: median(values), min: Math.min(...values), max: Math.max(...values) });

  if (only === 'hnsw' || only === 'both') {
    const store = make('hnsw');
    if (truth === null) truth = queries.map(q => new Set(store.similaritySearch(q, k, undefined, 0, true).map(r => r.nodeRef[1])));
    const plainBytes = held() - baseline;                      // the store before any index
    const t0 = nowS();
    const stats = store.buildAnnIndex();
    report.hnsw = { buildS: nowS() - t0 };
    const h = report.hnsw;
    const withGraph = held() - baseline;
    h.storeMB = plainBytes / 1e6;
    h.storePlusGraphMB = withGraph / 1e6;
    h.rawMB = rawBytes / 1e6;
    h.graphBytes = stats.indexBytes;
    h.graphBytesPerNode = stats.indexBytes / n;
    h.graphOverRaw = stats.indexBytes / rawBytes;
    h.wholeStoreOverRaw = withGraph / rawBytes;                // rows + keys/refs/texts + graph, measured from the heap
    h.storeNoIndexOverRaw = plainBytes / rawBytes;
    h.byEf = {};
    for (const ef of efs) {
      const q = [];
      for (let r = 0; r < runs; r++) q.push(qps(store, ef));
      h.byEf[ef] = { recall: recallOf(store, ef), approximate: lastApproximate, qps: spread(q) };
    }
    // the auto ef, as a user would run it
    const auto = [];
    for (let r = 0; r < runs; r++) auto.push(qps(store, undefined));
    h.autoEf = { ef: store.annStats().efSearch, recall: recallOf(store, undefined), approximate: lastApproximate, qps: spread(auto) };
    // inserts and updates through the write path
    const insertMs = [];
    const extra = base.slice(0, writes).map(v => Array.from(v, (x, j) => x + 0.001 * Math.sin(j)));
    for (let i = 0; i < writes; i++) {
      const t = nowS();
      store.add(['new', i], '', extra[i]);
      insertMs.push((nowS() - t) * 1000);
    }
    const sorted = a => [...a].sort((x, y) => x - y);
    h.insertMs = { median: median(insertMs), p99: sorted(insertMs)[Math.floor(insertMs.length * 0.99)], min: Math.min(...insertMs), max: Math.max(...insertMs) };
    const updateMs = [];
    for (let i = 0; i < writes; i++) {
      const t = nowS();
      store.add(['doc', (i * 7919) % n], '', extra[i]);
      updateMs.push((nowS() - t) * 1000);
    }
    h.updateMs = { median: median(updateMs), p99: sorted(updateMs)[Math.floor(updateMs.length * 0.99)] };
    const tD = nowS();
    for (let i = 0; i < writes; i++) store.remove(['new', i]);
    h.deleteUs = ((nowS() - tD) / writes) * 1e6;
    const st = store.annStats();
    h.stats = { efSearch: st.efSearch, maxLevel: st.maxLevel, graphNodes: st.graphNodes, graphDead: st.graphDead, churn: st.churn, inserts: st.inserts };
  }

  if (only === 'ivf' || only === 'both') {
    const store = make('ivf');
    if (truth === null) truth = queries.map(q => new Set(store.similaritySearch(q, k, undefined, 0, true).map(r => r.nodeRef[1])));
    const t1 = nowS();
    const s = store.buildAnnIndex();
    report.ivf = { buildS: nowS() - t1, nlist: s.nlist, indexBytes: s.indexBytes, byNprobe: {} };
    const probes = String(flags.nprobe ?? '8,16,32,48,64,96,128').split(',').map(Number);
    for (const np of probes) {
      store.annNprobe = np;
      const q = [];
      for (let r = 0; r < runs; r++) q.push(qps(store, undefined));
      report.ivf.byNprobe[np] = { recall: recallOf(store, undefined), approximate: lastApproximate, qps: spread(q) };
    }
  }
  console.log(JSON.stringify(report, null, 2));
}

if (mode === 'xdump') xdump();
else await measure();
