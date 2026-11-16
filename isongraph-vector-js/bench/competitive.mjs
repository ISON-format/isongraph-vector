#!/usr/bin/env node
/**
 * Loader and measurement harness for benchmarks/competitive/ (the cross-library comparison).
 * Reads the same raw float32 vectors, queries and float64 ground truth the other harnesses feed their
 * systems. Bench-only: the library is not touched.
 *
 *   node --max-old-space-size=24000 bench/competitive.mjs --ds-dir <prep>/text384 --ds text384 --n 100000 \
 *        --exp core,mutation --queries 2000 --label js --out result.json
 *   ... --exp filter
 * (No save/load experiment: this port has no vector-persistence call besides exportEmbeddings.)
 */
import { openSync, readSync, closeSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { cpus } from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { resolve, dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (name, def) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : def; };
const dir = arg('--ds-dir', ''), ds = arg('--ds', ''), label = arg('--label', 'js');
const n = Number(arg('--n', '100000')), nq = Number(arg('--queries', '1000')), budget = Number(arg('--budget', '15'));
const exps = new Set(arg('--exp', 'core').split(','));
const outPath = arg('--out', '');
const entry = resolve(here, '../src/index.js');
const { EmbeddingStore } = await import(pathToFileURL(entry).href);

const K = 10;
const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
const d = meta.d;

function readF32(path, start, count) {
  const fd = openSync(path, 'r');
  const buf = Buffer.alloc(count * d * 4);
  readSync(fd, buf, 0, buf.length, start * d * 4);
  closeSync(fd);
  const f = new Float32Array(buf.buffer, buf.byteOffset, count * d);
  const rows = new Array(count);
  for (let i = 0; i < count; i++) rows[i] = Array.from(f.subarray(i * d, (i + 1) * d));
  return rows;
}
const readI32 = path => { const b = readFileSync(path); return new Int32Array(b.buffer, b.byteOffset, b.length / 4); };
const base = (start, count) => readF32(join(dir, 'base.f32'), start, count);
const gt = kind => { const f = readI32(join(dir, `gt_${kind}${n}.i32`)); const out = []; for (let i = 0; i < f.length / 100; i++) out.push(Array.from(f.subarray(i * 100, i * 100 + K))); return out; };
const queries = readF32(join(dir, 'queries.f32'), 0, nq);

const now = () => Number(process.hrtime.bigint()) / 1e9;
const pct = (xs, p) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.round((s.length - 1) * p)] : 0; };
const mem = () => { const m = process.memoryUsage(); return { rss_mb: m.rss / 1048576, peak_mb: m.rss / 1048576, heap_mb: m.heapUsed / 1048576 }; };
let peakRss = 0;
const memP = () => { const m = mem(); peakRss = Math.max(peakRss, m.rss_mb); return { ...m, peak_mb: peakRss }; };
const idOf = r => Number(r.nodeRef[1]);
const search = (s, q, exact, ty) => s.similaritySearch(q, K, ty, 0.0, exact).map(idOf);
const recallOf = (found, truth) => { let hit = 0; for (let i = 0; i < found.length; i++) { const want = new Set(truth[i]); for (const x of found[i].slice(0, K)) if (want.has(x)) hit++; } return hit / (K * Math.max(1, found.length)); };

function point(s, truth, exact, ty) {
  let w0 = now();
  for (let i = 0; i < Math.min(50, nq); i++) { search(s, queries[i], exact, ty); if (now() - w0 > 3) break; }
  const probe = [];
  for (let i = 0; i < Math.min(5, nq); i++) { const t = now(); search(s, queries[i], exact, ty); probe.push(now() - t); }
  const est = Math.max(pct(probe, 0.5), 1e-7);
  const m = Math.floor(Math.min(nq, Math.max(30, budget / est)));
  const runs = [], lats = [];
  let rec = 0;
  for (let r = 0; r < 3; r++) {
    const lat = [], res = [];
    for (let i = 0; i < m; i++) { const t = now(); const ids = search(s, queries[i], exact, ty); lat.push(now() - t); if (r === 0) res.push(ids); }
    if (r === 0) rec = recallOf(res, truth.slice(0, m));
    const total = lat.reduce((a, b) => a + b, 0);
    runs.push(m / total); lats.push(...lat);
    if (total > budget * 1.5) break;
  }
  return { nq: m, recall: rec, qps_1t: pct(runs, 0.5), qps_1t_min: Math.min(...runs), qps_1t_max: Math.max(...runs), qps_1t_runs: runs,
    median_ms: pct(lats, 0.5) * 1000, p95_ms: pct(lats, 0.95) * 1000, p99_ms: pct(lats, 0.99) * 1000, mean_ms: lats.reduce((a, b) => a + b, 0) / lats.length * 1000 };
}
const nlistFor = x => Math.min(Math.max(Math.round(4 * Math.sqrt(x)), 16), 65536);
function knobs() { const nl = nlistFor(n); const v = [1, 2, 4, 8, 16, 32, 64, 128, 256].filter(x => x < nl); if (n <= 200000) v.push(nl); return v; }
function ingest(s, grp) {
  let secs = 0;
  const names = ['t10', 't01', 'rest'];
  for (let at = 0; at < n; at += 20000) {
    const m = Math.min(20000, n - at);
    const rows = base(at, m);
    const t = now();
    const entries = rows.map((v, i) => [[grp ? names[grp[at + i]] : 'doc', at + i], '', v]);
    s.addBatch(entries);
    secs += now() - t;
  }
  return secs;
}
const header = sys => ({ system: sys, ds, n, d, k: K, queries: nq, all_threads: cpus().length, version: 'isongraph-vector-js (repo HEAD, Node ' + process.version + ')' });

const runs = [];
const memBase = memP();
console.log(`${label}: ds=${ds} n=${n} d=${d} queries=${nq}`);
if (exps.has('core') || exps.has('mutation')) {
  let s = new EmbeddingStore(undefined, {});
  let ingestS = ingest(s, null);
  let memIng = memP();
  console.log(`  ingest ${ingestS.toFixed(2)}s`);
  const truth = gt('');
  let t0 = now();
  search(s, queries[0], true);
  const firstMs = (now() - t0) * 1000;
  const ex = header(`${label}_exact`);
  ex.build = { ingest_s: ingestS, index_s: 0, total_s: ingestS, first_query_ms: firstMs, ready_s: ingestS + firstMs / 1000 };
  ex.mem_baseline = memBase; ex.mem_after_ingest = memIng; ex.mem_after_build = memP(); ex.mem_after_first_query = memP();
  const p1 = point(s, truth, true); p1.knob = null;
  console.log(`  exact recall=${p1.recall} qps=${p1.qps_1t.toFixed(1)}`);
  ex.sweep = [p1]; ex.mem_after_sweep = memP();
  runs.push(ex);
  s = null;
  if (global.gc) global.gc();

  const st = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annSeed: 0 });
  ingestS = ingest(st, null);
  memIng = memP();
  t0 = now();
  const stats = st.buildAnnIndex();
  const idxS = now() - t0;
  console.log(`  ivf build ${idxS.toFixed(2)}s nlist=${stats.nlist}`);
  const iv = header(`${label}_ivf`);
  iv.knob_name = 'nprobe';
  iv.build = { ingest_s: ingestS, index_s: idxS, total_s: ingestS + idxS, first_query_ms: 0, ready_s: ingestS + idxS };
  iv.mem_baseline = memBase; iv.mem_after_ingest = memIng; iv.mem_after_build = memP();
  iv.index_stats = { nlist: stats.nlist, trained_n: stats.trainedN, index_bytes: stats.indexBytes, build_seconds: stats.buildSeconds };
  const pts = [];
  let refKnob = null, refRecall = 0;
  for (const kn of knobs()) {
    st.annNprobe = kn;
    const p = point(st, truth, false); p.knob = kn;
    console.log(`  ivf nprobe=${kn} recall=${p.recall.toFixed(4)} qps1t=${p.qps_1t.toFixed(0)}`);
    if (refKnob === null && p.recall >= 0.95) { refKnob = kn; refRecall = p.recall; }
    pts.push(p);
  }
  if (refKnob === null) refKnob = knobs().slice(-1)[0];
  iv.sweep = pts; iv.ref_knob = refKnob; iv.ref_recall_pre = refRecall; iv.mem_after_sweep = memP();
  if (exps.has('mutation')) {
    const gtm = join(dir, `gt_mut_${n}.i32`);
    if (existsSync(gtm)) {
      const upd = readI32(join(dir, `mut_${n}_upd.i32`)), del = readI32(join(dir, `mut_${n}_del.i32`));
      const f = readI32(gtm); const truthM = [];
      for (let i = 0; i < Math.min(nq, f.length / 100); i++) truthM.push(Array.from(f.subarray(i * 100, i * 100 + K)));
      st.annNprobe = refKnob;
      const m = { ref_knob: refKnob, recall_before: refRecall };
      const ins = base(n, 10000);
      let t = now();
      for (let i = 0; i < ins.length; i++) st.add(['doc', n + i], '', ins[i]);
      let dt = now() - t; m.insert_10k = { seconds: dt, per_s: 10000 / dt };
      const nv = base(n + 10000, 10000);
      t = now();
      for (let j = 0; j < nv.length; j++) st.add(['doc', upd[j]], '', nv[j]);
      dt = now() - t; m.update_10k = { seconds: dt, per_s: 10000 / dt };
      t = now();
      for (const id of del) st.remove(['doc', id]);
      dt = now() - t; m.delete_10k = { seconds: dt, per_s: 10000 / dt };
      t = now(); search(st, queries[0], false); m.first_query_after_ms = (now() - t) * 1000;
      const p = point(st, truthM, false);
      m.recall_after = p.recall; m.after = p; m.mem_after = memP();
      const s2 = st.annStats();
      m.index_stats_after = { pending: s2.pending, churn: s2.churn, trained_n: s2.trainedN };
      console.log(`  mutation recall_after=${p.recall}`);
      iv.mutation = m;
    } else iv.mutation = { skipped: 'dataset has no spare rows for this n' };
  }
  runs.push(iv);
}
if (exps.has('filter')) {
  const g = readFileSync(join(dir, `grp_${n}.u8`));
  const st = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annSeed: 0 });
  const ingestS = ingest(st, g);
  const t0 = now();
  st.buildAnnIndex();
  const idxS = now() - t0;
  const res = header(`${label}_ivf`);
  const subsets = {};
  res.filter = { build_with_metadata_s: ingestS + idxS, mem: memP(), subsets };
  const kn = knobs();
  const pick = [...new Set([1, Math.floor(kn.length / 3), Math.floor(kn.length / 2), Math.floor(2 * kn.length / 3), kn.length - 1])].sort((a, b) => a - b);
  for (const [ty, lab] of [['t10', 'f10'], ['t01', 'f01']]) {
    const truth = gt(lab + '_');
    const pts = [];
    for (const i of pick) {
      st.annNprobe = kn[i];
      const p = point(st, truth, false, ty); p.knob = kn[i];
      console.log(`  filter ${lab} nprobe=${kn[i]} recall=${p.recall.toFixed(4)} qps=${p.qps_1t.toFixed(0)}`);
      pts.push(p);
    }
    const pe = point(st, truth, true, ty); pe.knob = 'exact'; pts.push(pe);
    subsets[lab] = pts;
  }
  runs.push(res);
}
if (outPath) { writeFileSync(outPath, JSON.stringify({ status: 'ok', label, runs }, null, 1)); console.log(`wrote ${outPath}`); }
