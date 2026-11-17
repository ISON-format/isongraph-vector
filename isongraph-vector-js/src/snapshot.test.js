/**
 * The optional binary snapshot (DESIGN.md section 11 of the Rust crate): the
 * golden files the Rust crate wrote, a save -> load round trip of our own,
 * every rejection rule with its fall-back to the JSON sidecar, the save
 * protocol, and the single-copy table against a plain reference model.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, rmSync, readFileSync, writeFileSync, copyFileSync, existsSync, readdirSync, } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { EmbeddingStore, SemanticGraph, MockEncoder, graphDigest, selectTopK, } from './index.js';
import { SnapshotError, snapshotToBytes, readSnapshotBytes, applySnapshot, } from './snapshot.js';
import { saveGraph, loadGraphWithReport, loadSnapshot, saveSnapshot, snapshotPathFor, sidecarIdentity, embeddingsPathFor, } from './node.js';
const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'tests', 'fixtures', 'snapshot');
const dirs = [];
function tmp() {
    const dir = mkdtempSync(join(tmpdir(), 'igv-snap-'));
    dirs.push(dir);
    return dir;
}
afterEach(() => {
    while (dirs.length)
        rmSync(dirs.pop(), { recursive: true, force: true });
});
function sha() {
    const h = createHash('sha256');
    return { update: b => { h.update(b); }, digest: () => new Uint8Array(h.digest()) };
}
/** Recompute the trailer so that only the structural rule under test can reject the file. */
function reseal(bytes) {
    const body = bytes.subarray(0, bytes.length - 32);
    const out = new Uint8Array(bytes.length);
    out.set(body);
    out.set(createHash('sha256').update(body).digest(), body.length);
    return out;
}
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
/** Everything observable about a store's contents: order, keys, ids, texts, models and the vector bits. */
function dump(store) {
    return store.exportEmbeddings().embeddings.map(e => [
        e.type, e.id, e.id_type, e.text, e.model,
        Array.from(new Uint32Array(new Float32Array(e.vector).buffer)),
    ]);
}
/** A graph with numeric and string ids, two models, an update, a delete and a duplicate vector. */
function sampleGraph() {
    const graph = new SemanticGraph({ name: 'snap', encoder: new MockEncoder(8, 'v1'), autoEmbed: false });
    const next = rng(11);
    const vec = () => Array.from({ length: 8 }, () => Math.fround(next() * 2 - 1));
    for (let i = 1; i <= 30; i++) {
        graph.addNode('person', i, { name: `Person ${i}` });
        if (i === 12)
            graph.embeddingStore.add(['person', i], 'Person 12'); // from the encoder
        else
            graph.embeddingStore.add(['person', i], `Person ${i} \u{1F600} "quoted"\nnewline`, vec());
    }
    graph.embeddingStore.add(['doc', 'a:b'], 'id with a colon', vec()); // an embedding without a graph node (an id may hold a colon here)
    graph.embeddingStore.add(['person', 3], 'updated', vec()); // update in place
    graph.embeddingStore.remove(['person', 7]); // tombstone
    const dup = vec();
    graph.embeddingStore.add(['doc', 'x'], 'dup 1', dup); // an exact tie
    graph.addNode('doc', 'x', { name: 'x' });
    graph.embeddingStore.add(['doc', 'y'], 'dup 2', dup);
    graph.addNode('doc', 'y', { name: 'y' });
    graph.embeddingStore.add(['doc', 'z'], '', new Array(8).fill(0)); // a zero vector
    graph.addNode('doc', 'z', { name: 'z' });
    return graph;
}
function sameAnswers(a, b, dim, seed = 5) {
    const next = rng(seed);
    for (let i = 0; i < 12; i++) {
        const q = Array.from({ length: dim }, () => next() * 2 - 1);
        for (const [k, type, th] of [[10, undefined, -1], [3, 'doc', 0], [null, undefined, 0.2]]) {
            expect(JSON.stringify(a.similaritySearch(q, k, type, th))).toBe(JSON.stringify(b.similaritySearch(q, k, type, th)));
        }
    }
}
describe('snapshot written by the Rust crate (golden fixture)', () => {
    function copyFixture(dir, withSnap) {
        const path = join(dir, 'golden.ison');
        for (const name of ['golden.ison', 'golden.ison.embeddings.json', ...(withSnap ? ['golden.ison.embeddings.snap'] : [])]) {
            copyFileSync(join(FIXTURES, name), join(dir, name));
        }
        return path;
    }
    it('is read through the snapshot, and equals a JSON-only load of the same graph and sidecar', () => {
        const withSnap = loadGraphWithReport(copyFixture(tmp(), true));
        const jsonOnly = loadGraphWithReport(copyFixture(tmp(), false));
        expect(withSnap.report.source).toBe('snapshot');
        expect(withSnap.report.snapshotSkipped).toBeNull();
        expect(withSnap.report.embeddings).toBe(36);
        expect(jsonOnly.report.source).toBe('json');
        const a = withSnap.graph.embeddingStore;
        const b = jsonOnly.graph.embeddingStore;
        expect(a.count()).toBe(36);
        expect(a.dimension).toBe(12);
        expect(dump(a)).toEqual(dump(b)); // keys, texts, models, vector bits, order
        expect(a.models()).toEqual(b.models());
        expect(a.lastImportMeta).toEqual(b.lastImportMeta);
        expect(a.lastImportMeta.encoder_model).toBe('second-model');
        sameAnswers(withSnap.graph, jsonOnly.graph, 12);
    });
    it('is reproduced byte for byte by our writer from the loaded store', () => {
        const dir = tmp();
        const path = copyFixture(dir, true);
        const encoder = new MockEncoder(12);
        encoder.modelName = 'second-model'; // the header records the encoder's model
        const { graph } = loadGraphWithReport(path, { encoder });
        const meta = graph.embeddingStore.lastImportMeta;
        const out = join(dir, 'again.snap');
        saveSnapshot(graph.embeddingStore, out, {
            graphDigest: meta.graph_digest,
            sidecar: sidecarIdentity(embeddingsPathFor(path)),
        });
        expect(readFileSync(out)).toEqual(readFileSync(snapshotPathFor(path)));
    });
    it('keeps unicode, colons, empty and multi-line texts, a zero vector and a tie', () => {
        const { graph } = loadGraphWithReport(copyFixture(tmp(), true));
        const store = graph.embeddingStore;
        const rows = dump(store);
        expect(rows.some(r => r[3].includes('\u{1F600}'))).toBe(true);
        expect(rows.some(r => r[1].includes('id:with:colons'))).toBe(true);
        expect(rows.some(r => r[3] === '')).toBe(true);
        expect(rows.some(r => r[3].includes('\n'))).toBe(true);
        expect(rows.some(r => r[5].every(bits => bits === 0))).toBe(true);
        expect(new Set(rows.map(r => r[4])).size).toBe(2);
    });
    it('falls back to the JSON, untouched, when a copy of the snapshot is corrupt or truncated', () => {
        const jsonOnly = loadGraphWithReport(copyFixture(tmp(), false)).graph.embeddingStore;
        const original = new Uint8Array(readFileSync(join(FIXTURES, 'golden.ison.embeddings.snap')));
        const damaged = [
            ['a flipped float byte', (() => { const b = original.slice(); b[b.length - 100] ^= 0x40; return b; })()],
            ['a flipped trailer byte', (() => { const b = original.slice(); b[b.length - 1] ^= 1; return b; })()],
            ['half of the file', original.slice(0, original.length >> 1)],
            ['one byte short', original.slice(0, original.length - 1)],
            ['only the magic', original.slice(0, 8)],
            ['an empty file', new Uint8Array(0)],
        ];
        for (const [what, bytes] of damaged) {
            const dir = tmp();
            const path = copyFixture(dir, false);
            writeFileSync(snapshotPathFor(path), bytes);
            const { graph, report } = loadGraphWithReport(path);
            expect(report.source, what).toBe('json');
            expect(report.snapshotSkipped, what).not.toBeNull();
            expect(dump(graph.embeddingStore), what).toEqual(dump(jsonOnly));
        }
    });
});
describe('snapshot round trip of our own save', () => {
    it('saveGraph({ snapshot: true }) then loadGraph reads the snapshot and gives the same store', () => {
        const dir = tmp();
        const path = join(dir, 'g.ison');
        const graph = sampleGraph();
        saveGraph(graph, path, { snapshot: true });
        expect(existsSync(snapshotPathFor(path))).toBe(true);
        const viaSnap = loadGraphWithReport(path, { encoder: new MockEncoder(8, 'v1') });
        expect(viaSnap.report.source).toBe('snapshot');
        expect(viaSnap.report.embeddings).toBe(graph.embeddingStore.count());
        const noSnap = tmp();
        copyFileSync(path, join(noSnap, 'g.ison'));
        copyFileSync(embeddingsPathFor(path), join(noSnap, 'g.ison.embeddings.json'));
        const viaJson = loadGraphWithReport(join(noSnap, 'g.ison'), { encoder: new MockEncoder(8, 'v1') });
        expect(viaJson.report.source).toBe('json');
        expect(dump(viaSnap.graph.embeddingStore)).toEqual(dump(viaJson.graph.embeddingStore));
        expect(dump(viaSnap.graph.embeddingStore)).toEqual(dump(graph.embeddingStore)); // numeric ids stayed numbers
        expect(viaSnap.graph.getEmbedding(['person', 3]).nodeRef).toEqual(['person', 3]);
        expect(typeof viaSnap.graph.getEmbedding(['person', 3]).nodeRef[1]).toBe('number');
        expect(viaSnap.graph.getEmbedding(['person', 7])).toBeNull();
        sameAnswers(viaSnap.graph, viaJson.graph, 8);
        // text search works against the restored store too
        expect(viaSnap.graph.similaritySearch('Person 12', 1)[0].nodeRef).toEqual(['person', 12]);
    });
    it('writes the graph file and the JSON sidecar exactly as it does without a snapshot', () => {
        const graph = sampleGraph();
        const a = tmp();
        const b = tmp();
        saveGraph(graph, join(a, 'g.ison'));
        saveGraph(graph, join(b, 'g.ison'), { snapshot: true });
        expect(readFileSync(join(b, 'g.ison'))).toEqual(readFileSync(join(a, 'g.ison')));
        expect(readFileSync(embeddingsPathFor(join(b, 'g.ison')))).toEqual(readFileSync(embeddingsPathFor(join(a, 'g.ison'))));
    });
    it('spans several blocks, and the DataView path writes and reads the same bytes as the fast path', () => {
        const store = new EmbeddingStore();
        const next = rng(3);
        for (let i = 0; i < 2600; i++) {
            store.add(['n', `id${i}`], `text ${i}`, Array.from({ length: 6 }, () => Math.fround(next() - 0.5)));
        }
        const fast = snapshotToBytes(store, {}, sha());
        const portable = snapshotToBytes(store, {}, sha(), { portable: true });
        expect(portable.bytes).toEqual(fast.bytes);
        expect(fast.info.count).toBe(2600);
        for (const options of [{}, { portable: true }, { windowBytes: 64 }, { windowBytes: 1000, portable: true }]) {
            const copy = new EmbeddingStore();
            applySnapshot(copy, readSnapshotBytes(fast.bytes, sha(), options));
            expect(dump(copy)).toEqual(dump(store));
        }
        const small = snapshotToBytes(store, {}, sha(), { blockRows: 7 }); // the reader honours the header's block size
        const copy = new EmbeddingStore();
        applySnapshot(copy, readSnapshotBytes(small.bytes, sha()));
        expect(dump(copy)).toEqual(dump(store));
    });
    it('round-trips an empty store', () => {
        const store = new EmbeddingStore();
        const { bytes } = snapshotToBytes(store, {}, sha());
        const again = new EmbeddingStore();
        applySnapshot(again, readSnapshotBytes(bytes, sha()));
        expect(again.count()).toBe(0);
        expect(again.dimension).toBeNull();
    });
    it('matches the byte layout of the specification example (123 bytes)', () => {
        const store = new EmbeddingStore();
        store.add(['t', 'a'], 'x', [1.0, -2.0]);
        store.version = null;
        const { bytes } = snapshotToBytes(store, {}, sha());
        const u32 = (n) => [n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255];
        const ascii = (s) => Array.from(s).map(c => c.charCodeAt(0));
        const expected = [
            ...ascii('ISONVSNP'), ...u32(1), ...u32(0),
            ...u32(2), ...u32(1024), ...u32(1), ...u32(0),
            ...u32(0xffffffff), ...u32(0xffffffff), ...u32(0xffffffff), ...u32(0xffffffff),
            ...u32(1), ...u32(8), ...ascii('provided'),
            ...u32(1), ...ascii('t'), ...u32(1), ...ascii('a'), ...u32(1), ...ascii('x'), ...u32(0),
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0,
        ];
        expect(expected).toHaveLength(91);
        expect(bytes).toHaveLength(123);
        expect(Array.from(bytes.subarray(0, 91))).toEqual(expected);
        expect(Buffer.from(bytes.subarray(91)).toString('hex')).toBe(createHash('sha256').update(Uint8Array.from(expected)).digest('hex'));
    });
});
describe('snapshot rejection rules', () => {
    /** A two-row snapshot ('t','a','x') and ('t','b','y'), the store it came from, and its bytes. */
    function twoRows() {
        const store = new EmbeddingStore();
        store.add(['t', 'a'], 'x', [1, 0, 0]);
        store.add(['t', 'b'], 'y', [0, 1, 0]);
        return { store, bytes: snapshotToBytes(store, {}, sha()).bytes };
    }
    const dv = (b) => new DataView(b.buffer, b.byteOffset, b.byteLength);
    function find(bytes, needle, from = 0) {
        outer: for (let i = from; i <= bytes.length - needle.length; i++) {
            for (let j = 0; j < needle.length; j++)
                if (bytes[i + j] !== needle[j])
                    continue outer;
            return i;
        }
        throw new Error('pattern not found');
    }
    const reject = (bytes, reason, options = {}) => {
        const target = new EmbeddingStore();
        target.add(['keep', 1], 'kept', [9, 9, 9]);
        const before = dump(target);
        expect(() => applySnapshot(target, readSnapshotBytes(bytes, sha(), options))).toThrow(SnapshotError);
        try {
            readSnapshotBytes(bytes, sha(), options);
        }
        catch (error) {
            expect(error.reason).toMatch(reason);
        }
        expect(dump(target)).toEqual(before); // nothing was installed
        expect(target.dimension).toBe(3);
    };
    it('rejects bad magic, an unknown version, unknown flags and an implausible shape', () => {
        const { bytes } = twoRows();
        let b = bytes.slice();
        b[0] = 0x58;
        reject(reseal(b), /bad magic/);
        b = bytes.slice();
        dv(b).setUint32(8, 2, true);
        reject(reseal(b), /version 2/);
        b = bytes.slice();
        dv(b).setUint32(12, 2, true);
        reject(reseal(b), /unknown flags/);
        b = bytes.slice();
        dv(b).setUint32(20, 0, true);
        reject(reseal(b), /block size/);
        b = bytes.slice();
        dv(b).setUint32(20, (1 << 20) + 1, true);
        reject(reseal(b), /block size/);
        b = bytes.slice();
        dv(b).setUint32(16, 0, true);
        reject(reseal(b), /implausible shape/);
        b = bytes.slice();
        dv(b).setUint32(24, 0xffffffff, true);
        dv(b).setUint32(28, 1, true);
        reject(reseal(b), /implausible shape/);
        b = bytes.slice();
        dv(b).setUint32(24, 1000, true);
        reject(reseal(b), /fewer bytes than the rows need/);
    });
    it('rejects a repeated key, a model index out of range and a string past the end', () => {
        const { bytes } = twoRows();
        const dup = bytes.slice();
        const idB = find(dup, [1, 0, 0, 0, 0x74, 1, 0, 0, 0, 0x62]); // type 't', id 'b'
        dup[idB + 9] = 0x61; // ... now 'a': the key t:a twice
        reject(reseal(dup), /appears twice/);
        const model = bytes.slice();
        const rowA = find(model, [1, 0, 0, 0, 0x74, 1, 0, 0, 0, 0x61, 1, 0, 0, 0, 0x78]);
        dv(model).setUint32(rowA + 15, 5, true); // model_index 5 of 1
        reject(reseal(model), /model that is not listed/);
        const long = bytes.slice();
        dv(long).setUint32(find(long, [1, 0, 0, 0, 0x74, 1, 0, 0, 0, 0x61]), 0x7ffffff0, true);
        reject(reseal(long), /longer than the file/);
        const absent = bytes.slice();
        dv(absent).setUint32(find(absent, [1, 0, 0, 0, 0x74, 1, 0, 0, 0, 0x61]), 0xffffffff, true);
        reject(reseal(absent), /marked absent/);
    });
    it('rejects bytes between the last block and the trailer, bytes after it, and a wrong checksum', () => {
        const { bytes } = twoRows();
        const between = new Uint8Array(bytes.length + 1);
        between.set(bytes.subarray(0, bytes.length - 32));
        between.set(bytes.subarray(bytes.length - 32), bytes.length - 31);
        reject(reseal(between), /left over/);
        const after = new Uint8Array(bytes.length + 1);
        after.set(bytes);
        reject(after, /./);
        const bad = bytes.slice();
        bad[bytes.length - 40] ^= 1; // inside the last row's floats
        reject(bad, /checksum/);
    });
    it('rejects a truncation at every length', () => {
        const { bytes } = twoRows();
        for (let n = 0; n < bytes.length; n++) {
            expect(() => readSnapshotBytes(bytes.slice(0, n), sha()), `length ${n}`).toThrow(SnapshotError);
        }
    });
    it('rejects a string that is not UTF-8', () => {
        const { bytes } = twoRows();
        const b = bytes.slice();
        b[find(b, [1, 0, 0, 0, 0x74, 1, 0, 0, 0, 0x61, 1, 0, 0, 0, 0x78]) + 14] = 0xff; // the text of row a
        reject(reseal(b), /not UTF-8/);
    });
    it('refuses what `accept` refuses before reading any row', () => {
        const { bytes } = twoRows();
        expect(() => readSnapshotBytes(bytes, sha(), { accept: () => 'not for you' })).toThrow(/not for you/);
    });
    it('refuses to write a lone surrogate (UTF-8 cannot carry it)', () => {
        const store = new EmbeddingStore();
        store.add(['t', 'a'], 'bad \ud800 text', [1]);
        expect(() => snapshotToBytes(store, {}, sha())).toThrow(SnapshotError);
    });
});
describe('snapshot protocol: which file wins, and why a snapshot is skipped', () => {
    function saved() {
        const dir = tmp();
        const path = join(dir, 'g.ison');
        const graph = sampleGraph();
        saveGraph(graph, path, { snapshot: true });
        return { dir, path, graph };
    }
    it('skips a snapshot whose sidecar changed since (stale) and loads the new JSON', () => {
        const { path, graph } = saved();
        const sidecar = embeddingsPathFor(path);
        const payload = JSON.parse(readFileSync(sidecar, 'utf8'));
        payload.embeddings[0].text = 'edited after the snapshot';
        writeFileSync(sidecar, JSON.stringify(payload));
        const { graph: loaded, report } = loadGraphWithReport(path);
        expect(report.source).toBe('json');
        expect(report.snapshotSkipped).toMatch(/stale/);
        expect(loaded.embeddingStore.get(['person', 1]).text).toBe('edited after the snapshot');
        expect(graph.embeddingStore.get(['person', 1]).text).not.toBe('edited after the snapshot');
    });
    it('skips a snapshot saved for a different graph digest', () => {
        const { path, graph } = saved();
        saveSnapshot(graph.embeddingStore, snapshotPathFor(path), {
            graphDigest: 'sha256:' + '0'.repeat(64),
            sidecar: sidecarIdentity(embeddingsPathFor(path)),
        });
        const { report } = loadGraphWithReport(path);
        expect(report.source).toBe('json');
        expect(report.snapshotSkipped).toMatch(/different graph/);
    });
    it('skips a snapshot that records no digest or no sidecar identity', () => {
        const { path, graph } = saved();
        const digest = graphDigest(graph);
        saveSnapshot(graph.embeddingStore, snapshotPathFor(path), { sidecar: sidecarIdentity(embeddingsPathFor(path)) });
        expect(loadGraphWithReport(path).report.snapshotSkipped).toMatch(/different graph/);
        saveSnapshot(graph.embeddingStore, snapshotPathFor(path), { graphDigest: digest });
        expect(loadGraphWithReport(path).report.snapshotSkipped).toMatch(/different graph|no sidecar identity/);
    });
    it('skips a snapshot when the graph file was edited (with onMismatch: warn for the JSON)', () => {
        const { path } = saved();
        writeFileSync(path, readFileSync(path, 'utf8').replace('Person 5', 'Person five'));
        const quiet = () => { };
        const warn = console.warn;
        console.warn = quiet;
        try {
            const { report } = loadGraphWithReport(path, { onMismatch: 'warn' });
            expect(report.source).toBe('json');
            expect(report.snapshotSkipped).toMatch(/different graph/);
        }
        finally {
            console.warn = warn;
        }
    });
    it('never uses a snapshot when the sidecar is missing', () => {
        const { path } = saved();
        rmSync(embeddingsPathFor(path));
        const { graph, report } = loadGraphWithReport(path);
        expect(report.source).toBe('none');
        expect(graph.embeddingStore.count()).toBe(0);
    });
    it('embeddings: false ignores the snapshot too', () => {
        const { path } = saved();
        expect(loadGraphWithReport(path, { embeddings: false }).report.source).toBe('none');
    });
    it('a save without a snapshot deletes the old one, and a plain load then reads the JSON', () => {
        const { path, graph } = saved();
        graph.embeddingStore.add(['person', 1], 'changed', new Array(8).fill(0.25));
        saveGraph(graph, path);
        expect(existsSync(snapshotPathFor(path))).toBe(false);
        const { graph: loaded, report } = loadGraphWithReport(path);
        expect(report.source).toBe('json');
        expect(report.snapshotSkipped).toBeNull();
        expect(loaded.embeddingStore.get(['person', 1]).text).toBe('changed');
    });
    it('a snapshot that cannot be written still saves the pair, removes a stale snapshot and throws', () => {
        const { path, graph } = saved();
        graph.embeddingStore.add(['orphan', 5], 'an integer id for a node the graph does not have', new Array(8).fill(0.5));
        expect(() => saveGraph(graph, path, { snapshot: true })).toThrow(SnapshotError);
        expect(existsSync(snapshotPathFor(path))).toBe(false);
        expect(readdirSync(dirname(path)).filter(f => f.endsWith('.tmp'))).toEqual([]);
        const { graph: loaded, report } = loadGraphWithReport(path); // the pair was saved: JSON reads fine
        expect(report.source).toBe('json');
        expect(loaded.embeddingStore.get(['orphan', 5]).text).toMatch(/orphan|integer id/);
        expect(typeof loaded.embeddingStore.get(['orphan', 5]).nodeRef[1]).toBe('number');
    });
    it('a failing write leaves the previous snapshot untouched and no temp file behind', () => {
        const { dir, path, graph } = saved();
        const before = readFileSync(snapshotPathFor(path));
        const bad = new EmbeddingStore();
        bad.add(['t', 'a'], 'lone \ud800 surrogate', [1, 2]);
        expect(() => saveSnapshot(bad, snapshotPathFor(path), {})).toThrow(SnapshotError);
        expect(readFileSync(snapshotPathFor(path))).toEqual(before);
        expect(readdirSync(dir).filter(f => f.endsWith('.tmp'))).toEqual([]);
        expect(loadGraphWithReport(path).report.source).toBe('snapshot');
        expect(graph.embeddingStore.count()).toBeGreaterThan(0);
    });
    it('loadSnapshot replaces the store only when the whole file verifies', () => {
        const { path, graph } = saved();
        const snap = snapshotPathFor(path);
        const store = new EmbeddingStore();
        store.add(['keep', 1], 'kept', new Array(8).fill(1));
        const bytes = new Uint8Array(readFileSync(snap));
        const broken = bytes.slice();
        broken[bytes.length - 50] ^= 8;
        writeFileSync(snap, broken);
        expect(() => loadSnapshot(store, snap)).toThrow(SnapshotError);
        expect(store.count()).toBe(1);
        expect(store.get(['keep', 1]).text).toBe('kept');
        writeFileSync(snap, bytes);
        const info = loadSnapshot(store, snap);
        expect(info.count).toBe(graph.embeddingStore.count());
        expect(store.get(['keep', 1])).toBeNull();
        expect(store.count()).toBe(info.count);
        expect(store.get(['person', '1']).nodeRef).toEqual(['person', '1']); // no id type: a bare load reads ids as text
    });
    it('is deterministic: saving the same store twice gives the same bytes', () => {
        const { path, graph } = saved();
        const first = readFileSync(snapshotPathFor(path));
        saveGraph(graph, path, { snapshot: true });
        expect(readFileSync(snapshotPathFor(path))).toEqual(first);
    });
});
describe('the single-copy table against a plain reference', () => {
    /** What the store computed before the rework: float64 cosine over number[] copies, in arrival order. */
    class Reference {
        constructor() {
            this.rows = new Map();
        }
        add(ref, text, vector) {
            this.rows.set(`${ref[0]}:${ref[1]}`, { ref, text, vector: vector.map(Math.fround) });
        }
        remove(ref) {
            this.rows.delete(`${ref[0]}:${ref[1]}`);
        }
        static cosine(q, v) {
            let dot = 0, n1 = 0, n2 = 0;
            for (let i = 0; i < q.length; i++) {
                dot += q[i] * v[i];
                n1 += q[i] * q[i];
                n2 += v[i] * v[i];
            }
            n1 = Math.sqrt(n1);
            n2 = Math.sqrt(n2);
            return n1 === 0 || n2 === 0 ? 0 : dot / (n1 * n2);
        }
        search(q, k, type, threshold) {
            const out = [];
            for (const row of this.rows.values()) {
                if (type && row.ref[0] !== type)
                    continue;
                const score = Reference.cosine(q, row.vector);
                if (score >= threshold)
                    out.push({ nodeRef: row.ref, score, text: row.text });
            }
            return k === null ? out.sort((a, b) => b.score - a.score) : selectTopK(out, k);
        }
        scoreMap(q) {
            return [...this.rows].map(([key, row]) => [key, Reference.cosine(q, row.vector)]);
        }
    }
    it('returns the same ids, scores (bit for bit) and tie order after updates, deletes and compaction', () => {
        const next = rng(21);
        const dim = 9;
        const vec = () => Array.from({ length: dim }, () => Math.fround(next() * 2 - 1));
        const store = new EmbeddingStore();
        const ref = new Reference();
        const types = ['a', 'b', 'c:d'];
        const refOf = (i) => [types[i % 3], i % 5 === 0 ? `s${i}` : i];
        const pool = [];
        for (let i = 0; i < 2300; i++) { // crosses two chunk boundaries
            const r = next();
            const v = r < 0.06 && pool.length ? pool[Math.floor(next() * pool.length)] : r < 0.09 ? new Array(dim).fill(0) : vec();
            pool.push(v);
            store.add(refOf(i), `t${i}`, v);
            ref.add(refOf(i), `t${i}`, v);
        }
        for (let i = 0; i < 2300; i += 4) {
            const v = vec();
            store.add(refOf(i), `u${i}`, v);
            ref.add(refOf(i), `u${i}`, v);
        }
        for (let i = 1; i < 2300; i += 3) {
            store.remove(refOf(i));
            ref.remove(refOf(i));
        } // more than a quarter: compaction
        for (let i = 0; i < 30; i++) {
            const v = vec();
            store.add(refOf(i), `again${i}`, v);
            ref.add(refOf(i), `again${i}`, v);
        }
        expect(store.count()).toBe(ref.rows.size);
        for (let i = 0; i < 25; i++) {
            const q = i === 0 ? new Array(dim).fill(0) : i < 3 ? pool[i].slice() : vec();
            for (const k of [1, 7, 10, 300, 400, null])
                for (const type of [undefined, 'b', 'nope'])
                    for (const th of [0, -1, 0.4]) {
                        expect(store.similaritySearch(q, k, type, th), `k=${k} type=${type} th=${th}`).toEqual(ref.search(q, k, type, th));
                    }
            expect([...store.scoreMap(q)]).toEqual(ref.scoreMap(q));
        }
    });
    it('gives a get() copy: changing its vector changes nothing in the store', () => {
        const store = new EmbeddingStore();
        store.add(['t', 1], 'x', [1, 2, 3]);
        const record = store.get(['t', 1]);
        record.vector[0] = 99;
        expect(store.get(['t', 1]).vector).toEqual([1, 2, 3]);
        expect(store.similaritySearch([1, 2, 3], 1)[0].score).toBeCloseTo(1, 12);
    });
    it('keeps one copy: the index reports no cache and the vectors take about 4 bytes per component', () => {
        const store = new EmbeddingStore(undefined, { ann: 'ivf', annMinSize: 0, annNlist: 16, annNprobe: 4 });
        const next = rng(2);
        for (let i = 0; i < 3000; i++)
            store.add(['d', i], '', Array.from({ length: 32 }, () => next() - 0.5));
        store.buildAnnIndex();
        const stats = store.annStats();
        expect(stats.active).toBe(true);
        expect(stats.cacheBytes).toBe(0);
        expect(stats.vectorBytes).toBeGreaterThanOrEqual(3000 * 32 * 4);
        expect(stats.vectorBytes).toBeLessThan(3000 * 32 * 4 * 1.2);
    });
});
