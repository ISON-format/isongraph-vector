/**
 * Node-only persistence for SemanticGraph.
 *
 * The core entry (`isongraph-vector-js`) stays browser-safe, so everything
 * that touches the filesystem lives here (`isongraph-vector-js/node`).
 *
 * The ISON graph format carries nodes and edges only, so embeddings go to a
 * JSON sidecar at `<path>.embeddings.json` holding the same `ison-embeddings`
 * payload every port reads and writes. A graph file and sidecar written by the
 * Python port load here, and the other way round.
 */

import {
    readFileSync, writeFileSync, existsSync, statSync, openSync, fsyncSync, closeSync, renameSync,
    unlinkSync, fstatSync, readSync, writeSync,
} from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { basename, dirname, extname, join } from 'node:path';
import { ISONGraph } from 'ison-graph-js';
import {
    SemanticGraph, EmbeddingError, GraphEmbeddingMismatchError, GRAPH_DIGEST_BASIS,
    EMBEDDING_SIDECAR_SUFFIX, graphDigest,
} from './index.js';
import {
    SNAPSHOT_SUFFIX, SnapshotError, writeSnapshot, readSnapshot, applySnapshot, findUnrestorableId,
} from './snapshot.js';

/** Suffix appended to a graph path to locate its embeddings. */
export { EMBEDDING_SIDECAR_SUFFIX };

/** Sidecar path holding the embeddings for a saved graph. */
export function embeddingsPathFor(path) {
    return String(path) + EMBEDDING_SIDECAR_SUFFIX;
}

function resolveFormat(path, format) {
    if (format === 'ison' || format === 'isonl')
        return format;
    if (format !== undefined && format !== 'auto') {
        throw new Error(`unknown graph format '${format}' (expected 'ison', 'isonl' or 'auto')`);
    }
    return extname(String(path)) === '.isonl' ? 'isonl' : 'ison';
}

/**
 * Write `text` to `path` so a reader sees the old file or the new one.
 *
 * The text goes to a temp file in the same directory (same filesystem, so the
 * rename is not a copy), is flushed to disk, and is then renamed into place. A
 * crash mid-write leaves the previous file intact instead of a truncated one,
 * and a failed write removes its temp file.
 */
export function writeFileAtomic(path, text) {
    const target = String(path);
    const tmp = join(dirname(target), `.${basename(target)}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`);
    let fd;
    try {
        fd = openSync(tmp, 'w');
        writeFileSync(fd, text, 'utf8');
        try {
            fsyncSync(fd);
        }
        catch {
            // fsync unsupported on this filesystem; the rename still gives atomicity.
        }
        closeSync(fd);
        fd = undefined;
        renameSync(tmp, target);
    }
    catch (error) {
        if (fd !== undefined) {
            try { closeSync(fd); } catch { /* already failing */ }
        }
        try { unlinkSync(tmp); } catch { /* temp file never created */ }
        throw error;
    }
}

/**
 * Write every embedding of `graph` to a JSON sidecar, atomically. Returns the
 * count. `digest` (see graphDigest) ties the sidecar to one particular graph.
 *
 * Also accepts a bare store (an EmbeddingStore, or the SqliteEmbeddingStore
 * from `isongraph-vector-js/sqlite`) in place of the graph.
 */
export function saveEmbeddings(graph, path, digest) {
    const store = graph.embeddingStore ?? graph;
    const payload = store.exportEmbeddings(digest);
    writeFileAtomic(path, JSON.stringify(payload));
    return payload.count;
}

/**
 * Read a JSON sidecar written by saveEmbeddings (or any port) into `graph`.
 * Returns the number of embeddings loaded.
 */
export function loadEmbeddings(graph, path, replace = false) {
    const payload = JSON.parse(readFileSync(path, 'utf8'));
    return (graph.embeddingStore ?? graph).importEmbeddings(payload, replace);
}

// =============================================================================
// Binary snapshot (optional, beside the JSON sidecar)
// =============================================================================
export { SNAPSHOT_SUFFIX, SnapshotError };
/** `<graph path>.embeddings.snap`. */
export function snapshotPathFor(path) {
    return path + SNAPSHOT_SUFFIX;
}
/** SHA-256 over `node:crypto`, in the shape the browser-safe snapshot module asks for. */
function sha256() {
    const hash = createHash('sha256');
    return {
        update: bytes => {
            hash.update(bytes);
        },
        digest: () => new Uint8Array(hash.digest()),
    };
}
/** Bytes sampled from the head and tail of a sidecar, and per probe (see DESIGN.md 11.1). */
const FP_EDGE = 64 * 1024;
const FP_PROBE = 4 * 1024;
const FP_PROBES = 64;
/**
 * Identity of a sidecar file: its length and a SHA-256 over the length (u64 LE),
 * its first and last 64 KiB and 64 probes of 4 KiB at `floor(len * i / 65)`.
 * Any serializer rewriting the sidecar changes it; an in-place edit that keeps
 * the length and touches no sampled byte does not.
 */
export function sidecarIdentity(path) {
    const fd = openSync(path, 'r');
    try {
        const len = fstatSync(fd).size;
        const hash = createHash('sha256');
        const head = Buffer.alloc(8);
        head.writeBigUInt64LE(BigInt(len));
        hash.update(head);
        const take = (at, want) => {
            const n = Math.min(want, Math.max(0, len - at));
            const buf = Buffer.alloc(n);
            let got = 0;
            while (got < n) {
                const read = readSync(fd, buf, got, n - got, at + got);
                if (read === 0)
                    throw new EmbeddingError(`${path} changed while its identity was being read`);
                got += read;
            }
            hash.update(buf);
        };
        take(0, FP_EDGE);
        take(Math.max(0, len - FP_EDGE), FP_EDGE);
        for (let i = 1; i <= FP_PROBES; i++)
            take(Number((BigInt(len) * BigInt(i)) / BigInt(FP_PROBES + 1)), FP_PROBE);
        return { len, fingerprint: new Uint8Array(hash.digest()) };
    }
    finally {
        closeSync(fd);
    }
}
/**
 * Write whatever `fill` streams to `write` into `path`, so a reader sees the old
 * file or the new one: a temp file in the same directory, flushed, renamed over
 * `path`. A failure removes the temp file and leaves `path` as it was.
 */
function writeStreamAtomic(path, fill) {
    const tmp = join(dirname(path), `.${basename(path)}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`);
    let fd;
    try {
        fd = openSync(tmp, 'w');
        const handle = fd;
        const value = fill(bytes => {
            let at = 0;
            while (at < bytes.length)
                at += writeSync(handle, bytes, at, bytes.length - at);
        });
        try {
            fsyncSync(fd);
        }
        catch {
            // fsync unsupported on this filesystem; the rename still gives atomicity.
        }
        closeSync(fd);
        fd = undefined;
        renameSync(tmp, path);
        return value;
    }
    catch (error) {
        if (fd !== undefined) {
            try {
                closeSync(fd);
            }
            catch { /* already failing */ }
        }
        try {
            unlinkSync(tmp);
        }
        catch { /* temp file never created */ }
        throw error;
    }
}
/**
 * Write a binary snapshot of every embedding in `store` to `path`, atomically (a
 * temp file, flushed, renamed over `path`: a failure or crash leaves the
 * previous file untouched). `identity` ties it to the graph and sidecar it was
 * saved with; a load that finds a different identity refuses it. Prefer
 * `saveGraph(graph, path, { snapshot: true })`, which does all of that.
 */
export function saveSnapshot(store, path, identity = {}, options = {}) {
    return writeStreamAtomic(path, write => writeSnapshot(store, identity, sha256(), write, options));
}
function readSnapshotFile(path, options) {
    const fd = openSync(path, 'r');
    try {
        const size = fstatSync(fd).size;
        return readSnapshot(into => readSync(fd, into, 0, into.length, null), size, sha256(), options);
    }
    finally {
        closeSync(fd);
    }
}
/**
 * Replace the store's embeddings with those of a snapshot file. Nothing changes
 * unless the whole file verifies (structure and checksum); throws SnapshotError
 * otherwise. Does not check identity (pass `accept` to), and ids come back as
 * strings unless `resolveId` says otherwise: use `loadGraph`, which does both.
 */
export function loadSnapshot(store, path, options = {}) {
    const result = readSnapshotFile(path, options);
    applySnapshot(store, result);
    return result.info;
}
/**
 * How a snapshot's id strings map back to node ids, given the graph they were
 * saved with: an id that names a node of that type as a string stays a string;
 * a canonical integer string that names a node as a number becomes the number;
 * anything else stays a string. (The snapshot format carries no id type; the
 * JSON sidecar's `id_type` is what this stands in for.)
 */
function idResolverFor(graph) {
    return (type, id) => {
        if (graph.hasNode(type, id))
            return id;
        const n = Number(id);
        if (id !== '' && String(n) === id && graph.hasNode(type, n))
            return n;
        return id;
    };
}
function removeQuietly(path) {
    try {
        unlinkSync(path);
    }
    catch {
        // no such file, or it cannot be removed: either way there is nothing to keep stale
    }
}

/**
 * Save a graph to a file, and its embeddings alongside it.
 *
 * Both files are written atomically (temp file, fsync, rename), so a crash
 * never leaves a truncated file. The pair cannot be replaced as one unit, so
 * the sidecar also records a graphDigest of the graph it belongs to; loadGraph
 * checks it and catches a crash between the two writes, a sidecar copied from
 * another save, and a graph edited afterwards. The digest is taken of the
 * graph *as it reads back from the text just written*, so a round trip that
 * normalises a value does not cause a false mismatch.
 *
 * With `embeddings: false` an existing sidecar is left in place (and will fail
 * the digest check on load, as it no longer matches).
 *
 * @param graph      SemanticGraph (a plain ISONGraph saves without a sidecar)
 * @param path       graph file path
 * @param options.format      'ison', 'isonl' or 'auto' (by extension; default)
 * @param options.embeddings  write the sidecar (default true; false for
 *                            graph-only saves)
 * @param options.snapshot    also write a binary snapshot of the embeddings at
 *                            `<path>.embeddings.snap` (default false), after and
 *                            in addition to the JSON sidecar, which stays the
 *                            portable copy. loadGraph uses it, when it provably
 *                            belongs to the graph and sidecar on disk, instead
 *                            of parsing the JSON. A save without it deletes an
 *                            existing snapshot, which the new sidecar would make
 *                            stale. If the snapshot cannot be written the graph
 *                            and sidecar are saved anyway, a stale snapshot is
 *                            removed, and the error is thrown (a SnapshotError
 *                            when the store holds something the snapshot cannot
 *                            carry, such as an integer id for a node the graph
 *                            does not have).
 */
export function saveGraph(graph, path, options = {}) {
    const { format = 'auto', embeddings = true, snapshot = false } = options;
    const kind = resolveFormat(path, format);
    const text = kind === 'isonl' ? graph.toIsonl() : graph.toIson();
    writeFileAtomic(path, text);
    if (embeddings && graph.embeddingStore) {
        const stem = basename(String(path), extname(String(path)));
        const reread = kind === 'isonl' ? ISONGraph.fromIsonl(text, stem) : ISONGraph.fromIson(text, stem);
        const digest = graphDigest(reread);
        const sidecar = embeddingsPathFor(path);
        const snap = snapshotPathFor(path);
        if (!snapshot)
            removeQuietly(snap); // the sidecar is about to change: a snapshot of the old one is stale
        saveEmbeddings(graph, sidecar, digest);
        if (snapshot) {
            try {
                const problem = findUnrestorableId(graph.embeddingStore, idResolverFor(reread));
                if (problem !== null)
                    throw new SnapshotError(`not written: ${problem}`);
                saveSnapshot(graph.embeddingStore, snap, {
                    graphDigest: digest,
                    sidecar: sidecarIdentity(sidecar),
                });
            }
            catch (error) {
                removeQuietly(snap);
                throw error;
            }
        }
    }
}

/**
 * Number literals this port reads differently from Python, so a digest taken
 * here can differ from one taken there for an unedited file: integral floats
 * (`1.0` - a JS number cannot tell it from `1`) and integers beyond 2^53.
 */
function hasInexactNumberLiteral(text) {
    return /(?:^|[\s|])[-+]?(?:\d+\.0+|\d{16,})(?=$|[\s|])/m.test(text);
}

/** Warn when the sidecar's (or snapshot's) encoder version is not the one in use. */
function warnOnVersionSkew(graph) {
    const storedVersion = graph.embeddingStore.lastImportMeta.encoder_version;
    const currentVersion = graph.embeddingStore.version;
    if (storedVersion !== undefined && currentVersion !== null &&
        String(storedVersion) !== String(currentVersion)) {
        console.warn(`embeddings were made by encoder version '${storedVersion}', ` +
            `but the encoder in use is version '${currentVersion}'`);
    }
}

/**
 * Load a SemanticGraph from a file, restoring embeddings from its sidecar.
 *
 * Nodes are copied across without auto-embedding: the embeddings come from
 * the sidecar, already computed, and re-deriving them would need an encoder
 * and overwrite what was saved.
 *
 * If the sidecar records a `graph_digest` it is checked against the graph just
 * read (see `onMismatch`). The comparison is skipped when the digest is absent
 * or taken over a basis this build does not know. Where the file holds number
 * literals this port cannot read exactly like Python (see graphDigest), a
 * mismatch is reported as a warning whatever `onMismatch` says, since it may
 * be a false positive. If both the encoder's version and the sidecar's
 * `encoder_version` are set and differ, a warning is issued.
 *
 * @param path  graph file path
 * @param options.encoder      encoder for the loaded graph
 * @param options.format       'ison', 'isonl' or 'auto' (default)
 * @param options.embeddings   restore the sidecar when present (default true)
 * @param options.autoEmbed    auto-embed nodes added later (default true)
 * @param options.embedFields  fields concatenated for auto-embedding
 * @param options.name         graph name (default: the file name without its extension)
 * @param options.version      encoder version recorded for the loaded graph
 * @param options.onModelMismatch  'warn' (default), 'error' or 'ignore' for text
 *                             searches whose encoder model differs from the stored vectors'
 * @param options.onMismatch   what to do when the sidecar's graph digest does not match
 *                             the graph file: 'error' (default) throws
 *                             GraphEmbeddingMismatchError, 'warn' loads the vectors anyway,
 *                             'reembed' discards them and re-embeds every node
 *
 * A binary snapshot beside the sidecar (saveGraph with `snapshot: true`, or the
 * Rust crate's `save_with_snapshot`) is used instead of the JSON when its header
 * names this graph's digest and the sidecar file as it is now and its checksum
 * verifies; the result is the same store the JSON gives. Any other outcome
 * falls back to the JSON, exactly as without a snapshot. Use loadGraphWithReport
 * to see which one was used.
 */
export function loadGraph(path, options = {}) {
    return loadGraphWithReport(path, options).graph;
}

/**
 * loadGraph, also saying where the embeddings came from:
 * `{ graph, report: { source, snapshotSkipped, embeddings, seconds } }`, where
 * `source` is 'snapshot' (the binary file), 'json' (the sidecar) or 'none'
 * (embeddings off, or no sidecar) and `snapshotSkipped` is why a snapshot that
 * exists was not used (null when it was used or there is none).
 */
export function loadGraphWithReport(path, options = {}) {
    const started = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const {
        format = 'auto', embeddings = true, encoder, autoEmbed, embedFields,
        version, onModelMismatch, onMismatch = 'error',
        ann, annNlist, annNprobe, annMinSize, annSeed,
    } = options;
    if (onMismatch !== 'error' && onMismatch !== 'warn' && onMismatch !== 'reembed') {
        throw new EmbeddingError(`onMismatch must be 'error', 'warn' or 'reembed', got ${JSON.stringify(onMismatch)}`);
    }
    const kind = resolveFormat(path, format);
    // Python's write_text translates newlines on Windows, so a graph saved
    // there arrives CRLF-terminated, which the ISON parser rejects.
    const text = readFileSync(path, 'utf8').replace(/\r\n/g, '\n');
    const name = options.name ?? basename(String(path), extname(String(path)));
    const base = kind === 'isonl' ? ISONGraph.fromIsonl(text, name) : ISONGraph.fromIson(text, name);

    const graph = new SemanticGraph({
        name: base.name,
        directed: base.directed,
        encoder,
        autoEmbed,
        embedFields,
        version,
        onModelMismatch,
        ann,
        annNlist,
        annNprobe,
        annMinSize,
        annSeed,
    });
    for (const node of base.nodes()) {
        ISONGraph.prototype.addNode.call(graph, node.type, node.id, node.properties);
    }
    for (const relType of base.edgeTypes()) {
        for (const edge of base.edges(relType)) {
            graph.addEdge(edge.relType, edge.source, edge.target, edge.properties);
        }
    }

    const report = (source, snapshotSkipped, count) => ({
        graph,
        report: {
            source,
            snapshotSkipped,
            embeddings: count,
            seconds: ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - started) / 1000,
        },
    });
    if (!embeddings)
        return report('none', null, 0);
    const sidecar = embeddingsPathFor(path);
    if (!(existsSync(sidecar) && statSync(sidecar).isFile()))
        return report('none', null, 0);

    let actualDigest = null;
    const digestNow = () => (actualDigest ??= graphDigest(base));

    // The snapshot first: a graph digest equal to this graph's own (no leniency
    // for number literals read differently from Python), the sidecar as it is
    // now, a valid body and checksum. Anything else and the JSON is read.
    let snapshotSkipped = null;
    const snap = snapshotPathFor(path);
    if (existsSync(snap) && statSync(snap).isFile()) {
        try {
            const current = sidecarIdentity(sidecar);
            const digest = digestNow();
            const result = readSnapshotFile(snap, {
                resolveId: idResolverFor(base),
                accept: info => {
                    if (info.graphDigest !== digest || info.graphDigestBasis !== GRAPH_DIGEST_BASIS) {
                        return 'it was saved for a different graph (digest differs)';
                    }
                    if (info.sidecar === null)
                        return 'it records no sidecar identity';
                    if (info.sidecar.len !== current.len ||
                        !info.sidecar.fingerprint.every((b, i) => b === current.fingerprint[i])) {
                        return 'the JSON sidecar changed since the snapshot was written (stale snapshot)';
                    }
                    return null;
                },
            });
            applySnapshot(graph.embeddingStore, result);
            warnOnVersionSkew(graph);
            return report('snapshot', null, graph.embeddingStore.count());
        }
        catch (error) {
            snapshotSkipped = error instanceof SnapshotError ? error.reason : error.message;
        }
    }

    let payload;
    try {
        payload = JSON.parse(readFileSync(sidecar, 'utf8'));
    }
    catch (error) {
        throw new EmbeddingError(`unreadable embeddings sidecar ${sidecar}: ${error.message}`);
    }

    let problem = null;
    let unverifiable = false;
    const recorded = payload && typeof payload === 'object' ? payload.graph_digest : undefined;
    if (recorded !== undefined && recorded !== null &&
        (payload.graph_digest_basis ?? GRAPH_DIGEST_BASIS) === GRAPH_DIGEST_BASIS) {
        const actual = digestNow();
        if (recorded !== actual) {
            unverifiable = hasInexactNumberLiteral(text);
            problem =
                `embeddings sidecar ${sidecar} does not belong to graph ${path}: it was written ` +
                    `for graph digest ${recorded}, but the graph reads as ${actual} (the sidecar is ` +
                    `from a different save, or the graph was edited since). ` +
                    (unverifiable
                        ? `The graph file holds number literals (integral floats such as 1.0, or ` +
                            `integers beyond 2^53) that this port reads differently from Python, so the ` +
                            `digests may differ for an unedited file; using the sidecar's vectors anyway.`
                        : `Pass onMismatch: 'reembed' to rebuild the vectors or onMismatch: 'warn' to ` +
                            `use them anyway.`);
        }
    }

    if (problem !== null && onMismatch === 'error' && !unverifiable) {
        throw new GraphEmbeddingMismatchError(problem);
    }
    if (problem !== null && onMismatch === 'reembed' && !unverifiable) {
        graph.embedAllNodes();
    }
    else {
        if (problem !== null)
            console.warn(problem);
        graph.embeddingStore.importEmbeddings(payload);
        warnOnVersionSkew(graph);
    }
    return report('json', snapshotSkipped, graph.embeddingStore.count());
}
