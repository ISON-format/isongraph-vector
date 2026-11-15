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
    readFileSync, writeFileSync, existsSync, openSync, fsyncSync, closeSync, renameSync, unlinkSync,
} from 'node:fs';
import { randomBytes } from 'node:crypto';
import { basename, dirname, extname, join } from 'node:path';
import { ISONGraph } from 'ison-graph-js';
import {
    SemanticGraph, EmbeddingError, GraphEmbeddingMismatchError, GRAPH_DIGEST_BASIS,
    EMBEDDING_SIDECAR_SUFFIX, graphDigest,
} from './index.js';

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
 */
export function saveGraph(graph, path, options = {}) {
    const { format = 'auto', embeddings = true } = options;
    const kind = resolveFormat(path, format);
    const text = kind === 'isonl' ? graph.toIsonl() : graph.toIson();
    writeFileAtomic(path, text);
    if (embeddings && graph.embeddingStore) {
        const stem = basename(String(path), extname(String(path)));
        const reread = kind === 'isonl' ? ISONGraph.fromIsonl(text, stem) : ISONGraph.fromIson(text, stem);
        saveEmbeddings(graph, embeddingsPathFor(path), graphDigest(reread));
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
 */
export function loadGraph(path, options = {}) {
    const {
        format = 'auto', embeddings = true, encoder, autoEmbed, embedFields,
        version, onModelMismatch, onMismatch = 'error',
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
    });
    for (const node of base.nodes()) {
        ISONGraph.prototype.addNode.call(graph, node.type, node.id, node.properties);
    }
    for (const relType of base.edgeTypes()) {
        for (const edge of base.edges(relType)) {
            graph.addEdge(edge.relType, edge.source, edge.target, edge.properties);
        }
    }

    if (embeddings) {
        const sidecar = embeddingsPathFor(path);
        if (existsSync(sidecar)) {
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
                const actual = graphDigest(base);
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
                const storedVersion = graph.embeddingStore.lastImportMeta.encoder_version;
                const currentVersion = graph.embeddingStore.version;
                if (storedVersion !== undefined && currentVersion !== null &&
                    String(storedVersion) !== String(currentVersion)) {
                    console.warn(`embeddings were made by encoder version '${storedVersion}', ` +
                        `but the encoder in use is version '${currentVersion}'`);
                }
            }
        }
    }
    return graph;
}
