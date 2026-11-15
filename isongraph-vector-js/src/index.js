/**
 * ISONGraph Vector - Semantic Graph Extension for JavaScript
 *
 * Extends ISONGraph with embedding-based similarity search and semantic traversal.
 *
 * Generated from the TypeScript port's src/index.ts, so the two stay
 * behaviourally identical. Edit that file rather than this one.
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

import { ISONGraph, Direction } from 'ison-graph-js';
export const VERSION = "1.0.0";
/**
 * Portable embedding payload shared by every language port, so a store built
 * in one can be searched by another.
 */
export const EMBEDDING_FORMAT = "ison-embeddings";
export const EMBEDDING_FORMAT_VERSION = 1;
/** Suffix appended to a graph path to locate its embeddings (see `./node`). */
export const EMBEDDING_SIDECAR_SUFFIX = ".embeddings.json";
/**
 * Name of the serialization {@link graphDigest} hashes. Written next to the
 * digest so a reader that computes a different digest (or none) can tell it is
 * not comparable and skip the check instead of reporting a false mismatch.
 */
export const GRAPH_DIGEST_BASIS = "nodes-edges-json-v1";
/**
 * Model names that carry no information about which encoder made a vector:
 * "provided" (vector handed in, store has no encoder) and "unknown" (sidecars
 * from ports that do not record a model).
 */
export const UNKNOWN_MODELS = new Set(["unknown", "provided"]);
// =============================================================================
// Errors
// =============================================================================
/** Base class for embedding-specific failures. */
export class EmbeddingError extends Error {
    constructor(message) {
        super(message);
        this.name = 'EmbeddingError';
    }
}
/**
 * A vector was stored or queried with the wrong number of dimensions.
 *
 * Cosine similarity over a mismatched pair returns a plausible-looking number
 * rather than an error, so a store that quietly accepted mixed widths would
 * degrade silently. A store pins its dimension to the first vector it sees.
 */
export class DimensionMismatchError extends EmbeddingError {
    constructor(message) {
        super(message);
        this.name = 'DimensionMismatchError';
    }
}
/** Text was handed to a store with no encoder to embed it with. */
export class NoEncoderError extends EmbeddingError {
    constructor(message) {
        super(message);
        this.name = 'NoEncoderError';
    }
}
/**
 * A text query was encoded by a different model than the stored vectors.
 *
 * Two models of the same dimension give a plausible-looking cosine score
 * between vectors that live in unrelated spaces, so the ranking is noise
 * rather than an error. Thrown only with `onModelMismatch: 'error'`; the
 * default is a warning.
 */
export class ModelMismatchError extends EmbeddingError {
    constructor(message) {
        super(message);
        this.name = 'ModelMismatchError';
    }
}
/**
 * A graph file and its embeddings sidecar are not from the same save.
 *
 * Thrown by `loadGraph` (from `./node`) when the sidecar's recorded
 * `graph_digest` does not match the graph that was read - the sidecar came from
 * a different save, or the graph file was edited afterwards.
 */
export class GraphEmbeddingMismatchError extends EmbeddingError {
    constructor(message) {
        super(message);
        this.name = 'GraphEmbeddingMismatchError';
    }
}
/**
 * Deterministic encoder with no dependencies, for tests and offline work.
 *
 * The same text produces the same vector in *every* language port
 * (Python/TypeScript/JavaScript/Rust/C++/C#): a 32-bit FNV-1a hash over the UTF-8
 * bytes seeds an xorshift32 generator, and the top 24 bits of each state
 * become a value in [-1, 1).
 *
 * This replaces a scheme that emitted the *low* 16 bits of an LCG. The low
 * bits of an LCG evolve independently of the high bits, so every vector was
 * determined by `seed mod 65536`; in the exact-arithmetic ports (Rust/C++)
 * that meant 280 outright collisions in 5,000 texts, with colliding texts
 * scoring a perfect 1.0 against each other. JavaScript escaped that only
 * because `hash * 1103515245` overflows float64 and leaks high bits back in -
 * which also meant its vectors did not match the other ports.
 *
 * These are pseudo-embeddings with no semantic structure. Use them to test
 * plumbing, never to judge relevance quality.
 */
export class MockEncoder {
    constructor(dimension = 384, version) {
        if (!Number.isInteger(dimension) || dimension <= 0) {
            throw new EmbeddingError(`dimension must be a positive integer, got ${dimension}`);
        }
        this._dimension = dimension;
        this.modelName = `mock-${dimension}`;
        if (version !== undefined)
            this.version = version;
    }
    get dimension() {
        return this._dimension;
    }
    encode(text) {
        let state = MockEncoder.FNV_OFFSET_BASIS;
        // Math.imul keeps the multiply exact in 32 bits, matching the other ports.
        const fold = (byte) => {
            state = Math.imul(state ^ byte, MockEncoder.FNV_PRIME) >>> 0;
        };
        // UTF-8 bytes, hashed the same way every port hashes them. Written out
        // rather than using TextEncoder so this needs no DOM lib and behaves the
        // same in Node and the browser.
        for (let i = 0; i < text.length; i++) {
            let code = text.charCodeAt(i);
            if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
                const low = text.charCodeAt(i + 1);
                if (low >= 0xdc00 && low <= 0xdfff) {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    i++;
                }
            }
            if (code >= 0xd800 && code <= 0xdfff)
                code = 0xfffd; // lone surrogate
            if (code < 0x80) {
                fold(code);
            }
            else if (code < 0x800) {
                fold(0xc0 | (code >> 6));
                fold(0x80 | (code & 0x3f));
            }
            else if (code < 0x10000) {
                fold(0xe0 | (code >> 12));
                fold(0x80 | ((code >> 6) & 0x3f));
                fold(0x80 | (code & 0x3f));
            }
            else {
                fold(0xf0 | (code >> 18));
                fold(0x80 | ((code >> 12) & 0x3f));
                fold(0x80 | ((code >> 6) & 0x3f));
                fold(0x80 | (code & 0x3f));
            }
        }
        // xorshift32 has no way out of a zero state; steer off it.
        if (state === 0)
            state = 0x9e3779b9;
        const values = new Array(this._dimension);
        for (let i = 0; i < this._dimension; i++) {
            state ^= (state << 13) >>> 0;
            state >>>= 0;
            state ^= state >>> 17;
            state ^= (state << 5) >>> 0;
            state >>>= 0;
            values[i] = (state >>> 8) / 8388608.0 - 1.0; // top 24 bits
        }
        return values;
    }
    encodeBatch(texts) {
        return texts.map(t => this.encode(t));
    }
}
MockEncoder.FNV_OFFSET_BASIS = 0x811c9dc5;
MockEncoder.FNV_PRIME = 0x01000193;
/**
 * Keep the best `k` results without ordering all of them.
 *
 * Sorting every candidate to return ten is O(n log n) where this is O(n·k)
 * with early rejection — for the small k a search actually asks for, that is
 * a large saving over 20,000 candidates. Insertion goes *after* equal scores,
 * so the order matches what a stable descending sort would produce.
 */
export function selectTopK(results, k) {
    if (k <= 0)
        return [];
    if (results.length <= k)
        return results.sort((a, b) => b.score - a.score);
    const top = [];
    for (const candidate of results) {
        if (top.length === k && candidate.score <= top[k - 1].score)
            continue;
        let i = top.length < k ? top.length : k - 1;
        while (i > 0 && top[i - 1].score < candidate.score) {
            top[i] = top[i - 1];
            i--;
        }
        top[i] = candidate;
        if (top.length > k)
            top.length = k;
    }
    return top;
}
/**
 * In-memory embedding storage with brute-force cosine similarity search.
 */
export class EmbeddingStore {
    constructor(encoder, options = {}) {
        this._embeddings = new Map();
        this._dimension = null;
        this._onModelMismatch = 'warn';
        this._warnedMismatch = new Set();
        /** Metadata of the most recent imported sidecar (empty before any). */
        this.lastImportMeta = {};
        this._encoder = encoder || null;
        this._version = options.version ?? null;
        this.onModelMismatch = options.onModelMismatch ?? 'warn';
    }
    get encoder() {
        return this._encoder;
    }
    set encoder(encoder) {
        this._encoder = encoder;
    }
    /** Vector length this store holds, or null while it is still empty. */
    get dimension() {
        return this._dimension;
    }
    /** Encoder/model version recorded in the sidecar, or null if unknown. */
    get version() {
        if (this._version !== null)
            return this._version;
        const fromEncoder = this._encoder?.version;
        return fromEncoder === undefined || fromEncoder === null ? null : fromEncoder;
    }
    set version(value) {
        this._version = value;
    }
    /** How a text query reacts to vectors from another model (see EmbeddingStoreOptions). */
    get onModelMismatch() {
        return this._onModelMismatch;
    }
    set onModelMismatch(mode) {
        if (mode !== 'warn' && mode !== 'error' && mode !== 'ignore') {
            throw new EmbeddingError(`onModelMismatch must be 'warn', 'error' or 'ignore', got ${JSON.stringify(mode)}`);
        }
        this._onModelMismatch = mode;
    }
    _nodeRefToKey(ref) {
        return `${ref[0]}:${ref[1]}`;
    }
    _checkDimension(vector, what) {
        if (vector.length === 0) {
            throw new DimensionMismatchError(`${what} is empty`);
        }
        if (this._dimension === null) {
            this._dimension = vector.length;
        }
        else if (vector.length !== this._dimension) {
            throw new DimensionMismatchError(`${what} has ${vector.length} dimensions, store holds ${this._dimension}`);
        }
    }
    _cosineSimilarity(v1, v2) {
        let dot = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        if (norm1 === 0 || norm2 === 0)
            return 0;
        return dot / (norm1 * norm2);
    }
    _resolveQuery(query) {
        let vector;
        if (typeof query === 'string') {
            if (!this._encoder) {
                throw new NoEncoderError("No encoder provided for text query");
            }
            this._checkModelCompat();
            vector = this._encoder.encode(query);
        }
        else {
            vector = query;
        }
        if (vector.length === 0) {
            throw new DimensionMismatchError("query vector is empty");
        }
        if (this._dimension !== null && vector.length !== this._dimension) {
            throw new DimensionMismatchError(`query vector has ${vector.length} dimensions, store holds ${this._dimension}`);
        }
        return vector;
    }
    /** Model recorded on vectors written now: the encoder's, or "provided"/"unknown". */
    _modelName() {
        if (!this._encoder)
            return 'provided';
        return this._encoder.modelName || 'unknown';
    }
    /** The encoder's model, or null when it has none (or an uninformative one). */
    _currentModel() {
        const name = this._encoder?.modelName;
        return name && !UNKNOWN_MODELS.has(name) ? name : null;
    }
    /**
     * Which models produced the stored vectors: `{ modelName: count }`.
     *
     * "unknown" / "provided" appear as-is (vectors imported from a port that
     * records no model, or handed in with no encoder attached).
     */
    models() {
        const counts = new Map();
        for (const record of this._embeddings.values()) {
            counts.set(record.model, (counts.get(record.model) ?? 0) + 1);
        }
        return Object.fromEntries(counts);
    }
    /** Complain when a text query's encoder is not the stored vectors' model. */
    _checkModelCompat() {
        if (this._onModelMismatch === 'ignore')
            return;
        const current = this._currentModel();
        if (current === null)
            return;
        const foreign = Object.keys(this.models())
            .filter(m => !UNKNOWN_MODELS.has(m) && m !== current)
            .sort();
        if (foreign.length === 0)
            return;
        const message = `encoder model '${current}' differs from the model that produced the ` +
            `stored vectors (${foreign.map(m => `'${m}'`).join(', ')}); ` +
            `scores compare vectors from different spaces. Re-embed with ` +
            `reembedStale() or search with the original encoder.`;
        if (this._onModelMismatch === 'error')
            throw new ModelMismatchError(message);
        const key = JSON.stringify([current, foreign]);
        if (!this._warnedMismatch.has(key)) {
            this._warnedMismatch.add(key);
            if (typeof console !== 'undefined')
                console.warn(message);
        }
    }
    /**
     * Node refs whose stored vector does not belong to the current encoder.
     *
     * A vector is stale when its recorded model differs from `model` (default:
     * the encoder's `modelName`) or its length differs from `dimension`
     * (default: the encoder's `dimension`). Vectors whose model is
     * "unknown"/"provided" cannot be judged by model and are skipped unless
     * `includeUnknown` is true (a dimension mismatch still counts). With no
     * encoder and no arguments nothing can be judged and the result is empty.
     *
     * Takes either the positional form or an options object.
     */
    stale(modelOrOptions, dimensionArg, includeUnknownArg = false) {
        const options = modelOrOptions !== null && typeof modelOrOptions === 'object'
            ? modelOrOptions
            : { model: modelOrOptions, dimension: dimensionArg, includeUnknown: includeUnknownArg };
        const model = options.model ?? this._currentModel();
        let dimension = options.dimension ?? null;
        if (dimension === null && this._encoder) {
            const d = Number(this._encoder.dimension);
            dimension = Number.isFinite(d) ? Math.trunc(d) : null;
        }
        const includeUnknown = options.includeUnknown ?? false;
        const out = [];
        for (const record of this._embeddings.values()) {
            const byModel = model !== null &&
                record.model !== model &&
                (includeUnknown || !UNKNOWN_MODELS.has(record.model));
            const byDim = dimension !== null && record.vector.length !== dimension;
            if (byModel || byDim)
                out.push(record.nodeRef);
        }
        return out;
    }
    add(nodeRef, text, vector) {
        if (vector === undefined) {
            if (!this._encoder) {
                throw new NoEncoderError("No encoder provided and no vector given");
            }
            vector = this._encoder.encode(text);
        }
        this._checkDimension(vector, "vector");
        const key = this._nodeRefToKey(nodeRef);
        this._embeddings.set(key, { id: key, nodeRef, vector, text, model: this._modelName() });
        return key;
    }
    /** Add many embeddings, encoding their texts in one batched call. */
    addBatch(entries) {
        if (entries.length === 0)
            return 0;
        const needsEncoding = entries
            .map((e, i) => (e.length < 3 || e[2] === undefined ? i : -1))
            .filter(i => i >= 0);
        let encoded = [];
        if (needsEncoding.length > 0) {
            if (!this._encoder) {
                throw new NoEncoderError("No encoder provided and no vectors given");
            }
            encoded = this._encoder.encodeBatch(needsEncoding.map(i => entries[i][1]));
            if (encoded.length !== needsEncoding.length) {
                throw new EmbeddingError(`encoder returned ${encoded.length} vectors for ${needsEncoding.length} texts`);
            }
        }
        const vectorFor = new Map();
        needsEncoding.forEach((slot, i) => vectorFor.set(slot, encoded[i]));
        entries.forEach((entry, i) => {
            const vector = vectorFor.get(i) ?? entry[2];
            this.add(entry[0], entry[1], vector);
        });
        return entries.length;
    }
    get(nodeRef) {
        const key = this._nodeRefToKey(nodeRef);
        return this._embeddings.get(key) || null;
    }
    remove(nodeRef) {
        const key = this._nodeRefToKey(nodeRef);
        return this._embeddings.delete(key);
    }
    similaritySearch(query, topK = 10, nodeType, threshold = 0.0) {
        const queryVector = this._resolveQuery(query);
        const results = [];
        for (const record of this._embeddings.values()) {
            if (nodeType && record.nodeRef[0] !== nodeType)
                continue;
            const score = this._cosineSimilarity(queryVector, record.vector);
            if (score >= threshold) {
                results.push({ nodeRef: record.nodeRef, score, text: record.text });
            }
        }
        if (topK === null)
            return results.sort((a, b) => b.score - a.score);
        return selectTopK(results, topK);
    }
    /**
     * Cosine score for every embedded node, keyed by `type:id`.
     *
     * One scan serving both seed selection and the per-node relevance that
     * semanticMultiHop blends in, instead of two.
     */
    scoreMap(query, nodeType) {
        const queryVector = this._resolveQuery(query);
        const scores = new Map();
        for (const [key, record] of this._embeddings) {
            if (nodeType && record.nodeRef[0] !== nodeType)
                continue;
            scores.set(key, this._cosineSimilarity(queryVector, record.vector));
        }
        return scores;
    }
    count() {
        return this._embeddings.size;
    }
    /**
     * Clear all embeddings.
     *
     * @param resetDimension Also forget the pinned vector length, so the next
     *   vector may have a different one (switching to an encoder of a different
     *   dimension). Off by default: a store normally refuses vectors of the
     *   wrong length even when empty.
     */
    clear(resetDimension = false) {
        this._embeddings.clear();
        if (resetDimension)
            this._dimension = null;
    }
    /**
     * Serialize every embedding into the portable cross-language payload.
     *
     * Beyond the version-1 fields it appends, when known: `encoder_model` (the
     * encoder's modelName), `encoder_version` (this store's `version`) and, when
     * `graphDigest` is given, `graph_digest` plus `graph_digest_basis`. Absent
     * values are omitted, not written as null.
     */
    exportEmbeddings(graphDigest) {
        const embeddings = [];
        for (const record of this._embeddings.values()) {
            embeddings.push({
                type: String(record.nodeRef[0]),
                id: record.nodeRef[1],
                id_type: typeof record.nodeRef[1] === 'number' ? 'int' : 'str',
                text: record.text,
                model: record.model,
                vector: record.vector,
            });
        }
        const payload = {
            format: EMBEDDING_FORMAT,
            version: EMBEDDING_FORMAT_VERSION,
            dimension: this._dimension,
            count: embeddings.length,
            embeddings,
        };
        const encoderModel = this._currentModel();
        if (encoderModel !== null)
            payload.encoder_model = encoderModel;
        const version = this.version;
        if (version !== null)
            payload.encoder_version = String(version);
        if (graphDigest !== undefined && graphDigest !== null) {
            payload.graph_digest = graphDigest;
            payload.graph_digest_basis = GRAPH_DIGEST_BASIS;
        }
        return payload;
    }
    /** Load a payload produced by exportEmbeddings (in any language port). */
    importEmbeddings(payload, replace = false) {
        if (!payload || payload.format !== EMBEDDING_FORMAT) {
            throw new EmbeddingError(`not an ${EMBEDDING_FORMAT} payload: format=${payload && payload.format}`);
        }
        if (payload.version !== EMBEDDING_FORMAT_VERSION) {
            throw new EmbeddingError(`unsupported ${EMBEDDING_FORMAT} version ${payload.version} ` +
                `(this build reads version ${EMBEDDING_FORMAT_VERSION})`);
        }
        const meta = {};
        for (const key of ['encoder_model', 'encoder_version', 'graph_digest', 'graph_digest_basis']) {
            if (payload[key] !== undefined && payload[key] !== null)
                meta[key] = payload[key];
        }
        this.lastImportMeta = meta;
        if (replace) {
            this.clear();
            this._dimension = payload.dimension ?? null;
        }
        let written = 0;
        for (const item of payload.embeddings || []) {
            const id = item.id_type === 'int' ? Number(item.id) : item.id;
            const ref = [item.type, id];
            this._checkDimension(item.vector, `embedding for :${item.type}:${id}`);
            const key = this._nodeRefToKey(ref);
            this._embeddings.set(key, {
                id: key,
                nodeRef: ref,
                vector: item.vector,
                text: item.text ?? '',
                model: item.model ?? 'unknown',
            });
            written++;
        }
        return written;
    }
}
// =============================================================================
// Graph digest
// =============================================================================
/** SHA-256 of an ASCII string, as lowercase hex. Pure JS so the core stays browser-safe. */
function sha256Hex(ascii) {
    const primes = [];
    for (let n = 2; primes.length < 64; n++) {
        if (primes.every(p => n % p !== 0))
            primes.push(n);
    }
    const frac32 = (x) => Math.floor((x - Math.floor(x)) * 4294967296) >>> 0;
    const K = primes.map(p => frac32(Math.cbrt(p)));
    const H = primes.slice(0, 8).map(p => frac32(Math.sqrt(p)));
    const bytes = [];
    for (let i = 0; i < ascii.length; i++)
        bytes.push(ascii.charCodeAt(i) & 0xff);
    const bitLength = bytes.length * 8;
    bytes.push(0x80);
    while (bytes.length % 64 !== 56)
        bytes.push(0);
    const high = Math.floor(bitLength / 4294967296);
    const low = bitLength >>> 0;
    for (const word of [high, low]) {
        bytes.push((word >>> 24) & 0xff, (word >>> 16) & 0xff, (word >>> 8) & 0xff, word & 0xff);
    }
    const rotr = (x, n) => (x >>> n) | (x << (32 - n));
    const w = new Array(64);
    for (let offset = 0; offset < bytes.length; offset += 64) {
        for (let i = 0; i < 16; i++) {
            const j = offset + i * 4;
            w[i] = ((bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3]) | 0;
        }
        for (let i = 16; i < 64; i++) {
            const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
            const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
        }
        let [a, b, c, d, e, f, g, h] = H;
        for (let i = 0; i < 64; i++) {
            const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const ch = (e & f) ^ (~e & g);
            const t1 = (h + S1 + ch + K[i] + w[i]) | 0;
            const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const maj = (a & b) ^ (a & c) ^ (b & c);
            const t2 = (S0 + maj) | 0;
            h = g;
            g = f;
            f = e;
            e = (d + t1) | 0;
            d = c;
            c = b;
            b = a;
            a = (t1 + t2) | 0;
        }
        const sums = [a, b, c, d, e, f, g, h];
        for (let i = 0; i < 8; i++)
            H[i] = (H[i] + sums[i]) | 0;
    }
    return H.map(x => (x >>> 0).toString(16).padStart(8, '0')).join('');
}
/** Compare strings by code point, as Python does (JS `<` compares UTF-16 code units). */
function compareCodePoints(a, b) {
    if (a === b)
        return 0;
    const A = Array.from(a);
    const B = Array.from(b);
    const n = Math.min(A.length, B.length);
    for (let i = 0; i < n; i++) {
        const x = A[i].codePointAt(0);
        const y = B[i].codePointAt(0);
        if (x !== y)
            return x < y ? -1 : 1;
    }
    return A.length - B.length;
}
/** Python `repr(float)` for a non-integral (or non-finite) JS number. */
function pythonFloatRepr(x) {
    if (Number.isNaN(x))
        return 'NaN';
    if (x === Infinity)
        return 'Infinity';
    if (x === -Infinity)
        return '-Infinity';
    if (x === 0)
        return Object.is(x, -0) ? '-0.0' : '0.0';
    // toExponential() with no argument yields the shortest digits that round-trip,
    // the same digits Python's repr picks; only the layout differs.
    const m = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(x.toExponential());
    const sign = m[1];
    const digits = m[2] + (m[3] ?? '');
    const e = parseInt(m[4], 10);
    if (e < -4 || e >= 16) {
        const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
        const abs = Math.abs(e);
        return `${sign}${mantissa}e${e < 0 ? '-' : '+'}${abs < 10 ? '0' + abs : abs}`;
    }
    if (e >= 0) {
        if (digits.length <= e + 1)
            return `${sign}${digits}${'0'.repeat(e + 1 - digits.length)}.0`;
        return `${sign}${digits.slice(0, e + 1)}.${digits.slice(e + 1)}`;
    }
    return `${sign}0.${'0'.repeat(-e - 1)}${digits}`;
}
/** Python `json.dumps(str, ensure_ascii=True)`. */
function pythonJsonString(s) {
    let out = '"';
    for (let i = 0; i < s.length; i++) {
        const c = s.charCodeAt(i);
        if (c === 0x22)
            out += '\\"';
        else if (c === 0x5c)
            out += '\\\\';
        else if (c === 0x0a)
            out += '\\n';
        else if (c === 0x0d)
            out += '\\r';
        else if (c === 0x09)
            out += '\\t';
        else if (c === 0x08)
            out += '\\b';
        else if (c === 0x0c)
            out += '\\f';
        else if (c < 0x20 || c > 0x7e)
            out += '\\u' + c.toString(16).padStart(4, '0');
        else
            out += s[i];
    }
    return out + '"';
}
/**
 * Python `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
 * default=str)` for the value shapes a graph holds.
 *
 * A JS number with an integral value is written as a Python int (a graph read
 * back from ISON text cannot tell `1` from `1.0`, and Python reads an integral
 * literal without a decimal point as an int); anything else as Python's float
 * repr.
 */
function canonicalJson(value) {
    if (value === null || value === undefined)
        return 'null';
    switch (typeof value) {
        case 'boolean':
            return value ? 'true' : 'false';
        case 'number':
            return Number.isInteger(value) && !Object.is(value, -0)
                ? BigInt(value).toString()
                : pythonFloatRepr(value);
        case 'bigint':
            return value.toString();
        case 'string':
            return pythonJsonString(value);
        case 'object': {
            if (Array.isArray(value))
                return '[' + value.map(canonicalJson).join(',') + ']';
            const record = value;
            const keys = Object.keys(record).sort(compareCodePoints);
            return '{' + keys.map(k => pythonJsonString(k) + ':' + canonicalJson(record[k])).join(',') + '}';
        }
        default:
            return pythonJsonString(String(value));
    }
}
/**
 * Content digest of a graph's nodes and edges: `"sha256:<hex>"`.
 *
 * Hashes a canonical JSON rendering ({@link GRAPH_DIGEST_BASIS}): the
 * `directed` flag, every node as `[type, id-kind, id, properties]` and every
 * edge as `[relType, source, target, properties]`, each list sorted and every
 * object key-sorted. The same graph therefore hashes the same regardless of
 * insertion order. The graph *name* is excluded (it is derived from the file
 * name on load), and so are embeddings. Reproduces the Python port's
 * `graph_digest` exactly - see the golden fixture in `tests/fixtures`.
 *
 * Limit: a JS number cannot tell `1` from `1.0`, so an integral float written
 * by another port (`1.0`) digests as the int `1`; and an integer beyond 2^53
 * digests as the double it parsed to. `loadGraph` treats a mismatch on a file
 * containing such a literal as unverifiable rather than as an error.
 */
export function graphDigest(graph) {
    const refKey = (ref) => {
        const id = ref[1];
        const isInt = typeof id === 'number' && Number.isInteger(id);
        return [ref[0], isInt ? 'i' : 's', isInt ? BigInt(id).toString() : String(id)];
    };
    const nodes = [];
    for (const node of graph.nodes()) {
        nodes.push(canonicalJson([refKey(node.ref), node.properties]));
    }
    const edges = [];
    for (const edge of graph.edges()) {
        edges.push(canonicalJson([edge.relType, refKey(edge.source), refKey(edge.target), edge.properties]));
    }
    nodes.sort();
    edges.sort();
    const body = `{"directed":${graph.directed ? 'true' : 'false'},"edges":[${edges.join(',')}],` +
        `"nodes":[${nodes.join(',')}]}`;
    return 'sha256:' + sha256Hex(body);
}
/**
 * An edge's numeric `weightField` property clamped to [0, 1].
 *
 * A missing, non-numeric (including boolean) or NaN value counts as 1, so an
 * unweighted edge never penalises a hop.
 */
function edgeWeight(edge, weightField) {
    const value = edge.properties ? edge.properties[weightField] : undefined;
    if (typeof value !== 'number' || Number.isNaN(value))
        return 1.0;
    return Math.min(1.0, Math.max(0.0, value));
}
/**
 * ISONGraph extended with embedding-based semantic search.
 */
export class SemanticGraph extends ISONGraph {
    constructor(options = {}) {
        super(options.name || "semantic_graph", options.directed ?? true);
        this._encoder = options.encoder || null;
        this._embeddingStore = new EmbeddingStore(this._encoder || undefined, {
            version: options.version,
            onModelMismatch: options.onModelMismatch,
        });
        this._autoEmbed = options.autoEmbed ?? true;
        this._embedFields = options.embedFields || ['name', 'description', 'content', 'text'];
    }
    get embeddingStore() {
        return this._embeddingStore;
    }
    get encoder() {
        return this._encoder;
    }
    setEncoder(encoder) {
        this._encoder = encoder;
        this._embeddingStore.encoder = encoder;
    }
    /**
     * Render one property value as embedding text.
     *
     * ISONGraph property values are typed, so a property can be a number, a
     * bool or null. Null and undefined contribute nothing, so the same graph
     * produces the same text - and so the same vector - in every language port.
     */
    static propertyText(value) {
        if (value === null || value === undefined)
            return '';
        return String(value);
    }
    /** Concatenate the configured embedFields present in a property bag. */
    embedTextFor(properties) {
        const parts = [];
        for (const field of this._embedFields) {
            if (!(field in properties))
                continue;
            const text = SemanticGraph.propertyText(properties[field]);
            if (text)
                parts.push(text);
        }
        return parts.join(" ");
    }
    /**
     * Add a node, embedding it from its properties when autoEmbed is on.
     *
     * The base addNode is overridden rather than shadowed by a separate method,
     * so that `autoEmbed: true` actually applies to the call every caller
     * reaches for. It used to only apply inside addNodeWithEmbed, which meant a
     * plain addNode silently produced an unembedded, unsearchable node.
     */
    addNode(nodeType, nodeId, properties = {}) {
        const node = super.addNode(nodeType, nodeId, properties);
        if (this._autoEmbed && this._encoder) {
            const text = this.embedTextFor(properties);
            if (text)
                this.embedNode(node.ref, text);
        }
        return node;
    }
    /**
     * Add a node with an explicit embedding text.
     */
    addNodeWithEmbed(nodeType, nodeId, properties = {}, embedText) {
        if (embedText === undefined) {
            return this.addNode(nodeType, nodeId, properties);
        }
        const node = super.addNode(nodeType, nodeId, properties);
        this.embedNode(node.ref, embedText);
        return node;
    }
    /**
     * Embed a node with given text.
     */
    embedNode(nodeRef, text) {
        return this._embeddingStore.add(nodeRef, text);
    }
    /** Embed many nodes in a single batched encoder call. */
    embedNodes(items) {
        const pairs = items.filter(([, text]) => Boolean(text));
        if (pairs.length === 0)
            return 0;
        return this._embeddingStore.addBatch(pairs);
    }
    /**
     * Get embedding for a node.
     */
    getEmbedding(nodeRef) {
        return this._embeddingStore.get(nodeRef);
    }
    /**
     * Update node properties, keeping the embedding in step with them.
     *
     * With autoEmbed on, a change to any of embedFields re-embeds the node
     * from its new properties, so searches never rank it by text it no longer
     * carries. If the new properties leave nothing to embed, the stale vector
     * is dropped. Updates that touch no embed field, and graphs with autoEmbed
     * off (whose vectors the caller manages), leave the embedding alone. As in
     * addNode, re-embedding needs an encoder; without one the vector is kept.
     */
    updateNode(nodeType, nodeId, properties) {
        const node = super.updateNode(nodeType, nodeId, properties);
        if (this._autoEmbed && this._embedFields.some(f => f in (properties ?? {}))) {
            const text = this.embedTextFor(node.properties);
            if (!text) {
                this._embeddingStore.remove(node.ref);
            }
            else if (this._encoder) {
                this.embedNode(node.ref, text);
            }
        }
        return node;
    }
    /**
     * Remove node and its embedding.
     */
    removeNode(nodeType, nodeId) {
        const ref = [nodeType, nodeId];
        this._embeddingStore.remove(ref);
        super.removeNode(nodeType, nodeId);
    }
    /**
     * Build the `ref => boolean` predicate behind `nodeFilter` / `nodeTypes`.
     *
     * Returns null when neither is set, so unfiltered calls take exactly the
     * pre-existing code path. A node passes when its type is in `nodeTypes` (a
     * bare string means one type; an empty array passes nothing) *and*
     * `nodeFilter(node)` is truthy. A ref with no node in the graph cannot be
     * shown to `nodeFilter`, so it fails whenever a `nodeFilter` is set.
     * Answers are memoized for the one call.
     */
    _nodeGate(nodeFilter, nodeTypes) {
        const hasFilter = nodeFilter !== undefined && nodeFilter !== null;
        const hasTypes = nodeTypes !== undefined && nodeTypes !== null;
        if (!hasFilter && !hasTypes)
            return null;
        const types = hasTypes
            ? new Set(typeof nodeTypes === 'string' ? [nodeTypes] : nodeTypes)
            : null;
        const memo = new Map();
        return (ref) => {
            const key = `${ref[0]}:${ref[1]}`;
            const cached = memo.get(key);
            if (cached !== undefined)
                return cached;
            let ok = types === null || types.has(ref[0]);
            if (ok && hasFilter) {
                const node = this._resolveNode(ref);
                ok = node !== null && Boolean(nodeFilter(node));
            }
            memo.set(key, ok);
            return ok;
        };
    }
    /**
     * Top-k similarity search among the nodes that pass `gate`.
     *
     * With a gate the store cannot truncate first: the best k *unfiltered*
     * matches may all be filtered out, so every match is scored, filtered, and
     * only then cut to k.
     */
    _seedSearch(query, topK, nodeType, threshold, gate) {
        if (gate === null) {
            return this._embeddingStore.similaritySearch(query, topK, nodeType, threshold);
        }
        const results = this._embeddingStore
            .similaritySearch(query, null, nodeType, threshold)
            .filter(r => gate(r.nodeRef));
        return topK === null ? results : results.slice(0, Math.max(topK, 0));
    }
    /**
     * Find similar nodes by embedding.
     *
     * @param nodeFilter Keep only nodes for which `nodeFilter(node)` is truthy.
     *   Applied *before* `topK` is taken, so you still get `topK` matching nodes
     *   when that many exist. An embedding whose node is not in the graph fails
     *   the filter.
     * @param nodeTypes Keep only nodes whose type is in this list (a bare string
     *   means one type). Combines with `nodeType` and `nodeFilter` by AND.
     */
    similaritySearch(query, topK = 10, nodeType, threshold = 0.0, nodeFilter, nodeTypes) {
        const gate = this._nodeGate(nodeFilter, nodeTypes);
        return this._seedSearch(query, topK, nodeType, threshold, gate);
    }
    /**
     * Find nodes semantically closest to one already in the graph.
     *
     * The "more like this" query - no text and no encoder call: it searches with
     * the stored vector of the node and drops the node itself from the results.
     */
    similarToNode(nodeRef, topK = 10, nodeType, threshold = 0.0) {
        const record = this._embeddingStore.get(nodeRef);
        if (!record)
            return [];
        const key = `${nodeRef[0]}:${nodeRef[1]}`;
        const results = this._embeddingStore
            .similaritySearch(record.vector, topK === null ? null : topK + 1, nodeType, threshold)
            .filter(r => `${r.nodeRef[0]}:${r.nodeRef[1]}` !== key);
        return topK === null ? results : results.slice(0, topK);
    }
    _resolveNode(nodeRef) {
        try {
            return this.getNodeByRef(nodeRef);
        }
        catch {
            return null;
        }
    }
    /** (neighbor, connecting edge) pairs, in the order `neighbors()` yields. */
    _adjacent(nodeRef, relType, direction) {
        const pairs = [];
        if (direction === Direction.OUT || direction === Direction.BOTH) {
            for (const edge of this.edges(relType, nodeRef))
                pairs.push([edge.target, edge]);
        }
        if (direction === Direction.IN || direction === Direction.BOTH) {
            for (const edge of this.edges(relType, undefined, nodeRef))
                pairs.push([edge.source, edge]);
        }
        return pairs;
    }
    /**
     * Semantic multi-hop search.
     *
     * 1. Find top-k similar nodes as seeds (hop 0)
     * 2. Traverse the graph following relationships
     * 3. Score by max(seedSimilarity, 0) * product(decay * weight) over the
     *    hops - `weight` is 1 unless `weightField` is set - optionally blended
     *    with the relevance of each node itself
     *
     * Filtering (`nodeFilter` and `nodeTypes` combine with AND):
     *  - Seeds must pass. The top `topKSeeds` are chosen *among* passing nodes,
     *    so a filtered-out best match never occupies a seed slot.
     *  - Only passing nodes are returned.
     *  - By default a node that fails the filter can still be walked *through*:
     *    it is not returned, but it still costs a hop and its decay, and a
     *    returned node's `path` may list it. With `strictFilter` a failing node
     *    is never entered, so every node on every returned path passes.
     *
     * Takes either an options object or the original positional form (new
     * parameters were appended at the end, so existing callers are unaffected).
     */
    semanticMultiHop(query, relTypeOrOptions, maxHopsArg = 2, topKSeedsArg = 5, topKResultsArg = 10, directionArg = Direction.OUT, decayArg = 0.8, thresholdArg = 0.3, blendArg = 0.0, weightFieldArg, nodeFilterArg, nodeTypesArg, strictFilterArg = false) {
        // Accepts either an options object or the original positional form, so
        // existing callers keep working while new ones can name what they mean.
        const options = relTypeOrOptions !== null && typeof relTypeOrOptions === 'object'
            ? relTypeOrOptions
            : {
                relType: relTypeOrOptions,
                maxHops: maxHopsArg,
                topKSeeds: topKSeedsArg,
                topKResults: topKResultsArg,
                direction: directionArg,
                decay: decayArg,
                threshold: thresholdArg,
                blend: blendArg,
                weightField: weightFieldArg,
                nodeFilter: nodeFilterArg,
                nodeTypes: nodeTypesArg,
                strictFilter: strictFilterArg,
            };
        const { relType, maxHops = 2, topKSeeds = 5, topKResults = 10, direction = Direction.OUT, decay = 0.8, threshold = 0.3, blend = 0.0, weightField, nodeFilter, nodeTypes, strictFilter = false, } = options;
        if (blend < 0 || blend > 1) {
            throw new EmbeddingError(`blend must be in [0, 1], got ${blend}`);
        }
        const nodeRefToKey = (ref) => `${ref[0]}:${ref[1]}`;
        const weighted = weightField !== undefined && weightField !== null;
        const gate = this._nodeGate(nodeFilter, nodeTypes);
        const seeds = [...this._seedSearch(query, topKSeeds, undefined, threshold, gate)];
        if (seeds.length === 0)
            return [];
        // Blending needs a score for every node: one scan serves the per-node
        // relevance that is mixed into each traversal score.
        const scores = blend > 0 ? this._embeddingStore.scoreMap(query) : null;
        const combined = (base, key) => scores === null ? base : (1 - blend) * base + blend * (scores.get(key) ?? 0);
        const results = new Map();
        for (const seed of seeds) {
            const seedKey = nodeRefToKey(seed.nodeRef);
            // Cosine similarity is signed, and `base * decay ** hops` only decays a
            // non-negative base: multiplying a negative seed score by 0.8 moves it
            // toward zero, so everything it reached outranked the seed itself and
            // relevance grew with distance.
            const traversalBase = Math.max(seed.score, 0);
            // A seed is a hop-0 match on its own merit. An earlier traversal may
            // already have recorded it with a decayed score; that must not win over
            // the higher, direct similarity of the node. (Only the neighbour branch
            // below used to compare scores, so a direct match reachable from a
            // better seed came back at seed.score * decay, at hop 1, down a path it
            // never needed.)
            const seeded = results.get(seedKey);
            if (!seeded || seeded.score < seed.score) {
                results.set(seedKey, {
                    nodeRef: seed.nodeRef,
                    node: this._resolveNode(seed.nodeRef),
                    score: seed.score,
                    hopCount: 0,
                    path: [seed.nodeRef],
                });
            }
            // Breadth-first traversal. Each node keeps its best decay factor and its
            // fewest hops *separately*: with edge weights the route with the best
            // factor is not always the shortest one, and a shorter route must still
            // be expanded because it has hops left. Without weights the first
            // (shortest) arrival is both, so this reduces to a visited-set BFS.
            const bestFactor = new Map([[seedKey, 1.0]]);
            const bestHops = new Map([[seedKey, 0]]);
            const queue = [{ nodeRef: seed.nodeRef, path: [seed.nodeRef], keys: [seedKey], factor: 1.0, hop: 0 }];
            let head = 0;
            while (head < queue.length) {
                const { nodeRef, path, keys, factor, hop } = queue[head++];
                if (hop >= maxHops)
                    continue;
                for (const [neighbor, edge] of this._adjacent(nodeRef, relType, direction)) {
                    const neighborKey = nodeRefToKey(neighbor);
                    if (keys.includes(neighborKey))
                        continue;
                    if (strictFilter && gate !== null && !gate(neighbor))
                        continue;
                    const newHop = hop + 1;
                    let newFactor;
                    if (!weighted) {
                        if (bestHops.has(neighborKey))
                            continue;
                        // decay ** hops exactly, not a running product, so unweighted
                        // scores are bit-identical to the pre-weight releases.
                        newFactor = Math.pow(decay, newHop);
                    }
                    else {
                        newFactor = factor * decay * edgeWeight(edge, weightField);
                        const knownFactor = bestFactor.get(neighborKey) ?? -1.0;
                        const knownHops = bestHops.get(neighborKey);
                        if (newFactor <= knownFactor && newHop >= (knownHops ?? newHop + 1))
                            continue;
                    }
                    bestFactor.set(neighborKey, Math.max(newFactor, bestFactor.get(neighborKey) ?? -1.0));
                    bestHops.set(neighborKey, Math.min(newHop, bestHops.get(neighborKey) ?? newHop));
                    const newPath = [...path, neighbor];
                    if (gate === null || gate(neighbor)) {
                        const newScore = combined(traversalBase * newFactor, neighborKey);
                        const existing = results.get(neighborKey);
                        if (!existing || existing.score < newScore) {
                            results.set(neighborKey, {
                                nodeRef: neighbor,
                                node: this._resolveNode(neighbor),
                                score: newScore,
                                hopCount: newHop,
                                path: newPath,
                            });
                        }
                    }
                    queue.push({
                        nodeRef: neighbor,
                        path: newPath,
                        keys: [...keys, neighborKey],
                        factor: newFactor,
                        hop: newHop,
                    });
                }
            }
        }
        return Array.from(results.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, topKResults);
    }
    /** Fewest-hops path from `start` to `end` whose every node passes `gate`. */
    _gatedShortestPath(start, end, relType, maxHops, direction, gate) {
        if (!gate(start) || !gate(end))
            return null;
        const keyOf = (ref) => `${ref[0]}:${ref[1]}`;
        const endKey = keyOf(end);
        if (keyOf(start) === endKey)
            return [start];
        const visited = new Set([keyOf(start)]);
        const queue = [[start]];
        let head = 0;
        while (head < queue.length) {
            const path = queue[head++];
            if (path.length - 1 >= maxHops)
                continue;
            for (const [neighbor] of this._adjacent(path[path.length - 1], relType, direction)) {
                const neighborKey = keyOf(neighbor);
                if (visited.has(neighborKey) || !gate(neighbor))
                    continue;
                if (neighborKey === endKey)
                    return [...path, neighbor];
                visited.add(neighborKey);
                queue.push([...path, neighbor]);
            }
        }
        return null;
    }
    /**
     * Find semantically-rooted path to a target node.
     *
     * Starts from semantically similar nodes and finds the best path to the
     * target. Takes either an options object in place of `relType` or the
     * positional form (`direction` and the filter parameters come last).
     *
     * Filtering: seeds are chosen among passing nodes, and the *target must
     * pass* or the result is null. By default the path may run through failing
     * nodes (it is the fewest-hops path, as without a filter); with
     * `strictFilter` the path is the fewest-hops path that uses passing nodes
     * only, which can be longer or not exist.
     */
    semanticPath(query, targetRef, relTypeOrOptions, maxHopsArg = 5, topKSeedsArg = 3, thresholdArg = 0.0, decayArg = 0.9, directionArg = Direction.OUT, nodeFilterArg, nodeTypesArg, strictFilterArg = false) {
        const options = relTypeOrOptions !== null && typeof relTypeOrOptions === 'object'
            ? relTypeOrOptions
            : {
                relType: relTypeOrOptions,
                maxHops: maxHopsArg,
                topKSeeds: topKSeedsArg,
                threshold: thresholdArg,
                decay: decayArg,
                direction: directionArg,
                nodeFilter: nodeFilterArg,
                nodeTypes: nodeTypesArg,
                strictFilter: strictFilterArg,
            };
        const { relType, maxHops = 5, topKSeeds = 3, threshold = 0.0, decay = 0.9, direction = Direction.OUT, nodeFilter, nodeTypes, strictFilter = false, } = options;
        const gate = this._nodeGate(nodeFilter, nodeTypes);
        if (gate !== null && !gate(targetRef))
            return null;
        const seeds = this._seedSearch(query, topKSeeds, undefined, threshold, gate);
        // Every seed that reaches the target is a candidate, and the best score
        // wins: a slightly less similar seed one hop from the target beats a
        // better seed five hops away. Seeds arrive best-first, so a tie (strict
        // > below) goes to the more similar one. As in semanticMultiHop, a
        // negative seed passes no relevance along rather than "decaying" toward
        // zero and outscoring itself.
        let best = null;
        for (const seed of seeds) {
            let nodes;
            if (strictFilter && gate !== null) {
                const gated = this._gatedShortestPath(seed.nodeRef, targetRef, relType, maxHops, direction, gate);
                if (gated === null)
                    continue;
                nodes = gated;
            }
            else {
                const path = this.shortestPath(seed.nodeRef, targetRef, relType, maxHops, direction);
                if (!path)
                    continue;
                nodes = path.nodes;
            }
            const length = nodes.length - 1;
            const score = Math.max(seed.score, 0) * Math.pow(decay, length);
            if (best === null || score > best.score) {
                best = {
                    nodeRef: targetRef,
                    node: this._resolveNode(targetRef),
                    score,
                    hopCount: length,
                    path: nodes,
                };
            }
        }
        return best;
    }
    /**
     * Grow `keep` with connector nodes until it is as connected as the hop
     * budget allows. Returns the connectors added, in order.
     *
     * Greedy: each round finds, over all pairs of components, the shortest route
     * (at most `maxHops` edges, honouring `relType`/`direction`) whose interior
     * nodes lie outside `keep`, adds its interior, and recomputes components.
     * Each round merges at least two, so it terminates; it stops when one
     * component remains or no route fits the budget. `gate` (set for
     * `strictFilter`) bars failing nodes from serving as connectors.
     */
    _connectComponents(keep, relType, direction, maxHops, gate) {
        const added = [];
        const keyOf = (ref) => `${ref[0]}:${ref[1]}`;
        // Same ordering as Python's (type, type(id).__name__, str(id)) tuple.
        const compareRefs = (a, b) => compareCodePoints(String(a[0]), String(b[0])) ||
            compareCodePoints(typeof a[1] === 'number' ? 'int' : 'str', typeof b[1] === 'number' ? 'int' : 'str') ||
            compareCodePoints(String(a[1]), String(b[1]));
        while (keep.size > 1) {
            // Weakly connected components of the induced subgraph.
            const parent = new Map();
            for (const key of keep.keys())
                parent.set(key, key);
            const find = (key) => {
                while (parent.get(key) !== key) {
                    const grand = parent.get(parent.get(key));
                    parent.set(key, grand);
                    key = grand;
                }
                return key;
            };
            for (const edge of this.edges(relType)) {
                const s = keyOf(edge.source);
                const t = keyOf(edge.target);
                if (keep.has(s) && keep.has(t))
                    parent.set(find(s), find(t));
            }
            const groups = new Map();
            for (const [key, ref] of keep) {
                const root = find(key);
                const members = groups.get(root);
                if (members)
                    members.push(ref);
                else
                    groups.set(root, [ref]);
            }
            if (groups.size <= 1)
                break;
            const ordered = [...groups.values()]
                .map(g => g.sort(compareRefs))
                .sort((a, b) => compareRefs(a[0], b[0]));
            let bestRoute = null;
            for (const members of ordered) {
                const route = this._shortestLink(members, keep, relType, direction, maxHops, gate);
                if (route !== null && (bestRoute === null || route.length < bestRoute.length)) {
                    bestRoute = route;
                }
            }
            if (bestRoute === null)
                break;
            for (const ref of bestRoute.slice(1, -1)) {
                keep.set(keyOf(ref), ref);
                added.push(ref);
            }
        }
        return added;
    }
    /** Shortest route from any of `members` to a kept node outside them. */
    _shortestLink(members, keep, relType, direction, maxHops, gate) {
        const keyOf = (ref) => `${ref[0]}:${ref[1]}`;
        const visited = new Set(members.map(keyOf));
        const queue = members.map(m => [m]);
        let head = 0;
        while (head < queue.length) {
            const path = queue[head++];
            if (path.length - 1 >= maxHops)
                continue;
            for (const [neighbor] of this._adjacent(path[path.length - 1], relType, direction)) {
                const neighborKey = keyOf(neighbor);
                if (visited.has(neighborKey))
                    continue;
                if (keep.has(neighborKey))
                    return [...path, neighbor];
                if (gate !== null && !gate(neighbor))
                    continue;
                if (this._resolveNode(neighbor) === null)
                    continue;
                visited.add(neighborKey);
                queue.push([...path, neighbor]);
            }
        }
        return null;
    }
    /**
     * Extract the slice of the graph a query is actually about.
     *
     * Runs semanticMultiHop, then returns a new SemanticGraph holding exactly
     * those nodes, every edge induced between them, and their embeddings - so
     * the result can be traversed, searched or serialized on its own. This is
     * what makes an ISON graph useful as LLM context: a small, on-topic graph to
     * inject rather than the whole store.
     *
     * Takes every semanticMultiHop option (including `weightField`,
     * `nodeFilter`, `nodeTypes` and `strictFilter`), plus:
     *  - `name`: name for the returned graph (default `<name>_subgraph`)
     *  - `connect`: also add *connector* nodes so the selected nodes fall into as
     *    few connected components as possible. Components are weak (edge
     *    direction ignored when judging the result), but connector routes honour
     *    `relType` and `direction`. Each round adds the interior of the shortest
     *    route between two components (greedy; not a minimum Steiner tree),
     *    until one component remains or no route fits `connectMaxHops`. Nothing
     *    but those interior nodes is added, and they are not marked in any way.
     *  - `connectMaxHops`: longest route, counted in edges, that may join two
     *    components (default 2: one connector node per join; 3 allows two). A
     *    direct edge needs no connector (it is already an induced edge). Must be
     *    at least 1.
     *
     * With `connect` the result can hold **more than** `topKResults` nodes:
     * `topKResults` caps the *selected* nodes, and connectors are extra. Without
     * `strictFilter` a connector may be a node that fails `nodeFilter` /
     * `nodeTypes` (it is only a bridge); with it, connectors must pass too.
     */
    semanticSubgraph(query, options = {}) {
        const { connect = false, connectMaxHops = 2 } = options;
        if (connectMaxHops < 1) {
            throw new EmbeddingError(`connectMaxHops must be >= 1, got ${connectMaxHops}`);
        }
        const results = this.semanticMultiHop(query, options);
        const keyOf = (ref) => `${ref[0]}:${ref[1]}`;
        const keep = new Map(results.map(r => [keyOf(r.nodeRef), r.nodeRef]));
        if (connect && keep.size > 0) {
            const gate = options.strictFilter
                ? this._nodeGate(options.nodeFilter, options.nodeTypes)
                : null;
            this._connectComponents(keep, options.relType, options.direction ?? Direction.OUT, connectMaxHops, gate);
        }
        const sub = new SemanticGraph({
            name: options.name || `${this.name}_subgraph`,
            directed: this.directed,
            encoder: this._encoder || undefined,
            autoEmbed: false,
            embedFields: [...this._embedFields],
        });
        for (const ref of keep.values()) {
            const node = this._resolveNode(ref);
            if (!node)
                continue;
            sub.addNode(node.type, node.id, node.properties);
            const record = this._embeddingStore.get(ref);
            if (record)
                sub.embeddingStore.add(ref, record.text, record.vector);
        }
        for (const edge of this.edges(options.relType)) {
            if (keep.has(keyOf(edge.source)) && keep.has(keyOf(edge.target))) {
                sub.addEdge(edge.relType, edge.source, edge.target, edge.properties);
            }
        }
        return sub;
    }
    /**
     * Embed all nodes in the graph, in one batched encoder call.
     */
    embedAllNodes(textFn, skipExisting = false) {
        const defaultTextFn = (node) => this.embedTextFor(node.properties) || `${node.type}:${node.id}`;
        const fn = textFn || defaultTextFn;
        const batch = [];
        for (const node of this.nodes()) {
            if (skipExisting && this._embeddingStore.get(node.ref))
                continue;
            const text = fn(node);
            if (text)
                batch.push([node.ref, text]);
        }
        return this.embedNodes(batch);
    }
    /**
     * Nodes whose stored vector was not made by the current encoder.
     *
     * Stale means the vector's recorded model differs from the encoder's
     * `modelName`, or its length differs from the encoder's `dimension`. Vectors
     * recorded as "unknown"/"provided" (from ports that store no model, or
     * supplied without an encoder) cannot be judged by model and are left out
     * unless `includeUnknown` is true; a dimension mismatch always counts. With
     * no encoder nothing can be judged and the result is empty.
     */
    staleEmbeddings(includeUnknown = false) {
        return this._embeddingStore.stale({ includeUnknown });
    }
    /**
     * Re-encode every stale embedding with the current encoder.
     *
     * Each vector is rebuilt from the text it was originally embedded from (so
     * a custom embed text survives); a vector stored with no text falls back to
     * the node's embed fields. If the encoder's dimension differs from the
     * store's, every stored vector is stale, so the store is emptied, its pinned
     * dimension reset, and everything re-embedded.
     *
     * @param includeUnknown Also re-embed vectors of unknown provenance
     * @param batchSize Texts per encoder call
     * @returns Number of embeddings rewritten
     */
    reembedStale(includeUnknown = false, batchSize = 256) {
        const stale = this.staleEmbeddings(includeUnknown);
        if (stale.length === 0)
            return 0;
        const store = this._embeddingStore;
        const pairs = [];
        for (const ref of stale) {
            const record = store.get(ref);
            if (!record)
                continue;
            let text = record.text;
            if (!text) {
                const node = this._resolveNode(ref);
                text = (node ? this.embedTextFor(node.properties) : '') || `${ref[0]}:${ref[1]}`;
            }
            pairs.push([ref, text]);
        }
        const encoder = this._encoder;
        if (!encoder)
            throw new NoEncoderError("No encoder provided to re-embed with");
        if (store.dimension !== null && store.dimension !== encoder.dimension) {
            store.clear(true);
        }
        const step = Math.max(1, Math.floor(batchSize));
        let count = 0;
        for (let start = 0; start < pairs.length; start += step) {
            count += this.embedNodes(pairs.slice(start, start + step));
        }
        return count;
    }
    /**
     * Build a SemanticGraph from a plain ISONGraph (for example one parsed with
     * ISONGraph.fromIson), copying its nodes and edges.
     *
     * Nodes are copied through the base addNode so that nothing is embedded on
     * the way in: embeddings come from a saved payload, already computed, and
     * re-deriving them would need an encoder and overwrite what was saved.
     * This is pure (no I/O), so it is safe in a browser; see
     * `isongraph-vector-ts/node` for saving and loading files.
     */
    static fromISONGraph(base, options = {}) {
        const graph = new SemanticGraph({
            name: base.name,
            directed: base.directed,
            ...options,
        });
        for (const node of base.nodes()) {
            ISONGraph.prototype.addNode.call(graph, node.type, node.id, { ...node.properties });
        }
        for (const relType of base.edgeTypes()) {
            for (const edge of base.edges(relType)) {
                graph.addEdge(edge.relType, edge.source, edge.target, { ...edge.properties });
            }
        }
        return graph;
    }
    toString() {
        const embCount = this._embeddingStore.count();
        return `SemanticGraph(name=${this.name}, nodes=${this.nodeCount()}, edges=${this.edgeCount()}, embeddings=${embCount})`;
    }
}
// =============================================================================
// Exports
// =============================================================================
export { ISONGraph, Node, Direction } from 'ison-graph-js';
