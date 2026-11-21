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
/** True for an object with `encodeAsync` and `encodeBatchAsync` methods. */
export function isAsyncEncoder(value) {
    const v = value;
    return (v !== null && v !== undefined && typeof v === 'object' &&
        typeof v.encodeAsync === 'function' && typeof v.encodeBatchAsync === 'function');
}
/**
 * The synchronous face of an encoder that only has an asynchronous one, so the
 * stores (whose `encoder` is an EmbeddingEncoder) can hold it. Its metadata is
 * read live from the wrapped encoder; its synchronous methods refuse, naming
 * the asynchronous ones.
 */
class AsyncOnlyEncoder {
    constructor(inner) {
        this.inner = inner;
    }
    get dimension() {
        return this.inner.dimension;
    }
    get modelName() {
        return this.inner.modelName;
    }
    get version() {
        return this.inner.version;
    }
    encode(_text) {
        throw new EmbeddingError('this encoder is asynchronous: use the Async methods (addAsync, addNodeAsync, ' +
            'similaritySearchAsync, ...) or pass the vector yourself');
    }
    encodeBatch(texts) {
        return texts.map(text => this.encode(text));
    }
    encodeAsync(text) {
        return this.inner.encodeAsync(text);
    }
    encodeBatchAsync(texts) {
        return this.inner.encodeBatchAsync(texts);
    }
}
/** What a store or graph holds: sync encoders as they are, async-only ones wrapped. */
function adoptEncoder(encoder) {
    if (encoder === null || encoder === undefined)
        return null;
    const sync = encoder;
    if (typeof sync.encode === 'function' && typeof sync.encodeBatch === 'function') {
        return encoder;
    }
    return isAsyncEncoder(encoder) ? new AsyncOnlyEncoder(encoder) : encoder;
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
// =============================================================================
// HTTP Encoder
// =============================================================================
/** A request to an embeddings endpoint failed, or its answer was unusable. */
export class HttpEncoderError extends EmbeddingError {
    constructor(message, status) {
        super(message);
        this.name = 'HttpEncoderError';
        if (status !== undefined)
            this.status = status;
    }
}
const MAX_RETRY_WAIT_MS = 30000;
const ERROR_SNIPPET_CHARS = 200;
/**
 * Embeddings from an HTTP endpoint (OpenAI-compatible or Ollama), using the
 * platform `fetch` - no dependency, and no `node:` import, so it works in
 * Node 18+ and in browsers.
 *
 * **Every text you embed is sent to the configured endpoint**, including the
 * text of every node a graph embeds and every search query.
 *
 * It is an {@link AsyncEncoder} (`encodeAsync` / `encodeBatchAsync`), and that
 * is the way to use it: give it to a store or graph and call the `...Async`
 * methods (`addNodeAsync`, `updateNodeAsync`, `embedAllNodesAsync`,
 * `similaritySearchAsync`, ...), which await the network themselves. Identical
 * texts requested at the same moment share one request.
 *
 * `EmbeddingEncoder` is synchronous and a network call is not, so the
 * synchronous `encode` / `encodeBatch` serve from a cache: fetch first with
 * `await encoder.prefetch(texts)` (every `...Async` call fills it too), and a
 * text that was not fetched makes them throw an HttpEncoderError. That keeps
 * the plain synchronous methods (`addNode`, `addBatch`, `similaritySearch`)
 * usable for prefetched texts.
 *
 * @example
 * ```typescript
 * const encoder = new HttpEncoder({
 *   url: 'http://localhost:11434/api/embed', model: 'nomic-embed-text', api: 'ollama',
 * });
 * const graph = new SemanticGraph({ encoder });
 * await graph.addNodeAsync('person', 1, { name: 'Alice' });      // embeds over the network
 * const hits = await graph.similaritySearchAsync('who is Alice');
 *
 * // or, synchronously, after fetching the texts:
 * const texts = ['Alice is an engineer', 'Bob plans roadmaps'];
 * await encoder.prefetch([...texts, 'engineering']);
 * const store = new EmbeddingStore(encoder);
 * store.addBatch([[['person', 1], texts[0]], [['person', 2], texts[1]]]);
 * store.similaritySearch('engineering', 1);
 * ```
 */
export class HttpEncoder {
    constructor(options) {
        this._cache = new Map();
        this._inflight = new Map();
        if (!options || typeof options.url !== 'string' || typeof options.model !== 'string' || !options.model) {
            throw new EmbeddingError('HttpEncoder needs a url and a model');
        }
        let parsed;
        try {
            parsed = new URL(options.url);
        }
        catch {
            throw new EmbeddingError('HttpEncoder url is not a valid URL');
        }
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
            throw new EmbeddingError(`HttpEncoder url must be http or https, got ${parsed.protocol}`);
        }
        if (parsed.username || parsed.password) {
            throw new EmbeddingError('HttpEncoder url must not carry credentials: pass apiKey or headers instead');
        }
        const api = options.api ?? 'openai';
        if (api !== 'openai' && api !== 'ollama') {
            throw new EmbeddingError(`HttpEncoder api must be 'openai' or 'ollama', got ${JSON.stringify(api)}`);
        }
        const positive = (value, fallback, name, integer) => {
            if (value === undefined)
                return fallback;
            if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || (integer && !Number.isInteger(value))) {
                throw new EmbeddingError(`HttpEncoder ${name} must be a positive${integer ? ' integer' : ' number'}, got ${value}`);
            }
            return value;
        };
        const nonNegative = (value, fallback, name) => {
            if (value === undefined)
                return fallback;
            if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
                throw new EmbeddingError(`HttpEncoder ${name} must be a non-negative number, got ${value}`);
            }
            return value;
        };
        this.modelName = options.model;
        if (options.version !== undefined)
            this.version = options.version;
        this._api = api;
        this._url = options.url;
        // Origin and path only: a URL may carry credentials or a key in its query.
        this._safeUrl = `${parsed.protocol}//${parsed.host}${parsed.pathname}`;
        this._dimension = options.dimension === undefined ? null : positive(options.dimension, 0, 'dimension', true);
        this._batchSize = positive(options.batchSize, 64, 'batchSize', true);
        this._timeoutMs = positive(options.timeoutMs, 30000, 'timeoutMs', false);
        this._maxRetries = Math.floor(nonNegative(options.maxRetries, 2, 'maxRetries'));
        this._retryDelayMs = nonNegative(options.retryDelayMs, 500, 'retryDelayMs');
        this._maxCache = Math.floor(positive(options.maxCacheEntries, 100000, 'maxCacheEntries', false));
        this._concurrency = positive(options.concurrency, 1, 'concurrency', true);
        const fetchFn = options.fetch ?? globalThis.fetch;
        if (typeof fetchFn !== 'function') {
            throw new EmbeddingError('HttpEncoder needs a global fetch (Node 18+) or a fetch option');
        }
        this._fetch = fetchFn;
        const apiKey = options.apiKey;
        const extra = options.headers ?? {};
        this._buildHeaders = () => ({
            'Content-Type': 'application/json',
            ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
            ...extra,
        });
        const secrets = [apiKey, ...Object.entries(extra)
                .filter(([name]) => /key|auth|token|secret|cookie/i.test(name))
                .map(([, value]) => value)]
            .filter((s) => typeof s === 'string' && s.length > 0);
        this._redact = (text) => secrets.reduce((acc, s) => acc.split(s).join('***'), text);
    }
    /** Vector length; `NaN` until it is given or learned from a first response. */
    get dimension() {
        return this._dimension ?? NaN;
    }
    /** Number of texts whose vectors are cached for the synchronous `encode`. */
    get cacheSize() {
        return this._cache.size;
    }
    /** Forget every cached vector. */
    clearCache() {
        this._cache.clear();
    }
    /**
     * Fetch (and cache) the vectors for `texts`, so the synchronous `encode` /
     * `encodeBatch` can serve them. Texts already cached are not requested again.
     * More distinct texts than `maxCacheEntries` evicts the oldest.
     */
    async prefetch(texts) {
        await this.encodeBatchAsync(texts);
    }
    /** Fetch one vector (served from the cache when present). */
    async encodeAsync(text) {
        return (await this.encodeBatchAsync([text]))[0];
    }
    /**
     * Fetch vectors for many texts, in order. Only texts that are neither cached
     * nor already being fetched hit the network, `batchSize` per request and
     * `concurrency` requests at a time; the first failure rejects the call.
     */
    async encodeBatchAsync(texts) {
        const found = new Map();
        const fresh = [];
        const waits = [];
        const considered = new Set();
        for (const text of texts) {
            if (considered.has(text))
                continue;
            considered.add(text);
            const hit = this._cache.get(text);
            if (hit) {
                found.set(text, hit);
                continue;
            }
            const shared = this._inflight.get(text);
            if (shared) {
                waits.push(shared.then(fetched => { found.set(text, fetched.get(text)); }));
                continue;
            }
            fresh.push(text);
        }
        const chunks = [];
        for (let start = 0; start < fresh.length; start += this._batchSize) {
            chunks.push(fresh.slice(start, start + this._batchSize));
        }
        let next = 0;
        let failed = false;
        const worker = async () => {
            while (!failed && next < chunks.length) {
                const chunk = chunks[next++];
                const job = this._request(chunk).then(vectors => {
                    const fetched = new Map();
                    chunk.forEach((text, i) => {
                        fetched.set(text, vectors[i]);
                        this._remember(text, vectors[i]);
                    });
                    return fetched;
                });
                for (const text of chunk)
                    this._inflight.set(text, job);
                const release = () => {
                    for (const text of chunk)
                        if (this._inflight.get(text) === job)
                            this._inflight.delete(text);
                };
                job.then(release, release);
                try {
                    for (const [text, vector] of await job)
                        found.set(text, vector);
                }
                catch (error) {
                    failed = true;
                    throw error;
                }
            }
        };
        const workers = Array.from({ length: Math.min(this._concurrency, chunks.length) }, worker);
        await Promise.all([...waits, ...workers]);
        return texts.map(text => found.get(text).slice());
    }
    /** Synchronous: serves a prefetched text, otherwise throws HttpEncoderError. */
    encode(text) {
        const hit = this._cache.get(text);
        if (!hit) {
            throw new HttpEncoderError('HttpEncoder.encode is synchronous and this text was not fetched yet: ' +
                'await encoder.prefetch([...texts]) (or encodeAsync) first');
        }
        return hit.slice();
    }
    /** Synchronous: serves prefetched texts, otherwise throws HttpEncoderError. */
    encodeBatch(texts) {
        return texts.map(text => this.encode(text));
    }
    _remember(text, vector) {
        this._cache.delete(text);
        this._cache.set(text, vector);
        while (this._cache.size > this._maxCache) {
            const oldest = this._cache.keys().next().value;
            this._cache.delete(oldest);
        }
    }
    _fail(message, status) {
        return new HttpEncoderError(this._redact(message), status);
    }
    /** One POST (with bounded retries) for at most `batchSize` texts. */
    async _request(texts) {
        const body = JSON.stringify({ model: this.modelName, input: texts });
        for (let attempt = 0;; attempt++) {
            const retriable = attempt < this._maxRetries;
            let status = 0;
            let retryAfter = null;
            let text = '';
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), this._timeoutMs);
            try {
                const res = await this._fetch(this._url, {
                    method: 'POST',
                    headers: this._buildHeaders(),
                    body,
                    signal: controller.signal,
                });
                status = res.status;
                retryAfter = res.headers.get('retry-after');
                text = await res.text();
            }
            catch (err) {
                const timedOut = controller.signal.aborted;
                if (retriable) {
                    await this._sleep(this._backoff(attempt, null));
                    continue;
                }
                throw this._fail(timedOut
                    ? `request to ${this._safeUrl} timed out after ${this._timeoutMs} ms`
                    : `request to ${this._safeUrl} failed: ${err?.message ?? String(err)}`);
            }
            finally {
                clearTimeout(timer);
            }
            if (status >= 200 && status < 300)
                return this._parse(text, texts.length);
            if ((status === 429 || status >= 500) && retriable) {
                await this._sleep(this._backoff(attempt, retryAfter));
                continue;
            }
            const snippet = text.trim().slice(0, ERROR_SNIPPET_CHARS);
            throw this._fail(`embedding request to ${this._safeUrl} failed with HTTP ${status}` + (snippet ? `: ${snippet}` : ''), status);
        }
    }
    _backoff(attempt, retryAfter) {
        const seconds = retryAfter === null ? NaN : Number(retryAfter);
        if (Number.isFinite(seconds) && seconds >= 0)
            return Math.min(seconds * 1000, MAX_RETRY_WAIT_MS);
        return Math.min(this._retryDelayMs * 2 ** attempt, MAX_RETRY_WAIT_MS);
    }
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    _parse(text, expected) {
        let json;
        try {
            json = JSON.parse(text);
        }
        catch {
            throw this._fail(`${this._safeUrl} did not return JSON`);
        }
        let vectors;
        if (this._api === 'ollama') {
            if (!json || !Array.isArray(json.embeddings)) {
                throw this._fail(`${this._safeUrl} answer has no 'embeddings' array (is api: 'ollama' right?)`);
            }
            vectors = json.embeddings;
        }
        else {
            if (!json || !Array.isArray(json.data)) {
                throw this._fail(`${this._safeUrl} answer has no 'data' array (is api: 'openai' right?)`);
            }
            const items = json.data;
            const ordered = items.every(item => item && typeof item.index === 'number')
                ? [...items].sort((a, b) => a.index - b.index)
                : items;
            vectors = ordered.map(item => item && item.embedding);
        }
        if (vectors.length !== expected) {
            throw this._fail(`${this._safeUrl} returned ${vectors.length} embeddings for ${expected} texts`);
        }
        return vectors.map((vector, i) => {
            if (!Array.isArray(vector) || vector.length === 0 || !vector.every(v => typeof v === 'number' && Number.isFinite(v))) {
                throw this._fail(`${this._safeUrl} embedding ${i} is not a non-empty array of numbers`);
            }
            if (this._dimension === null)
                this._dimension = vector.length;
            else if (vector.length !== this._dimension) {
                throw new DimensionMismatchError(`${this.modelName} returned ${vector.length} dimensions, expected ${this._dimension}`);
            }
            return vector;
        });
    }
}
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
// =============================================================================
// Approximate (IVF) search
// =============================================================================
/** An exact search for at most this many results keeps them in a small sorted buffer; a larger k collects every hit first. */
const EXACT_TOPK_BUFFER = 256;
/** Stores below this many vectors answer exactly even with `ann` on. */
const ANN_MIN_SIZE = 50000;
/** New or updated rows are scored exactly until this many (or this fraction of the trained size) are pending. */
const ANN_FOLD_MIN = 512;
const ANN_FOLD_FRACTION = 0.02;
/** Retrain once writes + deletes since training exceed this fraction of the trained size. */
const ANN_RETRAIN_FRACTION = 0.5;
/**
 * k-means trains on up to this many sampled vectors per cell, for this many
 * rounds. Python samples 40 per cell; with no BLAS here the build cost matters,
 * and on 600k clustered vectors 24 per cell took 296 s against 390 s for 40
 * with the same recall at the default probe (0.978) and a little less at
 * `annNprobe` 1 (0.93 against 0.96).
 */
const ANN_SAMPLE_PER_CELL = 24;
const ANN_KMEANS_ITERS = 8;
/** From this many cells on, nearest-cell lookups go through a coarse layer of about 3 x sqrt(nlist) super-cells. */
const ANN_COARSE_MIN_CELLS = 256;
/**
 * ... and look inside this many of the super-cells nearest the vector. Measured
 * on 100k-300k clustered vectors: this finds the true nearest cell for 95-99%
 * of them at about a third of the brute-force cost; looking inside only 4
 * left 2-20% in a different cell, which capped recall whatever `annNprobe`.
 */
const ANN_COARSE_PROBE = 24;
/** k-means rounds that place the super-cells over the centroids. */
const ANN_COARSE_ROUNDS = 3;
/** Super-cells per sqrt(nlist): finer ones find the nearest cell more often for the same work. */
const ANN_COARSE_FACTOR = 3;
/** float32 rows vs the float64 exact score differ by about 1e-7; candidates this close under the threshold are re-scored. */
const ANN_THRESHOLD_SLACK = 1e-6;
/** Banker's rounding, as Python's `round`, so the default cell counts agree across ports. */
function roundHalfEven(x) {
    const floor = Math.floor(x);
    const diff = x - floor;
    if (diff < 0.5)
        return floor;
    if (diff > 0.5)
        return floor + 1;
    return floor % 2 === 0 ? floor : floor + 1;
}
const nowMs = () => typeof performance !== 'undefined' ? performance.now() : Date.now();
/** Small seeded generator (mulberry32): the same seed gives the same index. */
class Rng {
    constructor(seed) {
        this._a = (Math.trunc(seed) ^ 0x9e3779b9) >>> 0;
    }
    next() {
        this._a = (this._a + 0x6d2b79f5) >>> 0;
        let t = this._a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    /** Integer in [0, n). */
    below(n) {
        return Math.floor(this.next() * n);
    }
}
/** Dot product of a[ao..ao+dim) and b[bo..bo+dim), four accumulators wide. */
function dot2(a, ao, b, bo, dim) {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    let j = 0;
    const lim = dim - 3;
    for (; j < lim; j += 4) {
        s0 += a[ao + j] * b[bo + j];
        s1 += a[ao + j + 1] * b[bo + j + 1];
        s2 += a[ao + j + 2] * b[bo + j + 2];
        s3 += a[ao + j + 3] * b[bo + j + 3];
    }
    for (; j < dim; j++)
        s0 += a[ao + j] * b[bo + j];
    return s0 + s1 + s2 + s3;
}
/**
 * Nearest unit centroid to a vector, by dot product.
 *
 * With many cells a brute-force pass over all of them costs nlist x dim per
 * vector, which is what makes k-means and assignment slow in a language with no
 * BLAS. Past `ANN_COARSE_MIN_CELLS` the centroids are themselves grouped into
 * about sqrt(nlist) super-cells and only the members of the few super-cells
 * nearest the vector are compared: roughly 10x less work, and the cell it picks
 * is the true nearest one all but rarely. A vector put in a near-miss cell is
 * still found by any query that probes that cell.
 */
class CentroidSearch {
    constructor(centroids, nlist, dim, coarseFactor = ANN_COARSE_FACTOR, probe = ANN_COARSE_PROBE) {
        this._coarse = null;
        this._ncoarse = 0;
        this._members = new Int32Array(0);
        this._memberOffsets = new Int32Array(0);
        this._coarseScore = new Float64Array(0);
        this._pickIdx = new Int32Array(0);
        this._pickScore = new Float64Array(0);
        this._nlist = nlist;
        this._dim = dim;
        this._centroids = centroids;
        this._probe = probe;
        if (nlist >= ANN_COARSE_MIN_CELLS)
            this._buildCoarse(coarseFactor);
    }
    _buildCoarse(coarseFactor) {
        const nlist = this._nlist;
        const dim = this._dim;
        const centroids = this._centroids;
        const ncoarse = Math.max(2, Math.round(coarseFactor * Math.sqrt(nlist)));
        const coarse = new Float32Array(ncoarse * dim);
        for (let i = 0; i < ncoarse; i++) {
            const src = Math.floor((i * nlist) / ncoarse) * dim;
            coarse.set(centroids.subarray(src, src + dim), i * dim);
        }
        const owner = new Int32Array(nlist);
        const sums = new Float64Array(ncoarse * dim);
        const counts = new Int32Array(ncoarse);
        for (let round = 0; round < ANN_COARSE_ROUNDS; round++) {
            for (let c = 0; c < nlist; c++) {
                let best = 0;
                let bestScore = -Infinity;
                for (let k = 0; k < ncoarse; k++) {
                    const s = dot2(centroids, c * dim, coarse, k * dim, dim);
                    if (s > bestScore) {
                        bestScore = s;
                        best = k;
                    }
                }
                owner[c] = best;
            }
            if (round === ANN_COARSE_ROUNDS - 1)
                break;
            sums.fill(0);
            counts.fill(0);
            for (let c = 0; c < nlist; c++) {
                const k = owner[c];
                counts[k]++;
                const base = k * dim;
                const src = c * dim;
                for (let j = 0; j < dim; j++)
                    sums[base + j] += centroids[src + j];
            }
            for (let k = 0; k < ncoarse; k++) {
                if (counts[k] === 0)
                    continue;
                let len = 0;
                for (let j = 0; j < dim; j++)
                    len += sums[k * dim + j] * sums[k * dim + j];
                len = Math.sqrt(len);
                if (len === 0)
                    continue;
                for (let j = 0; j < dim; j++)
                    coarse[k * dim + j] = sums[k * dim + j] / len;
            }
        }
        const memberOffsets = new Int32Array(ncoarse + 1);
        for (let c = 0; c < nlist; c++)
            memberOffsets[owner[c] + 1]++;
        for (let k = 0; k < ncoarse; k++)
            memberOffsets[k + 1] += memberOffsets[k];
        const fill = memberOffsets.slice(0, ncoarse);
        const members = new Int32Array(nlist);
        for (let c = 0; c < nlist; c++)
            members[fill[owner[c]]++] = c;
        this._coarse = coarse;
        this._ncoarse = ncoarse;
        this._members = members;
        this._memberOffsets = memberOffsets;
        this._coarseScore = new Float64Array(ncoarse);
        const probe = Math.min(this._probe, ncoarse);
        this._pickIdx = new Int32Array(probe);
        this._pickScore = new Float64Array(probe);
    }
    /** Index of the centroid most similar to v[off..off+dim). */
    nearest(v, off) {
        const dim = this._dim;
        const centroids = this._centroids;
        let best = 0;
        let bestScore = -Infinity;
        const coarse = this._coarse;
        if (coarse === null) {
            for (let c = 0; c < this._nlist; c++) {
                const s = dot2(centroids, c * dim, v, off, dim);
                if (s > bestScore) {
                    bestScore = s;
                    best = c;
                }
            }
            return best;
        }
        const pickIdx = this._pickIdx;
        const pickScore = this._pickScore;
        const probe = pickIdx.length;
        let picked = 0;
        for (let k = 0; k < this._ncoarse; k++) {
            const s = dot2(coarse, k * dim, v, off, dim);
            if (picked === probe && s <= pickScore[probe - 1])
                continue;
            let i = picked < probe ? picked++ : probe - 1;
            while (i > 0 && pickScore[i - 1] < s) {
                pickScore[i] = pickScore[i - 1];
                pickIdx[i] = pickIdx[i - 1];
                i--;
            }
            pickScore[i] = s;
            pickIdx[i] = k;
        }
        const members = this._members;
        const offsets = this._memberOffsets;
        for (let p = 0; p < picked; p++) {
            const k = pickIdx[p];
            for (let m = offsets[k]; m < offsets[k + 1]; m++) {
                const c = members[m];
                const s = dot2(centroids, c * dim, v, off, dim);
                if (s > bestScore) {
                    bestScore = s;
                    best = c;
                }
            }
        }
        return best;
    }
}
/** Rows per matrix chunk (= one scan block). Chunks never move once allocated, so growth never copies stored rows. */
const CHUNK_ROWS = 1024;
const CHUNK_SHIFT = 10;
const CHUNK_MASK = CHUNK_ROWS - 1;
/** Rows the first chunk starts with; it doubles until it holds CHUNK_ROWS, later chunks are allocated whole. */
const FIRST_CHUNK_ROWS = 16;
/** True when this host stores a Float32Array in little-endian byte order (the snapshot's byte order). */
export function hostIsLittleEndian() {
    return new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
}
const HOST_LITTLE_ENDIAN = hostIsLittleEndian();
/**
 * The store's single copy of every vector, plus the per-row facts around it.
 *
 * A record *is a slot number*; nothing else owns the floats (see the Rust
 * crate's DESIGN.md, section 1). The rows live in `Float32Array` chunks of
 * {@link CHUNK_ROWS} rows each, `dim` floats per row, row-major; `norms`,
 * `codes` and `modelIx` are parallel columns, and `keys`/`refs`/`texts` hold the
 * strings. Slots are in arrival order, which is the order the store used to
 * iterate its records in, so ties still break the way they always did: an update
 * overwrites its slot in place, an insert takes `n`, a delete tombstones (key
 * `null`, norm `NaN`). Trailing tombstones are popped; once a quarter of the
 * slots are dead the live rows are moved down, keeping their order.
 *
 * Exact search, the IVF index and the HNSW graph all read the rows from here, so
 * an index adds no second copy of the vectors. Not part of the stable API: it is exported so
 * the snapshot module can read and fill it.
 */
export class VectorTable {
    constructor(dim) {
        this.chunks = [];
        /** Euclidean norm of each row, computed from the float32 values; NaN = tombstone. */
        this.norms = new Float64Array(0);
        /** Node type of each row, as an index into `typeNames`. */
        this.codes = new Int32Array(0);
        /** Model of each row, as an index into `modelNames`. */
        this.modelIx = new Int32Array(0);
        /** `"type:id"` of each row, or null for a tombstone. */
        this.keys = [];
        this.refs = [];
        this.texts = [];
        this.modelNames = [];
        this.typeNames = [];
        this.codeOf = new Map();
        /** key -> slot: the only lookup structure. */
        this.index = new Map();
        this._modelOf = new Map();
        /** Slots in use, tombstones included, and how many of those are tombstones. */
        this.n = 0;
        this.dead = 0;
        this.ivf = null;
        this.hnsw = null;
        this.dim = dim;
        this.query = new Float32Array(dim);
    }
    get capacity() {
        return this.norms.length;
    }
    /** Live rows. */
    get live() {
        return this.index.size;
    }
    _modelId(name) {
        let id = this._modelOf.get(name);
        if (id === undefined) {
            id = this.modelNames.length;
            this.modelNames.push(name);
            this._modelOf.set(name, id);
        }
        return id;
    }
    _growColumns() {
        const capacity = this.capacity + Math.max(1024, this.capacity >> 1);
        const norms = new Float64Array(capacity);
        norms.set(this.norms.subarray(0, this.n));
        const codes = new Int32Array(capacity);
        codes.set(this.codes.subarray(0, this.n));
        const modelIx = new Int32Array(capacity);
        modelIx.set(this.modelIx.subarray(0, this.n));
        this.norms = norms;
        this.codes = codes;
        this.modelIx = modelIx;
    }
    /** Make sure the chunk holding `slot` exists and is big enough. */
    _ensureRow(slot) {
        const c = slot >>> CHUNK_SHIFT;
        const dim = this.dim;
        if (c === this.chunks.length) {
            this.chunks.push(new Float32Array((c === 0 ? FIRST_CHUNK_ROWS : CHUNK_ROWS) * dim));
        }
        else if (c === 0) {
            const first = this.chunks[0];
            const need = ((slot & CHUNK_MASK) + 1) * dim;
            if (first.length < need) {
                const grown = new Float32Array(Math.min(CHUNK_ROWS * dim, Math.max(need, first.length * 2)));
                grown.set(first);
                this.chunks[0] = grown;
            }
        }
    }
    /** Find or create the slot of `key` and set its metadata; the caller writes the row and the norm. */
    _claim(key, ref, text, model) {
        const type = ref[0];
        let code = this.codeOf.get(type);
        if (code === undefined) {
            code = this.typeNames.length;
            this.typeNames.push(type);
            this.codeOf.set(type, code);
        }
        let slot = this.index.get(key);
        if (slot === undefined) {
            if (this.n >= this.capacity)
                this._growColumns();
            slot = this.n++;
            this._ensureRow(slot);
            this.keys[slot] = key;
            this.index.set(key, slot);
        }
        this.refs[slot] = ref;
        this.texts[slot] = text;
        this.codes[slot] = code;
        this.modelIx[slot] = this._modelId(model);
        return slot;
    }
    /** Insert or overwrite the row of `key`; returns its slot. The values are stored as float32. */
    put(key, ref, text, model, vector) {
        const slot = this._claim(key, ref, text, model);
        const dim = this.dim;
        const chunk = this.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        let sum = 0;
        for (let j = 0; j < dim; j++) {
            const f = Math.fround(vector[j]);
            chunk[base + j] = f;
            sum += f * f;
        }
        this.norms[slot] = Math.sqrt(sum);
        if (this.ivf !== null)
            this.ivf.notePut(slot, this.capacity);
        if (this.hnsw !== null)
            this.hnsw.notePut(this, slot);
        return slot;
    }
    /**
     * Like {@link put}, from `dim * 4` bytes of little-endian IEEE-754 float32
     * (the snapshot's row encoding). On a little-endian host the bytes are copied
     * as they are; otherwise (or with `portable`) each float is read through a
     * DataView.
     */
    putLittleEndian(key, ref, text, model, bytes, portable = false) {
        const slot = this._claim(key, ref, text, model);
        const dim = this.dim;
        const chunk = this.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        if (HOST_LITTLE_ENDIAN && !portable) {
            new Uint8Array(chunk.buffer, chunk.byteOffset + base * 4, dim * 4).set(bytes);
        }
        else {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            for (let j = 0; j < dim; j++)
                chunk[base + j] = view.getFloat32(4 * j, true);
        }
        let sum = 0;
        for (let j = 0; j < dim; j++) {
            const f = chunk[base + j];
            sum += f * f;
        }
        this.norms[slot] = Math.sqrt(sum);
        if (this.ivf !== null)
            this.ivf.notePut(slot, this.capacity);
        if (this.hnsw !== null)
            this.hnsw.notePut(this, slot);
        return slot;
    }
    /** Tombstone the row of `key`; false when there is none. */
    remove(key) {
        const slot = this.index.get(key);
        if (slot === undefined)
            return false;
        this.index.delete(key);
        this.keys[slot] = null;
        this.refs[slot] = null;
        this.texts[slot] = '';
        this.norms[slot] = NaN;
        this.dead++;
        if (this.ivf !== null)
            this.ivf.noteDrop(slot);
        if (this.hnsw !== null)
            this.hnsw.noteDrop();
        while (this.n > 0 && this.keys[this.n - 1] === null) { // trailing tombstones
            this.n--;
            this.keys.pop();
            this.refs.pop();
            this.texts.pop();
            this.dead--;
        }
        if (this.dead > 32 && this.dead * 4 > this.n)
            this._compact();
        return true;
    }
    /** Squeeze the tombstones out, keeping the live rows in their order. */
    _compact() {
        const dim = this.dim;
        const live = new Int32Array(this.n - this.dead);
        let kept = 0;
        for (let r = 0; r < this.n; r++)
            if (this.keys[r] !== null)
                live[kept++] = r;
        if (this.ivf !== null)
            this.ivf.compact(live);
        if (this.hnsw !== null) {
            const remap = new Int32Array(this.n).fill(-1);
            for (let w = 0; w < kept; w++)
                remap[live[w]] = w;
            this.hnsw = this.hnsw.compacted(remap, kept);
        }
        for (let w = 0; w < kept; w++) {
            const r = live[w];
            if (r === w)
                continue;
            const src = this.chunks[r >>> CHUNK_SHIFT];
            const dst = this.chunks[w >>> CHUNK_SHIFT];
            const so = (r & CHUNK_MASK) * dim;
            const d = (w & CHUNK_MASK) * dim;
            if (src === dst)
                src.copyWithin(d, so, so + dim);
            else
                dst.set(src.subarray(so, so + dim), d);
            this.norms[w] = this.norms[r];
            this.codes[w] = this.codes[r];
            this.modelIx[w] = this.modelIx[r];
            const key = this.keys[r];
            this.keys[w] = key;
            this.refs[w] = this.refs[r];
            this.texts[w] = this.texts[r];
            this.index.set(key, w);
        }
        this.keys.length = kept;
        this.refs.length = kept;
        this.texts.length = kept;
        this.chunks.length = Math.ceil(kept / CHUNK_ROWS);
        this.n = kept;
        this.dead = 0;
    }
    /** The vector of `slot` as a fresh array of numbers (the float32 values, widened). */
    vectorAt(slot) {
        const dim = this.dim;
        const chunk = this.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        const out = new Array(dim);
        for (let j = 0; j < dim; j++)
            out[j] = chunk[base + j];
        return out;
    }
    /** Bytes of the vectors and the numeric columns (not the strings). */
    bytes() {
        let total = this.norms.byteLength + this.codes.byteLength + this.modelIx.byteLength;
        for (const chunk of this.chunks)
            total += chunk.byteLength;
        return total;
    }
}
/**
 * Inverted-file ("IVF-flat") index over a {@link VectorTable}.
 *
 * Spherical k-means partitions the rows into `nlist` cells by direction, which
 * is what cosine similarity compares. A query scores the `nprobe` cells whose
 * centroids are nearest and only the rows inside them, with the same cosine
 * formula as the exact scan. The vectors are not copied again: the index is a
 * cell number per row (`assign`) plus the rows grouped by cell
 * (`order`/`offsets`), about 8 bytes per vector plus the centroids.
 *
 * It follows its table instead of being rebuilt on every write:
 *  - a **new** or **updated** row is *pending* (cell -1): every query scores
 *    the pending rows exactly, so a write is visible at once and costs the
 *    index nothing;
 *  - pending rows are folded into their nearest cell in one pass once there
 *    are enough of them (see `EmbeddingStore._annIndex`);
 *  - a **delete** clears the row's cell; stale entries in `order` are skipped
 *    at query time;
 *  - the table's compaction renumbers rows, and `compact` carries the cells
 *    across without retraining;
 *  - `churn` counts writes and deletes since training; the store retrains when
 *    it passes a fraction of the trained size.
 */
class IvfIndex {
    constructor(centroids, nlist, dim, assign) {
        this.order = new Int32Array(0);
        this.offsets = new Int32Array(0);
        this.pending = new Set();
        this.churn = 0;
        this.trainedN = 0;
        this.buildSeconds = 0;
        // Scratch kept between queries: a search allocates nothing proportional to the store.
        this._chosen = new Int32Array(0);
        this._chosenScore = new Float64Array(0);
        this.candRows = new Int32Array(0);
        this.candScores = new Float64Array(0);
        this.nlist = nlist;
        this.dim = dim;
        this.centroids = centroids;
        this.assign = assign;
        this._search = new CentroidSearch(centroids, nlist, dim);
    }
    /** Train centroids on a sample, then give every live row its cell. */
    static build(cache, nlistWanted, seed) {
        const started = nowMs();
        const { n, dim, chunks, norms, keys } = cache;
        const live = new Int32Array(n - cache.dead);
        let total = 0;
        for (let r = 0; r < n; r++)
            if (keys[r] !== null)
                live[total++] = r;
        const nlist = Math.max(1, Math.min(Math.floor(nlistWanted), total));
        const rng = new Rng(seed);
        // Sample up to 40 vectors per cell (a partial Fisher-Yates), as unit rows.
        const sampleN = Math.min(total, ANN_SAMPLE_PER_CELL * nlist);
        let sample;
        if (sampleN === total) {
            sample = live;
        }
        else {
            const pool = live.slice();
            for (let i = 0; i < sampleN; i++) {
                const j = i + rng.below(total - i);
                const t = pool[i];
                pool[i] = pool[j];
                pool[j] = t;
            }
            sample = pool.subarray(0, sampleN);
            sample.sort();
        }
        const x = new Float32Array(sampleN * dim);
        for (let i = 0; i < sampleN; i++) {
            const r = sample[i];
            const inv = norms[r] > 0 ? 1 / norms[r] : 0;
            const chunk = chunks[r >>> CHUNK_SHIFT];
            const src = (r & CHUNK_MASK) * dim;
            const dst = i * dim;
            for (let j = 0; j < dim; j++)
                x[dst + j] = chunk[src + j] * inv;
        }
        // Initial centroids: distinct sample points.
        const centroids = new Float32Array(nlist * dim);
        const picks = new Int32Array(sampleN);
        for (let i = 0; i < sampleN; i++)
            picks[i] = i;
        for (let c = 0; c < nlist; c++) {
            const j = c + rng.below(sampleN - c);
            const t = picks[c];
            picks[c] = picks[j];
            picks[j] = t;
            centroids.set(x.subarray(picks[c] * dim, (picks[c] + 1) * dim), c * dim);
        }
        // Spherical k-means: assign to the most similar centroid, re-centre on the sum, re-normalise.
        const cell = new Int32Array(sampleN);
        const counts = new Int32Array(nlist);
        const sums = new Float64Array(nlist * dim);
        for (let iter = 0; iter < ANN_KMEANS_ITERS; iter++) {
            const search = new CentroidSearch(centroids, nlist, dim);
            counts.fill(0);
            sums.fill(0);
            for (let i = 0; i < sampleN; i++) {
                const c = search.nearest(x, i * dim);
                cell[i] = c;
                counts[c]++;
                const base = c * dim;
                const src = i * dim;
                for (let j = 0; j < dim; j++)
                    sums[base + j] += x[src + j];
            }
            for (let c = 0; c < nlist; c++) {
                let len = 0;
                if (counts[c] > 0) {
                    for (let j = 0; j < dim; j++)
                        len += sums[c * dim + j] * sums[c * dim + j];
                    len = Math.sqrt(len);
                }
                if (counts[c] === 0 || len === 0) { // a dead centroid: re-seed on a random point
                    const i = rng.below(sampleN);
                    centroids.set(x.subarray(i * dim, (i + 1) * dim), c * dim);
                }
                else {
                    for (let j = 0; j < dim; j++)
                        centroids[c * dim + j] = sums[c * dim + j] / len;
                }
            }
        }
        const index = new IvfIndex(centroids, nlist, dim, new Int32Array(cache.capacity).fill(-1));
        index.trainedN = total;
        for (let t = 0; t < total; t++) {
            const r = live[t];
            index.assign[r] = index._search.nearest(chunks[r >>> CHUNK_SHIFT], (r & CHUNK_MASK) * dim);
        }
        index.relist(n);
        index.buildSeconds = (nowMs() - started) / 1000;
        return index;
    }
    /** Regroup the rows by cell (a stable counting sort of `assign`). */
    relist(n) {
        const assign = this.assign;
        const offsets = new Int32Array(this.nlist + 1);
        let total = 0;
        for (let r = 0; r < n; r++) {
            const c = assign[r];
            if (c >= 0) {
                offsets[c + 1]++;
                total++;
            }
        }
        for (let c = 0; c < this.nlist; c++)
            offsets[c + 1] += offsets[c];
        const fill = offsets.slice(0, this.nlist);
        const order = new Int32Array(total);
        for (let r = 0; r < n; r++) {
            const c = assign[r];
            if (c >= 0)
                order[fill[c]++] = r;
        }
        this.order = order;
        this.offsets = offsets;
    }
    /** Give every pending row its nearest cell and regroup. */
    fold(cache) {
        if (this.pending.size > 0) {
            const chunks = cache.chunks;
            for (const r of this.pending) {
                this.assign[r] = this._search.nearest(chunks[r >>> CHUNK_SHIFT], (r & CHUNK_MASK) * this.dim);
            }
            this.pending.clear();
        }
        this.relist(cache.n);
    }
    notePut(i, capacity) {
        if (i >= this.assign.length) {
            const grown = new Int32Array(Math.max(capacity, i + 1)).fill(-1);
            grown.set(this.assign);
            this.assign = grown;
        }
        this.assign[i] = -1;
        this.pending.add(i);
        this.churn++;
    }
    noteDrop(i) {
        if (i < this.assign.length)
            this.assign[i] = -1;
        this.pending.delete(i);
        this.churn++;
    }
    /** Follow the table's compaction (`live` = the old rows kept, in order). */
    compact(live) {
        const kept = live.length;
        const moved = new Int32Array(kept);
        for (let w = 0; w < kept; w++)
            moved[w] = this.assign[live[w]];
        this.assign.fill(-1);
        this.assign.set(moved);
        if (this.pending.size > 0) {
            const old = Array.from(this.pending).sort((a, b) => a - b);
            this.pending = new Set(old.map(r => lowerBound(live, r)));
        }
        this.relist(kept);
    }
    bytes() {
        return this.centroids.byteLength + this.assign.byteLength + this.order.byteLength + this.offsets.byteLength;
    }
    /**
     * Score the rows of the `lists` cells nearest `q`, plus every pending row,
     * into `candRows`/`candScores`; returns how many were kept. `code` < 0 means
     * any node type; rows scoring under `floor` are left out.
     */
    gather(cache, q, qnorm, code, floor, lists) {
        const { nlist, dim, centroids } = this;
        if (this._chosen.length < lists) {
            this._chosen = new Int32Array(lists);
            this._chosenScore = new Float64Array(lists);
        }
        const chosen = this._chosen;
        let nChosen = 0;
        if (lists >= nlist) {
            for (let c = 0; c < nlist; c++)
                chosen[c] = c;
            nChosen = nlist;
        }
        else {
            const chosenScore = this._chosenScore;
            for (let c = 0; c < nlist; c++) {
                const s = dot2(centroids, c * dim, q, 0, dim);
                if (nChosen === lists && s <= chosenScore[lists - 1])
                    continue;
                let i = nChosen < lists ? nChosen++ : lists - 1;
                while (i > 0 && chosenScore[i - 1] < s) {
                    chosenScore[i] = chosenScore[i - 1];
                    chosen[i] = chosen[i - 1];
                    i--;
                }
                chosenScore[i] = s;
                chosen[i] = c;
            }
        }
        const order = this.order;
        const offsets = this.offsets;
        let bound = this.pending.size;
        for (let k = 0; k < nChosen; k++)
            bound += offsets[chosen[k] + 1] - offsets[chosen[k]];
        if (this.candRows.length < bound) {
            const size = Math.max(bound, this.candRows.length * 2, 256);
            this.candRows = new Int32Array(size);
            this.candScores = new Float64Array(size);
        }
        const candRows = this.candRows;
        const candScores = this.candScores;
        const { chunks, norms, codes } = cache;
        const assign = this.assign;
        let m = 0;
        for (let k = 0; k < nChosen; k++) {
            const c = chosen[k];
            for (let p = offsets[c], end = offsets[c + 1]; p < end; p++) {
                const r = order[p];
                if (assign[r] < 0)
                    continue; // deleted or updated since the last regroup
                if (code >= 0 && codes[r] !== code)
                    continue;
                const denom = norms[r] * qnorm;
                const s = denom > 0 ? dot2(chunks[r >>> CHUNK_SHIFT], (r & CHUNK_MASK) * dim, q, 0, dim) / denom : 0;
                if (s < floor)
                    continue;
                candRows[m] = r;
                candScores[m] = s;
                m++;
            }
        }
        for (const r of this.pending) {
            if (code >= 0 && codes[r] !== code)
                continue;
            const denom = norms[r] * qnorm;
            const s = denom > 0 ? dot2(chunks[r >>> CHUNK_SHIFT], (r & CHUNK_MASK) * dim, q, 0, dim) / denom : 0;
            if (s < floor)
                continue;
            candRows[m] = r;
            candScores[m] = s;
            m++;
        }
        return m;
    }
    /** The `k` best of the first `m` candidates: score descending, row ascending on ties. */
    best(m, k) {
        const rows = this.candRows;
        const scores = this.candScores;
        const size = Math.min(k, m);
        // Min-heap of candidate slots whose root is the worst kept one.
        const heap = new Int32Array(size);
        const worse = (a, b) => scores[a] < scores[b] || (scores[a] === scores[b] && rows[a] > rows[b]);
        let count = 0;
        for (let i = 0; i < m; i++) {
            if (count === size) {
                const root = heap[0];
                if (!(scores[i] > scores[root] || (scores[i] === scores[root] && rows[i] < rows[root])))
                    continue;
                heap[0] = i;
                let at = 0;
                for (;;) {
                    const l = 2 * at + 1;
                    if (l >= size)
                        break;
                    const r = l + 1;
                    const child = r < size && worse(heap[r], heap[l]) ? r : l;
                    if (!worse(heap[child], heap[at]))
                        break;
                    const t = heap[at];
                    heap[at] = heap[child];
                    heap[child] = t;
                    at = child;
                }
            }
            else {
                let at = count++;
                heap[at] = i;
                while (at > 0) {
                    const parent = (at - 1) >> 1;
                    if (!worse(heap[at], heap[parent]))
                        break;
                    const t = heap[at];
                    heap[at] = heap[parent];
                    heap[parent] = t;
                    at = parent;
                }
            }
        }
        // Drain worst-first into the back of the output.
        const out = new Int32Array(count);
        for (let o = count - 1; o >= 0; o--) {
            out[o] = heap[0];
            const last = heap[o];
            if (o === 0)
                break;
            heap[0] = last;
            let at = 0;
            for (;;) {
                const l = 2 * at + 1;
                if (l >= o)
                    break;
                const r = l + 1;
                const child = r < o && worse(heap[r], heap[l]) ? r : l;
                if (!worse(heap[child], heap[at]))
                    break;
                const t = heap[at];
                heap[at] = heap[child];
                heap[child] = t;
                at = child;
            }
        }
        return out;
    }
}
/** Index of the first element of sorted `values` that is >= `target`. */
function lowerBound(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (values[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
/**
 * Dot products of one query with four rows, written to `out[0..3]`. Each sum
 * runs over the dimensions in order, exactly as a loop over one row would, so
 * the results are bit-identical to four separate loops; only the interleaving
 * differs.
 */
function dotRows4(q, a, ao, b, bo, c, co, d, dof, dim, out) {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let j = 0; j < dim; j++) {
        const x = q[j];
        s0 += x * a[ao + j];
        s1 += x * b[bo + j];
        s2 += x * c[co + j];
        s3 += x * d[dof + j];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}
// =============================================================================
// HNSW graph index (opt-in, beside IVF)
//
// A port of the Rust crate's `src/hnsw.rs` (DESIGN.md section 12, the portable
// rules): a layered neighbour graph over the table's slots. It holds links only
// - every score is computed from the one stored copy of each vector with the
// same dot product, in the same order, and the same cosine formula as the
// exact scan, so a hit's score is the exact scan's number for that id, and
// ties order by slot. This port is single-threaded: there is no pool and no
// parallel planning, so the graph is built one batch at a time on the calling
// thread (the graph does not depend on threads in any port: plans read a
// frozen graph and are applied in a fixed order).
// =============================================================================
/** Largest `m`: layer 0 holds `2 * m` links, so a link block is at most 128 ints. */
const HNSW_MAX_M = 64;
const HNSW_DEFAULT_M = 16;
const HNSW_DEFAULT_EF_CONSTRUCTION = 200;
const HNSW_ABSENT = 255;
const HNSW_MAX_LEVEL = 24;
/** Largest batch planned against one frozen graph while building, and the batch's share of the nodes so far (size / 16). */
const HNSW_BUILD_BATCH = 512;
const HNSW_BUILD_RATIO = 16;
/** Updates + deletes since the build beyond this many times the node count call for a rebuild. */
const HNSW_REBUILD_FRACTION = 1;
const HNSW_MAX_CAP = 2 * HNSW_MAX_M;
/** Layer-0 link blocks live in chunks of this many nodes, so growth never copies them. */
const HNSW_CHUNK_SHIFT = 10;
const HNSW_CHUNK_ROWS = 1 << HNSW_CHUNK_SHIFT;
const HNSW_CHUNK_MASK = HNSW_CHUNK_ROWS - 1;
/** `seed XOR` this seeds the level generator (the Rust crate's constant), as two 32-bit halves. */
const HNSW_SEED_XOR_HI = 0x48575357;
const HNSW_SEED_XOR_LO = 0x48575357;
/** `ef_search` by store size when none is configured (DESIGN.md section 12.1). */
function hnswAutoEf(live) {
    if (live < 10000)
        return 64;
    if (live < 200000)
        return 96;
    if (live < 600000)
        return 128;
    return 192;
}
let mulHi = 0;
let mulLo = 0;
/** (ah:al) * (bh:bl) mod 2^64, 32-bit halves in and out (through `mulHi`/`mulLo`); no BigInt, no allocation. */
function mul64(ah, al, bh, bl) {
    const a0 = al & 0xffff;
    const a1 = al >>> 16;
    const b0 = bl & 0xffff;
    const b1 = bl >>> 16;
    const p00 = a0 * b0;
    const mid = a0 * b1 + a1 * b0 + (p00 >>> 16);
    mulLo = (((mid & 0xffff) << 16) | (p00 & 0xffff)) >>> 0;
    const hi = a1 * b1 + Math.floor(mid / 65536);
    mulHi = (hi + Math.imul(ah, bl) + Math.imul(al, bh)) >>> 0;
}
/**
 * SplitMix64, the generator of the level draw and of the cross-port test data:
 * `state += 0x9E3779B97F4A7C15`, then the usual mix; `below(n)` is the high 64
 * bits of `next * n`. Integers only (two 32-bit halves, no BigInt per draw, no
 * logarithm). Low level: exported for the tests and the cross-port checks.
 */
export class SplitMix64 {
    /** `seed` is taken modulo 2^64 (a negative number or bigint wraps, as an unsigned cast would). */
    constructor(seed = 0) {
        /** High and low 32 bits of the last value `next()` produced. */
        this.hi = 0;
        this.lo = 0;
        const s = BigInt.asUintN(64, typeof seed === 'bigint' ? seed : BigInt(Math.trunc(Number.isFinite(seed) ? seed : 0)));
        this._sh = Number(s >> BigInt(32));
        this._sl = Number(s & BigInt(0xffffffff));
    }
    /** Advance one step; the value is in `hi`/`lo`. */
    next() {
        const sl = this._sl + 0x7f4a7c15;
        this._sl = sl >>> 0;
        this._sh = (this._sh + 0x9e3779b9 + (sl > 0xffffffff ? 1 : 0)) >>> 0;
        let zh = this._sh;
        let zl = this._sl;
        // z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        let th = zh >>> 30;
        let tl = ((zh << 2) | (zl >>> 30)) >>> 0;
        mul64((zh ^ th) >>> 0, (zl ^ tl) >>> 0, 0xbf58476d, 0x1ce4e5b9);
        zh = mulHi;
        zl = mulLo;
        // z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        th = zh >>> 27;
        tl = ((zh << 5) | (zl >>> 27)) >>> 0;
        mul64((zh ^ th) >>> 0, (zl ^ tl) >>> 0, 0x94d049bb, 0x133111eb);
        zh = mulHi;
        zl = mulLo;
        // z ^ (z >> 31)
        this.hi = (zh ^ (zh >>> 31)) >>> 0;
        this.lo = (zl ^ (((zh << 1) | (zl >>> 31)) >>> 0)) >>> 0;
    }
    /** The last value as a bigint (for tests; the hot paths read `hi`/`lo`). */
    value() {
        return (BigInt(this.hi) << BigInt(32)) | BigInt(this.lo);
    }
    /** Uniform in [0, n): the high 64 bits of `next * n`. */
    below(n) {
        this.next();
        if (n < 2097152) { // n < 2^21: both partial products stay exact in a double
            const carry = Math.floor((this.lo * n) / 4294967296);
            return Math.floor((this.hi * n + carry) / 4294967296);
        }
        return Number((this.value() * BigInt(n)) >> BigInt(64));
    }
}
/** The 64-bit state of a store seed's level generator: `seed XOR 0x4857535748575357` (a negative seed sign-extends). */
function hnswRngFor(seed) {
    const base = BigInt.asUintN(64, BigInt(Math.trunc(Number.isFinite(seed) ? seed : 0)));
    const mask = (BigInt(HNSW_SEED_XOR_HI) << BigInt(32)) | BigInt(HNSW_SEED_XOR_LO);
    return new SplitMix64(base ^ mask);
}
/** The first `count` levels the generator draws for a store seed and `m` (`level = 0; while level < 24 and rng.below(m) == 0: level++`). */
export function hnswLevelSequence(seed, m, count) {
    const rng = hnswRngFor(seed);
    const out = [];
    for (let i = 0; i < count; i++) {
        let level = 0;
        while (level < HNSW_MAX_LEVEL && rng.below(m) === 0)
            level++;
        out.push(level);
    }
    return out;
}
/**
 * Dot products of one float64 query with four rows, into `out[0..3]`. Each sum
 * runs over the dimensions in order from 0, exactly as `dotRows4` and the
 * one-row loop do, so a score equals the exact scan's; the query is always a
 * Float64Array here, which keeps this function monomorphic (`dotRows4` takes a
 * number[] too).
 */
function dotRows4F64(q, a, ao, b, bo, c, co, d, dof, dim, out) {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let j = 0; j < dim; j++) {
        const x = q[j];
        s0 += x * a[ao + j];
        s1 += x * b[bo + j];
        s2 += x * c[co + j];
        s3 += x * d[dof + j];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}
/** Cosine from a dot product and the two norms, as the exact scan computes it (0 when either norm is 0); NaN ranks last. */
function hnswCosine(dot, qn, rn) {
    if (qn === 0 || rn === 0)
        return 0;
    const s = dot / (qn * rn);
    return s !== s ? -Infinity : s;
}
/** A binary heap of (score, slot) pairs in typed arrays, the best on top (a max-heap) or the worst on top. "Better": higher score, then lower slot. */
class HnswHeap {
    constructor(bestOnTop, capacity = 256) {
        this.n = 0;
        /** The pair `pop()` removed last. */
        this.topS = 0;
        this.topI = 0;
        this._best = bestOnTop;
        this.s = new Float64Array(capacity);
        this.id = new Int32Array(capacity);
    }
    clear() {
        this.n = 0;
    }
    _grow() {
        const s = new Float64Array(this.s.length * 2);
        s.set(this.s);
        const id = new Int32Array(this.id.length * 2);
        id.set(this.id);
        this.s = s;
        this.id = id;
    }
    /** True when (sa, ia) belongs above (sb, ib) in this heap. */
    _above(sa, ia, sb, ib) {
        return this._best ? sa > sb || (sa === sb && ia < ib) : sb > sa || (sb === sa && ib < ia);
    }
    push(score, slot) {
        if (this.n === this.s.length)
            this._grow();
        const s = this.s;
        const id = this.id;
        let i = this.n++;
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (!this._above(score, slot, s[p], id[p]))
                break;
            s[i] = s[p];
            id[i] = id[p];
            i = p;
        }
        s[i] = score;
        id[i] = slot;
    }
    /** Remove the top into `topS`/`topI`. */
    pop() {
        const s = this.s;
        const id = this.id;
        this.topS = s[0];
        this.topI = id[0];
        const last = --this.n;
        if (last > 0) {
            const ls = s[last];
            const li = id[last];
            let i = 0;
            for (;;) {
                const l = 2 * i + 1;
                if (l >= last)
                    break;
                let c = l;
                const r = l + 1;
                if (r < last && this._above(s[r], id[r], s[l], id[l]))
                    c = r;
                if (!this._above(s[c], id[c], ls, li))
                    break;
                s[i] = s[c];
                id[i] = id[c];
                i = c;
            }
            s[i] = ls;
            id[i] = li;
        }
    }
}
/** Visited marks, 1 bit per slot, cleared by the words a query touched, so a query pays for what it visited. */
class HnswVisited {
    constructor() {
        this._bits = new Int32Array(64);
        this._words = new Int32Array(256);
        this._count = 0;
    }
    begin(slots) {
        this.clear();
        const need = (slots + 31) >>> 5;
        if (this._bits.length < need)
            this._bits = new Int32Array(Math.max(need, this._bits.length * 2));
    }
    clear() {
        const bits = this._bits;
        const words = this._words;
        for (let i = 0; i < this._count; i++)
            bits[words[i]] = 0;
        this._count = 0;
    }
    /** Mark `slot`; true if it already was. */
    testSet(slot) {
        const w = slot >>> 5;
        const m = 1 << (slot & 31);
        const cur = this._bits[w];
        if ((cur & m) !== 0)
            return true;
        if (cur === 0) {
            if (this._count === this._words.length) {
                const grown = new Int32Array(this._count * 2);
                grown.set(this._words);
                this._words = grown;
            }
            this._words[this._count++] = w;
        }
        this._bits[w] = cur | m;
        return false;
    }
}
/**
 * The graph over a {@link VectorTable}: per table slot a level (255 = not in the
 * graph), a block of `2m + 1` ints on layer 0 (link count, then links) in chunks
 * of 1024 nodes, and for a node of level L, L blocks of `m + 1` ints in one
 * arena for layers 1..L. About 146 bytes a node at m = 16 and no copy of any
 * vector.
 *
 * Follows its table instead of being rebuilt on every write (section 12.5): a
 * write links the row at once, a delete only tombstones (the table's NaN norm
 * hides the row; the node and its links stay so routes keep working), the
 * table's compaction renumbers the graph and bridges over the tombstones, and
 * updates + deletes beyond the node count mark it stale so the next search
 * rebuilds it. Never persisted.
 *
 * Single-threaded: all scratch (visited bitmap, both heaps, score and result
 * buffers) is kept on the instance and reused, so a query allocates nothing
 * proportional to the store.
 */
class HnswIndex {
    constructor(m, efConstruction, seed, dim) {
        /** Per slot: highest layer, or 255. */
        this.levels = new Uint8Array(0);
        /** Per slot: start of its upper blocks in `upper`, or -1. */
        this.upperAt = new Int32Array(0);
        /** Slots covered; at least the table's, more while trailing tombstones the table popped still have nodes. */
        this.slots = 0;
        this.l0 = [];
        this.upper = new Int32Array(0);
        this.upperLen = 0;
        this.entry = -1;
        this.top = 0;
        /** Nodes in the graph, tombstones included. */
        this.members = 0;
        this.churn = 0;
        this.buildSeconds = 0;
        this.inserts = 0;
        this.insertSeconds = 0;
        /** The last search stopped after scoring more nodes than its budget. */
        this.budgetHit = false;
        /** The best-first result of the last layer search: `count` pairs in `outS`/`outI`. */
        this.outS = new Float64Array(256);
        this.outI = new Int32Array(256);
        this._visited = new HnswVisited();
        this._cand = new HnswHeap(true);
        this._kept = new HnswHeap(false);
        this._sort = new HnswHeap(true);
        this._fresh = new Int32Array(HNSW_MAX_CAP);
        this._scores = new Float64Array(HNSW_MAX_CAP);
        this._d4 = new Float64Array(4);
        this._epS = new Float64Array(256);
        this._epI = new Int32Array(256);
        this._candS = new Float64Array(256);
        this._candI = new Int32Array(256);
        this._selS = new Float64Array(256);
        this._selI = new Int32Array(256);
        this._memI = new Int32Array(64);
        this._memS = new Float64Array(64);
        this._curS = 0;
        this._curI = 0;
        this._one = {
            slots: new Int32Array(1),
            levels: new Int32Array(1),
            relink: new Uint8Array(1),
        };
        this.m = m;
        this.m0 = 2 * m;
        this.efC = Math.max(efConstruction, m);
        this._rng = hnswRngFor(seed);
        this._dim = dim;
        this._qSearch = new Float64Array(dim);
        this._qPlan = new Float64Array(dim);
        this._qSel = new Float64Array(dim);
    }
    get stale() {
        return this.churn > HNSW_REBUILD_FRACTION * Math.max(1, this.members);
    }
    /** Memory of the graph itself: levels, upper-layer offsets, the arena and the layer-0 chunks. */
    bytes() {
        let total = this.levels.byteLength + this.upperAt.byteLength + this.upper.byteLength;
        for (const chunk of this.l0)
            total += chunk.byteLength;
        return total;
    }
    _cap(level) {
        return level === 0 ? this.m0 : this.m;
    }
    _drawLevel() {
        let level = 0;
        while (level < HNSW_MAX_LEVEL && this._rng.below(this.m) === 0)
            level++;
        return level;
    }
    // -- storage -----------------------------------------------------------------
    /** Cover slot `s`: new slots start absent, and the layer-0 chunk holding it exists. */
    _ensureSlot(s) {
        if (s >= this.slots) {
            if (s >= this.levels.length) {
                const cap = Math.max(this.levels.length * 2, s + 1, 16);
                const levels = new Uint8Array(cap);
                levels.set(this.levels);
                this.levels = levels;
                const upperAt = new Int32Array(cap);
                upperAt.set(this.upperAt);
                this.upperAt = upperAt;
            }
            for (let i = this.slots; i <= s; i++) {
                this.levels[i] = HNSW_ABSENT;
                this.upperAt[i] = -1;
            }
            this.slots = s + 1;
        }
        while (this.l0.length <= s >>> HNSW_CHUNK_SHIFT)
            this.l0.push(new Int32Array(HNSW_CHUNK_ROWS * (this.m0 + 1)));
    }
    /** The array holding the link block of `s` on `level`, and where in it the block starts (count, then links). */
    _arr(s, level) {
        return level === 0 ? this.l0[s >>> HNSW_CHUNK_SHIFT] : this.upper;
    }
    _off(s, level) {
        return level === 0 ? (s & HNSW_CHUNK_MASK) * (this.m0 + 1) : this.upperAt[s] + (level - 1) * (this.m + 1);
    }
    /** Replace the links of `s` on `level` with `items[0..count)`. */
    _setList(s, level, items, count) {
        const block = this._arr(s, level);
        const at = this._off(s, level);
        block[at] = count;
        for (let i = 0; i < count; i++)
            block[at + 1 + i] = items[i];
    }
    /** A copy of the links of `s` on `level`. */
    linksOf(s, level) {
        const block = this._arr(s, level);
        const at = this._off(s, level);
        return block.slice(at + 1, at + 1 + block[at]);
    }
    _addNode(s, level) {
        this._ensureSlot(s);
        this.levels[s] = level;
        if (level >= 1) {
            const need = level * (this.m + 1);
            if (this.upperLen + need > this.upper.length) {
                const grown = new Int32Array(Math.max(this.upper.length * 2, this.upperLen + need, 256));
                grown.set(this.upper);
                this.upper = grown;
            }
            this.upperAt[s] = this.upperLen;
            this.upperLen += need;
        }
        this.members++;
    }
    inGraph(s) {
        return s < this.slots && this.levels[s] !== HNSW_ABSENT;
    }
    levelOf(s) {
        return s < this.slots && this.levels[s] !== HNSW_ABSENT ? this.levels[s] : -1;
    }
    // -- scores ------------------------------------------------------------------------
    /** The norm of a tombstone's still-present floats (the table marks its own norm NaN), summed as `VectorTable.put` sums it. */
    _floatNorm(t, slot) {
        const dim = t.dim;
        const chunk = t.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        let sum = 0;
        for (let j = 0; j < dim; j++) {
            const f = chunk[base + j];
            sum += f * f;
        }
        return Math.sqrt(sum);
    }
    _rowNorm(t, slot) {
        const n = t.norms[slot];
        return n !== n ? this._floatNorm(t, slot) : n;
    }
    /** Copy the floats of `slot` into `into` (widened), so a stored row can be the query. */
    _loadRow(t, slot, into) {
        const dim = t.dim;
        const chunk = t.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        for (let j = 0; j < dim; j++)
            into[j] = chunk[base + j];
    }
    /**
     * Cosine of the query (`q`, norm `qn`) with each of `slots[start .. start+n)`,
     * four rows at a time and then one, into `out[0..n)`. Every sum runs in
     * dimension order, so a live row scores exactly as the exact scan scores it.
     * A tombstone is scored from its floats; anything that stays NaN ranks last.
     */
    _scoreInto(t, q, qn, slots, start, n, out) {
        const dim = t.dim;
        const chunks = t.chunks;
        const norms = t.norms;
        const d4 = this._d4;
        const end = start + n;
        let p = start;
        let o = 0;
        for (; p + 4 <= end; p += 4, o += 4) {
            const s0 = slots[p];
            const s1 = slots[p + 1];
            const s2 = slots[p + 2];
            const s3 = slots[p + 3];
            dotRows4F64(q, chunks[s0 >>> CHUNK_SHIFT], (s0 & CHUNK_MASK) * dim, chunks[s1 >>> CHUNK_SHIFT], (s1 & CHUNK_MASK) * dim, chunks[s2 >>> CHUNK_SHIFT], (s2 & CHUNK_MASK) * dim, chunks[s3 >>> CHUNK_SHIFT], (s3 & CHUNK_MASK) * dim, dim, d4);
            let r = norms[s0];
            out[o] = hnswCosine(d4[0], qn, r !== r ? this._floatNorm(t, s0) : r);
            r = norms[s1];
            out[o + 1] = hnswCosine(d4[1], qn, r !== r ? this._floatNorm(t, s1) : r);
            r = norms[s2];
            out[o + 2] = hnswCosine(d4[2], qn, r !== r ? this._floatNorm(t, s2) : r);
            r = norms[s3];
            out[o + 3] = hnswCosine(d4[3], qn, r !== r ? this._floatNorm(t, s3) : r);
        }
        for (; p < end; p++, o++) {
            const s = slots[p];
            const chunk = chunks[s >>> CHUNK_SHIFT];
            const base = (s & CHUNK_MASK) * dim;
            let dot = 0;
            for (let j = 0; j < dim; j++)
                dot += q[j] * chunk[base + j];
            const r = norms[s];
            out[o] = hnswCosine(dot, qn, r !== r ? this._floatNorm(t, s) : r);
        }
    }
    /** Score one slot into `_curS`/`_curI`. */
    _scoreOne(t, q, qn, slot) {
        const one = this._fresh;
        const out = this._scores;
        one[0] = slot;
        this._scoreInto(t, q, qn, one, 0, 1, out);
        this._curS = out[0];
        this._curI = slot;
    }
    // -- search ------------------------------------------------------------------------
    _ensureOut(n) {
        if (this.outS.length < n) {
            const size = Math.max(n, this.outS.length * 2);
            this.outS = new Float64Array(size);
            this.outI = new Int32Array(size);
        }
    }
    /**
     * Walk to the best node reachable by single steps on `level`, from
     * (`_curS`, `_curI`); the answer is left there. "Better": higher score, then
     * lower slot.
     */
    _greedy(t, q, qn, level) {
        const scores = this._scores;
        for (;;) {
            const cur = this._curI;
            const block = this._arr(cur, level);
            const at = this._off(cur, level);
            const count = block[at];
            if (count === 0)
                return;
            this._scoreInto(t, q, qn, block, at + 1, count, scores);
            let bestS = this._curS;
            let bestI = cur;
            let moved = false;
            for (let i = 0; i < count; i++) {
                const s = scores[i];
                const id = block[at + 1 + i];
                if (s > bestS || (s === bestS && id < bestI)) {
                    bestS = s;
                    bestI = id;
                    moved = true;
                }
            }
            if (!moved)
                return;
            this._curS = bestS;
            this._curI = bestI;
        }
    }
    /**
     * Best-first search of one layer from `nEps` entry points (`eps*`), keeping
     * the `ef` best admitted nodes into `outS`/`outI`, best first; returns how
     * many. Tombstones and non-admitted nodes (a node is admitted when its row is
     * live, its type code matches `code` (-1 = any), its score is at least
     * `threshold` and it is not `exclude`) are routed through but never kept.
     * Stops when the best unexpanded node is worse than the worst kept one (once
     * `ef` are kept), or, setting `budgetHit`, after `budget` nodes were scored.
     */
    _searchLayer(t, q, qn, epS, epI, nEps, ef, level, code, threshold, exclude, budget) {
        const visited = this._visited;
        visited.clear();
        const cand = this._cand;
        const kept = this._kept;
        cand.clear();
        kept.clear();
        const norms = t.norms;
        const codes = t.codes;
        for (let e = 0; e < nEps; e++) {
            const slot = epI[e];
            if (visited.testSet(slot))
                continue;
            const score = epS[e];
            cand.push(score, slot);
            if (slot !== exclude && norms[slot] === norms[slot] && score >= threshold && (code < 0 || codes[slot] === code)) {
                kept.push(score, slot);
                if (kept.n > ef)
                    kept.pop();
            }
        }
        const fresh = this._fresh;
        const scores = this._scores;
        let scored = 0;
        this.budgetHit = false;
        while (cand.n > 0) {
            cand.pop();
            const cs = cand.topS;
            const ci = cand.topI;
            if (kept.n >= ef) { // stop when the candidate is worse than the worst kept
                const ws = kept.s[0];
                if (ws > cs || (ws === cs && kept.id[0] < ci))
                    break;
            }
            const block = this._arr(ci, level);
            const at = this._off(ci, level);
            const count = block[at];
            let nf = 0;
            for (let i = 1; i <= count; i++) {
                const x = block[at + i];
                if (!visited.testSet(x))
                    fresh[nf++] = x;
            }
            if (nf === 0)
                continue;
            this._scoreInto(t, q, qn, fresh, 0, nf, scores);
            scored += nf;
            for (let i = 0; i < nf; i++) {
                const score = scores[i];
                const slot = fresh[i];
                if (kept.n < ef || score > kept.s[0] || (score === kept.s[0] && slot < kept.id[0])) {
                    cand.push(score, slot);
                    if (slot !== exclude && norms[slot] === norms[slot] && score >= threshold && (code < 0 || codes[slot] === code)) {
                        kept.push(score, slot);
                        if (kept.n > ef)
                            kept.pop();
                    }
                }
            }
            if (scored > budget) {
                this.budgetHit = true;
                break;
            }
        }
        const n = kept.n;
        this._ensureOut(n);
        const outS = this.outS;
        const outI = this.outI;
        for (let i = n - 1; i >= 0; i--) { // worst first, so fill from the back
            kept.pop();
            outS[i] = kept.topS;
            outI[i] = kept.topI;
        }
        return n;
    }
    /**
     * The `ef` best admitted nodes for a query, best first, into `outS`/`outI`;
     * returns how many (`budgetHit` says the search gave up scoring too many).
     * `query` holds the raw numbers the exact scan would use; `qn` is their norm.
     */
    search(t, query, qn, code, threshold, ef, budget) {
        this.budgetHit = false;
        if (this.entry === -1)
            return 0;
        const q = this._qSearch;
        for (let j = 0; j < this._dim; j++)
            q[j] = query[j];
        this._visited.begin(this.slots);
        try {
            this._scoreOne(t, q, qn, this.entry);
            for (let level = this.top; level >= 1; level--)
                this._greedy(t, q, qn, level);
            const epS = this._epS;
            const epI = this._epI;
            epS[0] = this._curS;
            epI[0] = this._curI;
            return this._searchLayer(t, q, qn, epS, epI, 1, ef, 0, code, threshold, -1, budget);
        }
        finally {
            this._visited.clear();
        }
    }
    // -- linking -----------------------------------------------------------------------
    _grow(name, nameI, need) {
        if (this[name].length >= need)
            return;
        const size = Math.max(need, this[name].length * 2);
        const s = new Float64Array(size);
        s.set(this[name]);
        this[name] = s;
        const i = new Int32Array(size);
        i.set(this[nameI]);
        this[nameI] = i;
    }
    /**
     * The diversity heuristic: walk the `n` candidates (`S`/`I`, best first) and
     * keep one only if it scores no better with any node already kept than with
     * the base (`sim(c, kept) > sim(c, base)` rejects); stop at `cap`. With at
     * most `cap` candidates, keep all.
     */
    _select(t, S, I, n, cap) {
        if (n <= cap)
            return I.slice(0, n);
        const chosen = new Int32Array(cap);
        let count = 0;
        const scores = this._scores;
        const q = this._qSel;
        for (let i = 0; i < n; i++) {
            if (count >= cap)
                break;
            const c = I[i];
            let reject = false;
            if (count > 0) {
                this._loadRow(t, c, q);
                this._scoreInto(t, q, this._rowNorm(t, c), chosen, 0, count, scores);
                const cs = S[i];
                for (let j = 0; j < count; j++) {
                    if (scores[j] > cs) {
                        reject = true;
                        break;
                    }
                }
            }
            if (!reject)
                chosen[count++] = c;
        }
        return count === cap ? chosen : chosen.slice(0, count);
    }
    /** Sort `n` pairs (`S`/`I`) best first, in place, through the sorting heap. */
    _sortBest(S, I, n) {
        const h = this._sort;
        h.clear();
        for (let i = 0; i < n; i++)
            h.push(S[i], I[i]);
        for (let i = 0; i < n; i++) {
            h.pop();
            S[i] = h.topS;
            I[i] = h.topI;
        }
    }
    /**
     * The neighbours chosen for the job (`x`, `level`) on every layer, against the
     * graph as it is now plus the other new members of its batch (the intra-batch
     * slots with level >= the layer, scored by brute force).
     */
    _plan(t, x, level, intraSlots, intraLevels, intraCount) {
        const visited = this._visited;
        visited.begin(this.slots);
        const q = this._qPlan;
        this._loadRow(t, x, q);
        const qn = this._rowNorm(t, x);
        const top = this.top;
        const lists = new Array(level + 1);
        let nEps = 0;
        if (this.entry !== -1) {
            this._scoreOne(t, q, qn, this.entry);
            for (let l = top; l >= level + 1; l--)
                this._greedy(t, q, qn, l);
            this._epS[0] = this._curS;
            this._epI[0] = this._curI;
            nEps = 1;
        }
        for (let l = level; l >= 0; l--) {
            let nc = 0;
            if (l <= top && nEps > 0) {
                nc = this._searchLayer(t, q, qn, this._epS, this._epI, nEps, this.efC, l, -1, -Infinity, x, 0x7fffffff);
                this._grow('_candS', '_candI', nc);
                this._candS.set(this.outS.subarray(0, nc));
                this._candI.set(this.outI.subarray(0, nc));
                if (nc > 0) {
                    this._grow('_epS', '_epI', nc);
                    this._epS.set(this._candS.subarray(0, nc));
                    this._epI.set(this._candI.subarray(0, nc));
                    nEps = nc;
                }
            }
            let members = 0;
            for (let i = 0; i < intraCount; i++)
                if (intraSlots[i] !== x && intraLevels[i] >= l)
                    members++;
            if (members > 0) {
                this._grow('_memS', '_memI', members);
                const memI = this._memI;
                const memS = this._memS;
                for (let i = 0, j = 0; i < intraCount; i++)
                    if (intraSlots[i] !== x && intraLevels[i] >= l)
                        memI[j++] = intraSlots[i];
                this._scoreInto(t, q, qn, memI, 0, members, memS);
                this._grow('_candS', '_candI', nc + members);
                const candS = this._candS;
                const candI = this._candI;
                for (let i = 0; i < members; i++) {
                    candS[nc + i] = memS[i];
                    candI[nc + i] = memI[i];
                }
                nc += members;
                this._sortBest(candS, candI, nc);
            }
            lists[l] = this._select(t, this._candS, this._candI, nc, this.m);
        }
        return lists;
    }
    /**
     * The list of `target` on `layer` with `incoming` added, pruned to the
     * layer's capacity when it overflows (tombstones are dropped then, and the
     * rest re-pruned with the diversity heuristic). Null if nothing changes.
     */
    _grownList(t, layer, target, incoming) {
        const cap = this._cap(layer);
        const block = this._arr(target, layer);
        const at = this._off(target, layer);
        const len = block[at];
        const all = new Int32Array(len + incoming.length);
        for (let i = 0; i < len; i++)
            all[i] = block[at + 1 + i];
        let count = len;
        for (const nb of incoming) {
            if (nb === target)
                continue;
            let seen = false;
            for (let i = 0; i < count; i++) {
                if (all[i] === nb) {
                    seen = true;
                    break;
                }
            }
            if (!seen)
                all[count++] = nb;
        }
        if (count === len)
            return null;
        if (count <= cap)
            return all.slice(0, count);
        const norms = t.norms;
        let live = 0;
        for (let i = 0; i < count; i++) {
            const nm = norms[all[i]];
            if (nm === nm)
                all[live++] = all[i];
        }
        count = live;
        if (count <= cap)
            return all.slice(0, count);
        const scores = new Float64Array(count);
        this._loadRow(t, target, this._qSel);
        this._scoreInto(t, this._qSel, this._rowNorm(t, target), all, 0, count, scores);
        const S = new Float64Array(count);
        const I = new Int32Array(count);
        for (let i = 0; i < count; i++) {
            S[i] = scores[i];
            I[i] = all[i];
        }
        this._sortBest(S, I, count);
        return this._select(t, S, I, count, cap);
    }
    /**
     * Link `count` jobs (slot, level, relink) into the graph: plan every job
     * against the graph as it is (read-only), then apply the plans in job order.
     * One routine for build, insert and update.
     */
    _link(t, slots, levels, relink, count) {
        const intraSlots = new Int32Array(count);
        const intraLevels = new Int32Array(count);
        let intra = 0;
        for (let i = 0; i < count; i++) {
            if (relink[i] === 0) {
                intraSlots[intra] = slots[i];
                intraLevels[intra++] = levels[i];
            }
        }
        const plans = new Array(count);
        for (let i = 0; i < count; i++)
            plans[i] = this._plan(t, slots[i], levels[i], intraSlots, intraLevels, intra);
        this._visited.clear();
        // forward links
        for (let i = 0; i < count; i++) {
            if (relink[i] === 0)
                this._addNode(slots[i], levels[i]);
            const plan = plans[i];
            for (let l = 0; l < plan.length; l++)
                this._setList(slots[i], l, plan[l], plan[l].length);
        }
        // backward links, grouped by (layer, target) in a fixed order
        const norms = t.norms;
        const groups = new Map();
        for (let i = 0; i < count; i++) {
            const plan = plans[i];
            for (let l = 0; l < plan.length; l++) {
                const list = plan[l];
                for (let k = 0; k < list.length; k++) {
                    const nb = list[k];
                    if (norms[nb] !== norms[nb])
                        continue;
                    const key = l * 4294967296 + nb;
                    const group = groups.get(key);
                    if (group === undefined)
                        groups.set(key, [slots[i]]);
                    else
                        group.push(slots[i]);
                }
            }
        }
        const keys = Float64Array.from(groups.keys()).sort();
        for (let g = 0; g < keys.length; g++) {
            const key = keys[g];
            const layer = Math.floor(key / 4294967296);
            const target = key - layer * 4294967296;
            const grown = this._grownList(t, layer, target, groups.get(key));
            if (grown !== null)
                this._setList(target, layer, grown, grown.length);
        }
        // the entry point is the first new node of the highest level
        for (let i = 0; i < count; i++) {
            if (relink[i] !== 0)
                continue;
            if (this.entry === -1 || levels[i] > this.top) {
                this.entry = slots[i];
                this.top = levels[i];
            }
        }
    }
    /**
     * A graph over every live row of `t` with a finite norm, built in slot order
     * in batches of size/16 (1 at the start, at most 512), one `_link` call each.
     */
    static build(t, m, efConstruction, seed) {
        const started = nowMs();
        const h = new HnswIndex(m, efConstruction, seed, t.dim);
        const n = t.n;
        if (n > 0)
            h._ensureSlot(n - 1);
        const norms = t.norms;
        const order = new Int32Array(n);
        let live = 0;
        for (let s = 0; s < n; s++)
            if (Number.isFinite(norms[s]))
                order[live++] = s; // a tombstone's norm is NaN
        const slots = new Int32Array(HNSW_BUILD_BATCH);
        const levels = new Int32Array(HNSW_BUILD_BATCH);
        const relink = new Uint8Array(HNSW_BUILD_BATCH);
        let at = 0;
        while (at < live) {
            const size = Math.min(Math.max(Math.floor(at / HNSW_BUILD_RATIO), 1), HNSW_BUILD_BATCH);
            const end = Math.min(at + size, live);
            let c = 0;
            for (let i = at; i < end; i++) {
                slots[c] = order[i];
                levels[c++] = h._drawLevel();
            }
            h._link(t, slots, levels, relink, c);
            at = end;
        }
        h.buildSeconds = (nowMs() - started) / 1000;
        return h;
    }
    /**
     * A row was written (new, or its vector replaced): link it into the graph now.
     * An update keeps the node's level and relinks it; other nodes' links to its
     * old position stay (routing hints). Counts as churn.
     */
    notePut(t, slot) {
        const started = nowMs();
        this._ensureSlot(slot);
        const norm = t.norms[slot];
        if (!Number.isFinite(norm)) {
            if (this.inGraph(slot))
                this.churn++;
            return;
        }
        const one = this._one;
        one.slots[0] = slot;
        if (this.inGraph(slot)) {
            this.churn++;
            one.levels[0] = this.levels[slot];
            one.relink[0] = 1;
        }
        else {
            one.levels[0] = this._drawLevel();
            one.relink[0] = 0;
        }
        this._link(t, one.slots, one.levels, one.relink, 1);
        this.inserts++;
        this.insertSeconds += (nowMs() - started) / 1000;
    }
    /** A row was deleted: the table's NaN norm hides it; the node and its links stay. */
    noteDrop() {
        this.churn++;
    }
    // -- compaction --------------------------------------------------------------------
    /**
     * The graph after the table's compaction (`remap[old]` is the new slot, or
     * -1 for a dead one; slots past `remap.length` are dead): tombstones leave,
     * links are renumbered, and a node that lost links takes up the links of the
     * tombstones it pointed to ("bridging"), computing no distance. The entry
     * point becomes the first node of the highest level.
     */
    compacted(remap, newSlots) {
        const oldN = this.slots;
        const fresh = new HnswIndex(this.m, this.efC, 0, this._dim);
        fresh._rng = this._rng;
        fresh.churn = this.churn;
        fresh.buildSeconds = this.buildSeconds;
        fresh.inserts = this.inserts;
        fresh.insertSeconds = this.insertSeconds;
        if (newSlots > 0)
            fresh._ensureSlot(newSlots - 1);
        const lost = [];
        const out = new Int32Array(HNSW_MAX_CAP);
        const mapped = (s) => (s < remap.length ? remap[s] : -1);
        for (let old = 0; old < oldN; old++) {
            const nw = mapped(old);
            if (nw < 0)
                continue;
            const level = this.levels[old];
            if (level === HNSW_ABSENT)
                continue;
            fresh._addNode(nw, level);
            for (let l = 0; l <= level; l++) {
                const cap = this._cap(l);
                let oc = 0;
                lost.length = 0;
                const block = this._arr(old, l);
                const at = this._off(old, l);
                const len = block[at];
                for (let i = 1; i <= len; i++) {
                    const nb = block[at + i];
                    const n2 = mapped(nb);
                    if (n2 >= 0)
                        out[oc++] = n2;
                    else
                        lost.push(nb);
                }
                for (const d of lost) {
                    if (oc >= cap)
                        break;
                    if (d >= oldN || this.levels[d] === HNSW_ABSENT || this.levels[d] < l)
                        continue;
                    const db = this._arr(d, l);
                    const da = this._off(d, l);
                    const dl = db[da];
                    for (let i = 1; i <= dl; i++) {
                        const nw2 = mapped(db[da + i]);
                        if (nw2 < 0 || nw2 === nw)
                            continue;
                        let seen = false;
                        for (let k = 0; k < oc; k++) {
                            if (out[k] === nw2) {
                                seen = true;
                                break;
                            }
                        }
                        if (seen)
                            continue;
                        out[oc++] = nw2;
                        if (oc >= cap)
                            break;
                    }
                }
                fresh._setList(nw, l, out, oc);
            }
            if (fresh.entry === -1 || level > fresh.top) {
                fresh.entry = nw;
                fresh.top = level;
            }
        }
        return fresh;
    }
    // -- checks ------------------------------------------------------------------------
    /** DESIGN.md section 12.6 invariants 8-10 against the table; null when they hold. */
    verify(t) {
        const n = t.n;
        if (this.slots < n)
            return `graph covers ${this.slots} rows, table has ${n}`;
        if (this.l0.length !== Math.ceil(this.slots / HNSW_CHUNK_ROWS))
            return `${this.l0.length} link chunks for ${this.slots} rows`;
        let members = 0;
        let maxLevel = 0;
        const norms = t.norms;
        for (let s = 0; s < this.slots; s++) {
            const level = this.levels[s];
            if (level === HNSW_ABSENT) {
                if (s < n && t.keys[s] !== null && Number.isFinite(norms[s]))
                    return `live slot ${s} is not in the graph`;
                continue;
            }
            if (s >= n && !(norms[s] !== norms[s]))
                return `slot ${s} past the table's end is not marked dead`;
            members++;
            maxLevel = Math.max(maxLevel, level);
            for (let l = 0; l <= level; l++) {
                const list = this.linksOf(s, l);
                if (list.length > this._cap(l))
                    return `slot ${s} layer ${l} holds ${list.length} links`;
                const seen = list.slice().sort();
                for (let i = 1; i < seen.length; i++)
                    if (seen[i] === seen[i - 1])
                        return `slot ${s} layer ${l} links a node twice`;
                for (const nb of list) {
                    if (nb < 0 || nb >= this.slots || nb === s)
                        return `slot ${s} layer ${l} links ${nb}`;
                    if (this.levels[nb] === HNSW_ABSENT || this.levels[nb] < l)
                        return `slot ${s} layer ${l} links ${nb}, which is not on that layer`;
                }
            }
        }
        if (members !== this.members)
            return `member counter ${this.members} but ${members} nodes`;
        if (members === 0) {
            if (this.entry !== -1)
                return 'an empty graph has an entry point';
        }
        else if (this.entry === -1 || this.levels[this.entry] !== this.top || maxLevel !== this.top) {
            return `entry ${this.entry} does not sit on top level ${this.top} (highest node ${maxLevel})`;
        }
        return null;
    }
    /** Nodes whose rows are tombstones (or past the table's end). */
    dead(t) {
        let dead = 0;
        for (let s = 0; s < this.slots; s++) {
            if (this.levels[s] === HNSW_ABSENT)
                continue;
            const nm = t.norms[s];
            if (nm !== nm)
                dead++;
        }
        return dead;
    }
    /** A comparable digest of the whole structure (levels, links, entry), to tell two graphs apart (tests). */
    structureHash() {
        let h1 = 0x811c9dc5;
        let h2 = 0x01000193;
        const mix = (v) => {
            h1 = Math.imul(h1 ^ (v >>> 0), 0x01000193) >>> 0;
            h2 = Math.imul(h2 + (v >>> 0) + 0x9e3779b9, 0x85ebca6b) >>> 0;
            h2 ^= h2 >>> 13;
        };
        mix(this.slots);
        mix(this.entry);
        mix(this.top);
        mix(this.members);
        for (let s = 0; s < this.slots; s++) {
            mix(this.levels[s]);
            if (this.levels[s] === HNSW_ABSENT)
                continue;
            for (let l = 0; l <= this.levels[s]; l++) {
                const list = this.linksOf(s, l);
                mix(list.length);
                for (const x of list)
                    mix(x);
            }
        }
        return `${h1.toString(16)}-${h2.toString(16)}`;
    }
}
/**
 * In-memory embedding storage with brute-force cosine similarity search.
 *
 * Search is exact by default. For large stores an opt-in approximate index
 * (`ann: 'ivf'`, see {@link AnnOptions}) trades a little recall for much lower
 * latency; this is built for an application's own knowledge graph, not as a
 * vector-database replacement.
 */
export class EmbeddingStore {
    constructor(encoder, options = {}) {
        this._table = null;
        this._dimension = null;
        this._onModelMismatch = 'warn';
        this._warnedMismatch = new Set();
        this._ann = null;
        this._annNlist = null;
        this._annNprobe = null;
        this._annMinSize = ANN_MIN_SIZE;
        this._annSeed = 0;
        this._annHnswM = HNSW_DEFAULT_M;
        this._annHnswEfConstruction = HNSW_DEFAULT_EF_CONSTRUCTION;
        this._annEfSearch = null;
        this._annCounts = { approximate: 0, exact: 0, widened: 0, builds: 0 };
        /** Metadata of the most recent imported sidecar (empty before any). */
        this.lastImportMeta = {};
        this._encoder = adoptEncoder(encoder);
        this._version = options.version ?? null;
        this.onModelMismatch = options.onModelMismatch ?? 'warn';
        const ann = options.ann ?? null;
        if (ann !== null && ann !== 'ivf' && ann !== 'hnsw') {
            throw new EmbeddingError(`ann must be null, 'ivf' or 'hnsw', got ${JSON.stringify(ann)}`);
        }
        const whole = (label, value, floor) => {
            if (value === undefined || value === null)
                return null;
            if (typeof value !== 'number' || !Number.isInteger(value) || value < floor) {
                throw new EmbeddingError(`${label} must be an integer >= ${floor}, got ${value}`);
            }
            return value;
        };
        this._ann = ann;
        this._annNlist = whole('annNlist', options.annNlist, 1);
        this._annNprobe = whole('annNprobe', options.annNprobe, 1);
        this._annMinSize = whole('annMinSize', options.annMinSize, 0) ?? ANN_MIN_SIZE;
        this._annSeed = Math.trunc(Number(options.annSeed ?? 0)) || 0;
        const hnsw = options.annHnsw ?? {};
        const m = whole('annHnsw.m', hnsw.m, 2) ?? HNSW_DEFAULT_M;
        if (m > HNSW_MAX_M)
            throw new EmbeddingError(`annHnsw.m must be an integer between 2 and ${HNSW_MAX_M}, got ${m}`);
        this._annHnswM = m;
        this._annHnswEfConstruction = whole('annHnsw.efConstruction', hnsw.efConstruction, 1) ?? HNSW_DEFAULT_EF_CONSTRUCTION;
        this._annEfSearch = whole('annHnsw.efSearch', hnsw.efSearch, 1);
    }
    /** The approximate index in use (`'ivf'` or `'hnsw'`), or null when search is exact. */
    get ann() {
        return this._ann;
    }
    /** Cells scanned per query; null means "automatic" (about 5% of the cells). */
    get annNprobe() {
        return this._annNprobe;
    }
    set annNprobe(value) {
        if (value !== null && (typeof value !== 'number' || !Number.isInteger(value) || value < 1)) {
            throw new EmbeddingError(`annNprobe must be an integer >= 1, got ${value}`);
        }
        this._annNprobe = value;
    }
    /**
     * Candidates per HNSW query; null means automatic (by store size, see
     * {@link HnswOptions}). Takes effect on the next search; the graph is not
     * rebuilt. Ignored by the exact scan and by IVF.
     */
    get annEfSearch() {
        return this._annEfSearch;
    }
    set annEfSearch(value) {
        if (value !== null && (typeof value !== 'number' || !Number.isInteger(value) || value < 1)) {
            throw new EmbeddingError(`annEfSearch must be an integer >= 1, got ${value}`);
        }
        this._annEfSearch = value;
    }
    /** Set {@link annEfSearch} (null = automatic); the same as assigning the property. */
    setAnnEfSearch(efSearch) {
        this.annEfSearch = efSearch;
    }
    get encoder() {
        return this._encoder;
    }
    set encoder(encoder) {
        this._encoder = adoptEncoder(encoder);
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
    /** Euclidean norm, summed in index order (the order the stored norms are summed in). */
    static _norm(v) {
        let sum = 0;
        for (let i = 0; i < v.length; i++)
            sum += v[i] * v[i];
        return Math.sqrt(sum);
    }
    /**
     * Cosine of a query against the row at `slot`: float64, the dot product summed
     * in dimension order, 0 when either norm is 0. The one definition every exact
     * path shares.
     */
    static _rowScore(table, q, qnorm, slot) {
        const rn = table.norms[slot];
        if (qnorm === 0 || rn === 0)
            return 0;
        const dim = table.dim;
        const chunk = table.chunks[slot >>> CHUNK_SHIFT];
        const base = (slot & CHUNK_MASK) * dim;
        let dot = 0;
        for (let j = 0; j < dim; j++)
            dot += q[j] * chunk[base + j];
        return dot / (qnorm * rn);
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
        return this._checkQuery(vector);
    }
    _checkQuery(vector) {
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
        const table = this._table;
        if (table !== null) {
            for (let slot = 0; slot < table.n; slot++) {
                if (table.keys[slot] === null)
                    continue;
                const model = table.modelNames[table.modelIx[slot]];
                counts.set(model, (counts.get(model) ?? 0) + 1);
            }
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
        const table = this._table;
        if (table === null)
            return out;
        const byDim = dimension !== null && table.dim !== dimension;
        for (let slot = 0; slot < table.n; slot++) {
            if (table.keys[slot] === null)
                continue;
            const rowModel = table.modelNames[table.modelIx[slot]];
            const byModel = model !== null &&
                rowModel !== model &&
                (includeUnknown || !UNKNOWN_MODELS.has(rowModel));
            if (byModel || byDim)
                out.push(table.refs[slot]);
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
        this._put(key, nodeRef, text, this._modelName(), vector);
        return key;
    }
    /**
     * Every write goes through here: the row is copied into the table (the one
     * copy of the vector), and an existing approximate index is told about it.
     */
    _put(key, ref, text, model, vector) {
        let table = this._table;
        if (table === null)
            table = this._table = new VectorTable(this._dimension);
        table.put(key, ref, text, model, vector);
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
    /**
     * Encode texts with the store's encoder, awaiting it: an {@link AsyncEncoder}
     * (such as HttpEncoder) over the network, any other encoder as `encodeBatch`.
     * Nothing is stored. Rejects with NoEncoderError when there is no encoder.
     */
    async encodeTextsAsync(texts) {
        const encoder = this._encoder;
        if (!encoder) {
            throw new NoEncoderError("No encoder provided and no vectors given");
        }
        const vectors = isAsyncEncoder(encoder)
            ? await encoder.encodeBatchAsync(texts)
            : encoder.encodeBatch(texts);
        if (vectors.length !== texts.length) {
            throw new EmbeddingError(`encoder returned ${vectors.length} vectors for ${texts.length} texts`);
        }
        return vectors;
    }
    /**
     * Like {@link add}, but awaits the encoder, so it works with one that makes
     * a network call. On a failure nothing is stored.
     */
    async addAsync(nodeRef, text, vector) {
        if (vector === undefined) {
            vector = (await this.encodeTextsAsync([text]))[0];
        }
        return this.add(nodeRef, text, vector);
    }
    /**
     * Like {@link addBatch}, but awaits the encoder: the texts that need
     * encoding go out in one `encodeBatchAsync` call (HttpEncoder splits it into
     * `batchSize` requests). All entries are stored only once every vector is in.
     */
    async addBatchAsync(entries) {
        if (entries.length === 0)
            return 0;
        const needsEncoding = entries
            .map((e, i) => (e.length < 3 || e[2] === undefined ? i : -1))
            .filter(i => i >= 0);
        const vectorFor = new Map();
        if (needsEncoding.length > 0) {
            const encoded = await this.encodeTextsAsync(needsEncoding.map(i => entries[i][1]));
            needsEncoding.forEach((slot, i) => vectorFor.set(slot, encoded[i]));
        }
        entries.forEach((entry, i) => {
            this.add(entry[0], entry[1], vectorFor.get(i) ?? entry[2]);
        });
        return entries.length;
    }
    /**
     * The vector a query stands for: a vector as it is, a text through the
     * encoder (awaited), with the same model-mismatch check as a synchronous
     * text query. Used by the `...Async` searches.
     */
    async resolveQueryAsync(query) {
        if (typeof query !== 'string')
            return this._checkQuery(query);
        if (!this._encoder) {
            throw new NoEncoderError("No encoder provided for text query");
        }
        this._checkModelCompat();
        return this._checkQuery((await this.encodeTextsAsync([query]))[0]);
    }
    /**
     * The stored record of a node, or null. The record is a copy: `vector` is a
     * fresh array of the stored float32 values, so changing it changes nothing in
     * the store (use {@link add} to replace a vector).
     */
    get(nodeRef) {
        const table = this._table;
        if (table === null)
            return null;
        const slot = table.index.get(this._nodeRefToKey(nodeRef));
        if (slot === undefined)
            return null;
        return {
            id: table.keys[slot],
            nodeRef: table.refs[slot],
            vector: table.vectorAt(slot),
            text: table.texts[slot],
            model: table.modelNames[table.modelIx[slot]],
        };
    }
    /** Whether a node has an embedding (no vector is copied). */
    has(nodeRef) {
        return this._table !== null && this._table.index.has(this._nodeRefToKey(nodeRef));
    }
    remove(nodeRef) {
        const table = this._table;
        if (table === null)
            return false;
        const removed = table.remove(this._nodeRefToKey(nodeRef));
        if (removed && table.hnsw !== null && table.live < Math.max(2, this._annMinSize))
            table.hnsw = null; // too small to index now
        return removed;
    }
    /**
     * Find the most similar nodes by embedding.
     *
     * Exact by default. With `ann: 'ivf'` or `'hnsw'`, a bounded `topK` on a store
     * of at least `annMinSize` vectors, with `topK` below the live count, is answered
     * from the approximate index (IVF re-scores its hits exactly; HNSW's hits are
     * the exact scan's own numbers; either way a true neighbour can be missing);
     * `exact: true` forces the full scan, and has no effect on a store without
     * `ann`. `topK: null` always scans everything.
     */
    similaritySearch(query, topK = 10, nodeType, threshold = 0.0, exact = false) {
        return this._search(this._resolveQuery(query), topK, nodeType, threshold, exact, 0);
    }
    /**
     * {@link similaritySearch} (never forced exact) with `efSearch` candidates for
     * this call only: the HNSW knob, ignored by the exact scan and by IVF.
     */
    similaritySearchEf(query, topK, nodeType, threshold, efSearch) {
        if (typeof efSearch !== 'number' || !Number.isInteger(efSearch) || efSearch < 1) {
            throw new EmbeddingError(`efSearch must be an integer >= 1, got ${efSearch}`);
        }
        return this._search(this._resolveQuery(query), topK, nodeType, threshold, false, efSearch);
    }
    _search(queryVector, topK, nodeType, threshold, exact, efOver) {
        if (this._ann !== null) {
            if (exact || topK === null || topK <= 0) {
                this._annCounts.exact++;
            }
            else {
                const answer = this._annSearch(queryVector, topK, nodeType, threshold, efOver);
                if (answer !== null)
                    return answer;
                this._annCounts.exact++;
            }
        }
        return this._scanExact(queryVector, topK, nodeType, threshold);
    }
    /**
     * The brute-force scan every search used before the approximate index existed.
     *
     * Reads the rows straight from the table, in slot (= arrival) order: float64
     * dot products over the float32 values, summed in dimension order, so a score
     * is the same number whatever else is in the store. A bounded `topK` is kept
     * in a small sorted buffer (a hit is only allocated as a result if it makes
     * the cut); ties keep arrival order, exactly as {@link selectTopK} does.
     */
    _scanExact(queryVector, topK, nodeType, threshold) {
        const table = this._table;
        if (table === null || table.live === 0)
            return [];
        if (topK !== null && topK <= 0)
            return [];
        let code = -1;
        if (nodeType) {
            code = table.codeOf.get(nodeType) ?? -2;
            if (code === -2)
                return []; // no node of that type exists
        }
        const bounded = topK !== null && Number.isInteger(topK) && topK <= EXACT_TOPK_BUFFER;
        const k = bounded ? topK : 0;
        const topRows = new Int32Array(k);
        const topScores = new Float64Array(k);
        let count = 0;
        const all = [];
        const { dim, chunks, norms, codes, keys, refs, texts } = table;
        const qnorm = EmbeddingStore._norm(queryVector);
        const consider = (slot, score) => {
            if (!(score >= threshold))
                return;
            if (bounded) {
                if (count === k && score <= topScores[k - 1])
                    return;
                let i = count < k ? count++ : k - 1;
                while (i > 0 && topScores[i - 1] < score) {
                    topScores[i] = topScores[i - 1];
                    topRows[i] = topRows[i - 1];
                    i--;
                }
                topScores[i] = score;
                topRows[i] = slot;
            }
            else {
                all.push({ nodeRef: refs[slot], score, text: texts[slot] });
            }
        };
        // Four rows at a time: four independent running sums hide the latency of the
        // adds. Each is still summed in dimension order, so a score is the same
        // number the one-row loop would give. Rows are handed to `consider` in slot
        // order, so ties break as they always did.
        const pend = new Int32Array(4);
        const dots = new Float64Array(4);
        let np = 0;
        const flush = () => {
            if (np === 4) {
                const s0 = pend[0], s1 = pend[1], s2 = pend[2], s3 = pend[3];
                dotRows4(queryVector, chunks[s0 >>> CHUNK_SHIFT], (s0 & CHUNK_MASK) * dim, chunks[s1 >>> CHUNK_SHIFT], (s1 & CHUNK_MASK) * dim, chunks[s2 >>> CHUNK_SHIFT], (s2 & CHUNK_MASK) * dim, chunks[s3 >>> CHUNK_SHIFT], (s3 & CHUNK_MASK) * dim, dim, dots);
            }
            else {
                for (let i = 0; i < np; i++) {
                    const slot = pend[i];
                    const chunk = chunks[slot >>> CHUNK_SHIFT];
                    const base = (slot & CHUNK_MASK) * dim;
                    let dot = 0;
                    for (let j = 0; j < dim; j++)
                        dot += queryVector[j] * chunk[base + j];
                    dots[i] = dot;
                }
            }
            for (let i = 0; i < np; i++) {
                const rn = norms[pend[i]];
                consider(pend[i], rn === 0 ? 0 : dots[i] / (qnorm * rn));
            }
            np = 0;
        };
        for (let slot = 0; slot < table.n; slot++) {
            if (keys[slot] === null)
                continue;
            if (code >= 0 && codes[slot] !== code)
                continue;
            if (qnorm === 0) {
                consider(slot, 0);
                continue;
            }
            pend[np++] = slot;
            if (np === 4)
                flush();
        }
        if (np > 0)
            flush();
        if (bounded) {
            const out = new Array(count);
            for (let i = 0; i < count; i++) {
                out[i] = { nodeRef: refs[topRows[i]], score: topScores[i], text: texts[topRows[i]] };
            }
            return out;
        }
        if (topK === null)
            return all.sort((a, b) => b.score - a.score);
        return selectTopK(all, topK);
    }
    /** {@link similaritySearch} with a text query encoded by awaiting the encoder. */
    async similaritySearchAsync(query, topK = 10, nodeType, threshold = 0.0, exact = false) {
        return this.similaritySearch(await this.resolveQueryAsync(query), topK, nodeType, threshold, exact);
    }
    /** The vector a query stands for (a text goes through the synchronous encoder). */
    resolveQuery(query) {
        return this._resolveQuery(query);
    }
    // -- approximate (HNSW) search ------------------------------------------------
    /** The ef a query uses: the per-call override, else the configured value, else the size-based default; never below k. */
    _hnswEf(over, k, live) {
        const ef = over > 0 ? over : this._annEfSearch ?? hnswAutoEf(live);
        return Math.max(ef, k, 1);
    }
    /**
     * The table holding the graph, with the graph built (or rebuilt once churn
     * made it stale) as the policy says; null means "answer exactly": the mode is
     * not HNSW, or the store holds fewer than `annMinSize` vectors (the graph is
     * dropped then). Built lazily by the first search, never persisted; writes
     * keep it current (see {@link VectorTable.put}).
     */
    _hnswTable() {
        if (this._ann !== 'hnsw' || this._dimension === null)
            return null;
        const table = this._table;
        if (table === null || table.live < Math.max(2, this._annMinSize)) {
            if (table !== null)
                table.hnsw = null;
            return null;
        }
        let graph = table.hnsw;
        if (graph !== null && graph.stale)
            graph = table.hnsw = null;
        if (graph === null) {
            table.hnsw = HnswIndex.build(table, this._annHnswM, this._annHnswEfConstruction, this._annSeed);
            this._annCounts.builds++;
        }
        return table;
    }
    /** A hit list from the graph's last result buffer (best first): the first `count` of them. */
    static _hnswHits(table, graph, count) {
        const hits = new Array(count);
        for (let i = 0; i < count; i++) {
            const slot = graph.outI[i];
            hits[i] = { nodeRef: table.refs[slot], score: graph.outS[i], text: table.texts[slot] };
        }
        return hits;
    }
    /**
     * Top-k from the graph, or null to fall back to the exact scan. A filter
     * (`nodeType`, `threshold`) leaves some answers short: ef is then widened 4x
     * and the search retried, and the scan answers instead once ef would cover
     * half the store or one search scores more nodes than the scan would cost
     * (a filter so selective that scanning is cheaper). A hit's score is the
     * exact scan's score for that node, bit for bit, and ties order by arrival.
     */
    _hnswSearch(queryVector, topK, nodeType, threshold, efOver) {
        const table = this._hnswTable();
        if (table === null)
            return null;
        const graph = table.hnsw;
        const qnorm = EmbeddingStore._norm(queryVector);
        if (!(qnorm > 0) || !Number.isFinite(qnorm))
            return null; // no direction to walk by: the scan answers
        let code = -1;
        if (nodeType) {
            code = table.codeOf.get(nodeType) ?? -2;
            if (code === -2)
                return []; // no node of that type exists
        }
        const live = table.live;
        if (!Number.isInteger(topK) || topK >= live)
            return null; // everything is wanted: the scan is the answer
        const baseEf = this._hnswEf(efOver, topK, live);
        for (let scale = 1;; scale *= 4) {
            const ef = baseEf * scale;
            if (scale === 1 ? ef >= live : ef * 2 >= live)
                return null;
            const found = graph.search(table, queryVector, qnorm, code, threshold, ef, Math.max(Math.floor(live / 8), ef * 16));
            if (graph.budgetHit)
                return null;
            if (found >= topK) {
                this._annCounts.approximate++;
                if (scale > 1)
                    this._annCounts.widened++;
                return EmbeddingStore._hnswHits(table, graph, topK);
            }
        }
    }
    // -- approximate (IVF) search -------------------------------------------------
    /** Cells for a store of `n` vectors: about 4 * sqrt(n), within [16, 65536]. */
    static _annAutoNlist(n) {
        return Math.min(Math.max(roundHalfEven(4 * Math.sqrt(n)), 16), 65536);
    }
    /**
     * The cache holding the index, with the index built or maintained as the
     * policy says; null means "answer exactly": `ann` is off, or the store holds
     * fewer than `annMinSize` vectors.
     *
     * Policy:
     *  - built lazily by the first search (never persisted): k-means on a sample
     *    of up to 40 x nlist vectors, then every row assigned a cell;
     *  - new and updated rows are scored exactly until at least
     *    `max(512, 2%)` of the indexed rows are pending, then folded into cells;
     *  - deletes tombstone (nothing to do but skip them);
     *  - once writes + deletes since training exceed 50% of the trained size,
     *    the index is retrained from the current data, with an automatic `nlist`
     *    re-chosen for the new size;
     *  - a cleared store, a dimension change, or a store that shrinks below
     *    `annMinSize` drops the index.
     *
     * The index reads the store's own rows; there is no second copy of the vectors.
     */
    _annIndex() {
        if (this._ann === null || this._dimension === null)
            return null;
        const table = this._table;
        if (table === null || table.live < Math.max(2, this._annMinSize)) {
            if (table !== null)
                table.ivf = null;
            return null;
        }
        const live = table.live;
        let ivf = table.ivf;
        if (ivf !== null && ivf.churn > ANN_RETRAIN_FRACTION * ivf.trainedN)
            ivf = table.ivf = null;
        if (ivf === null) {
            const nlist = this._annNlist ?? EmbeddingStore._annAutoNlist(live);
            table.ivf = IvfIndex.build(table, nlist, this._annSeed);
            this._annCounts.builds++;
        }
        else if (ivf.pending.size >= Math.max(ANN_FOLD_MIN, ANN_FOLD_FRACTION * ivf.trainedN)) {
            ivf.fold(table);
        }
        return table;
    }
    _annProbe(ivf) {
        const wanted = this._annNprobe ?? Math.max(1, roundHalfEven(0.05 * ivf.nlist));
        return Math.min(wanted, ivf.nlist);
    }
    /**
     * Cells to probe at widening `scale`; null = widened enough to just scan.
     * Scale 1 honours `annNprobe` as configured (`annNprobe >= nlist` is the
     * exact scan); later steps multiply it by 4 and give up on the index once
     * they would touch half the cells, where scanning everything costs about the
     * same.
     */
    _annLists(ivf, scale) {
        const lists = this._annProbe(ivf) * scale;
        if (scale === 1)
            return lists >= ivf.nlist ? null : lists;
        return lists * 2 >= ivf.nlist ? null : lists;
    }
    /** Load the query into the table's float32 buffer; null when it has no direction (all zeros). */
    _annQuery(scan, queryVector, nodeType) {
        const q = scan.query;
        let sum = 0;
        for (let j = 0; j < scan.dim; j++) {
            const f = Math.fround(queryVector[j]);
            q[j] = f;
            sum += f * f;
        }
        const qnorm = Math.sqrt(sum);
        if (qnorm === 0)
            return null;
        let code = -1;
        if (nodeType)
            code = scan.codeOf.get(nodeType) ?? -2; // -2: no node of that type exists
        return { qnorm, code };
    }
    /**
     * Top-k from the index, or null to fall back to the exact scan.
     *
     * Filters (`nodeType`, `threshold`) apply to the probed rows. If fewer than
     * `topK` survive, the probe is widened 4x and tried again, and past half the
     * cells the exact scan answers instead - so a rare type or a high threshold
     * costs time rather than silently returning a short list.
     */
    _annSearch(queryVector, topK, nodeType, threshold, efOver = 0) {
        if (this._ann === 'hnsw')
            return this._hnswSearch(queryVector, topK, nodeType, threshold, efOver);
        const scan = this._annIndex();
        if (scan === null)
            return null;
        const ivf = scan.ivf;
        const prepared = this._annQuery(scan, queryVector, nodeType);
        if (prepared === null)
            return null;
        if (prepared.code === -2)
            return [];
        const floor = threshold - ANN_THRESHOLD_SLACK;
        for (let scale = 1;; scale *= 4) {
            const lists = this._annLists(ivf, scale);
            if (lists === null)
                return null;
            const m = ivf.gather(scan, scan.query, prepared.qnorm, prepared.code, floor, lists);
            if (m < topK)
                continue;
            const best = ivf.best(m, topK);
            const results = [];
            for (let i = 0; i < best.length; i++) {
                const row = ivf.candRows[best[i]];
                results.push({ nodeRef: scan.refs[row], score: ivf.candScores[best[i]], text: scan.texts[row] });
            }
            this._annCounts.approximate++;
            if (scale > 1)
                this._annCounts.widened++;
            return this.annPolish(results, queryVector, threshold);
        }
    }
    /**
     * Re-score approximate results with the exact float64 cosine the full scan
     * uses (the index scores float32 copies), drop any that fall under
     * `threshold` once exact, and order them best first.
     */
    annPolish(results, queryVector, threshold) {
        if (this._ann === 'hnsw')
            return results; // the graph scored every hit exactly (same rows, same arithmetic) and ordered them
        const out = [];
        const table = this._table;
        if (table === null)
            return out;
        const qnorm = EmbeddingStore._norm(queryVector);
        for (const result of results) {
            const slot = table.index.get(this._nodeRefToKey(result.nodeRef));
            if (slot === undefined)
                continue;
            const score = EmbeddingStore._rowScore(table, queryVector, qnorm, slot);
            if (score >= threshold)
                out.push({ nodeRef: result.nodeRef, score, text: result.text });
        }
        return out.sort((a, b) => b.score - a.score);
    }
    /**
     * Candidates for a search that has to filter them itself (a graph's
     * `nodeFilter`): every row of the probed cells that clears `threshold`, best
     * first, lazily, with *approximate* (float32) scores - pass what you keep to
     * {@link annPolish}. `scale` is the widening step (1, 4, 16, ...); null asks
     * for the exact scan. `approximate` is false when the answer is exact - the
     * caller has nothing left to widen to.
     */
    annRanked(queryVector, nodeType, threshold, scale) {
        if (scale !== null && this._ann === 'hnsw') {
            const table = this._hnswTable();
            const graph = table === null ? null : table.hnsw;
            const qnorm = EmbeddingStore._norm(queryVector);
            if (table !== null && graph !== null && qnorm > 0 && Number.isFinite(qnorm)) {
                let code = -1;
                if (nodeType)
                    code = table.codeOf.get(nodeType) ?? -2;
                if (code === -2)
                    return { ranked: [], approximate: false };
                const live = table.live;
                const ef = this._hnswEf(0, 1, live) * scale;
                if (!(scale === 1 ? ef >= live : ef * 2 >= live)) {
                    const found = graph.search(table, queryVector, qnorm, code, threshold, ef, Math.max(Math.floor(live / 8), ef * 16));
                    if (!graph.budgetHit) {
                        this._annCounts.approximate++;
                        if (scale > 1)
                            this._annCounts.widened++;
                        return { ranked: EmbeddingStore._hnswHits(table, graph, found), approximate: true };
                    }
                }
            }
        }
        else if (scale !== null && this._ann !== null) {
            const scan = this._annIndex();
            const ivf = scan === null ? null : scan.ivf;
            const lists = ivf === null ? null : this._annLists(ivf, scale);
            const prepared = scan === null || lists === null ? null : this._annQuery(scan, queryVector, nodeType);
            if (scan !== null && ivf !== null && lists !== null && prepared !== null) {
                if (prepared.code === -2)
                    return { ranked: [], approximate: false };
                const m = ivf.gather(scan, scan.query, prepared.qnorm, prepared.code, threshold - ANN_THRESHOLD_SLACK, lists);
                this._annCounts.approximate++;
                if (scale > 1)
                    this._annCounts.widened++;
                const rows = ivf.candRows.slice(0, m);
                const scores = ivf.candScores.slice(0, m);
                const slots = new Int32Array(m);
                for (let i = 0; i < m; i++)
                    slots[i] = i;
                slots.sort((a, b) => scores[b] - scores[a] || rows[a] - rows[b]);
                const refs = scan.refs;
                const texts = scan.texts;
                const ranked = (function* () {
                    for (let i = 0; i < m; i++) {
                        const row = rows[slots[i]];
                        yield { nodeRef: refs[row], score: scores[slots[i]], text: texts[row] };
                    }
                })();
                return { ranked, approximate: true };
            }
        }
        return { ranked: this._scanExact(queryVector, null, nodeType, threshold), approximate: false };
    }
    /**
     * Build the approximate index now instead of on the first search; returns
     * {@link annStats}. `force` retrains even if an index exists. Does nothing
     * (and says so in the stats) while the store is smaller than `annMinSize`.
     *
     * @throws EmbeddingError when the store was created without `ann`
     */
    buildAnnIndex(force = false) {
        if (this._ann === null) {
            throw new EmbeddingError("this store has no approximate index; create it with ann: 'ivf' or 'hnsw'");
        }
        if (this._ann === 'hnsw') {
            if (force && this._table !== null)
                this._table.hnsw = null;
            this._hnswTable();
            return this.annStats();
        }
        if (force && this._table !== null)
            this._table.ivf = null;
        const scan = this._annIndex();
        if (scan !== null && scan.ivf !== null && scan.ivf.pending.size > 0)
            scan.ivf.fold(scan);
        return this.annStats();
    }
    /** What the approximate index looks like right now. Never builds anything. */
    annStats() {
        const scan = this._table;
        if (this._ann === 'hnsw') {
            const graph = scan === null ? null : scan.hnsw;
            return {
                enabled: true,
                mode: 'hnsw',
                active: graph !== null,
                nlist: null,
                nprobe: null,
                trainedN: graph === null ? null : graph.members,
                pending: 0,
                churn: graph === null ? 0 : graph.churn,
                buildSeconds: graph === null ? null : graph.buildSeconds,
                indexBytes: graph === null ? 0 : graph.bytes(),
                cacheBytes: 0,
                vectorBytes: scan === null ? 0 : scan.bytes(),
                minSize: this._annMinSize,
                searches: { ...this._annCounts },
                efSearch: this._annEfSearch ?? hnswAutoEf(scan === null ? 0 : scan.live),
                maxLevel: graph === null ? null : graph.top,
                graphNodes: graph === null ? 0 : graph.members,
                graphDead: graph === null || scan === null ? 0 : graph.dead(scan),
                inserts: graph === null ? 0 : graph.inserts,
                insertSeconds: graph === null ? 0 : graph.insertSeconds,
            };
        }
        const ivf = scan === null ? null : scan.ivf;
        return {
            enabled: this._ann !== null,
            mode: this._ann,
            active: ivf !== null,
            nlist: ivf === null ? null : ivf.nlist,
            nprobe: ivf === null ? this._annNprobe : this._annProbe(ivf),
            trainedN: ivf === null ? null : ivf.trainedN,
            pending: ivf === null ? 0 : ivf.pending.size,
            churn: ivf === null ? 0 : ivf.churn,
            buildSeconds: ivf === null ? null : ivf.buildSeconds,
            indexBytes: ivf === null ? 0 : ivf.bytes(),
            cacheBytes: 0,
            vectorBytes: scan === null ? 0 : scan.bytes(),
            minSize: this._annMinSize,
            searches: { ...this._annCounts },
        };
    }
    /**
     * Recall@k of the approximate index against the exact scan.
     *
     * Runs every query both ways and compares the *sets* of top-k nodes.
     * `queries` is a text, a vector, or a list of either. Use real queries from
     * your own workload: recall depends on how the data clusters, and a score
     * measured on unrelated queries or data does not transfer. Raise `annNprobe`
     * until the recall you need holds.
     *
     * Without an index in use (`ann` off, a store under `annMinSize`, or
     * `annNprobe >= nlist`) both answers are the exact scan, recall is 1 and
     * `approximate` is false.
     */
    recallCheck(queries, k = 10, nodeType, threshold = 0.0) {
        if (!Number.isInteger(k) || k <= 0)
            throw new EmbeddingError(`k must be positive, got ${k}`);
        const vectors = EmbeddingStore._queryList(queries).map(q => this._resolveQuery(q));
        return this._recall(vectors, k, nodeType, threshold);
    }
    /** {@link recallCheck} with text queries encoded by awaiting the encoder. */
    async recallCheckAsync(queries, k = 10, nodeType, threshold = 0.0) {
        if (!Number.isInteger(k) || k <= 0)
            throw new EmbeddingError(`k must be positive, got ${k}`);
        const vectors = [];
        for (const query of EmbeddingStore._queryList(queries))
            vectors.push(await this.resolveQueryAsync(query));
        return this._recall(vectors, k, nodeType, threshold);
    }
    /** One query or many, as a list of texts and plain number arrays. */
    static _queryList(queries) {
        if (typeof queries === 'string')
            return [queries];
        const list = Array.from(queries);
        if (list.length > 0 && typeof list[0] === 'number')
            return [list];
        return list.map(q => (typeof q === 'string' ? q : Array.from(q)));
    }
    _recall(vectors, k, nodeType, threshold) {
        const recalls = [];
        const annTimes = [];
        const exactTimes = [];
        let used = false;
        for (const vector of vectors) {
            const before = this._annCounts.approximate;
            let started = nowMs();
            const found = this.similaritySearch(vector, k, nodeType, threshold);
            annTimes.push(nowMs() - started);
            used = used || this._annCounts.approximate > before;
            started = nowMs();
            const truth = this.similaritySearch(vector, k, nodeType, threshold, true);
            exactTimes.push(nowMs() - started);
            if (truth.length > 0) {
                const wanted = new Set(truth.map(r => this._nodeRefToKey(r.nodeRef)));
                const hits = found.filter(r => wanted.has(this._nodeRefToKey(r.nodeRef))).length;
                recalls.push(hits / wanted.size);
            }
        }
        const median = (values) => {
            if (values.length === 0)
                return 0;
            const ordered = [...values].sort((a, b) => a - b);
            const mid = ordered.length >> 1;
            return ordered.length % 2 ? ordered[mid] : 0.5 * (ordered[mid - 1] + ordered[mid]);
        };
        const ivf = this._table === null ? null : this._table.ivf;
        const hnsw = this._ann === 'hnsw';
        return {
            recall: recalls.length ? recalls.reduce((a, b) => a + b, 0) / recalls.length : 1,
            minRecall: recalls.length ? Math.min(...recalls) : 1,
            queries: vectors.length,
            k,
            nlist: ivf === null ? 0 : ivf.nlist,
            nprobe: ivf === null ? this._annNprobe ?? 0 : this._annProbe(ivf),
            approximate: used,
            annMs: median(annTimes),
            exactMs: median(exactTimes),
            ...(hnsw ? { efSearch: this._hnswEf(0, k, this._table === null ? 0 : this._table.live) } : {}),
        };
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
        const table = this._table;
        if (table === null)
            return scores;
        let code = -1;
        if (nodeType) {
            code = table.codeOf.get(nodeType) ?? -2;
            if (code === -2)
                return scores;
        }
        const qnorm = EmbeddingStore._norm(queryVector);
        for (let slot = 0; slot < table.n; slot++) {
            const key = table.keys[slot];
            if (key === null)
                continue;
            if (code >= 0 && table.codes[slot] !== code)
                continue;
            scores.set(key, EmbeddingStore._rowScore(table, queryVector, qnorm, slot));
        }
        return scores;
    }
    count() {
        return this._table === null ? 0 : this._table.live;
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
        this._table = null;
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
        const table = this._table;
        for (let slot = 0; table !== null && slot < table.n; slot++) {
            const ref = table.refs[slot];
            if (ref === null)
                continue;
            embeddings.push({
                type: String(ref[0]),
                id: ref[1],
                id_type: typeof ref[1] === 'number' ? 'int' : 'str',
                text: table.texts[slot],
                model: table.modelNames[table.modelIx[slot]],
                vector: table.vectorAt(slot),
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
            this._put(key, ref, item.text ?? '', item.model ?? 'unknown', item.vector);
            written++;
        }
        return written;
    }
    /**
     * The table, the encoder's model and the version, for the snapshot writer
     * (see `snapshot.ts`). Low-level: not part of the stable API.
     */
    snapshotState() {
        return { table: this._table, encoderModel: this._currentModel(), encoderVersion: this.version };
    }
    /**
     * Replace every embedding with the rows of a verified, fully built table
     * (see `snapshot.ts`), as `importEmbeddings(payload, true)` would, and record
     * the snapshot's metadata as the last import's. Low-level: not part of the
     * stable API. Any approximate index is rebuilt lazily.
     */
    adoptTable(table, meta) {
        this._table = table.live === 0 ? null : table;
        this._dimension = table.live === 0 ? null : table.dim;
        this.lastImportMeta = meta;
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
        this._encoder = adoptEncoder(options.encoder);
        this._embeddingStore = new EmbeddingStore(this._encoder || undefined, {
            version: options.version,
            onModelMismatch: options.onModelMismatch,
            ann: options.ann,
            annNlist: options.annNlist,
            annNprobe: options.annNprobe,
            annMinSize: options.annMinSize,
            annSeed: options.annSeed,
            annHnsw: options.annHnsw,
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
        this._encoder = adoptEncoder(encoder);
        this._embeddingStore.encoder = this._encoder;
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
     * Like {@link addNode}, but awaits the encoder, so auto-embedding works with
     * one that makes a network call (an HttpEncoder needs no prefetch).
     *
     * The text is embedded *before* the node is added: if the encoder fails the
     * graph is untouched and the call can simply be retried.
     */
    async addNodeAsync(nodeType, nodeId, properties = {}) {
        let text = '';
        let vector = null;
        if (this._autoEmbed && this._encoder) {
            text = this.embedTextFor(properties);
            if (text)
                vector = (await this._embeddingStore.encodeTextsAsync([text]))[0];
        }
        const node = super.addNode(nodeType, nodeId, properties);
        if (vector !== null)
            this._embeddingStore.add(node.ref, text, vector);
        return node;
    }
    /** Like {@link addNodeWithEmbed}, awaiting the encoder (see {@link addNodeAsync}). */
    async addNodeWithEmbedAsync(nodeType, nodeId, properties = {}, embedText) {
        if (embedText === undefined) {
            return this.addNodeAsync(nodeType, nodeId, properties);
        }
        const vector = (await this._embeddingStore.encodeTextsAsync([embedText]))[0];
        const node = super.addNode(nodeType, nodeId, properties);
        this._embeddingStore.add(node.ref, embedText, vector);
        return node;
    }
    /**
     * Embed a node with given text.
     */
    embedNode(nodeRef, text) {
        return this._embeddingStore.add(nodeRef, text);
    }
    /** Like {@link embedNode}, awaiting the encoder. */
    embedNodeAsync(nodeRef, text) {
        return this._embeddingStore.addAsync(nodeRef, text);
    }
    /** Like {@link embedNodes}, awaiting the encoder; stores nothing if it fails. */
    async embedNodesAsync(items) {
        const pairs = items.filter(([, text]) => Boolean(text));
        if (pairs.length === 0)
            return 0;
        return this._embeddingStore.addBatchAsync(pairs);
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
     * Like {@link updateNode}, but awaits the encoder when a re-embed is needed.
     *
     * The new text is embedded before anything changes, so an encoder failure
     * leaves the node's properties and vector exactly as they were.
     */
    async updateNodeAsync(nodeType, nodeId, properties) {
        const current = this.getNode(nodeType, nodeId);
        const reembed = this._autoEmbed && this._embedFields.some(f => f in (properties ?? {}));
        let text = '';
        let vector = null;
        if (reembed) {
            text = this.embedTextFor({ ...current.properties, ...properties });
            if (text && this._encoder)
                vector = (await this._embeddingStore.encodeTextsAsync([text]))[0];
        }
        const node = super.updateNode(nodeType, nodeId, properties);
        if (reembed) {
            // Another update may have landed while the encoder was working.
            const latest = this.embedTextFor(node.properties);
            if (!latest) {
                this._embeddingStore.remove(node.ref);
            }
            else if (this._encoder) {
                if (vector === null || latest !== text) {
                    vector = (await this._embeddingStore.encodeTextsAsync([latest]))[0];
                }
                this._embeddingStore.add(node.ref, latest, vector);
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
     *
     * With the store's approximate index the candidates are the probed cells
     * only. If fewer than k of them pass the gate, the probe is widened (4x each
     * step) and the search retried, ending in the exact scan, so a selective
     * filter costs time rather than results.
     */
    _seedSearch(query, topK, nodeType, threshold, gate, exact = false) {
        const store = this._embeddingStore;
        if (gate === null) {
            return store.similaritySearch(query, topK, nodeType, threshold, exact);
        }
        if (store.ann === null || exact || topK === null || topK <= 0) {
            const results = store
                .similaritySearch(query, null, nodeType, threshold)
                .filter(r => gate(r.nodeRef));
            return topK === null ? results : results.slice(0, Math.max(topK, 0));
        }
        const queryVector = store.resolveQuery(query);
        for (let scale = 1;; scale *= 4) {
            const { ranked, approximate } = store.annRanked(queryVector, nodeType, threshold, scale);
            const results = [];
            for (const result of ranked) {
                if (gate(result.nodeRef)) {
                    results.push(result);
                    if (results.length >= topK)
                        break;
                }
            }
            if (!approximate)
                return results;
            if (results.length >= topK)
                return store.annPolish(results, queryVector, threshold);
        }
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
     * @param exact Force the full scan even when the graph was created with an
     *   approximate index (`ann`); no effect otherwise.
     */
    similaritySearch(query, topK = 10, nodeType, threshold = 0.0, nodeFilter, nodeTypes, exact = false) {
        const gate = this._nodeGate(nodeFilter, nodeTypes);
        return this._seedSearch(query, topK, nodeType, threshold, gate, exact);
    }
    /** Recall@k of the approximate index against the exact scan (see {@link EmbeddingStore.recallCheck}). */
    recallCheck(queries, k = 10, nodeType, threshold = 0.0) {
        return this._embeddingStore.recallCheck(queries, k, nodeType, threshold);
    }
    /** {@link recallCheck} with text queries encoded by awaiting the encoder. */
    recallCheckAsync(queries, k = 10, nodeType, threshold = 0.0) {
        return this._embeddingStore.recallCheckAsync(queries, k, nodeType, threshold);
    }
    /** {@link similaritySearch} with a text query encoded by awaiting the encoder. */
    async similaritySearchAsync(query, topK = 10, nodeType, threshold = 0.0, nodeFilter, nodeTypes, exact = false) {
        const vector = await this._embeddingStore.resolveQueryAsync(query);
        return this.similaritySearch(vector, topK, nodeType, threshold, nodeFilter, nodeTypes, exact);
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
    /** {@link semanticMultiHop} with a text query encoded by awaiting the encoder (options form). */
    async semanticMultiHopAsync(query, options = {}) {
        return this.semanticMultiHop(await this._embeddingStore.resolveQueryAsync(query), options);
    }
    /** {@link semanticPath} with a text query encoded by awaiting the encoder (options form). */
    async semanticPathAsync(query, targetRef, options = {}) {
        return this.semanticPath(await this._embeddingStore.resolveQueryAsync(query), targetRef, options);
    }
    /** {@link semanticSubgraph} with a text query encoded by awaiting the encoder. */
    async semanticSubgraphAsync(query, options = {}) {
        return this.semanticSubgraph(await this._embeddingStore.resolveQueryAsync(query), options);
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
            if (skipExisting && this._embeddingStore.has(node.ref))
                continue;
            const text = fn(node);
            if (text)
                batch.push([node.ref, text]);
        }
        return this.embedNodes(batch);
    }
    /** {@link embedAllNodes} awaiting the encoder: all the texts, then the stores. */
    async embedAllNodesAsync(textFn, skipExisting = false) {
        const fn = textFn || ((node) => this.embedTextFor(node.properties) || `${node.type}:${node.id}`);
        const batch = [];
        for (const node of this.nodes()) {
            if (skipExisting && this._embeddingStore.has(node.ref))
                continue;
            const text = fn(node);
            if (text)
                batch.push([node.ref, text]);
        }
        return this.embedNodesAsync(batch);
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
     * {@link reembedStale} awaiting the encoder.
     *
     * Every stale text is encoded (`batchSize` per encoder call) before anything
     * is written, so a failure partway leaves the store as it was. The new
     * vectors' length decides whether the pinned dimension changes, which also
     * works for an encoder that has not learned its dimension yet.
     */
    async reembedStaleAsync(includeUnknown = false, batchSize = 256) {
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
        if (!this._encoder)
            throw new NoEncoderError("No encoder provided to re-embed with");
        const step = Math.max(1, Math.floor(batchSize));
        const vectors = [];
        for (let start = 0; start < pairs.length; start += step) {
            const chunk = pairs.slice(start, start + step);
            vectors.push(...await store.encodeTextsAsync(chunk.map(([, text]) => text)));
        }
        if (vectors.length > 0 && store.dimension !== null && store.dimension !== vectors[0].length) {
            store.clear(true);
        }
        pairs.forEach(([ref, text], i) => store.add(ref, text, vectors[i]));
        return pairs.length;
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
