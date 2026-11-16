/**
 * ISONGraph Vector - Semantic Graph Extension for TypeScript
 *
 * Extends ISONGraph with embedding-based similarity search and semantic traversal.
 *
 * @example
 * ```typescript
 * import { SemanticGraph, MockEncoder } from 'isongraph-vector-ts';
 *
 * const graph = new SemanticGraph({ encoder: new MockEncoder() });
 * graph.addNode('person', 1, { name: 'Alice' });   // auto-embedded
 *
 * const similar = graph.similaritySearch("engineering", 5);
 * ```
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

import { ISONGraph, Node, NodeRef, Edge, Direction } from 'ison-graph-ts';

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
export const UNKNOWN_MODELS: ReadonlySet<string> = new Set(["unknown", "provided"]);

// =============================================================================
// Errors
// =============================================================================

/** Base class for embedding-specific failures. */
export class EmbeddingError extends Error {
  constructor(message: string) {
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
  constructor(message: string) {
    super(message);
    this.name = 'DimensionMismatchError';
  }
}

/** Text was handed to a store with no encoder to embed it with. */
export class NoEncoderError extends EmbeddingError {
  constructor(message: string) {
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
  constructor(message: string) {
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
  constructor(message: string) {
    super(message);
    this.name = 'GraphEmbeddingMismatchError';
  }
}

// =============================================================================
// Embedding Encoder Interface
// =============================================================================

/**
 * Anything that turns text into vectors.
 *
 * Two optional pieces of metadata are recorded by the store: `modelName`
 * (stamped on every vector the encoder writes; decides whether stored vectors
 * are *stale* for this encoder) and `version` (a free-form string, e.g. a model
 * revision or weights hash, written to the sidecar).
 */
export interface EmbeddingEncoder {
  readonly dimension: number;
  modelName?: string;
  version?: string;
  encode(text: string): number[];
  encodeBatch(texts: string[]): number[][];
}

/**
 * An encoder whose work is asynchronous - typically a network call.
 *
 * Same metadata as {@link EmbeddingEncoder} (`modelName` stamps every vector
 * it writes and decides staleness, `version` goes to the sidecar). `dimension`
 * may be `NaN` until the encoder has learned it from a first answer.
 *
 * Hand one to a store or graph and use the `...Async` methods
 * (`addAsync`, `addNodeAsync`, `similaritySearchAsync`, ...). An encoder that
 * is *also* an `EmbeddingEncoder` - `HttpEncoder` is both - keeps working with
 * the synchronous methods too. The `...Async` methods accept either kind.
 */
export interface AsyncEncoder {
  readonly dimension: number;
  modelName?: string;
  version?: string;
  encodeAsync(text: string): Promise<number[]>;
  encodeBatchAsync(texts: string[]): Promise<number[][]>;
}

/** True for an object with `encodeAsync` and `encodeBatchAsync` methods. */
export function isAsyncEncoder(value: unknown): value is AsyncEncoder {
  const v = value as Partial<AsyncEncoder> | null | undefined;
  return (
    v !== null && v !== undefined && typeof v === 'object' &&
    typeof v.encodeAsync === 'function' && typeof v.encodeBatchAsync === 'function'
  );
}

/**
 * The synchronous face of an encoder that only has an asynchronous one, so the
 * stores (whose `encoder` is an EmbeddingEncoder) can hold it. Its metadata is
 * read live from the wrapped encoder; its synchronous methods refuse, naming
 * the asynchronous ones.
 */
class AsyncOnlyEncoder implements EmbeddingEncoder, AsyncEncoder {
  readonly inner: AsyncEncoder;

  constructor(inner: AsyncEncoder) {
    this.inner = inner;
  }

  get dimension(): number {
    return this.inner.dimension;
  }

  get modelName(): string | undefined {
    return this.inner.modelName;
  }

  get version(): string | undefined {
    return this.inner.version;
  }

  encode(_text: string): number[] {
    throw new EmbeddingError(
      'this encoder is asynchronous: use the Async methods (addAsync, addNodeAsync, ' +
      'similaritySearchAsync, ...) or pass the vector yourself'
    );
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map(text => this.encode(text));
  }

  encodeAsync(text: string): Promise<number[]> {
    return this.inner.encodeAsync(text);
  }

  encodeBatchAsync(texts: string[]): Promise<number[][]> {
    return this.inner.encodeBatchAsync(texts);
  }
}

/** What a store or graph holds: sync encoders as they are, async-only ones wrapped. */
function adoptEncoder(encoder: EmbeddingEncoder | AsyncEncoder | null | undefined): EmbeddingEncoder | null {
  if (encoder === null || encoder === undefined) return null;
  const sync = encoder as Partial<EmbeddingEncoder>;
  if (typeof sync.encode === 'function' && typeof sync.encodeBatch === 'function') {
    return encoder as EmbeddingEncoder;
  }
  return isAsyncEncoder(encoder) ? new AsyncOnlyEncoder(encoder) : (encoder as unknown as EmbeddingEncoder);
}

/**
 * Deterministic encoder with no dependencies, for tests and offline work.
 *
 * The same text produces the same vector in *every* language port
 * (Python/TypeScript/JavaScript/Rust/C++): a 32-bit FNV-1a hash over the UTF-8
 * bytes seeds an xorshift32 generator, and the top 24 bits of each state
 * become a value in [-1, 1).
 *
 * This replaces a scheme that emitted the *low* 16 bits of an LCG. The low
 * bits of an LCG evolve independently of the high bits, so every vector was
 * determined by `seed mod 65536`; in the exact-arithmetic ports (Rust/C++)
 * that meant 280 outright collisions in 5,000 texts, with colliding texts
 * scoring a perfect 1.0 against each other. TypeScript escaped that only
 * because `hash * 1103515245` overflows float64 and leaks high bits back in -
 * which also meant its vectors did not match the other ports.
 *
 * These are pseudo-embeddings with no semantic structure. Use them to test
 * plumbing, never to judge relevance quality.
 */
export class MockEncoder implements EmbeddingEncoder {
  private static readonly FNV_OFFSET_BASIS = 0x811c9dc5;
  private static readonly FNV_PRIME = 0x01000193;

  private _dimension: number;

  /** `mock-<dimension>`; vectors it writes are stamped with this model. */
  modelName: string;

  /** Optional free-form version, recorded in the sidecar. */
  version?: string;

  constructor(dimension: number = 384, version?: string) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new EmbeddingError(`dimension must be a positive integer, got ${dimension}`);
    }
    this._dimension = dimension;
    this.modelName = `mock-${dimension}`;
    if (version !== undefined) this.version = version;
  }

  get dimension(): number {
    return this._dimension;
  }

  encode(text: string): number[] {
    let state = MockEncoder.FNV_OFFSET_BASIS;
    // Math.imul keeps the multiply exact in 32 bits, matching the other ports.
    const fold = (byte: number): void => {
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
      if (code >= 0xd800 && code <= 0xdfff) code = 0xfffd;  // lone surrogate

      if (code < 0x80) {
        fold(code);
      } else if (code < 0x800) {
        fold(0xc0 | (code >> 6));
        fold(0x80 | (code & 0x3f));
      } else if (code < 0x10000) {
        fold(0xe0 | (code >> 12));
        fold(0x80 | ((code >> 6) & 0x3f));
        fold(0x80 | (code & 0x3f));
      } else {
        fold(0xf0 | (code >> 18));
        fold(0x80 | ((code >> 12) & 0x3f));
        fold(0x80 | ((code >> 6) & 0x3f));
        fold(0x80 | (code & 0x3f));
      }
    }

    // xorshift32 has no way out of a zero state; steer off it.
    if (state === 0) state = 0x9e3779b9;

    const values: number[] = new Array(this._dimension);
    for (let i = 0; i < this._dimension; i++) {
      state ^= (state << 13) >>> 0;
      state >>>= 0;
      state ^= state >>> 17;
      state ^= (state << 5) >>> 0;
      state >>>= 0;
      values[i] = (state >>> 8) / 8388608.0 - 1.0;  // top 24 bits
    }
    return values;
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map(t => this.encode(t));
  }
}

// =============================================================================
// HTTP Encoder
// =============================================================================

/** A request to an embeddings endpoint failed, or its answer was unusable. */
export class HttpEncoderError extends EmbeddingError {
  /** HTTP status, when the server answered at all. */
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'HttpEncoderError';
    if (status !== undefined) this.status = status;
  }
}

/** Wire format of the endpoint: OpenAI `/v1/embeddings` or Ollama `/api/embed`. */
export type HttpEncoderApi = 'openai' | 'ollama';

/** The slice of the Fetch API HttpEncoder uses; `globalThis.fetch` satisfies it. */
export type HttpEncoderFetch = (
  url: string,
  init: { method: string; headers: Record<string, string>; body: string; signal?: AbortSignal }
) => Promise<{
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
}>;

export interface HttpEncoderOptions {
  /** Full endpoint URL, e.g. `https://api.openai.com/v1/embeddings` or `http://localhost:11434/api/embed`. */
  url: string;
  /** Model name sent in the request; also the encoder's `modelName`. */
  model: string;
  /** Request/response shape. Default 'openai' (also what most compatible servers speak). */
  api?: HttpEncoderApi;
  /** Sent as `Authorization: Bearer <apiKey>`. Never included in an error message. */
  apiKey?: string;
  /**
   * Vector length. Optional: when omitted it is learned from the first
   * response (and is `NaN` until then). When given, every response is checked
   * against it.
   */
  dimension?: number;
  /** Texts per request. Default 64. */
  batchSize?: number;
  /** Per-request timeout in milliseconds. Default 30000. */
  timeoutMs?: number;
  /** Extra request headers; these win over the defaults (e.g. to replace `Authorization`). */
  headers?: Record<string, string>;
  /** Free-form model revision, recorded in the sidecar as `encoder_version`. */
  version?: string;
  /** Retries after a 429, a 5xx, a timeout or a network error. Default 2 (so up to 3 attempts). */
  maxRetries?: number;
  /** First retry delay in ms, doubled each attempt (a `Retry-After` header wins). Default 500. */
  retryDelayMs?: number;
  /** Vectors kept for the synchronous `encode`; oldest dropped first. Default 100000. */
  maxCacheEntries?: number;
  /**
   * Requests in flight at once when one call needs several batches
   * (`encodeBatchAsync` of more than `batchSize` texts). Default 1: batches go
   * one after another. Raise it for a server that likes parallel work.
   */
  concurrency?: number;
  /** Replacement for `globalThis.fetch` (a proxy-aware fetch, or a test double). */
  fetch?: HttpEncoderFetch;
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
export class HttpEncoder implements EmbeddingEncoder {
  /** The `model` option; vectors it writes are stamped with this name. */
  modelName: string;

  /** The `version` option, when given. */
  version?: string;

  private _dimension: number | null;
  private _api: HttpEncoderApi;
  private _safeUrl: string;
  private _url: string;
  private _batchSize: number;
  private _timeoutMs: number;
  private _maxRetries: number;
  private _retryDelayMs: number;
  private _maxCache: number;
  private _concurrency: number;
  private _cache: Map<string, number[]> = new Map();
  private _inflight: Map<string, Promise<Map<string, number[]>>> = new Map();
  private _fetch: HttpEncoderFetch;
  // Closures, not fields: they hold the credentials, so logging or
  // serialising the encoder cannot print them.
  private _buildHeaders: () => Record<string, string>;
  private _redact: (text: string) => string;

  constructor(options: HttpEncoderOptions) {
    if (!options || typeof options.url !== 'string' || typeof options.model !== 'string' || !options.model) {
      throw new EmbeddingError('HttpEncoder needs a url and a model');
    }
    let parsed: URL;
    try {
      parsed = new URL(options.url);
    } catch {
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
    const positive = (value: number | undefined, fallback: number, name: string, integer: boolean): number => {
      if (value === undefined) return fallback;
      if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || (integer && !Number.isInteger(value))) {
        throw new EmbeddingError(`HttpEncoder ${name} must be a positive${integer ? ' integer' : ' number'}, got ${value}`);
      }
      return value;
    };
    const nonNegative = (value: number | undefined, fallback: number, name: string): number => {
      if (value === undefined) return fallback;
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
        throw new EmbeddingError(`HttpEncoder ${name} must be a non-negative number, got ${value}`);
      }
      return value;
    };

    this.modelName = options.model;
    if (options.version !== undefined) this.version = options.version;
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

    const fetchFn = options.fetch ?? (globalThis as { fetch?: HttpEncoderFetch }).fetch;
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
      .filter((s): s is string => typeof s === 'string' && s.length > 0);
    this._redact = (text: string) => secrets.reduce((acc, s) => acc.split(s).join('***'), text);
  }

  /** Vector length; `NaN` until it is given or learned from a first response. */
  get dimension(): number {
    return this._dimension ?? NaN;
  }

  /** Number of texts whose vectors are cached for the synchronous `encode`. */
  get cacheSize(): number {
    return this._cache.size;
  }

  /** Forget every cached vector. */
  clearCache(): void {
    this._cache.clear();
  }

  /**
   * Fetch (and cache) the vectors for `texts`, so the synchronous `encode` /
   * `encodeBatch` can serve them. Texts already cached are not requested again.
   * More distinct texts than `maxCacheEntries` evicts the oldest.
   */
  async prefetch(texts: string[]): Promise<void> {
    await this.encodeBatchAsync(texts);
  }

  /** Fetch one vector (served from the cache when present). */
  async encodeAsync(text: string): Promise<number[]> {
    return (await this.encodeBatchAsync([text]))[0];
  }

  /**
   * Fetch vectors for many texts, in order. Only texts that are neither cached
   * nor already being fetched hit the network, `batchSize` per request and
   * `concurrency` requests at a time; the first failure rejects the call.
   */
  async encodeBatchAsync(texts: string[]): Promise<number[][]> {
    const found = new Map<string, number[]>();
    const fresh: string[] = [];
    const waits: Array<Promise<void>> = [];
    const considered = new Set<string>();
    for (const text of texts) {
      if (considered.has(text)) continue;
      considered.add(text);
      const hit = this._cache.get(text);
      if (hit) {
        found.set(text, hit);
        continue;
      }
      const shared = this._inflight.get(text);
      if (shared) {
        waits.push(shared.then(fetched => { found.set(text, fetched.get(text)!); }));
        continue;
      }
      fresh.push(text);
    }

    const chunks: string[][] = [];
    for (let start = 0; start < fresh.length; start += this._batchSize) {
      chunks.push(fresh.slice(start, start + this._batchSize));
    }
    let next = 0;
    let failed = false;
    const worker = async (): Promise<void> => {
      while (!failed && next < chunks.length) {
        const chunk = chunks[next++];
        const job = this._request(chunk).then(vectors => {
          const fetched = new Map<string, number[]>();
          chunk.forEach((text, i) => {
            fetched.set(text, vectors[i]);
            this._remember(text, vectors[i]);
          });
          return fetched;
        });
        for (const text of chunk) this._inflight.set(text, job);
        const release = (): void => {
          for (const text of chunk) if (this._inflight.get(text) === job) this._inflight.delete(text);
        };
        job.then(release, release);
        try {
          for (const [text, vector] of await job) found.set(text, vector);
        } catch (error) {
          failed = true;
          throw error;
        }
      }
    };
    const workers = Array.from({ length: Math.min(this._concurrency, chunks.length) }, worker);
    await Promise.all([...waits, ...workers]);
    return texts.map(text => found.get(text)!.slice());
  }

  /** Synchronous: serves a prefetched text, otherwise throws HttpEncoderError. */
  encode(text: string): number[] {
    const hit = this._cache.get(text);
    if (!hit) {
      throw new HttpEncoderError(
        'HttpEncoder.encode is synchronous and this text was not fetched yet: ' +
        'await encoder.prefetch([...texts]) (or encodeAsync) first'
      );
    }
    return hit.slice();
  }

  /** Synchronous: serves prefetched texts, otherwise throws HttpEncoderError. */
  encodeBatch(texts: string[]): number[][] {
    return texts.map(text => this.encode(text));
  }

  private _remember(text: string, vector: number[]): void {
    this._cache.delete(text);
    this._cache.set(text, vector);
    while (this._cache.size > this._maxCache) {
      const oldest = this._cache.keys().next().value as string;
      this._cache.delete(oldest);
    }
  }

  private _fail(message: string, status?: number): HttpEncoderError {
    return new HttpEncoderError(this._redact(message), status);
  }

  /** One POST (with bounded retries) for at most `batchSize` texts. */
  private async _request(texts: string[]): Promise<number[][]> {
    const body = JSON.stringify({ model: this.modelName, input: texts });

    for (let attempt = 0; ; attempt++) {
      const retriable = attempt < this._maxRetries;
      let status = 0;
      let retryAfter: string | null = null;
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
      } catch (err) {
        const timedOut = controller.signal.aborted;
        if (retriable) {
          await this._sleep(this._backoff(attempt, null));
          continue;
        }
        throw this._fail(
          timedOut
            ? `request to ${this._safeUrl} timed out after ${this._timeoutMs} ms`
            : `request to ${this._safeUrl} failed: ${(err as Error)?.message ?? String(err)}`
        );
      } finally {
        clearTimeout(timer);
      }

      if (status >= 200 && status < 300) return this._parse(text, texts.length);

      if ((status === 429 || status >= 500) && retriable) {
        await this._sleep(this._backoff(attempt, retryAfter));
        continue;
      }
      const snippet = text.trim().slice(0, ERROR_SNIPPET_CHARS);
      throw this._fail(
        `embedding request to ${this._safeUrl} failed with HTTP ${status}` + (snippet ? `: ${snippet}` : ''),
        status
      );
    }
  }

  private _backoff(attempt: number, retryAfter: string | null): number {
    const seconds = retryAfter === null ? NaN : Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds * 1000, MAX_RETRY_WAIT_MS);
    return Math.min(this._retryDelayMs * 2 ** attempt, MAX_RETRY_WAIT_MS);
  }

  private _sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private _parse(text: string, expected: number): number[][] {
    let json: any;
    try {
      json = JSON.parse(text);
    } catch {
      throw this._fail(`${this._safeUrl} did not return JSON`);
    }

    let vectors: unknown[];
    if (this._api === 'ollama') {
      if (!json || !Array.isArray(json.embeddings)) {
        throw this._fail(`${this._safeUrl} answer has no 'embeddings' array (is api: 'ollama' right?)`);
      }
      vectors = json.embeddings;
    } else {
      if (!json || !Array.isArray(json.data)) {
        throw this._fail(`${this._safeUrl} answer has no 'data' array (is api: 'openai' right?)`);
      }
      const items = json.data as Array<{ index?: number; embedding?: unknown }>;
      const ordered = items.every(item => item && typeof item.index === 'number')
        ? [...items].sort((a, b) => a.index! - b.index!)
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
      if (this._dimension === null) this._dimension = vector.length;
      else if (vector.length !== this._dimension) {
        throw new DimensionMismatchError(
          `${this.modelName} returned ${vector.length} dimensions, expected ${this._dimension}`
        );
      }
      return vector as number[];
    });
  }
}

// =============================================================================
// Embedding Store
// =============================================================================

export interface EmbeddingRecord {
  id: string;
  nodeRef: NodeRef;
  vector: number[];
  text: string;
  /** Name of the model that produced the vector ("provided"/"unknown" if not known). */
  model: string;
}

export interface SimilarityResult {
  nodeRef: NodeRef;
  score: number;
  text: string;
}

/** One entry of the portable embedding payload. */
export interface ExportedEmbedding {
  type: string;
  id: number | string;
  id_type: 'int' | 'str';
  text: string;
  /** Model that made the vector; readers default a missing value to "unknown". */
  model?: string;
  vector: number[];
}

/**
 * Payload written by exportEmbeddings, readable by every language port.
 *
 * Version 1 of the payload is unchanged; the optional metadata after
 * `embeddings` is appended as flat top-level fields, which readers that do not
 * know them ignore.
 */
export interface ExportedEmbeddings {
  format: string;
  version: number;
  dimension: number | null;
  count: number;
  embeddings: ExportedEmbedding[];
  /** The encoder's `modelName`, when it has one. */
  encoder_model?: string;
  /** The store/encoder `version`, when known. */
  encoder_version?: string;
  /** Digest of the graph this sidecar was saved with (see graphDigest). */
  graph_digest?: string;
  /** Serialization the digest was taken over (GRAPH_DIGEST_BASIS). */
  graph_digest_basis?: string;
}

/** Metadata fields of an imported sidecar that the store remembers. */
export interface ImportMeta {
  encoder_model?: string;
  encoder_version?: string;
  graph_digest?: string;
  graph_digest_basis?: string;
}

/** What a text query does when the encoder's model is not the stored vectors' model. */
export type ModelMismatchMode = 'warn' | 'error' | 'ignore';

/** Options for EmbeddingStore beyond its encoder. */
export interface EmbeddingStoreOptions extends AnnOptions {
  /**
   * Free-form version of the encoder/model that made these vectors, recorded
   * in the sidecar. Falls back to the encoder's own `version`.
   */
  version?: string;
  /**
   * What a *text* query does when the encoder's `modelName` differs from the
   * model that produced the stored vectors: 'warn' (default; once per distinct
   * pair per store), 'error' (throw ModelMismatchError) or 'ignore'. Vectors
   * whose model is "unknown"/"provided" never trigger it, and vector queries
   * are never checked.
   */
  onModelMismatch?: ModelMismatchMode;
}

/** Options for EmbeddingStore.stale, all optional. */
export interface StaleOptions {
  /** Model to compare against (default: the encoder's modelName). */
  model?: string | null;
  /** Dimension to compare against (default: the encoder's dimension). */
  dimension?: number | null;
  /** Also count vectors recorded as "unknown"/"provided". */
  includeUnknown?: boolean;
}

/**
 * Keep the best `k` results without ordering all of them.
 *
 * Sorting every candidate to return ten is O(n log n) where this is O(n·k)
 * with early rejection — for the small k a search actually asks for, that is
 * a large saving over 20,000 candidates. Insertion goes *after* equal scores,
 * so the order matches what a stable descending sort would produce.
 */
export function selectTopK(results: SimilarityResult[], k: number): SimilarityResult[] {
  if (k <= 0) return [];
  if (results.length <= k) return results.sort((a, b) => b.score - a.score);

  const top: SimilarityResult[] = [];
  for (const candidate of results) {
    if (top.length === k && candidate.score <= top[k - 1].score) continue;
    let i = top.length < k ? top.length : k - 1;
    while (i > 0 && top[i - 1].score < candidate.score) {
      top[i] = top[i - 1];
      i--;
    }
    top[i] = candidate;
    if (top.length > k) top.length = k;
  }
  return top;
}

// =============================================================================
// Approximate (IVF) search
// =============================================================================

/** Stores below this many vectors answer exactly even with `ann` on. */
const ANN_MIN_SIZE = 50_000;
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

/** The approximate index kinds. Only IVF-flat exists. */
export type AnnMode = 'ivf';

/** Approximate-search options shared by EmbeddingStore and SemanticGraph. */
export interface AnnOptions {
  /**
   * `null`/`undefined` (default) keeps search exact, exactly as it always was.
   * `'ivf'` answers bounded top-k searches (`topK` not null) from an in-memory
   * IVF-flat index: vectors are clustered into `annNlist` cells and only the
   * `annNprobe` cells nearest the query are scored, with the same cosine
   * formula as the exact scan. It is *approximate* - a true neighbour in an
   * unprobed cell is missed - so measure it with `recallCheck()`. The index is
   * built lazily on the first search, never persisted (a saved graph and its
   * sidecar are byte-for-byte what they would be without it), and kept
   * current through writes and deletes. It keeps a Float32Array copy of the
   * vectors (4 bytes per component) next to the stored ones.
   */
  ann?: AnnMode | null;
  /** Cells in the index. Default: about `4 * sqrt(N)` (at least 16, at most 65536) when it is built. */
  annNlist?: number;
  /** Cells scanned per query. Default: about 5% of `annNlist`. `annNprobe >= annNlist` is the exact scan. */
  annNprobe?: number;
  /** Below this many vectors the index is not used and search stays exact. Default 50000. */
  annMinSize?: number;
  /** Seed for the k-means initialisation, so the same data gives the same index. Default 0. */
  annSeed?: number;
}

/** What the approximate index looks like right now (see `annStats()`). */
export interface AnnStats {
  enabled: boolean;
  mode: AnnMode | null;
  /** An index exists and will answer. */
  active: boolean;
  nlist: number | null;
  nprobe: number | null;
  /** Vectors the centroids were trained on. */
  trainedN: number | null;
  /** Rows scored exactly on every query until they are folded into cells. */
  pending: number;
  /** Writes + deletes since training. */
  churn: number;
  buildSeconds: number | null;
  /** Memory of the index itself (centroids, cell numbers, cell lists). */
  indexBytes: number;
  /** Memory of the Float32Array copy of the vectors the index scores. */
  cacheBytes: number;
  minSize: number;
  /** Counts since the store was created. */
  searches: { approximate: number; exact: number; widened: number; builds: number };
}

/** Recall of the approximate index against the exact scan (see `recallCheck()`). */
export interface RecallReport {
  /** Mean over queries of |approximate top-k & exact top-k| / |exact top-k|. */
  recall: number;
  /** The worst single query. */
  minRecall: number;
  queries: number;
  k: number;
  nlist: number;
  nprobe: number;
  /** False when the index was not consulted (off, under `annMinSize`, or `annNprobe >= annNlist`); recall is then 1 by construction. */
  approximate: boolean;
  /** Median milliseconds per approximate query / per exact query. */
  annMs: number;
  exactMs: number;
}

/** Banker's rounding, as Python's `round`, so the default cell counts agree across ports. */
function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

const nowMs = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

/** Small seeded generator (mulberry32): the same seed gives the same index. */
class Rng {
  private _a: number;

  constructor(seed: number) {
    this._a = (Math.trunc(seed) ^ 0x9e3779b9) >>> 0;
  }

  next(): number {
    this._a = (this._a + 0x6d2b79f5) >>> 0;
    let t = this._a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [0, n). */
  below(n: number): number {
    return Math.floor(this.next() * n);
  }
}

/** Dot product of a[ao..ao+dim) and b[bo..bo+dim), four accumulators wide. */
function dot2(a: Float32Array, ao: number, b: Float32Array, bo: number, dim: number): number {
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
  for (; j < dim; j++) s0 += a[ao + j] * b[bo + j];
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
  private readonly _nlist: number;
  private readonly _dim: number;
  private readonly _centroids: Float32Array;
  private _coarse: Float32Array | null = null;
  private _ncoarse = 0;
  private _members = new Int32Array(0);
  private _memberOffsets = new Int32Array(0);
  private _coarseScore = new Float64Array(0);
  private _probe: number;
  private _pickIdx = new Int32Array(0);
  private _pickScore = new Float64Array(0);

  constructor(
    centroids: Float32Array,
    nlist: number,
    dim: number,
    coarseFactor: number = ANN_COARSE_FACTOR,
    probe: number = ANN_COARSE_PROBE
  ) {
    this._nlist = nlist;
    this._dim = dim;
    this._centroids = centroids;
    this._probe = probe;
    if (nlist >= ANN_COARSE_MIN_CELLS) this._buildCoarse(coarseFactor);
  }

  private _buildCoarse(coarseFactor: number): void {
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
      if (round === ANN_COARSE_ROUNDS - 1) break;
      sums.fill(0);
      counts.fill(0);
      for (let c = 0; c < nlist; c++) {
        const k = owner[c];
        counts[k]++;
        const base = k * dim;
        const src = c * dim;
        for (let j = 0; j < dim; j++) sums[base + j] += centroids[src + j];
      }
      for (let k = 0; k < ncoarse; k++) {
        if (counts[k] === 0) continue;
        let len = 0;
        for (let j = 0; j < dim; j++) len += sums[k * dim + j] * sums[k * dim + j];
        len = Math.sqrt(len);
        if (len === 0) continue;
        for (let j = 0; j < dim; j++) coarse[k * dim + j] = sums[k * dim + j] / len;
      }
    }
    const memberOffsets = new Int32Array(ncoarse + 1);
    for (let c = 0; c < nlist; c++) memberOffsets[owner[c] + 1]++;
    for (let k = 0; k < ncoarse; k++) memberOffsets[k + 1] += memberOffsets[k];
    const fill = memberOffsets.slice(0, ncoarse);
    const members = new Int32Array(nlist);
    for (let c = 0; c < nlist; c++) members[fill[owner[c]]++] = c;
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
  nearest(v: Float32Array, off: number): number {
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
      if (picked === probe && s <= pickScore[probe - 1]) continue;
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

/**
 * Float32Array snapshot of the store's vectors, kept in step with its writes.
 *
 * One contiguous matrix with spare row capacity, the vector norms, a type code
 * per row and a liveness flag. A write patches it in place: an insert appends
 * a row, an update overwrites one, a delete tombstones one (squeezed out once
 * a quarter of the rows are dead). Rows stay in insertion order, which is the
 * order the store's Map iterates, so ties break exactly as the exact scan's do.
 * Built lazily by the first approximate search, never persisted.
 */
class ScanCache {
  readonly dim: number;
  matrix: Float32Array;
  norms: Float64Array;
  codes: Int32Array;
  alive: Uint8Array;
  /** Rows in use, tombstones included, and how many of those are tombstones. */
  n = 0;
  dead = 0;
  keys: Array<string | null> = [];
  pos: Map<string, number> = new Map();
  codeOf: Map<string, number> = new Map();
  ivf: IvfIndex | null = null;
  /** The query as float32, reused between searches. */
  query: Float32Array;

  constructor(dim: number, capacity: number) {
    this.dim = dim;
    this.matrix = new Float32Array(capacity * dim);
    this.norms = new Float64Array(capacity);
    this.codes = new Int32Array(capacity);
    this.alive = new Uint8Array(capacity);
    this.query = new Float32Array(dim);
  }

  static build(records: Iterable<EmbeddingRecord>, count: number, dim: number): ScanCache {
    const cache = new ScanCache(dim, count + Math.max(1024, count >> 4));
    for (const record of records) cache.put(record.id, record);
    return cache;
  }

  get capacity(): number {
    return this.norms.length;
  }

  private _grow(): void {
    const capacity = this.capacity + Math.max(1024, this.capacity >> 2);
    const matrix = new Float32Array(capacity * this.dim);
    matrix.set(this.matrix.subarray(0, this.n * this.dim));
    const norms = new Float64Array(capacity);
    norms.set(this.norms.subarray(0, this.n));
    const codes = new Int32Array(capacity);
    codes.set(this.codes.subarray(0, this.n));
    const alive = new Uint8Array(capacity);
    alive.set(this.alive.subarray(0, this.n));
    this.matrix = matrix;
    this.norms = norms;
    this.codes = codes;
    this.alive = alive;
  }

  /** Mirror one upserted record. */
  put(key: string, record: EmbeddingRecord): void {
    const dim = this.dim;
    let i = this.pos.get(key);
    if (i === undefined) {
      if (this.n >= this.capacity) this._grow();
      i = this.n++;
      this.keys[i] = key;
      this.pos.set(key, i);
      const type = record.nodeRef[0];
      let code = this.codeOf.get(type);
      if (code === undefined) {
        code = this.codeOf.size;
        this.codeOf.set(type, code);
      }
      this.codes[i] = code;
      this.alive[i] = 1;
    }
    const vector = record.vector;
    const base = i * dim;
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      const f = Math.fround(vector[j]);
      this.matrix[base + j] = f;
      sum += f * f;
    }
    this.norms[i] = Math.sqrt(sum);
    if (this.ivf !== null) this.ivf.notePut(i, this.capacity);
  }

  /** Mirror one deleted record. */
  drop(key: string): void {
    const i = this.pos.get(key);
    if (i === undefined) return;
    this.pos.delete(key);
    this.matrix.fill(0, i * this.dim, (i + 1) * this.dim);
    this.norms[i] = 0;
    this.alive[i] = 0;
    this.keys[i] = null;
    this.dead++;
    if (this.ivf !== null) this.ivf.noteDrop(i);
    while (this.n > 0 && this.alive[this.n - 1] === 0) {      // trailing tombstones
      this.n--;
      this.keys.pop();
      this.dead--;
    }
    if (this.dead > 32 && this.dead * 4 > this.n) this._compact();
  }

  /** Squeeze the tombstones out, keeping the live rows in their order. */
  private _compact(): void {
    const dim = this.dim;
    const live = new Int32Array(this.n - this.dead);
    let kept = 0;
    for (let r = 0; r < this.n; r++) if (this.alive[r] === 1) live[kept++] = r;
    if (this.ivf !== null) this.ivf.compact(live);
    for (let w = 0; w < kept; w++) {
      const r = live[w];
      if (r !== w) {
        this.matrix.copyWithin(w * dim, r * dim, (r + 1) * dim);
        this.norms[w] = this.norms[r];
        this.codes[w] = this.codes[r];
        const key = this.keys[r]!;
        this.keys[w] = key;
        this.pos.set(key, w);
      }
    }
    this.alive.fill(1, 0, kept);
    this.alive.fill(0, kept, this.n);
    this.keys.length = kept;
    this.n = kept;
    this.dead = 0;
  }

  bytes(): number {
    return this.matrix.byteLength + this.norms.byteLength + this.codes.byteLength + this.alive.byteLength;
  }
}

/**
 * Inverted-file ("IVF-flat") index over a {@link ScanCache}.
 *
 * Spherical k-means partitions the rows into `nlist` cells by direction, which
 * is what cosine similarity compares. A query scores the `nprobe` cells whose
 * centroids are nearest and only the rows inside them, with the same cosine
 * formula as the exact scan. The vectors are not copied again: the index is a
 * cell number per row (`assign`) plus the rows grouped by cell
 * (`order`/`offsets`), about 8 bytes per vector plus the centroids.
 *
 * It follows its cache instead of being rebuilt on every write:
 *  - a **new** or **updated** row is *pending* (cell -1): every query scores
 *    the pending rows exactly, so a write is visible at once and costs the
 *    index nothing;
 *  - pending rows are folded into their nearest cell in one pass once there
 *    are enough of them (see `EmbeddingStore._annIndex`);
 *  - a **delete** clears the row's cell; stale entries in `order` are skipped
 *    at query time;
 *  - the cache's compaction renumbers rows, and `compact` carries the cells
 *    across without retraining;
 *  - `churn` counts writes and deletes since training; the store retrains when
 *    it passes a fraction of the trained size.
 */
class IvfIndex {
  readonly nlist: number;
  readonly dim: number;
  readonly centroids: Float32Array;
  assign: Int32Array;
  order = new Int32Array(0);
  offsets = new Int32Array(0);
  pending: Set<number> = new Set();
  churn = 0;
  trainedN = 0;
  buildSeconds = 0;
  private _search: CentroidSearch;
  // Scratch kept between queries: a search allocates nothing proportional to the store.
  private _chosen = new Int32Array(0);
  private _chosenScore = new Float64Array(0);
  candRows = new Int32Array(0);
  candScores = new Float64Array(0);

  constructor(centroids: Float32Array, nlist: number, dim: number, assign: Int32Array) {
    this.nlist = nlist;
    this.dim = dim;
    this.centroids = centroids;
    this.assign = assign;
    this._search = new CentroidSearch(centroids, nlist, dim);
  }

  /** Train centroids on a sample, then give every live row its cell. */
  static build(cache: ScanCache, nlistWanted: number, seed: number): IvfIndex {
    const started = nowMs();
    const { n, dim, matrix, norms, alive } = cache;
    const live = new Int32Array(n - cache.dead);
    let total = 0;
    for (let r = 0; r < n; r++) if (alive[r] === 1) live[total++] = r;
    const nlist = Math.max(1, Math.min(Math.floor(nlistWanted), total));
    const rng = new Rng(seed);

    // Sample up to 40 vectors per cell (a partial Fisher-Yates), as unit rows.
    const sampleN = Math.min(total, ANN_SAMPLE_PER_CELL * nlist);
    let sample: Int32Array;
    if (sampleN === total) {
      sample = live;
    } else {
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
      const src = r * dim;
      const dst = i * dim;
      for (let j = 0; j < dim; j++) x[dst + j] = matrix[src + j] * inv;
    }

    // Initial centroids: distinct sample points.
    const centroids = new Float32Array(nlist * dim);
    const picks = new Int32Array(sampleN);
    for (let i = 0; i < sampleN; i++) picks[i] = i;
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
        for (let j = 0; j < dim; j++) sums[base + j] += x[src + j];
      }
      for (let c = 0; c < nlist; c++) {
        let len = 0;
        if (counts[c] > 0) {
          for (let j = 0; j < dim; j++) len += sums[c * dim + j] * sums[c * dim + j];
          len = Math.sqrt(len);
        }
        if (counts[c] === 0 || len === 0) {                // a dead centroid: re-seed on a random point
          const i = rng.below(sampleN);
          centroids.set(x.subarray(i * dim, (i + 1) * dim), c * dim);
        } else {
          for (let j = 0; j < dim; j++) centroids[c * dim + j] = sums[c * dim + j] / len;
        }
      }
    }

    const index = new IvfIndex(centroids, nlist, dim, new Int32Array(cache.capacity).fill(-1));
    index.trainedN = total;
    for (let t = 0; t < total; t++) {
      const r = live[t];
      index.assign[r] = index._search.nearest(matrix, r * dim);
    }
    index.relist(n);
    index.buildSeconds = (nowMs() - started) / 1000;
    return index;
  }

  /** Regroup the rows by cell (a stable counting sort of `assign`). */
  relist(n: number): void {
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
    for (let c = 0; c < this.nlist; c++) offsets[c + 1] += offsets[c];
    const fill = offsets.slice(0, this.nlist);
    const order = new Int32Array(total);
    for (let r = 0; r < n; r++) {
      const c = assign[r];
      if (c >= 0) order[fill[c]++] = r;
    }
    this.order = order;
    this.offsets = offsets;
  }

  /** Give every pending row its nearest cell and regroup. */
  fold(cache: ScanCache): void {
    if (this.pending.size > 0) {
      for (const r of this.pending) {
        this.assign[r] = this._search.nearest(cache.matrix, r * this.dim);
      }
      this.pending.clear();
    }
    this.relist(cache.n);
  }

  notePut(i: number, capacity: number): void {
    if (i >= this.assign.length) {
      const grown = new Int32Array(Math.max(capacity, i + 1)).fill(-1);
      grown.set(this.assign);
      this.assign = grown;
    }
    this.assign[i] = -1;
    this.pending.add(i);
    this.churn++;
  }

  noteDrop(i: number): void {
    if (i < this.assign.length) this.assign[i] = -1;
    this.pending.delete(i);
    this.churn++;
  }

  /** Follow the cache's compaction (`live` = the old rows kept, in order). */
  compact(live: Int32Array): void {
    const kept = live.length;
    const moved = new Int32Array(kept);
    for (let w = 0; w < kept; w++) moved[w] = this.assign[live[w]];
    this.assign.fill(-1);
    this.assign.set(moved);
    if (this.pending.size > 0) {
      const old = Array.from(this.pending).sort((a, b) => a - b);
      this.pending = new Set(old.map(r => lowerBound(live, r)));
    }
    this.relist(kept);
  }

  bytes(): number {
    return this.centroids.byteLength + this.assign.byteLength + this.order.byteLength + this.offsets.byteLength;
  }

  /**
   * Score the rows of the `lists` cells nearest `q`, plus every pending row,
   * into `candRows`/`candScores`; returns how many were kept. `code` < 0 means
   * any node type; rows scoring under `floor` are left out.
   */
  gather(cache: ScanCache, q: Float32Array, qnorm: number, code: number, floor: number, lists: number): number {
    const { nlist, dim, centroids } = this;
    if (this._chosen.length < lists) {
      this._chosen = new Int32Array(lists);
      this._chosenScore = new Float64Array(lists);
    }
    const chosen = this._chosen;
    let nChosen = 0;
    if (lists >= nlist) {
      for (let c = 0; c < nlist; c++) chosen[c] = c;
      nChosen = nlist;
    } else {
      const chosenScore = this._chosenScore;
      for (let c = 0; c < nlist; c++) {
        const s = dot2(centroids, c * dim, q, 0, dim);
        if (nChosen === lists && s <= chosenScore[lists - 1]) continue;
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
    for (let k = 0; k < nChosen; k++) bound += offsets[chosen[k] + 1] - offsets[chosen[k]];
    if (this.candRows.length < bound) {
      const size = Math.max(bound, this.candRows.length * 2, 256);
      this.candRows = new Int32Array(size);
      this.candScores = new Float64Array(size);
    }
    const candRows = this.candRows;
    const candScores = this.candScores;
    const { matrix, norms, codes } = cache;
    const assign = this.assign;

    let m = 0;
    for (let k = 0; k < nChosen; k++) {
      const c = chosen[k];
      for (let p = offsets[c], end = offsets[c + 1]; p < end; p++) {
        const r = order[p];
        if (assign[r] < 0) continue;                       // deleted or updated since the last regroup
        if (code >= 0 && codes[r] !== code) continue;
        const denom = norms[r] * qnorm;
        const s = denom > 0 ? dot2(matrix, r * dim, q, 0, dim) / denom : 0;
        if (s < floor) continue;
        candRows[m] = r;
        candScores[m] = s;
        m++;
      }
    }
    for (const r of this.pending) {
      if (code >= 0 && codes[r] !== code) continue;
      const denom = norms[r] * qnorm;
      const s = denom > 0 ? dot2(matrix, r * dim, q, 0, dim) / denom : 0;
      if (s < floor) continue;
      candRows[m] = r;
      candScores[m] = s;
      m++;
    }
    return m;
  }

  /** The `k` best of the first `m` candidates: score descending, row ascending on ties. */
  best(m: number, k: number): Int32Array {
    const rows = this.candRows;
    const scores = this.candScores;
    const size = Math.min(k, m);
    // Min-heap of candidate slots whose root is the worst kept one.
    const heap = new Int32Array(size);
    const worse = (a: number, b: number): boolean =>
      scores[a] < scores[b] || (scores[a] === scores[b] && rows[a] > rows[b]);
    let count = 0;
    for (let i = 0; i < m; i++) {
      if (count === size) {
        const root = heap[0];
        if (!(scores[i] > scores[root] || (scores[i] === scores[root] && rows[i] < rows[root]))) continue;
        heap[0] = i;
        let at = 0;
        for (;;) {
          const l = 2 * at + 1;
          if (l >= size) break;
          const r = l + 1;
          const child = r < size && worse(heap[r], heap[l]) ? r : l;
          if (!worse(heap[child], heap[at])) break;
          const t = heap[at];
          heap[at] = heap[child];
          heap[child] = t;
          at = child;
        }
      } else {
        let at = count++;
        heap[at] = i;
        while (at > 0) {
          const parent = (at - 1) >> 1;
          if (!worse(heap[at], heap[parent])) break;
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
      if (o === 0) break;
      heap[0] = last;
      let at = 0;
      for (;;) {
        const l = 2 * at + 1;
        if (l >= o) break;
        const r = l + 1;
        const child = r < o && worse(heap[r], heap[l]) ? r : l;
        if (!worse(heap[child], heap[at])) break;
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
function lowerBound(values: Int32Array, target: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
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
  private _embeddings: Map<string, EmbeddingRecord> = new Map();
  private _encoder: EmbeddingEncoder | null;
  private _dimension: number | null = null;
  private _version: string | null;
  private _onModelMismatch: ModelMismatchMode = 'warn';
  private _warnedMismatch: Set<string> = new Set();
  private _ann: AnnMode | null = null;
  private _annNlist: number | null = null;
  private _annNprobe: number | null = null;
  private _annMinSize: number = ANN_MIN_SIZE;
  private _annSeed = 0;
  private _annCounts = { approximate: 0, exact: 0, widened: 0, builds: 0 };
  private _scan: ScanCache | null = null;

  /** Metadata of the most recent imported sidecar (empty before any). */
  lastImportMeta: ImportMeta = {};

  constructor(encoder?: EmbeddingEncoder | AsyncEncoder, options: EmbeddingStoreOptions = {}) {
    this._encoder = adoptEncoder(encoder);
    this._version = options.version ?? null;
    this.onModelMismatch = options.onModelMismatch ?? 'warn';

    const ann = options.ann ?? null;
    if (ann !== null && ann !== 'ivf') {
      throw new EmbeddingError(`ann must be null or 'ivf', got ${JSON.stringify(ann)}`);
    }
    const whole = (label: string, value: number | null | undefined, floor: number): number | null => {
      if (value === undefined || value === null) return null;
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
  }

  /** The approximate index in use (`'ivf'`), or null when search is exact. */
  get ann(): AnnMode | null {
    return this._ann;
  }

  /** Cells scanned per query; null means "automatic" (about 5% of the cells). */
  get annNprobe(): number | null {
    return this._annNprobe;
  }

  set annNprobe(value: number | null) {
    if (value !== null && (typeof value !== 'number' || !Number.isInteger(value) || value < 1)) {
      throw new EmbeddingError(`annNprobe must be an integer >= 1, got ${value}`);
    }
    this._annNprobe = value;
  }

  get encoder(): EmbeddingEncoder | null {
    return this._encoder;
  }

  set encoder(encoder: EmbeddingEncoder | AsyncEncoder | null) {
    this._encoder = adoptEncoder(encoder);
  }

  /** Vector length this store holds, or null while it is still empty. */
  get dimension(): number | null {
    return this._dimension;
  }

  /** Encoder/model version recorded in the sidecar, or null if unknown. */
  get version(): string | null {
    if (this._version !== null) return this._version;
    const fromEncoder = this._encoder?.version;
    return fromEncoder === undefined || fromEncoder === null ? null : fromEncoder;
  }

  set version(value: string | null) {
    this._version = value;
  }

  /** How a text query reacts to vectors from another model (see EmbeddingStoreOptions). */
  get onModelMismatch(): ModelMismatchMode {
    return this._onModelMismatch;
  }

  set onModelMismatch(mode: ModelMismatchMode) {
    if (mode !== 'warn' && mode !== 'error' && mode !== 'ignore') {
      throw new EmbeddingError(
        `onModelMismatch must be 'warn', 'error' or 'ignore', got ${JSON.stringify(mode)}`
      );
    }
    this._onModelMismatch = mode;
  }

  private _nodeRefToKey(ref: NodeRef): string {
    return `${ref[0]}:${ref[1]}`;
  }

  private _checkDimension(vector: number[], what: string): void {
    if (vector.length === 0) {
      throw new DimensionMismatchError(`${what} is empty`);
    }
    if (this._dimension === null) {
      this._dimension = vector.length;
    } else if (vector.length !== this._dimension) {
      throw new DimensionMismatchError(
        `${what} has ${vector.length} dimensions, store holds ${this._dimension}`
      );
    }
  }

  private _cosineSimilarity(v1: number[], v2: number[]): number {
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

    if (norm1 === 0 || norm2 === 0) return 0;
    return dot / (norm1 * norm2);
  }

  private _resolveQuery(query: string | number[]): number[] {
    let vector: number[];
    if (typeof query === 'string') {
      if (!this._encoder) {
        throw new NoEncoderError("No encoder provided for text query");
      }
      this._checkModelCompat();
      vector = this._encoder.encode(query);
    } else {
      vector = query;
    }
    return this._checkQuery(vector);
  }

  private _checkQuery(vector: number[]): number[] {
    if (vector.length === 0) {
      throw new DimensionMismatchError("query vector is empty");
    }
    if (this._dimension !== null && vector.length !== this._dimension) {
      throw new DimensionMismatchError(
        `query vector has ${vector.length} dimensions, store holds ${this._dimension}`
      );
    }
    return vector;
  }

  /** Model recorded on vectors written now: the encoder's, or "provided"/"unknown". */
  private _modelName(): string {
    if (!this._encoder) return 'provided';
    return this._encoder.modelName || 'unknown';
  }

  /** The encoder's model, or null when it has none (or an uninformative one). */
  private _currentModel(): string | null {
    const name = this._encoder?.modelName;
    return name && !UNKNOWN_MODELS.has(name) ? name : null;
  }

  /**
   * Which models produced the stored vectors: `{ modelName: count }`.
   *
   * "unknown" / "provided" appear as-is (vectors imported from a port that
   * records no model, or handed in with no encoder attached).
   */
  models(): Record<string, number> {
    const counts = new Map<string, number>();
    for (const record of this._embeddings.values()) {
      counts.set(record.model, (counts.get(record.model) ?? 0) + 1);
    }
    return Object.fromEntries(counts);
  }

  /** Complain when a text query's encoder is not the stored vectors' model. */
  private _checkModelCompat(): void {
    if (this._onModelMismatch === 'ignore') return;
    const current = this._currentModel();
    if (current === null) return;
    const foreign = Object.keys(this.models())
      .filter(m => !UNKNOWN_MODELS.has(m) && m !== current)
      .sort();
    if (foreign.length === 0) return;

    const message =
      `encoder model '${current}' differs from the model that produced the ` +
      `stored vectors (${foreign.map(m => `'${m}'`).join(', ')}); ` +
      `scores compare vectors from different spaces. Re-embed with ` +
      `reembedStale() or search with the original encoder.`;
    if (this._onModelMismatch === 'error') throw new ModelMismatchError(message);

    const key = JSON.stringify([current, foreign]);
    if (!this._warnedMismatch.has(key)) {
      this._warnedMismatch.add(key);
      if (typeof console !== 'undefined') console.warn(message);
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
  stale(
    modelOrOptions?: string | null | StaleOptions,
    dimensionArg?: number | null,
    includeUnknownArg: boolean = false
  ): NodeRef[] {
    const options: StaleOptions =
      modelOrOptions !== null && typeof modelOrOptions === 'object'
        ? modelOrOptions
        : { model: modelOrOptions, dimension: dimensionArg, includeUnknown: includeUnknownArg };

    const model = options.model ?? this._currentModel();
    let dimension = options.dimension ?? null;
    if (dimension === null && this._encoder) {
      const d = Number(this._encoder.dimension);
      dimension = Number.isFinite(d) ? Math.trunc(d) : null;
    }
    const includeUnknown = options.includeUnknown ?? false;

    const out: NodeRef[] = [];
    for (const record of this._embeddings.values()) {
      const byModel =
        model !== null &&
        record.model !== model &&
        (includeUnknown || !UNKNOWN_MODELS.has(record.model));
      const byDim = dimension !== null && record.vector.length !== dimension;
      if (byModel || byDim) out.push(record.nodeRef);
    }
    return out;
  }

  add(nodeRef: NodeRef, text: string, vector?: number[]): string {
    if (vector === undefined) {
      if (!this._encoder) {
        throw new NoEncoderError("No encoder provided and no vector given");
      }
      vector = this._encoder.encode(text);
    }
    this._checkDimension(vector, "vector");

    const key = this._nodeRefToKey(nodeRef);
    this._set(key, { id: key, nodeRef, vector, text, model: this._modelName() });
    return key;
  }

  /** Every write goes through here, so the approximate index's cache stays in step. */
  private _set(key: string, record: EmbeddingRecord): void {
    this._embeddings.set(key, record);
    if (this._scan !== null) this._scan.put(key, record);
  }

  /** Add many embeddings, encoding their texts in one batched call. */
  addBatch(entries: Array<[NodeRef, string] | [NodeRef, string, number[]]>): number {
    if (entries.length === 0) return 0;

    const needsEncoding = entries
      .map((e, i) => (e.length < 3 || e[2] === undefined ? i : -1))
      .filter(i => i >= 0);

    let encoded: number[][] = [];
    if (needsEncoding.length > 0) {
      if (!this._encoder) {
        throw new NoEncoderError("No encoder provided and no vectors given");
      }
      encoded = this._encoder.encodeBatch(needsEncoding.map(i => entries[i][1]));
      if (encoded.length !== needsEncoding.length) {
        throw new EmbeddingError(
          `encoder returned ${encoded.length} vectors for ${needsEncoding.length} texts`
        );
      }
    }

    const vectorFor = new Map<number, number[]>();
    needsEncoding.forEach((slot, i) => vectorFor.set(slot, encoded[i]));

    entries.forEach((entry, i) => {
      const vector = vectorFor.get(i) ?? (entry[2] as number[]);
      this.add(entry[0], entry[1], vector);
    });
    return entries.length;
  }

  /**
   * Encode texts with the store's encoder, awaiting it: an {@link AsyncEncoder}
   * (such as HttpEncoder) over the network, any other encoder as `encodeBatch`.
   * Nothing is stored. Rejects with NoEncoderError when there is no encoder.
   */
  async encodeTextsAsync(texts: string[]): Promise<number[][]> {
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
  async addAsync(nodeRef: NodeRef, text: string, vector?: number[]): Promise<string> {
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
  async addBatchAsync(entries: Array<[NodeRef, string] | [NodeRef, string, number[]]>): Promise<number> {
    if (entries.length === 0) return 0;
    const needsEncoding = entries
      .map((e, i) => (e.length < 3 || e[2] === undefined ? i : -1))
      .filter(i => i >= 0);

    const vectorFor = new Map<number, number[]>();
    if (needsEncoding.length > 0) {
      const encoded = await this.encodeTextsAsync(needsEncoding.map(i => entries[i][1]));
      needsEncoding.forEach((slot, i) => vectorFor.set(slot, encoded[i]));
    }
    entries.forEach((entry, i) => {
      this.add(entry[0], entry[1], vectorFor.get(i) ?? (entry[2] as number[]));
    });
    return entries.length;
  }

  /**
   * The vector a query stands for: a vector as it is, a text through the
   * encoder (awaited), with the same model-mismatch check as a synchronous
   * text query. Used by the `...Async` searches.
   */
  async resolveQueryAsync(query: string | number[]): Promise<number[]> {
    if (typeof query !== 'string') return this._checkQuery(query);
    if (!this._encoder) {
      throw new NoEncoderError("No encoder provided for text query");
    }
    this._checkModelCompat();
    return this._checkQuery((await this.encodeTextsAsync([query]))[0]);
  }

  get(nodeRef: NodeRef): EmbeddingRecord | null {
    const key = this._nodeRefToKey(nodeRef);
    return this._embeddings.get(key) || null;
  }

  remove(nodeRef: NodeRef): boolean {
    const key = this._nodeRefToKey(nodeRef);
    const removed = this._embeddings.delete(key);
    if (removed && this._scan !== null) this._scan.drop(key);
    return removed;
  }

  /**
   * Find the most similar nodes by embedding.
   *
   * Exact by default. With `ann: 'ivf'`, a bounded `topK` on a store of at
   * least `annMinSize` vectors is answered from the approximate index (the
   * scores are the same cosine values, but a true neighbour in an unprobed
   * cell can be missing); `exact: true` forces the full scan, and has no effect
   * on a store without `ann`. `topK: null` always scans everything.
   */
  similaritySearch(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0,
    exact: boolean = false
  ): SimilarityResult[] {
    const queryVector = this._resolveQuery(query);
    if (this._ann !== null) {
      if (exact || topK === null || topK <= 0) {
        this._annCounts.exact++;
      } else {
        const answer = this._annSearch(queryVector, topK, nodeType, threshold);
        if (answer !== null) return answer;
        this._annCounts.exact++;
      }
    }
    return this._scanExact(queryVector, topK, nodeType, threshold);
  }

  /** The brute-force scan every search used before the approximate index existed. */
  private _scanExact(
    queryVector: number[],
    topK: number | null,
    nodeType: string | undefined,
    threshold: number
  ): SimilarityResult[] {
    const results: SimilarityResult[] = [];

    for (const record of this._embeddings.values()) {
      if (nodeType && record.nodeRef[0] !== nodeType) continue;

      const score = this._cosineSimilarity(queryVector, record.vector);
      if (score >= threshold) {
        results.push({ nodeRef: record.nodeRef, score, text: record.text });
      }
    }

    if (topK === null) return results.sort((a, b) => b.score - a.score);
    return selectTopK(results, topK);
  }

  /** {@link similaritySearch} with a text query encoded by awaiting the encoder. */
  async similaritySearchAsync(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0,
    exact: boolean = false
  ): Promise<SimilarityResult[]> {
    return this.similaritySearch(await this.resolveQueryAsync(query), topK, nodeType, threshold, exact);
  }

  /** The vector a query stands for (a text goes through the synchronous encoder). */
  resolveQuery(query: string | number[]): number[] {
    return this._resolveQuery(query);
  }

  // -- approximate (IVF) search -------------------------------------------------

  /** Cells for a store of `n` vectors: about 4 * sqrt(n), within [16, 65536]. */
  private static _annAutoNlist(n: number): number {
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
   *    `annMinSize` drops the index with its cache.
   */
  private _annIndex(): ScanCache | null {
    if (this._ann === null || this._dimension === null) return null;
    const live = this._embeddings.size;
    if (live < Math.max(2, this._annMinSize)) {
      this._scan = null;
      return null;
    }
    let scan = this._scan;
    if (scan === null) {
      scan = this._scan = ScanCache.build(this._embeddings.values(), live, this._dimension);
    }
    let ivf = scan.ivf;
    if (ivf !== null && ivf.churn > ANN_RETRAIN_FRACTION * ivf.trainedN) ivf = scan.ivf = null;
    if (ivf === null) {
      const nlist = this._annNlist ?? EmbeddingStore._annAutoNlist(live);
      scan.ivf = IvfIndex.build(scan, nlist, this._annSeed);
      this._annCounts.builds++;
    } else if (ivf.pending.size >= Math.max(ANN_FOLD_MIN, ANN_FOLD_FRACTION * ivf.trainedN)) {
      ivf.fold(scan);
    }
    return scan;
  }

  private _annProbe(ivf: IvfIndex): number {
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
  private _annLists(ivf: IvfIndex, scale: number): number | null {
    const lists = this._annProbe(ivf) * scale;
    if (scale === 1) return lists >= ivf.nlist ? null : lists;
    return lists * 2 >= ivf.nlist ? null : lists;
  }

  /** Load the query into the cache's float32 buffer; null when it has no direction (all zeros). */
  private _annQuery(scan: ScanCache, queryVector: number[], nodeType: string | undefined): { qnorm: number; code: number } | null {
    const q = scan.query;
    let sum = 0;
    for (let j = 0; j < scan.dim; j++) {
      const f = Math.fround(queryVector[j]);
      q[j] = f;
      sum += f * f;
    }
    const qnorm = Math.sqrt(sum);
    if (qnorm === 0) return null;
    let code = -1;
    if (nodeType) code = scan.codeOf.get(nodeType) ?? -2;        // -2: no node of that type exists
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
  private _annSearch(
    queryVector: number[],
    topK: number,
    nodeType: string | undefined,
    threshold: number
  ): SimilarityResult[] | null {
    const scan = this._annIndex();
    if (scan === null) return null;
    const ivf = scan.ivf!;
    const prepared = this._annQuery(scan, queryVector, nodeType);
    if (prepared === null) return null;
    if (prepared.code === -2) return [];
    const floor = threshold - ANN_THRESHOLD_SLACK;
    for (let scale = 1; ; scale *= 4) {
      const lists = this._annLists(ivf, scale);
      if (lists === null) return null;
      const m = ivf.gather(scan, scan.query, prepared.qnorm, prepared.code, floor, lists);
      if (m < topK) continue;
      const best = ivf.best(m, topK);
      const results: SimilarityResult[] = [];
      for (let i = 0; i < best.length; i++) {
        const record = this._embeddings.get(scan.keys[ivf.candRows[best[i]]]!)!;
        results.push({ nodeRef: record.nodeRef, score: ivf.candScores[best[i]], text: record.text });
      }
      this._annCounts.approximate++;
      if (scale > 1) this._annCounts.widened++;
      return this.annPolish(results, queryVector, threshold);
    }
  }

  /**
   * Re-score approximate results with the exact float64 cosine the full scan
   * uses (the index scores float32 copies), drop any that fall under
   * `threshold` once exact, and order them best first.
   */
  annPolish(results: SimilarityResult[], queryVector: number[], threshold: number): SimilarityResult[] {
    const out: SimilarityResult[] = [];
    for (const result of results) {
      const record = this._embeddings.get(this._nodeRefToKey(result.nodeRef));
      if (!record) continue;
      const score = this._cosineSimilarity(queryVector, record.vector);
      if (score >= threshold) out.push({ nodeRef: result.nodeRef, score, text: result.text });
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
  annRanked(
    queryVector: number[],
    nodeType: string | undefined,
    threshold: number,
    scale: number | null
  ): { ranked: Iterable<SimilarityResult>; approximate: boolean } {
    if (scale !== null && this._ann !== null) {
      const scan = this._annIndex();
      const ivf = scan === null ? null : scan.ivf;
      const lists = ivf === null ? null : this._annLists(ivf, scale);
      const prepared = scan === null || lists === null ? null : this._annQuery(scan, queryVector, nodeType);
      if (scan !== null && ivf !== null && lists !== null && prepared !== null) {
        if (prepared.code === -2) return { ranked: [], approximate: false };
        const m = ivf.gather(scan, scan.query, prepared.qnorm, prepared.code, threshold - ANN_THRESHOLD_SLACK, lists);
        this._annCounts.approximate++;
        if (scale > 1) this._annCounts.widened++;
        const rows = ivf.candRows.slice(0, m);
        const scores = ivf.candScores.slice(0, m);
        const slots = new Int32Array(m);
        for (let i = 0; i < m; i++) slots[i] = i;
        slots.sort((a, b) => scores[b] - scores[a] || rows[a] - rows[b]);
        const keys = scan.keys;
        const embeddings = this._embeddings;
        const ranked = (function* (): Generator<SimilarityResult> {
          for (let i = 0; i < m; i++) {
            const record = embeddings.get(keys[rows[slots[i]]]!)!;
            yield { nodeRef: record.nodeRef, score: scores[slots[i]], text: record.text };
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
  buildAnnIndex(force: boolean = false): AnnStats {
    if (this._ann === null) {
      throw new EmbeddingError("this store has no approximate index; create it with ann: 'ivf'");
    }
    if (force && this._scan !== null) this._scan.ivf = null;
    const scan = this._annIndex();
    if (scan !== null && scan.ivf !== null && scan.ivf.pending.size > 0) scan.ivf.fold(scan);
    return this.annStats();
  }

  /** What the approximate index looks like right now. Never builds anything. */
  annStats(): AnnStats {
    const scan = this._scan;
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
      cacheBytes: scan === null ? 0 : scan.bytes(),
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
  recallCheck(
    queries: string | ArrayLike<number> | Array<string | ArrayLike<number>>,
    k: number = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): RecallReport {
    if (!Number.isInteger(k) || k <= 0) throw new EmbeddingError(`k must be positive, got ${k}`);
    const vectors = EmbeddingStore._queryList(queries).map(q => this._resolveQuery(q));
    return this._recall(vectors, k, nodeType, threshold);
  }

  /** {@link recallCheck} with text queries encoded by awaiting the encoder. */
  async recallCheckAsync(
    queries: string | ArrayLike<number> | Array<string | ArrayLike<number>>,
    k: number = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): Promise<RecallReport> {
    if (!Number.isInteger(k) || k <= 0) throw new EmbeddingError(`k must be positive, got ${k}`);
    const vectors: number[][] = [];
    for (const query of EmbeddingStore._queryList(queries)) vectors.push(await this.resolveQueryAsync(query));
    return this._recall(vectors, k, nodeType, threshold);
  }

  /** One query or many, as a list of texts and plain number arrays. */
  private static _queryList(
    queries: string | ArrayLike<number> | Array<string | ArrayLike<number>>
  ): Array<string | number[]> {
    if (typeof queries === 'string') return [queries];
    const list = Array.from(queries as ArrayLike<unknown>);
    if (list.length > 0 && typeof list[0] === 'number') return [list as number[]];
    return list.map(q => (typeof q === 'string' ? q : Array.from(q as ArrayLike<number>)));
  }

  private _recall(vectors: number[][], k: number, nodeType: string | undefined, threshold: number): RecallReport {
    const recalls: number[] = [];
    const annTimes: number[] = [];
    const exactTimes: number[] = [];
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
    const median = (values: number[]): number => {
      if (values.length === 0) return 0;
      const ordered = [...values].sort((a, b) => a - b);
      const mid = ordered.length >> 1;
      return ordered.length % 2 ? ordered[mid] : 0.5 * (ordered[mid - 1] + ordered[mid]);
    };
    const ivf = this._scan === null ? null : this._scan.ivf;
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
    };
  }

  /**
   * Cosine score for every embedded node, keyed by `type:id`.
   *
   * One scan serving both seed selection and the per-node relevance that
   * semanticMultiHop blends in, instead of two.
   */
  scoreMap(query: string | number[], nodeType?: string): Map<string, number> {
    const queryVector = this._resolveQuery(query);
    const scores = new Map<string, number>();
    for (const [key, record] of this._embeddings) {
      if (nodeType && record.nodeRef[0] !== nodeType) continue;
      scores.set(key, this._cosineSimilarity(queryVector, record.vector));
    }
    return scores;
  }

  count(): number {
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
  clear(resetDimension: boolean = false): void {
    this._embeddings.clear();
    this._scan = null;
    if (resetDimension) this._dimension = null;
  }

  /**
   * Serialize every embedding into the portable cross-language payload.
   *
   * Beyond the version-1 fields it appends, when known: `encoder_model` (the
   * encoder's modelName), `encoder_version` (this store's `version`) and, when
   * `graphDigest` is given, `graph_digest` plus `graph_digest_basis`. Absent
   * values are omitted, not written as null.
   */
  exportEmbeddings(graphDigest?: string | null): ExportedEmbeddings {
    const embeddings: ExportedEmbedding[] = [];
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
    const payload: ExportedEmbeddings = {
      format: EMBEDDING_FORMAT,
      version: EMBEDDING_FORMAT_VERSION,
      dimension: this._dimension,
      count: embeddings.length,
      embeddings,
    };
    const encoderModel = this._currentModel();
    if (encoderModel !== null) payload.encoder_model = encoderModel;
    const version = this.version;
    if (version !== null) payload.encoder_version = String(version);
    if (graphDigest !== undefined && graphDigest !== null) {
      payload.graph_digest = graphDigest;
      payload.graph_digest_basis = GRAPH_DIGEST_BASIS;
    }
    return payload;
  }

  /** Load a payload produced by exportEmbeddings (in any language port). */
  importEmbeddings(payload: ExportedEmbeddings, replace: boolean = false): number {
    if (!payload || payload.format !== EMBEDDING_FORMAT) {
      throw new EmbeddingError(
        `not an ${EMBEDDING_FORMAT} payload: format=${payload && payload.format}`
      );
    }
    if (payload.version !== EMBEDDING_FORMAT_VERSION) {
      throw new EmbeddingError(
        `unsupported ${EMBEDDING_FORMAT} version ${payload.version} ` +
        `(this build reads version ${EMBEDDING_FORMAT_VERSION})`
      );
    }

    const meta: ImportMeta = {};
    for (const key of ['encoder_model', 'encoder_version', 'graph_digest', 'graph_digest_basis'] as const) {
      if (payload[key] !== undefined && payload[key] !== null) meta[key] = payload[key];
    }
    this.lastImportMeta = meta;

    if (replace) {
      this.clear();
      this._dimension = payload.dimension ?? null;
    }

    let written = 0;
    for (const item of payload.embeddings || []) {
      const id = item.id_type === 'int' ? Number(item.id) : item.id;
      const ref = [item.type, id] as NodeRef;
      this._checkDimension(item.vector, `embedding for :${item.type}:${id}`);
      const key = this._nodeRefToKey(ref);
      this._set(key, {
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
function sha256Hex(ascii: string): string {
  const primes: number[] = [];
  for (let n = 2; primes.length < 64; n++) {
    if (primes.every(p => n % p !== 0)) primes.push(n);
  }
  const frac32 = (x: number): number => Math.floor((x - Math.floor(x)) * 4294967296) >>> 0;
  const K = primes.map(p => frac32(Math.cbrt(p)));
  const H = primes.slice(0, 8).map(p => frac32(Math.sqrt(p)));

  const bytes: number[] = [];
  for (let i = 0; i < ascii.length; i++) bytes.push(ascii.charCodeAt(i) & 0xff);
  const bitLength = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  const high = Math.floor(bitLength / 4294967296);
  const low = bitLength >>> 0;
  for (const word of [high, low]) {
    bytes.push((word >>> 24) & 0xff, (word >>> 16) & 0xff, (word >>> 8) & 0xff, word & 0xff);
  }

  const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));
  const w = new Array<number>(64);
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
      h = g; g = f; f = e; e = (d + t1) | 0;
      d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    const sums = [a, b, c, d, e, f, g, h];
    for (let i = 0; i < 8; i++) H[i] = (H[i] + sums[i]) | 0;
  }
  return H.map(x => (x >>> 0).toString(16).padStart(8, '0')).join('');
}

/** Compare strings by code point, as Python does (JS `<` compares UTF-16 code units). */
function compareCodePoints(a: string, b: string): number {
  if (a === b) return 0;
  const A = Array.from(a);
  const B = Array.from(b);
  const n = Math.min(A.length, B.length);
  for (let i = 0; i < n; i++) {
    const x = A[i].codePointAt(0)!;
    const y = B[i].codePointAt(0)!;
    if (x !== y) return x < y ? -1 : 1;
  }
  return A.length - B.length;
}

/** Python `repr(float)` for a non-integral (or non-finite) JS number. */
function pythonFloatRepr(x: number): string {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return 'Infinity';
  if (x === -Infinity) return '-Infinity';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  // toExponential() with no argument yields the shortest digits that round-trip,
  // the same digits Python's repr picks; only the layout differs.
  const m = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(x.toExponential())!;
  const sign = m[1];
  const digits = m[2] + (m[3] ?? '');
  const e = parseInt(m[4], 10);
  if (e < -4 || e >= 16) {
    const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const abs = Math.abs(e);
    return `${sign}${mantissa}e${e < 0 ? '-' : '+'}${abs < 10 ? '0' + abs : abs}`;
  }
  if (e >= 0) {
    if (digits.length <= e + 1) return `${sign}${digits}${'0'.repeat(e + 1 - digits.length)}.0`;
    return `${sign}${digits.slice(0, e + 1)}.${digits.slice(e + 1)}`;
  }
  return `${sign}0.${'0'.repeat(-e - 1)}${digits}`;
}

/** Python `json.dumps(str, ensure_ascii=True)`. */
function pythonJsonString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x22) out += '\\"';
    else if (c === 0x5c) out += '\\\\';
    else if (c === 0x0a) out += '\\n';
    else if (c === 0x0d) out += '\\r';
    else if (c === 0x09) out += '\\t';
    else if (c === 0x08) out += '\\b';
    else if (c === 0x0c) out += '\\f';
    else if (c < 0x20 || c > 0x7e) out += '\\u' + c.toString(16).padStart(4, '0');
    else out += s[i];
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
function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return 'null';
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
      if (Array.isArray(value)) return '[' + value.map(canonicalJson).join(',') + ']';
      const record = value as Record<string, unknown>;
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
export function graphDigest(graph: ISONGraph): string {
  const refKey = (ref: NodeRef): unknown[] => {
    const id = ref[1];
    const isInt = typeof id === 'number' && Number.isInteger(id);
    return [ref[0], isInt ? 'i' : 's', isInt ? BigInt(id).toString() : String(id)];
  };

  const nodes: string[] = [];
  for (const node of graph.nodes()) {
    nodes.push(canonicalJson([refKey(node.ref), node.properties]));
  }
  const edges: string[] = [];
  for (const edge of graph.edges()) {
    edges.push(canonicalJson([edge.relType, refKey(edge.source), refKey(edge.target), edge.properties]));
  }
  nodes.sort();
  edges.sort();
  const body =
    `{"directed":${graph.directed ? 'true' : 'false'},"edges":[${edges.join(',')}],` +
    `"nodes":[${nodes.join(',')}]}`;
  return 'sha256:' + sha256Hex(body);
}

// =============================================================================
// Semantic Search Result
// =============================================================================

export interface SemanticSearchResult {
  nodeRef: NodeRef;
  node: Node | null;
  score: number;
  hopCount: number;
  path: NodeRef[];
}

// =============================================================================
// Semantic Graph
// =============================================================================

export interface SemanticGraphOptions extends AnnOptions {
  name?: string;
  directed?: boolean;
  encoder?: EmbeddingEncoder | AsyncEncoder;
  autoEmbed?: boolean;
  embedFields?: string[];
  /**
   * Encoder/model version recorded in the sidecar (default: the encoder's own
   * `version`, if any).
   */
  version?: string;
  /**
   * What a text search does when the encoder's model differs from the stored
   * vectors' model: 'warn' (default), 'error' or 'ignore'. See
   * EmbeddingStoreOptions.
   */
  onModelMismatch?: ModelMismatchMode;
}

/** Node filtering shared by the search methods. */
export interface NodeFilterOptions {
  /** Predicate on a node; only nodes for which it returns truthy pass. */
  nodeFilter?: (node: Node) => boolean;
  /**
   * Node types allowed (a type name, or an array). An empty array passes
   * nothing. Combines with `nodeFilter` by AND.
   */
  nodeTypes?: string | string[];
  /**
   * Make the filter a hard wall for traversal too: a failing node is never
   * entered, so every node on every returned path passes. Without it a failing
   * node can still be walked *through* (it costs a hop and its decay but is
   * never returned).
   */
  strictFilter?: boolean;
}

/** Options for semanticMultiHop, all optional. */
export interface MultiHopOptions extends NodeFilterOptions {
  relType?: string;
  maxHops?: number;
  topKSeeds?: number;
  topKResults?: number;
  direction?: Direction;
  decay?: number;
  threshold?: number;
  /**
   * How much of the relevance of a node itself to mix into its traversal
   * score, in [0, 1]. 0 (the default) keeps pure seed-and-decay scoring.
   * Raise it so a node found two hops out but highly relevant in its own
   * right is not buried under a closer, unrelated neighbour. Hop-0 seeds
   * score the same either way.
   */
  blend?: number;
  /**
   * Name of an edge property whose numeric value multiplies the decay of every
   * hop that crosses that edge (clamped to [0, 1]; a missing, non-numeric,
   * boolean or NaN value counts as 1, so partly-weighted graphs work). A node
   * is scored by its *best* route, which with weights can be a longer one than
   * the fewest-hops route. Undefined (the default) ignores edge properties
   * entirely.
   */
  weightField?: string;
}

/** Options for semanticPath, all optional. */
export interface SemanticPathOptions extends NodeFilterOptions {
  relType?: string;
  maxHops?: number;
  topKSeeds?: number;
  /** Minimum seed similarity. Cosine similarity is signed; lower this to search from any start. */
  threshold?: number;
  decay?: number;
  direction?: Direction;
}

/** Options for semanticSubgraph: everything semanticMultiHop takes, plus these. */
export interface SubgraphOptions extends MultiHopOptions {
  /** Name for the returned graph (default `<name>_subgraph`). */
  name?: string;
  /**
   * Also add connector nodes so the selected nodes fall into as few connected
   * components as possible (see semanticSubgraph). Default false.
   */
  connect?: boolean;
  /** Longest route, in edges, that may join two components (default 2, minimum 1). */
  connectMaxHops?: number;
}

/**
 * An edge's numeric `weightField` property clamped to [0, 1].
 *
 * A missing, non-numeric (including boolean) or NaN value counts as 1, so an
 * unweighted edge never penalises a hop.
 */
function edgeWeight(edge: Edge, weightField: string): number {
  const value = edge.properties ? edge.properties[weightField] : undefined;
  if (typeof value !== 'number' || Number.isNaN(value)) return 1.0;
  return Math.min(1.0, Math.max(0.0, value));
}

/**
 * ISONGraph extended with embedding-based semantic search.
 */
export class SemanticGraph extends ISONGraph {
  private _encoder: EmbeddingEncoder | null;
  private _embeddingStore: EmbeddingStore;
  private _autoEmbed: boolean;
  private _embedFields: string[];

  constructor(options: SemanticGraphOptions = {}) {
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
    });
    this._autoEmbed = options.autoEmbed ?? true;
    this._embedFields = options.embedFields || ['name', 'description', 'content', 'text'];
  }

  get embeddingStore(): EmbeddingStore {
    return this._embeddingStore;
  }

  get encoder(): EmbeddingEncoder | null {
    return this._encoder;
  }

  setEncoder(encoder: EmbeddingEncoder | AsyncEncoder): void {
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
  static propertyText(value: unknown): string {
    if (value === null || value === undefined) return '';
    return String(value);
  }

  /** Concatenate the configured embedFields present in a property bag. */
  embedTextFor(properties: Record<string, any>): string {
    const parts: string[] = [];
    for (const field of this._embedFields) {
      if (!(field in properties)) continue;
      const text = SemanticGraph.propertyText(properties[field]);
      if (text) parts.push(text);
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
  override addNode(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any> = {}
  ): Node {
    const node = super.addNode(nodeType, nodeId, properties);
    if (this._autoEmbed && this._encoder) {
      const text = this.embedTextFor(properties);
      if (text) this.embedNode(node.ref, text);
    }
    return node;
  }

  /**
   * Add a node with an explicit embedding text.
   */
  addNodeWithEmbed(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any> = {},
    embedText?: string
  ): Node {
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
  async addNodeAsync(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any> = {}
  ): Promise<Node> {
    let text = '';
    let vector: number[] | null = null;
    if (this._autoEmbed && this._encoder) {
      text = this.embedTextFor(properties);
      if (text) vector = (await this._embeddingStore.encodeTextsAsync([text]))[0];
    }
    const node = super.addNode(nodeType, nodeId, properties);
    if (vector !== null) this._embeddingStore.add(node.ref, text, vector);
    return node;
  }

  /** Like {@link addNodeWithEmbed}, awaiting the encoder (see {@link addNodeAsync}). */
  async addNodeWithEmbedAsync(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any> = {},
    embedText?: string
  ): Promise<Node> {
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
  embedNode(nodeRef: NodeRef, text: string): string {
    return this._embeddingStore.add(nodeRef, text);
  }

  /** Like {@link embedNode}, awaiting the encoder. */
  embedNodeAsync(nodeRef: NodeRef, text: string): Promise<string> {
    return this._embeddingStore.addAsync(nodeRef, text);
  }

  /** Like {@link embedNodes}, awaiting the encoder; stores nothing if it fails. */
  async embedNodesAsync(items: Array<[NodeRef, string]>): Promise<number> {
    const pairs = items.filter(([, text]) => Boolean(text));
    if (pairs.length === 0) return 0;
    return this._embeddingStore.addBatchAsync(pairs);
  }

  /** Embed many nodes in a single batched encoder call. */
  embedNodes(items: Array<[NodeRef, string]>): number {
    const pairs = items.filter(([, text]) => Boolean(text));
    if (pairs.length === 0) return 0;
    return this._embeddingStore.addBatch(pairs);
  }

  /**
   * Get embedding for a node.
   */
  getEmbedding(nodeRef: NodeRef): EmbeddingRecord | null {
    return this._embeddingStore.get(nodeRef);
  }

  /**
   * Update node properties, keeping the embedding in step with them.
   *
   * With autoEmbed on, a change to any of the embedFields re-embeds the node
   * from its new properties, so searches never rank it by text it no longer
   * carries. If the new properties leave nothing to embed, the stale vector
   * is dropped. Updates that touch no embed field, and graphs with autoEmbed
   * off (whose vectors the caller manages), leave the embedding alone. As in
   * addNode, nothing is re-embedded while no encoder is set.
   */
  override updateNode(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any>
  ): Node {
    const node = super.updateNode(nodeType, nodeId, properties);
    if (this._autoEmbed && this._embedFields.some(f => f in properties)) {
      const text = this.embedTextFor(node.properties);
      if (!text) {
        this._embeddingStore.remove(node.ref);
      } else if (this._encoder) {
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
  async updateNodeAsync(
    nodeType: string,
    nodeId: number | string,
    properties: Record<string, any>
  ): Promise<Node> {
    const current = this.getNode(nodeType, nodeId);
    const reembed = this._autoEmbed && this._embedFields.some(f => f in (properties ?? {}));
    let text = '';
    let vector: number[] | null = null;
    if (reembed) {
      text = this.embedTextFor({ ...current.properties, ...properties });
      if (text && this._encoder) vector = (await this._embeddingStore.encodeTextsAsync([text]))[0];
    }
    const node = super.updateNode(nodeType, nodeId, properties);
    if (reembed) {
      // Another update may have landed while the encoder was working.
      const latest = this.embedTextFor(node.properties);
      if (!latest) {
        this._embeddingStore.remove(node.ref);
      } else if (this._encoder) {
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
  override removeNode(nodeType: string, nodeId: number | string): void {
    const ref: NodeRef = [nodeType, nodeId];
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
  private _nodeGate(
    nodeFilter?: ((node: Node) => boolean) | null,
    nodeTypes?: string | string[] | null
  ): ((ref: NodeRef) => boolean) | null {
    const hasFilter = nodeFilter !== undefined && nodeFilter !== null;
    const hasTypes = nodeTypes !== undefined && nodeTypes !== null;
    if (!hasFilter && !hasTypes) return null;

    const types = hasTypes
      ? new Set<string>(typeof nodeTypes === 'string' ? [nodeTypes] : nodeTypes)
      : null;
    const memo = new Map<string, boolean>();

    return (ref: NodeRef): boolean => {
      const key = `${ref[0]}:${ref[1]}`;
      const cached = memo.get(key);
      if (cached !== undefined) return cached;
      let ok = types === null || types.has(ref[0]);
      if (ok && hasFilter) {
        const node = this._resolveNode(ref);
        ok = node !== null && Boolean(nodeFilter!(node));
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
  private _seedSearch(
    query: string | number[],
    topK: number | null,
    nodeType: string | undefined,
    threshold: number,
    gate: ((ref: NodeRef) => boolean) | null,
    exact: boolean = false
  ): SimilarityResult[] {
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
    for (let scale = 1; ; scale *= 4) {
      const { ranked, approximate } = store.annRanked(queryVector, nodeType, threshold, scale);
      const results: SimilarityResult[] = [];
      for (const result of ranked) {
        if (gate(result.nodeRef)) {
          results.push(result);
          if (results.length >= topK) break;
        }
      }
      if (!approximate) return results;
      if (results.length >= topK) return store.annPolish(results, queryVector, threshold);
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
  similaritySearch(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0,
    nodeFilter?: (node: Node) => boolean,
    nodeTypes?: string | string[],
    exact: boolean = false
  ): SimilarityResult[] {
    const gate = this._nodeGate(nodeFilter, nodeTypes);
    return this._seedSearch(query, topK, nodeType, threshold, gate, exact);
  }

  /** Recall@k of the approximate index against the exact scan (see {@link EmbeddingStore.recallCheck}). */
  recallCheck(
    queries: string | ArrayLike<number> | Array<string | ArrayLike<number>>,
    k: number = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): RecallReport {
    return this._embeddingStore.recallCheck(queries, k, nodeType, threshold);
  }

  /** {@link recallCheck} with text queries encoded by awaiting the encoder. */
  recallCheckAsync(
    queries: string | ArrayLike<number> | Array<string | ArrayLike<number>>,
    k: number = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): Promise<RecallReport> {
    return this._embeddingStore.recallCheckAsync(queries, k, nodeType, threshold);
  }

  /** {@link similaritySearch} with a text query encoded by awaiting the encoder. */
  async similaritySearchAsync(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0,
    nodeFilter?: (node: Node) => boolean,
    nodeTypes?: string | string[],
    exact: boolean = false
  ): Promise<SimilarityResult[]> {
    const vector = await this._embeddingStore.resolveQueryAsync(query);
    return this.similaritySearch(vector, topK, nodeType, threshold, nodeFilter, nodeTypes, exact);
  }

  /**
   * Find nodes semantically closest to one already in the graph.
   *
   * The "more like this" query - no text and no encoder call: it searches with
   * the stored vector of the node and drops the node itself from the results.
   */
  similarToNode(
    nodeRef: NodeRef,
    topK: number = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): SimilarityResult[] {
    const record = this._embeddingStore.get(nodeRef);
    if (!record) return [];
    const key = `${nodeRef[0]}:${nodeRef[1]}`;
    const results = this._embeddingStore
      .similaritySearch(record.vector, topK === null ? null : topK + 1, nodeType, threshold)
      .filter(r => `${r.nodeRef[0]}:${r.nodeRef[1]}` !== key);
    return topK === null ? results : results.slice(0, topK);
  }

  private _resolveNode(nodeRef: NodeRef): Node | null {
    try {
      return this.getNodeByRef(nodeRef);
    } catch {
      return null;
    }
  }

  /** (neighbor, connecting edge) pairs, in the order `neighbors()` yields. */
  private _adjacent(
    nodeRef: NodeRef,
    relType: string | undefined,
    direction: Direction
  ): Array<[NodeRef, Edge]> {
    const pairs: Array<[NodeRef, Edge]> = [];
    if (direction === Direction.OUT || direction === Direction.BOTH) {
      for (const edge of this.edges(relType, nodeRef)) pairs.push([edge.target, edge]);
    }
    if (direction === Direction.IN || direction === Direction.BOTH) {
      for (const edge of this.edges(relType, undefined, nodeRef)) pairs.push([edge.source, edge]);
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
  semanticMultiHop(
    query: string | number[],
    relTypeOrOptions?: string | MultiHopOptions,
    maxHopsArg: number = 2,
    topKSeedsArg: number = 5,
    topKResultsArg: number = 10,
    directionArg: Direction = Direction.OUT,
    decayArg: number = 0.8,
    thresholdArg: number = 0.3,
    blendArg: number = 0.0,
    weightFieldArg?: string,
    nodeFilterArg?: (node: Node) => boolean,
    nodeTypesArg?: string | string[],
    strictFilterArg: boolean = false
  ): SemanticSearchResult[] {
    // Accepts either an options object or the original positional form, so
    // existing callers keep working while new ones can name what they mean.
    const options: MultiHopOptions =
      relTypeOrOptions !== null && typeof relTypeOrOptions === 'object'
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

    const {
      relType,
      maxHops = 2,
      topKSeeds = 5,
      topKResults = 10,
      direction = Direction.OUT,
      decay = 0.8,
      threshold = 0.3,
      blend = 0.0,
      weightField,
      nodeFilter,
      nodeTypes,
      strictFilter = false,
    } = options;

    if (blend < 0 || blend > 1) {
      throw new EmbeddingError(`blend must be in [0, 1], got ${blend}`);
    }

    const nodeRefToKey = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;
    const weighted = weightField !== undefined && weightField !== null;
    const gate = this._nodeGate(nodeFilter, nodeTypes);

    const seeds = [...this._seedSearch(query, topKSeeds, undefined, threshold, gate)];
    if (seeds.length === 0) return [];

    // Blending needs a score for every node: one scan serves the per-node
    // relevance that is mixed into each traversal score.
    const scores = blend > 0 ? this._embeddingStore.scoreMap(query) : null;
    const combined = (base: number, key: string): number =>
      scores === null ? base : (1 - blend) * base + blend * (scores.get(key) ?? 0);

    const results = new Map<string, SemanticSearchResult>();

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
      const bestFactor = new Map<string, number>([[seedKey, 1.0]]);
      const bestHops = new Map<string, number>([[seedKey, 0]]);
      const queue: Array<{
        nodeRef: NodeRef;
        path: NodeRef[];
        keys: string[];
        factor: number;
        hop: number;
      }> = [{ nodeRef: seed.nodeRef, path: [seed.nodeRef], keys: [seedKey], factor: 1.0, hop: 0 }];
      let head = 0;

      while (head < queue.length) {
        const { nodeRef, path, keys, factor, hop } = queue[head++];

        if (hop >= maxHops) continue;

        for (const [neighbor, edge] of this._adjacent(nodeRef, relType, direction)) {
          const neighborKey = nodeRefToKey(neighbor);
          if (keys.includes(neighborKey)) continue;
          if (strictFilter && gate !== null && !gate(neighbor)) continue;

          const newHop = hop + 1;
          let newFactor: number;
          if (!weighted) {
            if (bestHops.has(neighborKey)) continue;
            // decay ** hops exactly, not a running product, so unweighted
            // scores are bit-identical to the pre-weight releases.
            newFactor = Math.pow(decay, newHop);
          } else {
            newFactor = factor * decay * edgeWeight(edge, weightField!);
            const knownFactor = bestFactor.get(neighborKey) ?? -1.0;
            const knownHops = bestHops.get(neighborKey);
            if (newFactor <= knownFactor && newHop >= (knownHops ?? newHop + 1)) continue;
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
  private _gatedShortestPath(
    start: NodeRef,
    end: NodeRef,
    relType: string | undefined,
    maxHops: number,
    direction: Direction,
    gate: (ref: NodeRef) => boolean
  ): NodeRef[] | null {
    if (!gate(start) || !gate(end)) return null;
    const keyOf = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;
    const endKey = keyOf(end);
    if (keyOf(start) === endKey) return [start];

    const visited = new Set<string>([keyOf(start)]);
    const queue: NodeRef[][] = [[start]];
    let head = 0;
    while (head < queue.length) {
      const path = queue[head++];
      if (path.length - 1 >= maxHops) continue;
      for (const [neighbor] of this._adjacent(path[path.length - 1], relType, direction)) {
        const neighborKey = keyOf(neighbor);
        if (visited.has(neighborKey) || !gate(neighbor)) continue;
        if (neighborKey === endKey) return [...path, neighbor];
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
  semanticPath(
    query: string | number[],
    targetRef: NodeRef,
    relTypeOrOptions?: string | SemanticPathOptions,
    maxHopsArg: number = 5,
    topKSeedsArg: number = 3,
    thresholdArg: number = 0.0,
    decayArg: number = 0.9,
    directionArg: Direction = Direction.OUT,
    nodeFilterArg?: (node: Node) => boolean,
    nodeTypesArg?: string | string[],
    strictFilterArg: boolean = false
  ): SemanticSearchResult | null {
    const options: SemanticPathOptions =
      relTypeOrOptions !== null && typeof relTypeOrOptions === 'object'
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
    const {
      relType,
      maxHops = 5,
      topKSeeds = 3,
      threshold = 0.0,
      decay = 0.9,
      direction = Direction.OUT,
      nodeFilter,
      nodeTypes,
      strictFilter = false,
    } = options;

    const gate = this._nodeGate(nodeFilter, nodeTypes);
    if (gate !== null && !gate(targetRef)) return null;

    const seeds = this._seedSearch(query, topKSeeds, undefined, threshold, gate);

    // Every seed that reaches the target is a candidate, and the best score
    // wins: a slightly less similar seed one hop from the target beats a
    // better seed five hops away. Seeds arrive best-first, so a tie (strict
    // > below) goes to the more similar one. As in semanticMultiHop, a
    // negative seed passes no relevance along rather than "decaying" toward
    // zero and outscoring itself.
    let best: SemanticSearchResult | null = null;
    for (const seed of seeds) {
      let nodes: NodeRef[];
      if (strictFilter && gate !== null) {
        const gated = this._gatedShortestPath(
          seed.nodeRef, targetRef, relType, maxHops, direction, gate);
        if (gated === null) continue;
        nodes = gated;
      } else {
        const path = this.shortestPath(seed.nodeRef, targetRef, relType, maxHops, direction);
        if (!path) continue;
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
  private _connectComponents(
    keep: Map<string, NodeRef>,
    relType: string | undefined,
    direction: Direction,
    maxHops: number,
    gate: ((ref: NodeRef) => boolean) | null
  ): NodeRef[] {
    const added: NodeRef[] = [];
    const keyOf = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;
    // Same ordering as Python's (type, type(id).__name__, str(id)) tuple.
    const compareRefs = (a: NodeRef, b: NodeRef): number =>
      compareCodePoints(String(a[0]), String(b[0])) ||
      compareCodePoints(typeof a[1] === 'number' ? 'int' : 'str', typeof b[1] === 'number' ? 'int' : 'str') ||
      compareCodePoints(String(a[1]), String(b[1]));

    while (keep.size > 1) {
      // Weakly connected components of the induced subgraph.
      const parent = new Map<string, string>();
      for (const key of keep.keys()) parent.set(key, key);
      const find = (key: string): string => {
        while (parent.get(key) !== key) {
          const grand = parent.get(parent.get(key)!)!;
          parent.set(key, grand);
          key = grand;
        }
        return key;
      };
      for (const edge of this.edges(relType)) {
        const s = keyOf(edge.source);
        const t = keyOf(edge.target);
        if (keep.has(s) && keep.has(t)) parent.set(find(s), find(t));
      }
      const groups = new Map<string, NodeRef[]>();
      for (const [key, ref] of keep) {
        const root = find(key);
        const members = groups.get(root);
        if (members) members.push(ref);
        else groups.set(root, [ref]);
      }
      if (groups.size <= 1) break;

      const ordered = [...groups.values()]
        .map(g => g.sort(compareRefs))
        .sort((a, b) => compareRefs(a[0], b[0]));
      let bestRoute: NodeRef[] | null = null;
      for (const members of ordered) {
        const route = this._shortestLink(members, keep, relType, direction, maxHops, gate);
        if (route !== null && (bestRoute === null || route.length < bestRoute.length)) {
          bestRoute = route;
        }
      }
      if (bestRoute === null) break;
      for (const ref of bestRoute.slice(1, -1)) {
        keep.set(keyOf(ref), ref);
        added.push(ref);
      }
    }
    return added;
  }

  /** Shortest route from any of `members` to a kept node outside them. */
  private _shortestLink(
    members: NodeRef[],
    keep: Map<string, NodeRef>,
    relType: string | undefined,
    direction: Direction,
    maxHops: number,
    gate: ((ref: NodeRef) => boolean) | null
  ): NodeRef[] | null {
    const keyOf = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;
    const visited = new Set<string>(members.map(keyOf));
    const queue: NodeRef[][] = members.map(m => [m]);
    let head = 0;
    while (head < queue.length) {
      const path = queue[head++];
      if (path.length - 1 >= maxHops) continue;
      for (const [neighbor] of this._adjacent(path[path.length - 1], relType, direction)) {
        const neighborKey = keyOf(neighbor);
        if (visited.has(neighborKey)) continue;
        if (keep.has(neighborKey)) return [...path, neighbor];
        if (gate !== null && !gate(neighbor)) continue;
        if (this._resolveNode(neighbor) === null) continue;
        visited.add(neighborKey);
        queue.push([...path, neighbor]);
      }
    }
    return null;
  }

  /** {@link semanticMultiHop} with a text query encoded by awaiting the encoder (options form). */
  async semanticMultiHopAsync(
    query: string | number[],
    options: MultiHopOptions = {}
  ): Promise<SemanticSearchResult[]> {
    return this.semanticMultiHop(await this._embeddingStore.resolveQueryAsync(query), options);
  }

  /** {@link semanticPath} with a text query encoded by awaiting the encoder (options form). */
  async semanticPathAsync(
    query: string | number[],
    targetRef: NodeRef,
    options: SemanticPathOptions = {}
  ): Promise<SemanticSearchResult | null> {
    return this.semanticPath(await this._embeddingStore.resolveQueryAsync(query), targetRef, options);
  }

  /** {@link semanticSubgraph} with a text query encoded by awaiting the encoder. */
  async semanticSubgraphAsync(
    query: string | number[],
    options: SubgraphOptions = {}
  ): Promise<SemanticGraph> {
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
  semanticSubgraph(
    query: string | number[],
    options: SubgraphOptions = {}
  ): SemanticGraph {
    const { connect = false, connectMaxHops = 2 } = options;
    if (connectMaxHops < 1) {
      throw new EmbeddingError(`connectMaxHops must be >= 1, got ${connectMaxHops}`);
    }

    const results = this.semanticMultiHop(query, options);
    const keyOf = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;
    const keep = new Map<string, NodeRef>(results.map(r => [keyOf(r.nodeRef), r.nodeRef]));

    if (connect && keep.size > 0) {
      const gate = options.strictFilter
        ? this._nodeGate(options.nodeFilter, options.nodeTypes)
        : null;
      this._connectComponents(keep, options.relType, options.direction ?? Direction.OUT,
        connectMaxHops, gate);
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
      if (!node) continue;
      sub.addNode(node.type, node.id, node.properties);
      const record = this._embeddingStore.get(ref);
      if (record) sub.embeddingStore.add(ref, record.text, record.vector);
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
  embedAllNodes(textFn?: (node: Node) => string, skipExisting: boolean = false): number {
    const defaultTextFn = (node: Node): string =>
      this.embedTextFor(node.properties) || `${node.type}:${node.id}`;

    const fn = textFn || defaultTextFn;
    const batch: Array<[NodeRef, string]> = [];

    for (const node of this.nodes()) {
      if (skipExisting && this._embeddingStore.get(node.ref)) continue;
      const text = fn(node);
      if (text) batch.push([node.ref, text]);
    }

    return this.embedNodes(batch);
  }

  /** {@link embedAllNodes} awaiting the encoder: all the texts, then the stores. */
  async embedAllNodesAsync(
    textFn?: (node: Node) => string,
    skipExisting: boolean = false
  ): Promise<number> {
    const fn = textFn || ((node: Node): string =>
      this.embedTextFor(node.properties) || `${node.type}:${node.id}`);
    const batch: Array<[NodeRef, string]> = [];
    for (const node of this.nodes()) {
      if (skipExisting && this._embeddingStore.get(node.ref)) continue;
      const text = fn(node);
      if (text) batch.push([node.ref, text]);
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
  staleEmbeddings(includeUnknown: boolean = false): NodeRef[] {
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
  reembedStale(includeUnknown: boolean = false, batchSize: number = 256): number {
    const stale = this.staleEmbeddings(includeUnknown);
    if (stale.length === 0) return 0;

    const store = this._embeddingStore;
    const pairs: Array<[NodeRef, string]> = [];
    for (const ref of stale) {
      const record = store.get(ref);
      if (!record) continue;
      let text = record.text;
      if (!text) {
        const node = this._resolveNode(ref);
        text = (node ? this.embedTextFor(node.properties) : '') || `${ref[0]}:${ref[1]}`;
      }
      pairs.push([ref, text]);
    }

    const encoder = this._encoder;
    if (!encoder) throw new NoEncoderError("No encoder provided to re-embed with");
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
  async reembedStaleAsync(includeUnknown: boolean = false, batchSize: number = 256): Promise<number> {
    const stale = this.staleEmbeddings(includeUnknown);
    if (stale.length === 0) return 0;

    const store = this._embeddingStore;
    const pairs: Array<[NodeRef, string]> = [];
    for (const ref of stale) {
      const record = store.get(ref);
      if (!record) continue;
      let text = record.text;
      if (!text) {
        const node = this._resolveNode(ref);
        text = (node ? this.embedTextFor(node.properties) : '') || `${ref[0]}:${ref[1]}`;
      }
      pairs.push([ref, text]);
    }
    if (!this._encoder) throw new NoEncoderError("No encoder provided to re-embed with");

    const step = Math.max(1, Math.floor(batchSize));
    const vectors: number[][] = [];
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
  static fromISONGraph(base: ISONGraph, options: SemanticGraphOptions = {}): SemanticGraph {
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

  toString(): string {
    const embCount = this._embeddingStore.count();
    return `SemanticGraph(name=${this.name}, nodes=${this.nodeCount()}, edges=${this.edgeCount()}, embeddings=${embCount})`;
  }
}

// =============================================================================
// Exports
// =============================================================================

export {
  ISONGraph,
  Node,
  NodeRef,
  Direction
} from 'ison-graph-ts';
