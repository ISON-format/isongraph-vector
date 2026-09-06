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

import { ISONGraph, Node, NodeRef, Direction } from 'ison-graph-ts';

export const VERSION = "1.0.0";

/**
 * Portable embedding payload shared by every language port, so a store built
 * in one can be searched by another.
 */
export const EMBEDDING_FORMAT = "ison-embeddings";
export const EMBEDDING_FORMAT_VERSION = 1;

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

// =============================================================================
// Embedding Encoder Interface
// =============================================================================

export interface EmbeddingEncoder {
  readonly dimension: number;
  encode(text: string): number[];
  encodeBatch(texts: string[]): number[][];
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

  constructor(dimension: number = 384) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new EmbeddingError(`dimension must be a positive integer, got ${dimension}`);
    }
    this._dimension = dimension;
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
// Embedding Store
// =============================================================================

export interface EmbeddingRecord {
  id: string;
  nodeRef: NodeRef;
  vector: number[];
  text: string;
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
  vector: number[];
}

/** Payload written by exportEmbeddings, readable by every language port. */
export interface ExportedEmbeddings {
  format: string;
  version: number;
  dimension: number | null;
  count: number;
  embeddings: ExportedEmbedding[];
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

/**
 * In-memory embedding storage with brute-force cosine similarity search.
 */
export class EmbeddingStore {
  private _embeddings: Map<string, EmbeddingRecord> = new Map();
  private _encoder: EmbeddingEncoder | null;
  private _dimension: number | null = null;

  constructor(encoder?: EmbeddingEncoder) {
    this._encoder = encoder || null;
  }

  get encoder(): EmbeddingEncoder | null {
    return this._encoder;
  }

  set encoder(encoder: EmbeddingEncoder | null) {
    this._encoder = encoder;
  }

  /** Vector length this store holds, or null while it is still empty. */
  get dimension(): number | null {
    return this._dimension;
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
      vector = this._encoder.encode(query);
    } else {
      vector = query;
    }
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

  add(nodeRef: NodeRef, text: string, vector?: number[]): string {
    if (vector === undefined) {
      if (!this._encoder) {
        throw new NoEncoderError("No encoder provided and no vector given");
      }
      vector = this._encoder.encode(text);
    }
    this._checkDimension(vector, "vector");

    const key = this._nodeRefToKey(nodeRef);
    this._embeddings.set(key, { id: key, nodeRef, vector, text });
    return key;
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

  get(nodeRef: NodeRef): EmbeddingRecord | null {
    const key = this._nodeRefToKey(nodeRef);
    return this._embeddings.get(key) || null;
  }

  remove(nodeRef: NodeRef): boolean {
    const key = this._nodeRefToKey(nodeRef);
    return this._embeddings.delete(key);
  }

  similaritySearch(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): SimilarityResult[] {
    const queryVector = this._resolveQuery(query);
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

  clear(): void {
    this._embeddings.clear();
    this._dimension = null;
  }

  /** Serialize every embedding into the portable cross-language payload. */
  exportEmbeddings(): ExportedEmbeddings {
    const embeddings: ExportedEmbedding[] = [];
    for (const record of this._embeddings.values()) {
      embeddings.push({
        type: String(record.nodeRef[0]),
        id: record.nodeRef[1],
        id_type: typeof record.nodeRef[1] === 'number' ? 'int' : 'str',
        text: record.text,
        vector: record.vector,
      });
    }
    return {
      format: EMBEDDING_FORMAT,
      version: EMBEDDING_FORMAT_VERSION,
      dimension: this._dimension,
      count: embeddings.length,
      embeddings,
    };
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
    if (replace) this.clear();

    let written = 0;
    for (const item of payload.embeddings || []) {
      const id = item.id_type === 'int' ? Number(item.id) : item.id;
      this.add([item.type, id] as NodeRef, item.text ?? '', item.vector);
      written++;
    }
    return written;
  }
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

export interface SemanticGraphOptions {
  name?: string;
  directed?: boolean;
  encoder?: EmbeddingEncoder;
  autoEmbed?: boolean;
  embedFields?: string[];
}

/** Options for semanticMultiHop, all optional. */
export interface MultiHopOptions {
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

    this._encoder = options.encoder || null;
    this._embeddingStore = new EmbeddingStore(this._encoder || undefined);
    this._autoEmbed = options.autoEmbed ?? true;
    this._embedFields = options.embedFields || ['name', 'description', 'content', 'text'];
  }

  get embeddingStore(): EmbeddingStore {
    return this._embeddingStore;
  }

  get encoder(): EmbeddingEncoder | null {
    return this._encoder;
  }

  setEncoder(encoder: EmbeddingEncoder): void {
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
   * Embed a node with given text.
   */
  embedNode(nodeRef: NodeRef, text: string): string {
    return this._embeddingStore.add(nodeRef, text);
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
   * Remove node and its embedding.
   */
  override removeNode(nodeType: string, nodeId: number | string): void {
    const ref: NodeRef = [nodeType, nodeId];
    this._embeddingStore.remove(ref);
    super.removeNode(nodeType, nodeId);
  }

  /**
   * Find similar nodes by embedding.
   */
  similaritySearch(
    query: string | number[],
    topK: number | null = 10,
    nodeType?: string,
    threshold: number = 0.0
  ): SimilarityResult[] {
    return this._embeddingStore.similaritySearch(query, topK, nodeType, threshold);
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

  /**
   * Semantic multi-hop search.
   *
   * 1. Find top-k similar nodes as seeds (hop 0)
   * 2. Traverse the graph following relationships
   * 3. Score by max(seedSimilarity, 0) * decay^hopCount, optionally blended
   *    with the relevance of each node itself
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
    blendArg: number = 0.0
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
    } = options;

    if (blend < 0 || blend > 1) {
      throw new EmbeddingError(`blend must be in [0, 1], got ${blend}`);
    }

    const nodeRefToKey = (ref: NodeRef) => `${ref[0]}:${ref[1]}`;

    // One scan serves both seed selection and per-node relevance.
    const scores = this._embeddingStore.scoreMap(query);
    if (scores.size === 0) return [];

    const seeds = [...this._embeddingStore.similaritySearch(query, topKSeeds, undefined, threshold)];
    if (seeds.length === 0) return [];

    const combined = (base: number, key: string): number =>
      blend <= 0 ? base : (1 - blend) * base + blend * (scores.get(key) ?? 0);

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

      // BFS traversal
      const queue: Array<{ nodeRef: NodeRef; path: NodeRef[]; hop: number }> = [
        { nodeRef: seed.nodeRef, path: [seed.nodeRef], hop: 0 },
      ];
      const visited = new Set<string>([seedKey]);
      let head = 0;

      while (head < queue.length) {
        const { nodeRef, path, hop } = queue[head++];

        if (hop >= maxHops) continue;

        const neighbors = this.neighbors(nodeRef, relType, direction);

        for (const neighbor of neighbors) {
          const neighborKey = nodeRefToKey(neighbor);
          if (visited.has(neighborKey)) continue;

          visited.add(neighborKey);
          const newPath = [...path, neighbor];
          const newHop = hop + 1;
          const newScore = combined(traversalBase * Math.pow(decay, newHop), neighborKey);

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

          queue.push({ nodeRef: neighbor, path: newPath, hop: newHop });
        }
      }
    }

    return Array.from(results.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topKResults);
  }

  /**
   * Find semantically-rooted path to a target node.
   */
  semanticPath(
    query: string | number[],
    targetRef: NodeRef,
    relType?: string,
    maxHops: number = 5,
    topKSeeds: number = 3,
    threshold: number = 0.0,
    decay: number = 0.9
  ): SemanticSearchResult | null {
    const seeds = this.similaritySearch(query, topKSeeds, undefined, threshold);

    for (const seed of seeds) {
      const path = this.shortestPath(seed.nodeRef, targetRef, relType, maxHops);
      if (path) {
        return {
          nodeRef: targetRef,
          node: this._resolveNode(targetRef),
          score: seed.score * Math.pow(decay, path.length),
          hopCount: path.length,
          path: path.nodes,
        };
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
   */
  semanticSubgraph(
    query: string | number[],
    options: MultiHopOptions & { name?: string } = {}
  ): SemanticGraph {
    const results = this.semanticMultiHop(query, options);
    const keep = new Set(results.map(r => `${r.nodeRef[0]}:${r.nodeRef[1]}`));

    const sub = new SemanticGraph({
      name: options.name || `${this.name}_subgraph`,
      directed: this.directed,
      encoder: this._encoder || undefined,
      autoEmbed: false,
      embedFields: [...this._embedFields],
    });

    for (const result of results) {
      const node = result.node ?? this._resolveNode(result.nodeRef);
      if (!node) continue;
      sub.addNode(node.type, node.id, node.properties);
      const record = this._embeddingStore.get(result.nodeRef);
      if (record) sub.embeddingStore.add(result.nodeRef, record.text, record.vector);
    }

    for (const edge of this.edges(options.relType)) {
      const sourceKey = `${edge.source[0]}:${edge.source[1]}`;
      const targetKey = `${edge.target[0]}:${edge.target[1]}`;
      if (keep.has(sourceKey) && keep.has(targetKey)) {
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
