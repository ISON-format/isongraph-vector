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
    constructor(dimension = 384) {
        if (!Number.isInteger(dimension) || dimension <= 0) {
            throw new EmbeddingError(`dimension must be a positive integer, got ${dimension}`);
        }
        this._dimension = dimension;
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
    constructor(encoder) {
        this._embeddings = new Map();
        this._dimension = null;
        this._encoder = encoder || null;
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
    add(nodeRef, text, vector) {
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
    clear() {
        this._embeddings.clear();
        this._dimension = null;
    }
    /** Serialize every embedding into the portable cross-language payload. */
    exportEmbeddings() {
        const embeddings = [];
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
    importEmbeddings(payload, replace = false) {
        if (!payload || payload.format !== EMBEDDING_FORMAT) {
            throw new EmbeddingError(`not an ${EMBEDDING_FORMAT} payload: format=${payload && payload.format}`);
        }
        if (payload.version !== EMBEDDING_FORMAT_VERSION) {
            throw new EmbeddingError(`unsupported ${EMBEDDING_FORMAT} version ${payload.version} ` +
                `(this build reads version ${EMBEDDING_FORMAT_VERSION})`);
        }
        if (replace)
            this.clear();
        let written = 0;
        for (const item of payload.embeddings || []) {
            const id = item.id_type === 'int' ? Number(item.id) : item.id;
            this.add([item.type, id], item.text ?? '', item.vector);
            written++;
        }
        return written;
    }
}
/**
 * ISONGraph extended with embedding-based semantic search.
 */
export class SemanticGraph extends ISONGraph {
    constructor(options = {}) {
        super(options.name || "semantic_graph", options.directed ?? true);
        this._encoder = options.encoder || null;
        this._embeddingStore = new EmbeddingStore(this._encoder || undefined);
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
     * Remove node and its embedding.
     */
    removeNode(nodeType, nodeId) {
        const ref = [nodeType, nodeId];
        this._embeddingStore.remove(ref);
        super.removeNode(nodeType, nodeId);
    }
    /**
     * Find similar nodes by embedding.
     */
    similaritySearch(query, topK = 10, nodeType, threshold = 0.0) {
        return this._embeddingStore.similaritySearch(query, topK, nodeType, threshold);
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
    /**
     * Semantic multi-hop search.
     *
     * 1. Find top-k similar nodes as seeds (hop 0)
     * 2. Traverse the graph following relationships
     * 3. Score by max(seedSimilarity, 0) * decay^hopCount, optionally blended
     *    with the relevance of each node itself
     */
    semanticMultiHop(query, relTypeOrOptions, maxHopsArg = 2, topKSeedsArg = 5, topKResultsArg = 10, directionArg = Direction.OUT, decayArg = 0.8, thresholdArg = 0.3, blendArg = 0.0) {
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
            };
        const { relType, maxHops = 2, topKSeeds = 5, topKResults = 10, direction = Direction.OUT, decay = 0.8, threshold = 0.3, blend = 0.0, } = options;
        if (blend < 0 || blend > 1) {
            throw new EmbeddingError(`blend must be in [0, 1], got ${blend}`);
        }
        const nodeRefToKey = (ref) => `${ref[0]}:${ref[1]}`;
        // One scan serves both seed selection and per-node relevance.
        const scores = this._embeddingStore.scoreMap(query);
        if (scores.size === 0)
            return [];
        const seeds = [...this._embeddingStore.similaritySearch(query, topKSeeds, undefined, threshold)];
        if (seeds.length === 0)
            return [];
        const combined = (base, key) => blend <= 0 ? base : (1 - blend) * base + blend * (scores.get(key) ?? 0);
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
            // BFS traversal
            const queue = [
                { nodeRef: seed.nodeRef, path: [seed.nodeRef], hop: 0 },
            ];
            const visited = new Set([seedKey]);
            let head = 0;
            while (head < queue.length) {
                const { nodeRef, path, hop } = queue[head++];
                if (hop >= maxHops)
                    continue;
                const neighbors = this.neighbors(nodeRef, relType, direction);
                for (const neighbor of neighbors) {
                    const neighborKey = nodeRefToKey(neighbor);
                    if (visited.has(neighborKey))
                        continue;
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
    semanticPath(query, targetRef, relType, maxHops = 5, topKSeeds = 3, threshold = 0.0, decay = 0.9) {
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
    semanticSubgraph(query, options = {}) {
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
            if (!node)
                continue;
            sub.addNode(node.type, node.id, node.properties);
            const record = this._embeddingStore.get(result.nodeRef);
            if (record)
                sub.embeddingStore.add(result.nodeRef, record.text, record.vector);
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
    toString() {
        const embCount = this._embeddingStore.count();
        return `SemanticGraph(name=${this.name}, nodes=${this.nodeCount()}, edges=${this.edgeCount()}, embeddings=${embCount})`;
    }
}
// =============================================================================
// Exports
// =============================================================================
export { ISONGraph, Node, Direction } from 'ison-graph-js';
