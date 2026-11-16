/**
 * Node filtering, edge weighting, subgraph connectivity, model lifecycle and
 * save/load consistency. Mirrors the Python port's TestNodeFiltering,
 * TestEdgeWeighting, TestSubgraphConnectivity, TestModelLifecycle and
 * TestSaveLoadConsistency.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync, } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SemanticGraph, ISONGraph, MockEncoder, EmbeddingStore, Direction, EmbeddingError, ModelMismatchError, GraphEmbeddingMismatchError, EMBEDDING_FORMAT, GRAPH_DIGEST_BASIS, graphDigest, } from './index.js';
import { saveGraph, loadGraph, embeddingsPathFor, writeFileAtomic } from './node.js';
const dirs = [];
function tmp() {
    const d = mkdtempSync(join(tmpdir(), 'isongraph-vector-feat-'));
    dirs.push(d);
    return d;
}
afterEach(() => {
    vi.restoreAllMocks();
    while (dirs.length)
        rmSync(dirs.pop(), { recursive: true, force: true });
});
const Q = [1.0, 0.0, 0.0, 0.0];
function namedGraph(encoder, names = ['Alice', 'Bob', 'Carol']) {
    const graph = new SemanticGraph({ name: 'g', encoder, autoEmbed: true });
    names.forEach((name, i) => graph.addNode('person', i + 1, { name }));
    graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
    return graph;
}
/**
 * Hand-vectored graph for traversal tests (query [1,0,0,0] is the target).
 *
 *     person 1  --KNOWS--> company 100 --KNOWS--> person 3
 *     person 2  (isolated)                         person 3 has no vector
 *
 * company 100 is the single best match for the query, person 1 next.
 */
function vectorGraph() {
    const graph = new SemanticGraph({ name: 'v', autoEmbed: false });
    graph.addNode('person', 1, { name: 'P1', active: true });
    graph.addNode('person', 2, { name: 'P2', active: false });
    graph.addNode('person', 3, { name: 'P3', active: true });
    graph.addNode('company', 100, { name: 'C' });
    graph.addEdge('KNOWS', ['person', 1], ['company', 100]);
    graph.addEdge('KNOWS', ['company', 100], ['person', 3]);
    const store = graph.embeddingStore;
    store.add(['company', 100], 'c', [1.0, 0.0, 0.0, 0.0]);
    store.add(['person', 1], 'p1', [0.9, 0.1, 0.0, 0.0]);
    store.add(['person', 2], 'p2', [0.8, 0.2, 0.0, 0.0]);
    return graph;
}
const refs = (list) => list.map(r => r.nodeRef);
const refSet = (graph) => new Set([...graph.nodes()].map(n => `${n.ref[0]}:${n.ref[1]}`));
function retagged(dimension, name) {
    const encoder = new MockEncoder(dimension);
    encoder.modelName = name;
    return encoder;
}
// -----------------------------------------------------------------------------
// Node filtering
// -----------------------------------------------------------------------------
describe('node filtering', () => {
    it('similaritySearch filters before taking topK', () => {
        const graph = vectorGraph();
        // the single best match is the company; filtering must look past it
        expect(graph.similaritySearch(Q, 1)[0].nodeRef).toEqual(['company', 100]);
        const top = graph.similaritySearch(Q, 1, undefined, 0.0, undefined, ['person']);
        expect(refs(top)).toEqual([['person', 1]]);
    });
    it('nodeFilter and nodeTypes combine with AND', () => {
        const graph = vectorGraph();
        const active = graph.similaritySearch(Q, null, undefined, -1.0, n => n.properties.active);
        expect(refs(active)).toEqual([['person', 1]]);
        const none = graph.similaritySearch(Q, null, undefined, -1.0, n => n.properties.active, ['company']);
        expect(none).toEqual([]);
    });
    it('a bare string is one type and an empty array matches nothing', () => {
        const graph = vectorGraph();
        expect(refs(graph.similaritySearch(Q, null, undefined, 0.0, undefined, 'company')))
            .toEqual([['company', 100]]);
        expect(graph.similaritySearch(Q, null, undefined, 0.0, undefined, [])).toEqual([]);
    });
    it('embeddings without a node fail nodeFilter only', () => {
        const graph = vectorGraph();
        graph.embeddingStore.add(['ghost', 1], 'g', [1.0, 0.0, 0.0, 0.0]);
        const byType = graph.similaritySearch(Q, null, undefined, -1.0, undefined, ['ghost']);
        expect(refs(byType)).toContainEqual(['ghost', 1]);
        const byFilter = graph.similaritySearch(Q, null, undefined, -1.0, () => true);
        expect(refs(byFilter)).not.toContainEqual(['ghost', 1]);
    });
    it('multi-hop seeds and results are filtered', () => {
        const graph = vectorGraph();
        const found = graph.semanticMultiHop(Q, {
            maxHops: 2, topKSeeds: 1, threshold: -1.0, nodeTypes: ['person'],
        });
        const got = refs(found);
        // the best match (company) did not take the seed slot, and is not returned
        expect(got[0]).toEqual(['person', 1]);
        expect(got).not.toContainEqual(['company', 100]);
    });
    it('filtered nodes are traversed but not returned by default', () => {
        const graph = vectorGraph();
        const found = new Map(graph.semanticMultiHop(Q, {
            maxHops: 2, topKSeeds: 1, threshold: -1.0, nodeTypes: ['person'],
        }).map(r => [`${r.nodeRef[0]}:${r.nodeRef[1]}`, r]));
        const reach = found.get('person:3');
        expect(reach.path).toEqual([['person', 1], ['company', 100], ['person', 3]]);
        expect(reach.hopCount).toBe(2); // the company still cost a hop
        expect(reach.score).toBeCloseTo(found.get('person:1').score * 0.8 ** 2, 12);
    });
    it('strictFilter walls off the traversal', () => {
        const graph = vectorGraph();
        const found = refs(graph.semanticMultiHop(Q, {
            maxHops: 2, topKSeeds: 1, threshold: -1.0, nodeTypes: ['person'], strictFilter: true,
        }));
        expect(found).not.toContainEqual(['person', 3]);
        expect(found).not.toContainEqual(['company', 100]);
    });
    it('the blend path respects the filter too', () => {
        const graph = vectorGraph();
        const found = graph.semanticMultiHop(Q, {
            maxHops: 1, topKSeeds: 2, threshold: -1.0, blend: 0.5, nodeTypes: ['person'],
        });
        expect(found.length).toBeGreaterThan(0);
        expect(found.every(r => r.nodeRef[0] === 'person')).toBe(true);
    });
    it('positional multi-hop arguments accept the new filter parameters', () => {
        const graph = vectorGraph();
        const found = graph.semanticMultiHop(Q, undefined, 2, 1, 10, Direction.OUT, 0.8, -1.0, 0.0, undefined, undefined, ['person'], true);
        expect(refs(found)).toEqual([['person', 1]]);
    });
    it('semanticPath requires a passing target', () => {
        const graph = vectorGraph();
        expect(graph.semanticPath(Q, ['company', 100], { threshold: -1.0, nodeTypes: ['person'] }))
            .toBeNull();
        expect(graph.semanticPath(Q, ['company', 100], { threshold: -1.0 })).not.toBeNull();
    });
    it('semanticPath may cross failing nodes unless strict', () => {
        const graph = vectorGraph();
        const lax = graph.semanticPath(Q, ['person', 3], {
            threshold: -1.0, topKSeeds: 1, nodeTypes: ['person'],
        });
        expect(lax.path).toEqual([['person', 1], ['company', 100], ['person', 3]]);
        const strict = graph.semanticPath(Q, ['person', 3], {
            threshold: -1.0, topKSeeds: 1, nodeTypes: ['person'], strictFilter: true,
        });
        expect(strict).toBeNull();
    });
    it('semanticPath seeds come from passing nodes', () => {
        const graph = vectorGraph();
        const result = graph.semanticPath(Q, ['person', 3], {
            threshold: -1.0, topKSeeds: 1, nodeFilter: n => n.type === 'person',
        });
        expect(result.path[0]).toEqual(['person', 1]);
    });
    it('positional semanticPath keeps its argument order and gains direction + filters', () => {
        const graph = vectorGraph();
        // (relType, maxHops, topKSeeds, threshold, decay, direction, nodeFilter, nodeTypes, strictFilter)
        const fwd = graph.semanticPath(Q, ['person', 3], 'KNOWS', 5, 1, -1.0, 0.9, Direction.OUT);
        expect(fwd.path).toEqual([['company', 100], ['person', 3]]);
        expect(graph.semanticPath(Q, ['person', 3], 'KNOWS', 5, 1, -1.0, 0.9, Direction.IN)).toBeNull();
        const strict = graph.semanticPath(Q, ['person', 3], 'KNOWS', 5, 1, -1.0, 0.9, Direction.OUT, undefined, ['person'], true);
        expect(strict).toBeNull();
    });
    it('semanticPath direction IN walks edges backwards', () => {
        const graph = vectorGraph();
        // person 3 <- company 100 <- person 1, so from person 3 the company is reachable inbound
        graph.embeddingStore.add(['person', 3], 'p3', [1.0, 0.0, 0.0, 0.0]);
        const result = graph.semanticPath(Q, ['person', 1], { threshold: -1.0, topKSeeds: 10, direction: Direction.IN });
        expect(result).not.toBeNull();
        expect(result.path[result.path.length - 1]).toEqual(['person', 1]);
        expect(graph.semanticPath(Q, ['person', 3], { threshold: -1.0, topKSeeds: 1, direction: Direction.OUT }).path)
            .toEqual([['company', 100], ['person', 3]]);
    });
    it('the subgraph holds only passing nodes', () => {
        const graph = vectorGraph();
        const sub = graph.semanticSubgraph(Q, {
            maxHops: 2, topKSeeds: 2, threshold: -1.0, nodeTypes: ['person'],
        });
        expect(new Set([...sub.nodes()].map(n => n.type))).toEqual(new Set(['person']));
        expect(refSet(sub).has('person:3')).toBe(true);
    });
    it('defaults are unchanged', () => {
        const graph = vectorGraph();
        const plain = graph.semanticMultiHop(Q, { maxHops: 2, topKSeeds: 1, threshold: -1.0 });
        expect(plain[0].nodeRef).toEqual(['company', 100]);
        expect(refs(plain)).toContainEqual(['person', 3]);
    });
});
// -----------------------------------------------------------------------------
// Edge weighting
// -----------------------------------------------------------------------------
describe('edge weighting', () => {
    /** S -> P (weight pWeight) and S -> Q (weight qWeight, or none when 'missing'). */
    function fan(pWeight, qWeight = 'missing') {
        const graph = new SemanticGraph({ autoEmbed: false });
        for (const ref of ['S', 'P', 'Q'])
            graph.addNode('n', ref);
        graph.addEdge('R', ['n', 'S'], ['n', 'P'], { w: pWeight });
        if (qWeight === 'missing')
            graph.addEdge('R', ['n', 'S'], ['n', 'Q']);
        else
            graph.addEdge('R', ['n', 'S'], ['n', 'Q'], { w: qWeight });
        graph.embeddingStore.add(['n', 'S'], 's', [1.0, 0.0]);
        return graph;
    }
    function scores(graph, extra = {}) {
        const found = graph.semanticMultiHop([1.0, 0.0], {
            maxHops: 1, topKSeeds: 1, threshold: -1.0, ...extra,
        });
        return Object.fromEntries(found.map(r => [String(r.nodeRef[1]), r.score]));
    }
    it('the default ignores edge properties', () => {
        const s = scores(fan(0.1));
        expect(s.P).toBeCloseTo(0.8, 12);
        expect(s.Q).toBeCloseTo(0.8, 12);
    });
    it('weight multiplies the decay', () => {
        const s = scores(fan(0.5), { weightField: 'w' });
        expect(s.S).toBeCloseTo(1.0, 12);
        expect(s.P).toBeCloseTo(0.8 * 0.5, 12);
        expect(s.Q).toBeCloseTo(0.8, 12); // missing -> 1.0
    });
    it('weights are clamped and junk counts as one', () => {
        expect(scores(fan(5.0), { weightField: 'w' }).P).toBeCloseTo(0.8, 12);
        expect(scores(fan(-3.0), { weightField: 'w' }).P).toBeCloseTo(0.0, 12);
        expect(scores(fan('heavy'), { weightField: 'w' }).P).toBeCloseTo(0.8, 12);
        expect(scores(fan(true), { weightField: 'w' }).P).toBeCloseTo(0.8, 12);
        expect(scores(fan(NaN), { weightField: 'w' }).P).toBeCloseTo(0.8, 12);
    });
    it('weight 0 gives score 0 but the node is still reachable', () => {
        const s = scores(fan(0.0), { weightField: 'w' });
        expect('P' in s).toBe(true);
        expect(s.P).toBe(0);
    });
    function triangle() {
        const graph = new SemanticGraph({ autoEmbed: false });
        for (const ref of ['S', 'M', 'T'])
            graph.addNode('n', ref);
        graph.addEdge('R', ['n', 'S'], ['n', 'T'], { w: 0.1 }); // direct but weak
        graph.addEdge('R', ['n', 'S'], ['n', 'M'], { w: 1.0 });
        graph.addEdge('R', ['n', 'M'], ['n', 'T'], { w: 1.0 }); // 2 hops, strong
        graph.embeddingStore.add(['n', 'S'], 's', [1.0, 0.0]);
        return graph;
    }
    it('the best route can be longer than the shortest', () => {
        const graph = triangle();
        const byKey = (extra) => new Map(graph.semanticMultiHop([1.0, 0.0], { maxHops: 2, topKSeeds: 1, threshold: -1.0, ...extra })
            .map(r => [String(r.nodeRef[1]), r]));
        const plain = byKey({});
        expect(plain.get('T').hopCount).toBe(1);
        expect(plain.get('T').score).toBeCloseTo(0.8, 12);
        const weighted = byKey({ weightField: 'w' });
        expect(weighted.get('T').hopCount).toBe(2);
        expect(weighted.get('T').path).toEqual([['n', 'S'], ['n', 'M'], ['n', 'T']]);
        expect(weighted.get('T').score).toBeCloseTo(0.64, 12);
    });
    it('the hop budget still applies to the better route', () => {
        const found = new Map(triangle().semanticMultiHop([1.0, 0.0], {
            maxHops: 1, topKSeeds: 1, threshold: -1.0, weightField: 'w',
        }).map(r => [String(r.nodeRef[1]), r]));
        expect(found.get('T').score).toBeCloseTo(0.8 * 0.1, 12);
    });
    it('subgraph selection follows the weights', () => {
        const sub = fan(0.1).semanticSubgraph([1.0, 0.0], {
            maxHops: 1, topKSeeds: 1, topKResults: 2, threshold: -1.0, weightField: 'w',
        });
        expect(new Set([...sub.nodes()].map(n => n.id))).toEqual(new Set(['S', 'Q']));
    });
    it('unweighted scores are bit-identical to decay ** hops', () => {
        const graph = vectorGraph();
        const found = graph.semanticMultiHop(Q, { maxHops: 2, topKSeeds: 1, threshold: -1.0, decay: 0.7 });
        const seed = found.find(r => r.hopCount === 0);
        for (const r of found) {
            expect(r.score).toBe(Math.max(seed.score, 0) * Math.pow(0.7, r.hopCount));
        }
    });
});
// -----------------------------------------------------------------------------
// Subgraph connectivity
// -----------------------------------------------------------------------------
describe('subgraph connectivity', () => {
    /** A and B embedded; X (hub, unembedded) links them: A -> X -> B by default. */
    function bridge(opts = {}) {
        const { relAB = 'R', relXB = 'R', aToX = true, bToX = false } = opts;
        const graph = new SemanticGraph({ autoEmbed: false });
        for (const ref of ['A', 'B', 'X'])
            graph.addNode('n', ref, { kind: ref === 'X' ? 'hub' : 'leaf' });
        if (aToX)
            graph.addEdge(relAB, ['n', 'A'], ['n', 'X']);
        else
            graph.addEdge(relAB, ['n', 'X'], ['n', 'A']);
        if (bToX)
            graph.addEdge(relXB, ['n', 'B'], ['n', 'X']);
        else
            graph.addEdge(relXB, ['n', 'X'], ['n', 'B']);
        graph.embeddingStore.add(['n', 'A'], 'a', [1.0, 0.0]);
        graph.embeddingStore.add(['n', 'B'], 'b', [0.9, 0.1]);
        return graph;
    }
    const sub = (graph, extra = {}) => graph.semanticSubgraph([1.0, 0.0], {
        maxHops: 0, topKSeeds: 2, topKResults: 2, threshold: -1.0, ...extra,
    });
    it('the default leaves the selection disconnected', () => {
        const s = sub(bridge());
        expect(refSet(s)).toEqual(new Set(['n:A', 'n:B']));
        expect(s.edgeCount()).toBe(0);
    });
    it('connect adds the bridging node and its edges', () => {
        const s = sub(bridge(), { connect: true });
        expect(refSet(s)).toEqual(new Set(['n:A', 'n:B', 'n:X']));
        expect(s.edgeCount()).toBe(2);
        // the connector carries its own properties
        expect(s.getNodeByRef(['n', 'X']).properties).toEqual({ kind: 'hub' });
    });
    it('connector nodes may exceed topKResults', () => {
        expect(sub(bridge(), { connect: true, topKResults: 2 }).nodeCount()).toBe(3);
    });
    it('the hop budget limits the search', () => {
        const graph = bridge();
        expect(refSet(sub(graph, { connect: true, connectMaxHops: 1 }))).toEqual(new Set(['n:A', 'n:B']));
        expect(refSet(sub(graph, { connect: true, connectMaxHops: 2 })).has('n:X')).toBe(true);
        expect(() => sub(graph, { connect: true, connectMaxHops: 0 })).toThrow(EmbeddingError);
    });
    it('a longer budget adds a chain of connectors', () => {
        const graph = new SemanticGraph({ autoEmbed: false });
        for (const ref of ['A', 'M1', 'M2', 'B'])
            graph.addNode('n', ref);
        graph.addEdge('R', ['n', 'A'], ['n', 'M1']);
        graph.addEdge('R', ['n', 'M1'], ['n', 'M2']);
        graph.addEdge('R', ['n', 'M2'], ['n', 'B']);
        graph.embeddingStore.add(['n', 'A'], 'a', [1.0, 0.0]);
        graph.embeddingStore.add(['n', 'B'], 'b', [0.9, 0.1]);
        expect(sub(graph, { connect: true, connectMaxHops: 2 }).nodeCount()).toBe(2);
        expect(sub(graph, { connect: true, connectMaxHops: 3 }).nodeCount()).toBe(4);
    });
    it('direction is respected', () => {
        const graph = bridge({ aToX: true, bToX: true }); // A -> X <- B
        expect(refSet(sub(graph, { connect: true }))).toEqual(new Set(['n:A', 'n:B']));
        expect(refSet(sub(graph, { connect: true, direction: Direction.BOTH })).has('n:X')).toBe(true);
    });
    it('relType is respected', () => {
        const graph = bridge({ relAB: 'KNOWS', relXB: 'LIKES' });
        expect(refSet(sub(graph, { connect: true, relType: 'KNOWS' }))).toEqual(new Set(['n:A', 'n:B']));
        expect(refSet(sub(graph, { connect: true })).has('n:X')).toBe(true);
    });
    it('one hub joins three components', () => {
        const graph = new SemanticGraph({ autoEmbed: false });
        graph.addNode('n', 'HUB');
        ['A', 'B', 'C'].forEach((ref, i) => {
            graph.addNode('n', ref);
            graph.addEdge('R', ['n', ref], ['n', 'HUB']);
            graph.embeddingStore.add(['n', ref], ref, [1.0 - i * 0.05, 0.1 * i]);
        });
        const s = graph.semanticSubgraph([1.0, 0.0], {
            maxHops: 0, topKSeeds: 3, topKResults: 3, threshold: -1.0, connect: true, direction: Direction.BOTH,
        });
        expect(refSet(s)).toEqual(new Set(['n:A', 'n:B', 'n:C', 'n:HUB']));
    });
    it('strictFilter bars failing nodes as connectors', () => {
        const graph = bridge();
        const kw = { connect: true, nodeFilter: (n) => n.properties.kind === 'leaf' };
        // lax: X fails the filter but may still bridge
        expect(refSet(sub(graph, kw)).has('n:X')).toBe(true);
        expect(refSet(sub(graph, { ...kw, strictFilter: true }))).toEqual(new Set(['n:A', 'n:B']));
    });
});
// -----------------------------------------------------------------------------
// Model lifecycle
// -----------------------------------------------------------------------------
describe('model lifecycle', () => {
    it('models() counts vectors per model', () => {
        const encoder = new MockEncoder(16);
        expect(namedGraph(encoder).embeddingStore.models()).toEqual({ [encoder.modelName]: 3 });
    });
    it('is stale by model name, then reembeds', () => {
        const graph = namedGraph(new MockEncoder(16));
        graph.setEncoder(retagged(16, 'next-gen-16'));
        expect(graph.staleEmbeddings().sort()).toEqual([['person', 1], ['person', 2], ['person', 3]]);
        expect(graph.embeddingStore.models()).toEqual({ 'mock-16': 3 });
        expect(graph.reembedStale()).toBe(3);
        expect(graph.staleEmbeddings()).toEqual([]);
        expect(graph.embeddingStore.models()).toEqual({ 'next-gen-16': 3 });
    });
    it('nothing is stale under the same encoder', () => {
        const graph = namedGraph(new MockEncoder(16));
        expect(graph.staleEmbeddings()).toEqual([]);
        expect(graph.reembedStale()).toBe(0);
    });
    it('a dimension change resets the store', () => {
        const graph = namedGraph(new MockEncoder(8));
        graph.setEncoder(new MockEncoder(16));
        expect(graph.staleEmbeddings().length).toBe(3);
        expect(graph.reembedStale()).toBe(3);
        expect(graph.embeddingStore.dimension).toBe(16);
        expect(graph.getEmbedding(['person', 1]).vector.length).toBe(16);
        expect(graph.staleEmbeddings()).toEqual([]);
    });
    it('reembed keeps custom embed text', () => {
        const graph = new SemanticGraph({ encoder: new MockEncoder(8), autoEmbed: false });
        graph.addNodeWithEmbed('person', 1, { name: 'A' }, 'a very specific blurb');
        graph.setEncoder(retagged(8, 'v2'));
        graph.reembedStale();
        const record = graph.getEmbedding(['person', 1]);
        expect(record.text).toBe('a very specific blurb');
        expect(record.model).toBe('v2');
    });
    it('an unknown model is not stale unless asked', () => {
        const graph = new SemanticGraph({ encoder: new MockEncoder(8), autoEmbed: false });
        graph.addNode('person', 1);
        graph.embeddingStore.encoder = null; // a vector handed in -> "provided"
        graph.embeddingStore.add(['person', 1], 'x', new Array(8).fill(0.1));
        graph.embeddingStore.encoder = graph.encoder;
        expect(graph.embeddingStore.models()).toEqual({ provided: 1 });
        expect(graph.staleEmbeddings()).toEqual([]);
        expect(graph.staleEmbeddings(true)).toEqual([['person', 1]]);
    });
    it('stale accepts positional and options forms', () => {
        const graph = namedGraph(new MockEncoder(16));
        expect(graph.embeddingStore.stale('other-model').length).toBe(3);
        expect(graph.embeddingStore.stale({ model: 'other-model' }).length).toBe(3);
        expect(graph.embeddingStore.stale(null, 99).length).toBe(3);
        expect(graph.embeddingStore.stale({ dimension: 16 })).toEqual([]);
    });
    it('with no encoder and no arguments nothing can be judged', () => {
        const store = new EmbeddingStore();
        store.add(['a', 1], 'x', [1, 2]);
        expect(store.stale()).toEqual([]);
    });
    it('a text search with a different model warns once', () => {
        const graph = namedGraph(new MockEncoder(16));
        graph.setEncoder(retagged(16, 'other-16'));
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        graph.similaritySearch('hello', 1, undefined, -1.0);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(String(warn.mock.calls[0][0])).toContain('other-16');
        graph.similaritySearch('hello again', 1, undefined, -1.0);
        expect(warn).toHaveBeenCalledTimes(1);
    });
    it('a text search with a different model can throw', () => {
        const graph = new SemanticGraph({
            encoder: new MockEncoder(16), autoEmbed: true, onModelMismatch: 'error',
        });
        graph.addNode('person', 1, { name: 'Alice' });
        graph.setEncoder(retagged(16, 'other-16'));
        expect(() => graph.similaritySearch('hello', 1)).toThrow(ModelMismatchError);
        expect(() => graph.similaritySearch('hello', 1)).toThrow(/mock-16/);
        expect(() => graph.semanticMultiHop('hello')).toThrow(ModelMismatchError);
        // re-embedding is the cure
        graph.reembedStale();
        expect(graph.similaritySearch('hello', 1, undefined, -1.0).length).toBeGreaterThan(0);
    });
    it('vector queries and ignore mode are never checked', () => {
        const graph = namedGraph(new MockEncoder(16));
        graph.setEncoder(retagged(16, 'other-16'));
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        graph.similaritySearch(new MockEncoder(16).encode('x'), 1, undefined, -1.0);
        const quiet = new SemanticGraph({ encoder: new MockEncoder(16), onModelMismatch: 'ignore' });
        quiet.addNode('person', 1, { name: 'Alice' });
        quiet.setEncoder(retagged(16, 'other-16'));
        quiet.similaritySearch('x', 1, undefined, -1.0);
        expect(warn).not.toHaveBeenCalled();
    });
    it('rejects an unknown mismatch mode', () => {
        expect(() => new EmbeddingStore(undefined, { onModelMismatch: 'loud' })).toThrow(EmbeddingError);
    });
    it('the version is recorded in the sidecar', () => {
        const encoder = new MockEncoder(8, '2026.1');
        const graph = namedGraph(encoder);
        const path = join(tmp(), 'g.ison');
        saveGraph(graph, path);
        const payload = JSON.parse(readFileSync(embeddingsPathFor(path), 'utf8'));
        expect(payload.encoder_version).toBe('2026.1');
        // an explicit version wins over the encoder's
        const store = new EmbeddingStore(encoder, { version: 'explicit' });
        store.add(['a', 1], 'x');
        expect(store.exportEmbeddings().encoder_version).toBe('explicit');
        // and nothing is written when no version is known
        const plain = namedGraph(new MockEncoder(8));
        expect('encoder_version' in plain.embeddingStore.exportEmbeddings()).toBe(false);
    });
    it('loading with a different version warns', () => {
        const graph = namedGraph(new MockEncoder(8, '1'));
        const path = join(tmp(), 'g.ison');
        saveGraph(graph, path);
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        loadGraph(path, { encoder: new MockEncoder(8, '2') });
        expect(warn).toHaveBeenCalledTimes(1);
        expect(String(warn.mock.calls[0][0])).toContain("version '1'");
        warn.mockClear();
        loadGraph(path, { encoder: new MockEncoder(8, '1') });
        loadGraph(path, { encoder: new MockEncoder(8) }); // unversioned: no claim
        expect(warn).not.toHaveBeenCalled();
    });
});
// -----------------------------------------------------------------------------
// Save/load consistency
// -----------------------------------------------------------------------------
describe('save/load consistency', () => {
    const enc = () => new MockEncoder(16);
    it('the sidecar records the digest and stays version 1', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        const payload = JSON.parse(readFileSync(embeddingsPathFor(path), 'utf8'));
        expect(payload.format).toBe(EMBEDDING_FORMAT);
        expect(payload.version).toBe(1);
        expect(payload.graph_digest).toMatch(/^sha256:[0-9a-f]{64}$/);
        expect(payload.graph_digest_basis).toBe(GRAPH_DIGEST_BASIS);
        expect(payload.encoder_model).toBe(encoder.modelName);
        // Optional metadata trails the array so naive readers meet
        // "format"/"version"/"embeddings" first.
        expect(Object.keys(payload).slice(0, 5)).toEqual(['format', 'version', 'dimension', 'count', 'embeddings']);
        expect(payload.embeddings[0].model).toBe(encoder.modelName);
    });
    it('saves leave no temp files', () => {
        const dir = tmp();
        saveGraph(namedGraph(enc()), join(dir, 'g.ison'));
        expect(readdirSync(dir).sort()).toEqual(['g.ison', 'g.ison.embeddings.json']);
    });
    it('a failed write keeps the previous file and leaves no temp file', () => {
        const dir = tmp();
        const target = join(dir, 'g.ison');
        writeFileAtomic(target, 'before');
        // renaming a file onto a directory fails; the previous state must survive
        const blocked = join(dir, 'blocked');
        mkdirSync(blocked);
        expect(() => writeFileAtomic(blocked, 'x')).toThrow();
        expect(readFileSync(target, 'utf8')).toBe('before');
        expect(readdirSync(dir).filter(n => n.endsWith('.tmp'))).toEqual([]);
    });
    it('a matching pair loads clean (typed values do not look mismatched)', () => {
        const encoder = enc();
        const graph = new SemanticGraph({ name: 't', encoder, autoEmbed: false });
        graph.addNode('person', 1, { name: 'A b', age: 30, ratio: 0.5, ok: true, note: null });
        graph.addNode('account', '007', { name: 'Bond' });
        graph.addEdge('KNOWS', ['person', 1], ['account', '007'], { w: 0.5, n: 2 });
        graph.embedNode(['person', 1], 'a');
        const path = join(tmp(), 't.ison');
        saveGraph(graph, path);
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        const loaded = loadGraph(path, { encoder });
        expect(loaded.embeddingStore.count()).toBe(1);
        expect(warn).not.toHaveBeenCalled();
    });
    it('ISONL round trips and verifies', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.isonl');
        saveGraph(namedGraph(encoder), path);
        expect(loadGraph(path, { encoder }).embeddingStore.count()).toBe(3);
    });
    it('a CRLF graph file loads and still verifies', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        writeFileSync(path, readFileSync(path, 'utf8').replace(/\n/g, '\r\n'), 'utf8');
        const loaded = loadGraph(path, { encoder });
        expect(loaded.nodeCount()).toBe(3);
        expect(loaded.embeddingStore.count()).toBe(3);
    });
    it('a sidecar from another save is rejected', () => {
        const encoder = enc();
        const dir = tmp();
        const pa = join(dir, 'a.ison');
        const pb = join(dir, 'b.ison');
        saveGraph(namedGraph(encoder, ['Alice', 'Bob', 'Carol']), pa);
        saveGraph(namedGraph(encoder, ['Xavier', 'Yolanda', 'Zed']), pb);
        // b's vectors dropped next to a's graph
        writeFileSync(embeddingsPathFor(pa), readFileSync(embeddingsPathFor(pb), 'utf8'), 'utf8');
        expect(() => loadGraph(pa, { encoder })).toThrow(GraphEmbeddingMismatchError);
        expect(() => loadGraph(pa, { encoder })).toThrow(/digest/);
    });
    function savedThenEdited() {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        writeFileSync(path, readFileSync(path, 'utf8').replace('Bob', 'Robert'), 'utf8');
        return { path, encoder };
    }
    it('an edited graph is rejected by default', () => {
        const { path, encoder } = savedThenEdited();
        expect(() => loadGraph(path, { encoder })).toThrow(GraphEmbeddingMismatchError);
    });
    it('warn mode loads the old vectors anyway', () => {
        const { path, encoder } = savedThenEdited();
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        const loaded = loadGraph(path, { encoder, onMismatch: 'warn' });
        expect(String(warn.mock.calls[0][0])).toContain('digest');
        expect(loaded.getEmbedding(['person', 2]).text).toBe('Bob'); // the stale text
    });
    it('reembed mode rebuilds from the edited graph', () => {
        const { path, encoder } = savedThenEdited();
        const loaded = loadGraph(path, { encoder, onMismatch: 'reembed' });
        const record = loaded.getEmbedding(['person', 2]);
        expect(record.text).toBe('Robert');
        record.vector.forEach((v, i) => expect(v).toBeCloseTo(encoder.encode('Robert')[i], 6));
        expect(loaded.embeddingStore.count()).toBe(3);
    });
    it('a sidecar without a digest loads as before', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        const sidecar = embeddingsPathFor(path);
        const payload = JSON.parse(readFileSync(sidecar, 'utf8'));
        for (const key of ['graph_digest', 'graph_digest_basis', 'encoder_model'])
            delete payload[key];
        writeFileSync(sidecar, JSON.stringify(payload), 'utf8');
        // even an edited graph is accepted: there is nothing to compare with
        writeFileSync(path, readFileSync(path, 'utf8').replace('Bob', 'Robert'), 'utf8');
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        expect(loadGraph(path, { encoder }).embeddingStore.count()).toBe(3);
        expect(warn).not.toHaveBeenCalled();
    });
    it('a digest over an unknown basis is not compared', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        const sidecar = embeddingsPathFor(path);
        const payload = JSON.parse(readFileSync(sidecar, 'utf8'));
        payload.graph_digest = 'sha256:' + '0'.repeat(64);
        payload.graph_digest_basis = 'some-other-port-v9';
        writeFileSync(sidecar, JSON.stringify(payload), 'utf8');
        expect(loadGraph(path, { encoder }).embeddingStore.count()).toBe(3);
    });
    it('a payload with unknown fields imports and remembers the metadata', () => {
        const encoder = enc();
        const source = new EmbeddingStore(encoder);
        source.add(['person', 1], 'Alice');
        const payload = source.exportEmbeddings('sha256:abc');
        payload.some_future_field = { x: 1 };
        const target = new EmbeddingStore(encoder);
        expect(target.importEmbeddings(payload)).toBe(1);
        expect(target.lastImportMeta.graph_digest).toBe('sha256:abc');
        expect(target.lastImportMeta.graph_digest_basis).toBe(GRAPH_DIGEST_BASIS);
        expect(target.lastImportMeta.encoder_model).toBe(encoder.modelName);
        expect(target.get(['person', 1]).model).toBe(encoder.modelName);
    });
    it('an imported payload with no per-vector model reads as unknown', () => {
        const source = new EmbeddingStore(new MockEncoder(8));
        source.add(['person', 1], 'Alice');
        const payload = source.exportEmbeddings();
        delete payload.embeddings[0].model;
        const target = new EmbeddingStore();
        target.importEmbeddings(payload);
        expect(target.models()).toEqual({ unknown: 1 });
    });
    it('rejects an unknown onMismatch mode', () => {
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        saveGraph(namedGraph(encoder), path);
        expect(() => loadGraph(path, { encoder, onMismatch: 'ignore' })).toThrow(/onMismatch/);
    });
    it('degrades a mismatch to a warning when the file has number literals it cannot read exactly', () => {
        // A Python-saved file can hold 1.0, which this port reads as 1 and so digests differently.
        const encoder = enc();
        const path = join(tmp(), 'g.ison');
        const graph = new SemanticGraph({ encoder, autoEmbed: false });
        graph.addNode('t', 1, { name: 'x', score: 3 });
        graph.embedNode(['t', 1], 'x');
        saveGraph(graph, path);
        // 3 -> 3.0 reads back identically here, so also edit a value to force a digest difference
        writeFileSync(path, readFileSync(path, 'utf8').replace(/\b3\b/, '3.0').replace('x', 'y'), 'utf8');
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => { });
        const loaded = loadGraph(path, { encoder }); // default 'error' must not throw here
        expect(loaded.embeddingStore.count()).toBe(1);
        expect(String(warn.mock.calls[0][0])).toContain('number literals');
    });
    it('the digest ignores insertion order but sees content', () => {
        const a = new SemanticGraph({ autoEmbed: false });
        const b = new SemanticGraph({ autoEmbed: false });
        a.addNode('t', 1, { x: 1 });
        a.addNode('t', 2, { x: 2 });
        a.addEdge('R', ['t', 1], ['t', 2]);
        b.addNode('t', 2, { x: 2 });
        b.addNode('t', 1, { x: 1 });
        b.addEdge('R', ['t', 1], ['t', 2]);
        expect(graphDigest(a)).toBe(graphDigest(b));
        b.updateNode('t', 1, { x: 99 });
        expect(graphDigest(a)).not.toBe(graphDigest(b));
        // an int id and a str id that print alike are different nodes
        const c = new SemanticGraph({ autoEmbed: false });
        const d = new SemanticGraph({ autoEmbed: false });
        c.addNode('t', 7);
        d.addNode('t', '7');
        expect(graphDigest(c)).not.toBe(graphDigest(d));
    });
});
// -----------------------------------------------------------------------------
// graphDigest: golden fixture shared with the Python port
// -----------------------------------------------------------------------------
describe('graphDigest golden fixture', () => {
    const fixture = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/graph_digest_golden.json', import.meta.url)), 'utf8'));
    it('uses the expected basis', () => {
        expect(fixture.basis).toBe(GRAPH_DIGEST_BASIS);
    });
    for (const c of fixture.cases) {
        it(`reproduces the Python digest for "${c.name}"`, () => {
            const graph = new ISONGraph(c.name, c.directed);
            for (const n of c.nodes)
                graph.addNode(n.type, n.id, n.properties);
            for (const e of c.edges)
                graph.addEdge(e.rel_type, e.source, e.target, e.properties);
            expect(graphDigest(graph)).toBe(c.digest);
        });
    }
    it('matches Python float repr, escapes and large ints (digest computed by the Python port)', () => {
        const graph = new ISONGraph('x');
        graph.addNode('t', 1, {
            a: 0.1, b: 0.0001, c: 0.00001, d: 123456.789, e: -0.75,
            f: 1234567890123.4567, g: 2.5e-8, h: 1000000000000000.5,
            big: 9007199254740991, n: null, ok: false,
        });
        graph.addNode('t', 's1', { s: 'ü\u{1F600}\u0007\u007f\n"\\ /' });
        graph.addNode('t', '007', { k: 0.30000000000000004 });
        graph.addEdge('R', ['t', 1], ['t', 's1'], { w: 0.5, z: -2 });
        expect(graphDigest(graph))
            .toBe('sha256:126c23dd68cf53c960fb9769a1083d331d332bd36223702f77ba6a8274ccf109');
    });
    it('the pure-JS SHA-256 agrees with node:crypto on the canonical body', () => {
        // Exercise the padding boundaries through graphs whose bodies straddle 55/56/64 bytes.
        for (let n = 0; n < 40; n++) {
            const graph = new ISONGraph('p');
            graph.addNode('t', 1, { s: 'x'.repeat(n) });
            const digest = graphDigest(graph);
            const body = `{"directed":true,"edges":[],"nodes":[[["t","i","1"],{"s":"${'x'.repeat(n)}"}]]}`;
            expect(digest).toBe('sha256:' + createHash('sha256').update(body).digest('hex'));
        }
    });
    it('saveGraph and loadGraph agree through real files for every fixture graph', () => {
        const encoder = new MockEncoder(8);
        for (const c of fixture.cases) {
            const graph = new SemanticGraph({ name: c.name, directed: c.directed, encoder, autoEmbed: false });
            for (const n of c.nodes)
                graph.addNode(n.type, n.id, n.properties);
            for (const e of c.edges)
                graph.addEdge(e.rel_type, e.source, e.target, e.properties);
            const path = join(tmp(), 'g.ison');
            saveGraph(graph, path);
            const payload = JSON.parse(readFileSync(embeddingsPathFor(path), 'utf8'));
            // The digest is of the graph as it reads back from the text (ISON gives
            // a node that lacks a column a null there), not of the in-memory graph.
            const loaded = loadGraph(path, { encoder }); // would throw on a mismatch
            expect(graphDigest(loaded)).toBe(payload.graph_digest);
            expect(existsSync(path)).toBe(true);
        }
    });
});
