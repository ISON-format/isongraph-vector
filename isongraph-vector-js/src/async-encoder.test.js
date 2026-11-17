/**
 * The asynchronous encoder path: AsyncEncoder, the *Async methods on the store
 * and the graph, and HttpEncoder driving auto-embedding with no prefetch.
 * Everything runs against a local node:http server - never the real network.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { createServer } from 'node:http';
import {
  SemanticGraph, EmbeddingStore, MockEncoder, HttpEncoder, HttpEncoderError, EmbeddingError,
  DimensionMismatchError, ModelMismatchError, NoEncoderError, Direction, isAsyncEncoder,
} from './index.js';

const servers = [];
afterEach(async () => {
  vi.restoreAllMocks();
  while (servers.length) {
    const server = servers.pop();
    server.closeAllConnections();
    await new Promise(resolve => server.close(() => resolve()));
  }
});

/** A server whose answers come from `respond(request, n)`; may delay with `wait` ms or `hang`. */
async function serve(respond) {
  const seen = [];
  let inflight = 0;
  let peak = 0;
  const server = createServer((req, res) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      const request = {
        method: req.method, url: req.url, headers: req.headers,
        body: raw ? JSON.parse(raw) : null,
      };
      const n = seen.length;
      seen.push(request);
      const answer = respond(request, n);
      if (answer.hang) return;
      inflight++;
      peak = Math.max(peak, inflight);
      setTimeout(() => {
        inflight--;
        res.writeHead(answer.status ?? 200, { 'content-type': 'application/json', ...(answer.headers ?? {}) });
        res.end(typeof answer.body === 'string' ? answer.body : JSON.stringify(answer.body));
      }, answer.wait ?? 0);
    });
  });
  servers.push(server);
  await new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve()));
  const { port } = server.address();
  return { base: `http://127.0.0.1:${port}`, seen, peak: () => peak };
}

/** Topic vectors: texts that mention a topic word point along its axis. */
function vectorOf(text) {
  const t = text.toLowerCase();
  return [
    t.includes('engineer') ? 1 : 0,
    t.includes('manager') ? 1 : 0,
    t.includes('design') ? 1 : 0,
    0.1,
  ];
}

const openaiAnswer = request => ({
  body: {
    data: request.body.input.map((text, index) => ({ index, embedding: vectorOf(text) })).reverse(),
  },
});
const ollamaAnswer = request => ({ body: { embeddings: request.body.input.map(vectorOf) } });

const fast = { retryDelayMs: 1 };

describe('AsyncEncoder', () => {
  it('HttpEncoder is one, MockEncoder is not', () => {
    const http = new HttpEncoder({ url: 'http://localhost:1/e', model: 'm' });
    expect(isAsyncEncoder(http)).toBe(true);
    expect(isAsyncEncoder(new MockEncoder(8))).toBe(false);
    expect(isAsyncEncoder(null)).toBe(false);
    expect(isAsyncEncoder({ encodeAsync: () => 1 })).toBe(false);
  });

  const remote = () => ({
    dimension: 4,
    modelName: 'only-async',
    version: 'v9',
    encodeAsync: async text => vectorOf(text),
    encodeBatchAsync: async texts => texts.map(vectorOf),
  });

  it('a pure async encoder works with the Async methods and stamps its model', async () => {
    const store = new EmbeddingStore(remote());
    expect(store.encoder).not.toBeNull();
    await store.addAsync(['p', 1], 'Alice the engineer');
    await store.addBatchAsync([[['p', 2], 'Bob the manager'], [['p', 3], 'Carol the designer']]);

    expect(store.count()).toBe(3);
    expect(store.models()).toEqual({ 'only-async': 3 });
    expect(store.version).toBe('v9');
    const hits = await store.similaritySearchAsync('engineering', 1);
    expect(hits[0].nodeRef).toEqual(['p', 1]);
  });

  it('its synchronous face says to use the Async method', async () => {
    const store = new EmbeddingStore(remote());
    expect(() => store.add(['p', 1], 'text')).toThrow(EmbeddingError);
    expect(() => store.add(['p', 1], 'text')).toThrow(/Async/);
    expect(() => store.similaritySearch('text')).toThrow(/Async/);
    // vectors you already have never need the encoder
    store.add(['p', 1], 'text', [1, 0, 0, 0]);
    expect(store.similaritySearch([1, 0, 0, 0], 1).length).toBe(1);
  });

  it('works through a graph and reports stale vectors when the model changes', async () => {
    const graph = new SemanticGraph({ encoder: remote() });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer' });
    expect(graph.staleEmbeddings()).toEqual([]);

    graph.setEncoder({ ...remote(), modelName: 'newer' });
    expect(graph.staleEmbeddings()).toEqual([['person', 1]]);
    expect(await graph.reembedStaleAsync()).toBe(1);
    expect(graph.embeddingStore.models()).toEqual({ newer: 1 });
  });

  it('async methods also work with a synchronous encoder', async () => {
    const encoder = new MockEncoder(16);
    const sync = new SemanticGraph({ encoder });
    sync.addNode('person', 1, { name: 'Alice' });
    const viaAsync = new SemanticGraph({ encoder });
    await viaAsync.addNodeAsync('person', 1, { name: 'Alice' });
    expect(viaAsync.getEmbedding(['person', 1]).vector).toEqual(sync.getEmbedding(['person', 1]).vector);
    const hits = await viaAsync.similaritySearchAsync('Alice', 1);
    expect(hits[0].score).toBeCloseTo(1, 6);
  });

  it('refuses a text query with no encoder', async () => {
    const store = new EmbeddingStore();
    await expect(store.similaritySearchAsync('text')).rejects.toThrow(NoEncoderError);
    await expect(store.addAsync(['p', 1], 'text')).rejects.toThrow(NoEncoderError);
    store.add(['p', 1], 'x', [1, 0]);
    expect(await store.similaritySearchAsync([1, 0], 1)).toHaveLength(1);
  });

  it('rejects an encoder that returns the wrong number of vectors', async () => {
    const broken = {
      dimension: 4, encodeAsync: async () => [1, 0, 0, 0], encodeBatchAsync: async () => [[1, 0, 0, 0]],
    };
    const store = new EmbeddingStore(broken);
    await expect(store.addBatchAsync([[['p', 1], 'a'], [['p', 2], 'b']])).rejects.toThrow(/1 vectors for 2 texts/);
    expect(store.count()).toBe(0);
  });
});

describe('SemanticGraph auto-embed over HTTP (openai shape)', () => {
  it('addNodeAsync embeds with no prefetch; sync addNode still needs one', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/v1/embeddings`, model: 'remote-4', apiKey: 'sk-test-key' });
    const graph = new SemanticGraph({ encoder });

    const node = await graph.addNodeAsync('person', 1, { name: 'Alice', description: 'an engineer' });
    expect(node.ref).toEqual(['person', 1]);
    expect(seen).toHaveLength(1);
    expect(seen[0].body.input).toEqual(['Alice an engineer']);
    expect(seen[0].headers.authorization).toBe('Bearer sk-test-key');
    const record = graph.getEmbedding(['person', 1]);
    expect(record.vector).toEqual(vectorOf('Alice an engineer'));
    expect(record.model).toBe('remote-4');
    expect(graph.embeddingStore.dimension).toBe(4);

    // the synchronous path is unchanged: it needs the text prefetched
    expect(() => graph.addNode('person', 2, { name: 'Bob the manager' })).toThrow(HttpEncoderError);
    await encoder.prefetch(['Bob the manager']);
    graph.addNode('person', 3, { name: 'Bob the manager' });
    expect(graph.getEmbedding(['person', 3])).not.toBeNull();
  });

  it('a node with nothing to embed makes no request', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await graph.addNodeAsync('person', 1, { age: 30 });
    expect(seen).toHaveLength(0);
    expect(graph.getEmbedding(['person', 1])).toBeNull();
    expect(graph.getNode('person', 1).properties.age).toBe(30);
  });

  it('autoEmbed off leaves the node unembedded; addNodeWithEmbedAsync embeds the given text', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm' });
    const off = new SemanticGraph({ encoder, autoEmbed: false });
    await off.addNodeAsync('person', 1, { name: 'Alice' });
    expect(seen).toHaveLength(0);
    await off.addNodeWithEmbedAsync('person', 2, { name: 'Bob' }, 'a manager');
    expect(seen[0].body.input).toEqual(['a manager']);
    expect(off.getEmbedding(['person', 2]).text).toBe('a manager');
    await off.addNodeWithEmbedAsync('person', 3, { name: 'Carol the designer' });   // autoEmbed off -> no embed text, no vector
    expect(off.getEmbedding(['person', 3])).toBeNull();
  });

  it('updateNodeAsync re-embeds on an embed-field change and only then', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer', age: 30 });
    expect(seen).toHaveLength(1);

    await graph.updateNodeAsync('person', 1, { age: 31 });
    expect(seen).toHaveLength(1);                                // no embed field touched

    await graph.updateNodeAsync('person', 1, { name: 'Alice the manager' });
    expect(seen).toHaveLength(2);
    expect(graph.getEmbedding(['person', 1]).text).toBe('Alice the manager');
    expect(graph.getEmbedding(['person', 1]).vector).toEqual(vectorOf('manager'));

    await graph.updateNodeAsync('person', 1, { name: null });    // nothing left to embed
    expect(graph.getEmbedding(['person', 1])).toBeNull();
    expect(seen).toHaveLength(2);
  });

  it('searches by text over the network and ranks by meaning', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await graph.embedAllNodesAsync();                             // empty graph: nothing to do
    for (const [id, name] of [[1, 'Alice the engineer'], [2, 'Bob the manager'], [3, 'Carol the designer']]) {
      await graph.addNodeAsync('person', id, { name });
    }
    const before = seen.length;
    const hits = await graph.similaritySearchAsync('who is the engineering lead', 2, undefined, 0);
    expect(seen.length).toBe(before + 1);
    expect(seen[before].body.input).toEqual(['who is the engineering lead']);
    expect(hits[0].nodeRef).toEqual(['person', 1]);

    const filtered = await graph.similaritySearchAsync('engineering', 5, undefined, -1, n => n.id !== 1);
    expect(filtered.every(r => r.nodeRef[1] !== 1)).toBe(true);
  });

  it('semanticMultiHopAsync, semanticPathAsync and semanticSubgraphAsync encode the query once', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer' });
    await graph.addNodeAsync('person', 2, { name: 'Bob the manager' });
    await graph.addNodeAsync('person', 3, { name: 'Carol the designer' });
    graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
    graph.addEdge('KNOWS', ['person', 2], ['person', 3]);

    const mark = seen.length;
    const hops = await graph.semanticMultiHopAsync('engineering lead', { relType: 'KNOWS', maxHops: 2, threshold: 0.5 });
    expect(seen.length).toBe(mark + 1);
    expect(hops.map(r => r.nodeRef[1])).toContain(2);

    const path = await graph.semanticPathAsync('an engineer', ['person', 3], { relType: 'KNOWS', threshold: 0.5 });
    expect(path.path.map(r => r[1])).toEqual([1, 2, 3]);
    expect(seen.length).toBe(mark + 2);

    const sub = await graph.semanticSubgraphAsync('the engineers', { relType: 'KNOWS', maxHops: 1, threshold: 0.5, direction: Direction.OUT });
    expect(sub.nodeCount()).toBe(2);
    expect(seen.length).toBe(mark + 3);

    // the sync versions still take a vector, no encoder involved
    expect(graph.semanticMultiHop(vectorOf('engineer'), { relType: 'KNOWS', threshold: 0.5 }).length).toBeGreaterThan(0);
    expect(seen.length).toBe(mark + 3);
  });

  it('similarToNode needs no encoder: a plain sync call keeps working', async () => {
    const { base } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer' });
    await graph.addNodeAsync('person', 2, { name: 'Bob the engineer' });
    expect(graph.similarToNode(['person', 1], 1)[0].nodeRef).toEqual(['person', 2]);
  });
});

describe('ollama shape', () => {
  it('auto-embeds and searches', async () => {
    const { base, seen } = await serve(ollamaAnswer);
    const encoder = new HttpEncoder({ url: `${base}/api/embed`, model: 'nomic', api: 'ollama' });
    const graph = new SemanticGraph({ encoder });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer' });
    await graph.addNodeAsync('person', 2, { name: 'Bob the manager' });
    expect(seen[0].url).toBe('/api/embed');
    expect(seen[0].body).toEqual({ model: 'nomic', input: ['Alice the engineer'] });
    const hits = await graph.similaritySearchAsync('engineer', 1);
    expect(hits[0].nodeRef).toEqual(['person', 1]);
    expect(graph.embeddingStore.models()).toEqual({ nomic: 2 });
  });

  it('a vector of the wrong length is a DimensionMismatchError and changes nothing', async () => {
    const { base } = await serve(ollamaAnswer);
    const encoder = new HttpEncoder({ url: `${base}/api/embed`, model: 'nomic', api: 'ollama', dimension: 8 });
    const graph = new SemanticGraph({ encoder });
    await expect(graph.addNodeAsync('person', 1, { name: 'Alice' })).rejects.toThrow(DimensionMismatchError);
    expect(graph.nodeCount()).toBe(0);
  });
});

describe('batching and concurrency', () => {
  it('embedAllNodesAsync sends the texts in batchSize requests', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', batchSize: 40 });
    const graph = new SemanticGraph({ encoder, autoEmbed: false });
    for (let i = 0; i < 100; i++) graph.addNode('doc', i, { name: `document ${i} engineer` });

    expect(await graph.embedAllNodesAsync()).toBe(100);
    expect(seen.map(r => r.body.input.length)).toEqual([40, 40, 20]);
    expect(graph.embeddingStore.count()).toBe(100);

    // skipExisting: nothing is requested for nodes that already have a vector
    expect(await graph.embedAllNodesAsync(undefined, true)).toBe(0);
    expect(seen).toHaveLength(3);
  });

  it('concurrency runs chunk requests in parallel, up to the limit', async () => {
    const { base, seen, peak } = await serve(request => ({ ...openaiAnswer(request), wait: 40 }));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', batchSize: 1, concurrency: 3 });
    const vectors = await encoder.encodeBatchAsync(['a', 'b', 'c', 'd', 'e', 'f', 'g']);
    expect(vectors).toEqual(['a', 'b', 'c', 'd', 'e', 'f', 'g'].map(vectorOf));
    expect(seen).toHaveLength(7);
    expect(peak()).toBe(3);

    const serial = await serve(request => ({ ...openaiAnswer(request), wait: 10 }));
    await new HttpEncoder({ url: `${serial.base}/e`, model: 'm', batchSize: 1 }).encodeBatchAsync(['a', 'b', 'c']);
    expect(serial.peak()).toBe(1);                                // default stays sequential
  });

  it('validates concurrency', () => {
    expect(() => new HttpEncoder({ url: 'http://localhost:1/e', model: 'm', concurrency: 0 })).toThrow(/concurrency/);
    expect(() => new HttpEncoder({ url: 'http://localhost:1/e', model: 'm', concurrency: 1.5 })).toThrow(/concurrency/);
  });

  it('identical texts requested at the same moment share one request', async () => {
    const { base, seen } = await serve(request => ({ ...openaiAnswer(request), wait: 30 }));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm' });
    const results = await Promise.all([
      encoder.encodeAsync('same engineer'), encoder.encodeAsync('same engineer'), encoder.encodeAsync('same engineer'),
    ]);
    expect(seen).toHaveLength(1);
    expect(results[0]).toEqual(vectorOf('same engineer'));
    expect(results[1]).not.toBe(results[0]);                      // each caller owns its copy
  });

  it('many nodes added at once all land', async () => {
    const { base, seen } = await serve(request => ({ ...openaiAnswer(request), wait: 5 }));
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm' }) });
    await Promise.all(Array.from({ length: 25 }, (_, i) => graph.addNodeAsync('doc', i, { name: `doc ${i} designer` })));
    expect(graph.nodeCount()).toBe(25);
    expect(graph.embeddingStore.count()).toBe(25);
    expect(seen.length).toBeGreaterThan(0);
    expect((await graph.similaritySearchAsync('design', 25, undefined, 0.5)).length).toBe(25);
  });

  it('a failed request in the middle of a batch rejects the whole call and stores nothing', async () => {
    const { base } = await serve((request, n) =>
      n === 1 ? { status: 400, body: 'bad' } : openaiAnswer(request));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', batchSize: 2, ...fast });
    const graph = new SemanticGraph({ encoder, autoEmbed: false });
    for (let i = 0; i < 6; i++) graph.addNode('doc', i, { name: `doc ${i}` });
    await expect(graph.embedAllNodesAsync()).rejects.toThrow(HttpEncoderError);
    expect(graph.embeddingStore.count()).toBe(0);
  });
});

describe('stale detection and re-embedding over the network', () => {
  it('reembedStaleAsync rewrites vectors from another model, in batches', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'old', batchSize: 100 }) });
    for (let i = 0; i < 5; i++) await graph.addNodeAsync('doc', i, { name: `doc ${i} engineer` });
    expect(graph.staleEmbeddings()).toEqual([]);

    graph.setEncoder(new HttpEncoder({ url: `${base}/e`, model: 'new', batchSize: 100 }));
    expect(graph.staleEmbeddings()).toHaveLength(5);
    const mark = seen.length;
    expect(await graph.reembedStaleAsync(false, 2)).toBe(5);
    expect(seen.length - mark).toBe(3);                           // 2 + 2 + 1 texts
    expect(graph.embeddingStore.models()).toEqual({ new: 5 });
    expect(graph.staleEmbeddings()).toEqual([]);
    expect(seen[mark].body.model).toBe('new');
  });

  it('a different dimension empties the store and re-embeds everything, learning it from the answer', async () => {
    const wide = await serve(request => ({
      body: { data: request.body.input.map((t, index) => ({ index, embedding: [...vectorOf(t), 0.5] })) },
    }));
    const { base } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'four' }) });
    await graph.addNodeAsync('doc', 1, { name: 'Alice the engineer' });
    expect(graph.embeddingStore.dimension).toBe(4);

    graph.setEncoder(new HttpEncoder({ url: `${wide.base}/e`, model: 'five' }));   // dimension not known yet
    expect(await graph.reembedStaleAsync()).toBe(1);
    expect(graph.embeddingStore.dimension).toBe(5);
    expect(graph.getEmbedding(['doc', 1]).vector).toHaveLength(5);
  });

  it('a failing re-embed leaves the old vectors alone', async () => {
    const { base } = await serve((request, n) => n === 0 ? openaiAnswer(request) : { status: 400, body: 'no' });
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'old', ...fast }) });
    await graph.addNodeAsync('doc', 1, { name: 'Alice the engineer' });
    graph.setEncoder(new HttpEncoder({ url: `${base}/e`, model: 'new', ...fast }));
    await expect(graph.reembedStaleAsync()).rejects.toThrow(HttpEncoderError);
    expect(graph.getEmbedding(['doc', 1]).model).toBe('old');
    expect(graph.staleEmbeddings()).toHaveLength(1);
  });

  it('a text query from another model follows onModelMismatch', async () => {
    const { base } = await serve(openaiAnswer);
    const store = new EmbeddingStore(new HttpEncoder({ url: `${base}/e`, model: 'old' }), { onModelMismatch: 'error' });
    await store.addAsync(['p', 1], 'Alice the engineer');
    store.encoder = new HttpEncoder({ url: `${base}/e`, model: 'new' });
    await expect(store.similaritySearchAsync('engineer')).rejects.toThrow(ModelMismatchError);
    expect(await store.similaritySearchAsync([1, 0, 0, 0.1], 1)).toHaveLength(1);   // vectors are never checked

    store.onModelMismatch = 'warn';
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await store.similaritySearchAsync('engineer');
    await store.similaritySearchAsync('engineer');
    expect(warn).toHaveBeenCalledTimes(1);
  });
});

describe('failures from the network', () => {
  it('a server error leaves the graph as it was', async () => {
    const { base } = await serve(() => ({ status: 500, body: 'boom' }));
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm', maxRetries: 0 }) });
    const error = await graph.addNodeAsync('person', 1, { name: 'Alice' }).catch(e => e);
    expect(error).toBeInstanceOf(HttpEncoderError);
    expect(error.status).toBe(500);
    expect(graph.nodeCount()).toBe(0);
    expect(graph.embeddingStore.count()).toBe(0);
    // ... so the call can simply be retried
    await expect(graph.addNodeAsync('person', 1, { name: 'Alice' })).rejects.toThrow(HttpEncoderError);
  });

  it('updateNodeAsync that cannot embed keeps the old properties and vector', async () => {
    const { base } = await serve((request, n) => n === 0 ? openaiAnswer(request) : { status: 503, body: 'down' });
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm', maxRetries: 0 }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice the engineer' });
    await expect(graph.updateNodeAsync('person', 1, { name: 'Alice the manager' })).rejects.toThrow(HttpEncoderError);
    expect(graph.getNode('person', 1).properties.name).toBe('Alice the engineer');
    expect(graph.getEmbedding(['person', 1]).text).toBe('Alice the engineer');
  });

  it('retries a 429 then succeeds', async () => {
    const { base, seen } = await serve((request, n) =>
      n === 0 ? { status: 429, headers: { 'retry-after': '0' }, body: 'slow' } : openaiAnswer(request));
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm', ...fast }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice' });
    expect(seen).toHaveLength(2);
    expect(graph.embeddingStore.count()).toBe(1);
  });

  it('a request that never completes times out, and the node is not added', async () => {
    const { base, seen } = await serve(() => ({ hang: true }));
    const graph = new SemanticGraph({
      encoder: new HttpEncoder({ url: `${base}/e`, model: 'm', timeoutMs: 50, maxRetries: 1, ...fast }),
    });
    const error = await graph.addNodeAsync('person', 1, { name: 'Alice' }).catch(e => e);
    expect(error).toBeInstanceOf(HttpEncoderError);
    expect(error.message).toContain('timed out');
    expect(seen).toHaveLength(2);
    expect(graph.nodeCount()).toBe(0);
    await expect(graph.similaritySearchAsync('text')).rejects.toThrow(/timed out/);
  });

  it('a refused connection is reported, not hung', async () => {
    const { base } = await serve(openaiAnswer);
    const dead = base.replace(/:\d+$/, ':1');
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${dead}/e`, model: 'm', maxRetries: 0 }) });
    await expect(graph.addNodeAsync('person', 1, { name: 'Alice' })).rejects.toThrow(/failed/);
  });

  it('the api key never appears in an error from any async path', async () => {
    const secret = 'sk-do-not-leak-0123456789';
    // a server that echoes the credentials back, the worst case
    const { base } = await serve(request => ({ status: 401, body: `rejected ${request.headers.authorization}` }));
    const encoder = new HttpEncoder({
      url: `${base}/e?api_key=query-secret`, model: 'm', apiKey: secret, headers: { 'X-Api-Token': 'header-secret' }, ...fast,
    });
    const graph = new SemanticGraph({ encoder });
    const errors = [];
    errors.push(await graph.addNodeAsync('person', 1, { name: 'Alice' }).catch(e => e));
    errors.push(await graph.similaritySearchAsync('Alice').catch(e => e));
    const manual = new SemanticGraph({ encoder, autoEmbed: false });
    manual.addNode('person', 1, { name: 'Alice' });
    errors.push(await manual.embedAllNodesAsync().catch(e => e));
    errors.push(await graph.embeddingStore.addBatchAsync([[['p', 1], 'x']]).catch(e => e));
    for (const error of errors) {
      expect(error).toBeInstanceOf(Error);
      for (const leaked of [secret, 'query-secret', 'header-secret']) {
        expect(String(error.message)).not.toContain(leaked);
        expect(String(error.stack)).not.toContain(leaked);
      }
    }
    expect(errors.some(e => e instanceof HttpEncoderError && e.status === 401)).toBe(true);
    expect(JSON.stringify(graph.encoder)).not.toContain(secret);
  });

  it('sends the key as a bearer token and lets custom headers win', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const graph = new SemanticGraph({ encoder: new HttpEncoder({ url: `${base}/e`, model: 'm', apiKey: 'sk-abc' }) });
    await graph.addNodeAsync('person', 1, { name: 'Alice' });
    await graph.similaritySearchAsync('Alice');
    expect(seen.every(r => r.headers.authorization === 'Bearer sk-abc')).toBe(true);

    const custom = await serve(openaiAnswer);
    const g2 = new SemanticGraph({
      encoder: new HttpEncoder({ url: `${custom.base}/e`, model: 'm', apiKey: 'ignored', headers: { Authorization: 'Token zzz' } }),
    });
    await g2.addNodeAsync('person', 1, { name: 'Alice' });
    expect(custom.seen[0].headers.authorization).toBe('Token zzz');
  });
});
