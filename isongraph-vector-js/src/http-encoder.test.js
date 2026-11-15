/**
 * HttpEncoder against a local node:http server - never the real network.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { createServer } from 'node:http';
import {
  HttpEncoder, HttpEncoderError, EmbeddingStore, EmbeddingError, DimensionMismatchError,
} from './index.js';

const servers = [];
afterEach(async () => {
  while (servers.length) {
    const server = servers.pop();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});

/**
 * Start a server whose answers come from `respond(request, n)`, where n counts
 * requests from 0. `respond` returns { status?, headers?, body?, hang? }.
 */
async function serve(respond) {
  const seen = [];
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
      if (answer.hang) return;            // never answer: exercises the timeout
      res.writeHead(answer.status ?? 200, { 'content-type': 'application/json', ...(answer.headers ?? {}) });
      res.end(typeof answer.body === 'string' ? answer.body : JSON.stringify(answer.body));
    });
  });
  servers.push(server);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  return { base: `http://127.0.0.1:${port}`, seen };
}

/** A deterministic 4-d vector per text, so tests can predict the answer. */
function vectorOf(text) {
  return [text.length, 1, 0, 0.5];
}

const openaiAnswer = (request) => ({
  body: {
    object: 'list',
    // deliberately reversed: the client must sort by `index`
    data: request.body.input.map((text, index) => ({ object: 'embedding', index, embedding: vectorOf(text) })).reverse(),
  },
});

const ollamaAnswer = (request) => ({ body: { model: request.body.model, embeddings: request.body.input.map(vectorOf) } });

const fast = { retryDelayMs: 1 };

describe('HttpEncoder (openai)', () => {
  it('posts model and input, sends the key, sorts by index', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/v1/embeddings`, model: 'text-embed-x', apiKey: 'sk-secret-123' });

    expect(Number.isNaN(encoder.dimension)).toBe(true);        // not known yet
    const vectors = await encoder.encodeBatchAsync(['a', 'bbb', 'cc']);

    expect(vectors).toEqual([vectorOf('a'), vectorOf('bbb'), vectorOf('cc')]);
    expect(seen.length).toBe(1);
    expect(seen[0].method).toBe('POST');
    expect(seen[0].url).toBe('/v1/embeddings');
    expect(seen[0].headers.authorization).toBe('Bearer sk-secret-123');
    expect(seen[0].headers['content-type']).toBe('application/json');
    expect(seen[0].body).toEqual({ model: 'text-embed-x', input: ['a', 'bbb', 'cc'] });
    expect(encoder.modelName).toBe('text-embed-x');
    expect(encoder.dimension).toBe(4);
  });

  it('omits Authorization without a key, and custom headers win', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const plain = new HttpEncoder({ url: `${base}/e`, model: 'm' });
    await plain.encodeAsync('x');
    expect(seen[0].headers.authorization).toBeUndefined();

    const custom = new HttpEncoder({
      url: `${base}/e`, model: 'm', apiKey: 'ignored', headers: { Authorization: 'Token abc', 'X-Team': 'blue' },
    });
    await custom.encodeAsync('x');
    expect(seen[1].headers.authorization).toBe('Token abc');
    expect(seen[1].headers['x-team']).toBe('blue');
  });

  it('splits into batches, dedupes, and caches', async () => {
    const { base, seen } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', batchSize: 2 });

    await encoder.prefetch(['t1', 't2', 't3', 't4', 't5', 't1']);
    expect(seen.map(r => r.body.input)).toEqual([['t1', 't2'], ['t3', 't4'], ['t5']]);
    expect(encoder.cacheSize).toBe(5);

    await encoder.prefetch(['t3', 't5']);          // already cached
    expect(seen.length).toBe(3);
    expect(await encoder.encodeBatchAsync([])).toEqual([]);
    expect(seen.length).toBe(3);
  });

  it('the synchronous interface needs a prefetch', async () => {
    const { base } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm' });

    expect(() => encoder.encode('hello')).toThrow(HttpEncoderError);
    expect(() => encoder.encode('hello')).toThrow(/prefetch/);
    await encoder.prefetch(['hello', 'world']);
    expect(encoder.encode('hello')).toEqual(vectorOf('hello'));
    expect(encoder.encodeBatch(['world', 'hello'])).toEqual([vectorOf('world'), vectorOf('hello')]);

    const copy = encoder.encode('hello');
    copy[0] = 99;                                   // callers cannot corrupt the cache
    expect(encoder.encode('hello')).toEqual(vectorOf('hello'));

    encoder.clearCache();
    expect(encoder.cacheSize).toBe(0);
    expect(() => encoder.encode('hello')).toThrow(HttpEncoderError);
  });

  it('bounds the cache, oldest first', async () => {
    const { base } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', maxCacheEntries: 2, batchSize: 1 });
    await encoder.prefetch(['a', 'b', 'c']);
    expect(encoder.cacheSize).toBe(2);
    expect(() => encoder.encode('a')).toThrow(HttpEncoderError);
    expect(encoder.encode('c')).toEqual(vectorOf('c'));
  });

  it('drives an EmbeddingStore once the texts are prefetched', async () => {
    const { base } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'remote-4', version: 'rev-7' });
    const texts = ['Alice the engineer', 'Bob', 'engineering'];
    await encoder.prefetch(texts);

    const store = new EmbeddingStore(encoder);
    store.addBatch([[['p', 1], texts[0]], [['p', 2], texts[1]]]);
    expect(store.models()).toEqual({ 'remote-4': 2 });
    expect(store.dimension).toBe(4);
    expect(store.similaritySearch('engineering', 1, undefined, -1.0).length).toBe(1);
    expect(store.exportEmbeddings().encoder_model).toBe('remote-4');
    expect(store.exportEmbeddings().encoder_version).toBe('rev-7');
    expect(() => store.similaritySearch('never fetched', 1)).toThrow(HttpEncoderError);

    store.encoder = new HttpEncoder({ url: `${base}/e`, model: 'remote-4b' });
    expect(store.stale().length).toBe(2);
  });
});

describe('HttpEncoder (ollama)', () => {
  it('parses embeddings and keeps the declared dimension', async () => {
    const { base, seen } = await serve(ollamaAnswer);
    const encoder = new HttpEncoder({ url: `${base}/api/embed`, model: 'nomic', api: 'ollama', dimension: 4 });
    expect(encoder.dimension).toBe(4);

    expect(await encoder.encodeBatchAsync(['aa', 'b'])).toEqual([vectorOf('aa'), vectorOf('b')]);
    expect(seen[0].url).toBe('/api/embed');
    expect(seen[0].body).toEqual({ model: 'nomic', input: ['aa', 'b'] });
  });

  it('rejects a vector of the wrong length', async () => {
    const { base } = await serve(ollamaAnswer);
    const encoder = new HttpEncoder({ url: `${base}/api/embed`, model: 'nomic', api: 'ollama', dimension: 8 });
    await expect(encoder.encodeAsync('x')).rejects.toThrow(DimensionMismatchError);
  });

  it('says so when the answer is not the configured shape', async () => {
    const { base } = await serve(openaiAnswer);
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', api: 'ollama' });
    await expect(encoder.encodeAsync('x')).rejects.toThrow(/'embeddings'/);
  });
});

describe('HttpEncoder failures', () => {
  it('retries a 429 and a 5xx, then succeeds', async () => {
    const { base, seen } = await serve((request, n) => {
      if (n === 0) return { status: 429, headers: { 'retry-after': '0' }, body: { error: 'slow down' } };
      if (n === 1) return { status: 503, body: 'unavailable' };
      return openaiAnswer(request);
    });
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', ...fast });
    expect(await encoder.encodeAsync('hi')).toEqual(vectorOf('hi'));
    expect(seen.length).toBe(3);
  });

  it('gives up after maxRetries with the status', async () => {
    const { base, seen } = await serve(() => ({ status: 500, body: 'boom' }));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', maxRetries: 2, ...fast });
    const error = await encoder.encodeAsync('hi').catch(e => e);
    expect(error).toBeInstanceOf(HttpEncoderError);
    expect(error).toBeInstanceOf(EmbeddingError);
    expect(error.status).toBe(500);
    expect(error.message).toContain('HTTP 500');
    expect(seen.length).toBe(3);                    // 1 try + 2 retries, no more
  });

  it('does not retry a client error', async () => {
    const { base, seen } = await serve(() => ({ status: 400, body: { error: 'bad input' } }));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', ...fast });
    const error = await encoder.encodeAsync('hi').catch(e => e);
    expect(error.status).toBe(400);
    expect(error.message).toContain('bad input');
    expect(seen.length).toBe(1);
  });

  it('never puts the api key or url credentials in an error', async () => {
    // a server that echoes the credentials back, the worst case
    const { base } = await serve(request => ({ status: 401, body: `rejected ${request.headers.authorization}` }));
    const encoder = new HttpEncoder({ url: `${base}/e?api_key=query-secret`, model: 'm', apiKey: 'sk-very-secret', ...fast });
    const error = await encoder.encodeAsync('hi').catch(e => e);

    expect(error.status).toBe(401);
    for (const secret of ['sk-very-secret', 'query-secret']) {
      expect(error.message).not.toContain(secret);
    }
    expect(JSON.stringify(Object.entries(encoder))).not.toContain('sk-very-secret');
    expect(JSON.stringify(encoder)).not.toContain('sk-very-secret');
  });

  it('times out a server that never answers', async () => {
    const { base, seen } = await serve(() => ({ hang: true }));
    const encoder = new HttpEncoder({ url: `${base}/e`, model: 'm', timeoutMs: 60, maxRetries: 1, ...fast });
    const error = await encoder.encodeAsync('hi').catch(e => e);
    expect(error).toBeInstanceOf(HttpEncoderError);
    expect(error.message).toContain('timed out');
    expect(seen.length).toBe(2);
  });

  it('reports a refused connection', async () => {
    const { base } = await serve(openaiAnswer);
    const dead = base.replace(/:\d+$/, ':1');       // nothing listens on port 1
    const encoder = new HttpEncoder({ url: `${dead}/e`, model: 'm', maxRetries: 0 });
    await expect(encoder.encodeAsync('hi')).rejects.toThrow(/failed/);
  });

  it('rejects a malformed answer', async () => {
    const wrongCount = await serve(() => ({ body: { data: [{ index: 0, embedding: [1] }] } }));
    await expect(new HttpEncoder({ url: `${wrongCount.base}/e`, model: 'm' }).encodeBatchAsync(['a', 'b']))
      .rejects.toThrow(/1 embeddings for 2 texts/);

    const notJson = await serve(() => ({ body: 'not json at all' }));
    await expect(new HttpEncoder({ url: `${notJson.base}/e`, model: 'm' }).encodeAsync('a'))
      .rejects.toThrow(/did not return JSON/);

    const junk = await serve(() => ({ body: { data: [{ index: 0, embedding: ['x'] }] } }));
    await expect(new HttpEncoder({ url: `${junk.base}/e`, model: 'm' }).encodeAsync('a'))
      .rejects.toThrow(/numbers/);
  });

  it('validates its options', () => {
    const ok = { url: 'http://localhost:1/e', model: 'm' };
    expect(() => new HttpEncoder({ ...ok, url: 'not a url' })).toThrow(EmbeddingError);
    expect(() => new HttpEncoder({ ...ok, url: 'file:///etc/passwd' })).toThrow(/http or https/);
    expect(() => new HttpEncoder({ ...ok, url: 'http://user:pw-secret@localhost/e' })).toThrow(/credentials/);
    expect(() => new HttpEncoder({ ...ok, model: '' })).toThrow(EmbeddingError);
    expect(() => new HttpEncoder({ ...ok, api: 'cohere' })).toThrow(/openai/);
    expect(() => new HttpEncoder({ ...ok, batchSize: 0 })).toThrow(/batchSize/);
    expect(() => new HttpEncoder({ ...ok, timeoutMs: -5 })).toThrow(/timeoutMs/);
    expect(() => new HttpEncoder({ ...ok, dimension: 1.5 })).toThrow(/dimension/);
    expect(() => new HttpEncoder({ ...ok, maxRetries: -1 })).toThrow(/maxRetries/);
  });
});
