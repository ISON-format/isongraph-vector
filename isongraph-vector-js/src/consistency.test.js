/**
 * updateNode re-embedding, semanticPath best-seed selection, and graph
 * persistence with the embeddings sidecar (src/node.js). Mirrors the Python
 * port's behaviour.
 */
import { describe, it, expect } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SemanticGraph, MockEncoder } from './index.js';
import { saveGraph, loadGraph, embeddingsPathFor, EMBEDDING_SIDECAR_SUFFIX } from './node.js';

describe('updateNode keeps embeddings in step', () => {
  const enc = new MockEncoder(16);

  it('re-embeds when an embed field changes', () => {
    const g = new SemanticGraph({ encoder: enc });
    g.addNode('person', 1, { name: 'Alice', age: 30 });
    g.updateNode('person', 1, { name: 'Bob' });
    expect(g.getEmbedding(['person', 1]).text).toBe('Bob');
    expect(g.getEmbedding(['person', 1]).vector).toEqual(enc.encode('Bob'));
  });

  it('embeds a node that had no embeddable text before', () => {
    const g = new SemanticGraph({ encoder: enc });
    g.addNode('person', 1, { age: 30 });
    expect(g.getEmbedding(['person', 1])).toBeNull();
    g.updateNode('person', 1, { name: 'Carol' });
    expect(g.getEmbedding(['person', 1]).text).toBe('Carol');
  });

  it('drops the vector when nothing is left to embed', () => {
    const g = new SemanticGraph({ encoder: enc });
    g.addNode('person', 1, { name: 'Alice' });
    g.updateNode('person', 1, { name: null });
    expect(g.getEmbedding(['person', 1])).toBeNull();
  });

  it('leaves the embedding alone when no embed field changed', () => {
    const g = new SemanticGraph({ encoder: enc });
    g.addNode('person', 1, { name: 'Alice' });
    const before = g.getEmbedding(['person', 1]);
    g.updateNode('person', 1, { age: 31 });
    expect(g.getEmbedding(['person', 1])).toBe(before);
  });

  it('leaves vectors alone when autoEmbed is off', () => {
    const g = new SemanticGraph({ encoder: enc, autoEmbed: false });
    g.addNodeWithEmbed('person', 1, { name: 'Alice' }, 'custom text');
    g.updateNode('person', 1, { name: 'Bob' });
    expect(g.getEmbedding(['person', 1]).text).toBe('custom text');
  });

  it('returns the updated node and still throws for a missing one', () => {
    const g = new SemanticGraph({ encoder: enc });
    g.addNode('person', 1, { name: 'Alice' });
    expect(g.updateNode('person', 1, { name: 'Bob' }).properties.name).toBe('Bob');
    expect(() => g.updateNode('person', 99, { name: 'x' })).toThrow();
  });
});

describe('semanticPath picks the best seed', () => {
  // Seeds A (cosine 1.0, 3 hops from T) and B (cosine 0.9, 1 hop from T).
  function pathGraph(bVector) {
    const g = new SemanticGraph({ autoEmbed: false });
    for (const id of ['A', 'M1', 'M2', 'B', 'T']) g.addNode('n', id, {});
    g.addEdge('R', ['n', 'A'], ['n', 'M1']);
    g.addEdge('R', ['n', 'M1'], ['n', 'M2']);
    g.addEdge('R', ['n', 'M2'], ['n', 'T']);
    g.addEdge('R', ['n', 'B'], ['n', 'T']);
    g.embeddingStore.add(['n', 'A'], 'A', [1, 0]);
    g.embeddingStore.add(['n', 'B'], 'B', bVector);
    return g;
  }

  it('prefers a closer, slightly less similar seed over a better distant one', () => {
    const g = pathGraph([0.9, Math.sqrt(1 - 0.81)]);
    const result = g.semanticPath([1, 0], ['n', 'T'], 'R', 5, 3, 0.0, 0.9);
    expect(result.path[0]).toEqual(['n', 'B']);
    expect(result.hopCount).toBe(1);
    expect(result.score).toBeCloseTo(0.9 * 0.9, 6);
  });

  it('keeps the earlier seed when two seeds tie exactly', () => {
    const g = new SemanticGraph({ autoEmbed: false });
    for (const id of ['A', 'B', 'T']) g.addNode('n', id, {});
    g.addEdge('R', ['n', 'B'], ['n', 'T']);
    g.addEdge('R', ['n', 'A'], ['n', 'T']);
    g.embeddingStore.add(['n', 'A'], 'A', [1, 0]);
    g.embeddingStore.add(['n', 'B'], 'B', [1, 0]);
    const first = g.similaritySearch([1, 0], 3)[0].nodeRef;
    const result = g.semanticPath([1, 0], ['n', 'T'], 'R');
    expect(result.score).toBeCloseTo(0.9, 9);
    expect(result.path[0]).toEqual(first);
  });

  it('clamps a negative seed score to zero', () => {
    const g = new SemanticGraph({ autoEmbed: false });
    g.addNode('n', 'A', {});
    g.addNode('n', 'T', {});
    g.addEdge('R', ['n', 'A'], ['n', 'T']);
    g.embeddingStore.add(['n', 'A'], 'A', [-1, 0]);
    const result = g.semanticPath([1, 0], ['n', 'T'], 'R', 5, 3, -1.0, 0.9);
    expect(result.score).toBe(0);
  });

  it('returns null when no seed reaches the target', () => {
    const g = pathGraph([0, 1]);
    g.addNode('n', 'Z', {});
    expect(g.semanticPath([1, 0], ['n', 'Z'], 'R')).toBeNull();
  });
});

describe('graph persistence with embeddings sidecar', () => {
  const enc = new MockEncoder(16);
  const RESUME = 'résumé';
  function withTmp(fn) {
    const dir = mkdtempSync(join(tmpdir(), 'isongraph-vector-'));
    try { return fn(dir); } finally { rmSync(dir, { recursive: true, force: true }); }
  }
  function build() {
    const g = new SemanticGraph({ name: 'people', encoder: enc });
    g.addNode('person', '1', { name: 'Alice', role: 'engineer' });
    g.addNode('person', 2, { name: 'Bob' });
    g.addNode('doc', RESUME, { text: 'Carol plans roadmaps' });
    g.addEdge('KNOWS', ['person', '1'], ['person', 2], { since: 2020 });
    return g;
  }

  it('round-trips nodes, edges and embeddings', () => withTmp(dir => {
    const path = join(dir, 'people.ison');
    saveGraph(build(), path);
    expect(existsSync(path + EMBEDDING_SIDECAR_SUFFIX)).toBe(true);
    expect(embeddingsPathFor(path)).toBe(path + '.embeddings.json');

    const loaded = loadGraph(path, { encoder: enc });
    expect(loaded.name).toBe('people');
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.edgeCount()).toBe(1);
    expect(loaded.embeddingStore.count()).toBe(3);
    expect(loaded.getEmbedding(['person', 2]).vector).toEqual(enc.encode('Bob'));
    expect(loaded.getEmbedding(['person', '1']).text).toBe('Alice');
    expect(loaded.getEmbedding(['doc', RESUME]).text).toBe('Carol plans roadmaps');
    expect(loaded.similaritySearch('Bob', 1)[0].nodeRef).toEqual(['person', 2]);
  }));

  it('does not re-embed on load', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    const g = new SemanticGraph({ encoder: enc });
    g.addNodeWithEmbed('person', 1, { name: 'Alice' }, 'custom text');
    saveGraph(g, path);
    const loaded = loadGraph(path, { encoder: enc });
    expect(loaded.getEmbedding(['person', 1]).text).toBe('custom text');
  }));

  it('graph-only save writes no sidecar, and load without one is empty', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    saveGraph(build(), path, { embeddings: false });
    expect(existsSync(path + EMBEDDING_SIDECAR_SUFFIX)).toBe(false);
    expect(loadGraph(path).embeddingStore.count()).toBe(0);
  }));

  it('can skip the sidecar on load', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    saveGraph(build(), path);
    expect(loadGraph(path, { embeddings: false }).embeddingStore.count()).toBe(0);
  }));

  it('supports isonl by extension', () => withTmp(dir => {
    const path = join(dir, 'g.isonl');
    saveGraph(build(), path);
    const loaded = loadGraph(path, { encoder: enc });
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.embeddingStore.count()).toBe(3);
  }));

  it('writes the shared ison-embeddings payload', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    saveGraph(build(), path);
    const payload = JSON.parse(readFileSync(embeddingsPathFor(path), 'utf8'));
    expect(payload.format).toBe('ison-embeddings');
    expect(payload.version).toBe(1);
    expect(payload.count).toBe(3);
    expect(payload.embeddings.find(e => e.type === 'person' && e.id === 2).id_type).toBe('int');
  }));

  it('loads a graph file with CRLF line endings (Python on Windows)', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    saveGraph(build(), path);
    writeFileSync(path, readFileSync(path, 'utf8').replace(/\n/g, '\r\n'));
    const loaded = loadGraph(path, { encoder: enc });
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.edgeCount()).toBe(1);
    expect(loaded.embeddingStore.count()).toBe(3);
  }));

  it('rejects a sidecar that is not an ison-embeddings payload', () => withTmp(dir => {
    const path = join(dir, 'g.ison');
    saveGraph(build(), path);
    writeFileSync(embeddingsPathFor(path), '{"format":"other"}');
    expect(() => loadGraph(path)).toThrow();
  }));
});
