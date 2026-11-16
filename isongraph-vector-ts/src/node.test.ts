/**
 * Graph persistence (Node entry point): save/load with the embeddings sidecar.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SemanticGraph, MockEncoder, EMBEDDING_FORMAT } from './index.js';
import { saveGraph, loadGraph, embeddingsPathFor } from './node.js';

const dirs: string[] = [];
function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), 'isongraph-vector-'));
  dirs.push(d);
  return d;
}
afterEach(() => {
  while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true });
});

function buildGraph(): SemanticGraph {
  const graph = new SemanticGraph({ name: 'social', encoder: new MockEncoder(16) });
  graph.addNode('person', 1, { name: 'Alice', description: 'software engineer' });
  graph.addNode('person', '007', { name: 'Bond' });
  graph.addNode('doc', 'résumé', { text: 'Carol plans roadmaps' });
  graph.addEdge('KNOWS', ['person', 1], ['person', '007'], { since: 2020 });
  return graph;
}

describe('saveGraph / loadGraph', () => {
  it('writes the graph file and an embeddings sidecar next to it', () => {
    const path = join(tmp(), 'g.ison');
    saveGraph(buildGraph(), path);
    expect(existsSync(path)).toBe(true);
    const side = embeddingsPathFor(path);
    expect(side).toBe(path + '.embeddings.json');
    const payload = JSON.parse(readFileSync(side, 'utf8'));
    expect(payload.format).toBe(EMBEDDING_FORMAT);
    expect(payload.count).toBe(3);
  });

  it('round-trips nodes, edges and embeddings so search works after load', () => {
    const path = join(tmp(), 'g.ison');
    const original = buildGraph();
    saveGraph(original, path);

    const loaded = loadGraph(path, { encoder: new MockEncoder(16) });
    expect(loaded.name).toBe('g');
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.edgeCount()).toBe(1);
    expect(loaded.getNode('person', '007').properties.name).toBe('Bond');
    expect(loaded.getEdge('KNOWS', ['person', 1], ['person', '007']).properties.since).toBe(2020);
    expect(loaded.embeddingStore.count()).toBe(3);
    expect(loaded.getEmbedding(['person', 1])!.vector)
      .toEqual(original.getEmbedding(['person', 1])!.vector);
    expect(loaded.getEmbedding(['doc', 'résumé'])).not.toBeNull();

    const hit = loaded.similaritySearch('Bond', 1)[0];
    expect(hit.nodeRef).toEqual(['person', '007']);
  });

  it('does not re-embed on load: saved vectors are restored as they were', () => {
    const path = join(tmp(), 'g.ison');
    const g = new SemanticGraph({ encoder: new MockEncoder(4), autoEmbed: false });
    g.addNode('n', 1, { name: 'Alice' });
    g.embeddingStore.add(['n', 1], 'custom', [1, 0, 0, 0]);
    saveGraph(g, path);

    // autoEmbed defaults to true on the loaded graph
    const loaded = loadGraph(path, { encoder: new MockEncoder(4) });
    expect(loaded.getEmbedding(['n', 1])!.text).toBe('custom');
    expect(loaded.getEmbedding(['n', 1])!.vector).toEqual([1, 0, 0, 0]);
  });

  it('skips the sidecar on request, and loads without one', () => {
    const path = join(tmp(), 'g.ison');
    saveGraph(buildGraph(), path, { embeddings: false });
    expect(existsSync(embeddingsPathFor(path))).toBe(false);
    const loaded = loadGraph(path, { encoder: new MockEncoder(16), autoEmbed: false });
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.embeddingStore.count()).toBe(0);
  });

  it('can ignore an existing sidecar', () => {
    const path = join(tmp(), 'g.ison');
    saveGraph(buildGraph(), path);
    const loaded = loadGraph(path, {
      encoder: new MockEncoder(16), autoEmbed: false, embeddings: false,
    });
    expect(loaded.embeddingStore.count()).toBe(0);
  });

  it('round-trips ISONL by extension', () => {
    const path = join(tmp(), 'g.isonl');
    saveGraph(buildGraph(), path);
    expect(readFileSync(path, 'utf8').split('\n')[0]).toContain('|');
    const loaded = loadGraph(path, { encoder: new MockEncoder(16) });
    expect(loaded.nodeCount()).toBe(3);
    expect(loaded.edgeCount()).toBe(1);
    expect(loaded.embeddingStore.count()).toBe(3);
  });

  it('loads a sidecar written by the Python port', () => {
    // Shape and typing as written by the Python port's SemanticGraph.save.
    const path = join(tmp(), 'py.ison');
    const g = new SemanticGraph({ encoder: new MockEncoder(4), autoEmbed: false });
    g.addNode('person', 1, { name: 'Alice' });
    g.addNode('account', '007', { name: 'Bond' });
    saveGraph(g, path, { embeddings: false });
    const payload = {
      format: 'ison-embeddings', version: 1, dimension: 4, count: 2,
      embeddings: [
        { type: 'person', id: 1, id_type: 'int', text: 'Alice', model: 'mock-4',
          vector: [-0.8610728979110718, -0.33433854579925537, 0.8408418893814087, -0.2771846055984497] },
        { type: 'account', id: '007', id_type: 'str', text: 'Bond', model: 'mock-4',
          vector: [-0.9531009197235107, 0.9167747497558594, 0.5073114633560181, -0.0420149564743042] },
      ],
    };
    writeFileSync(embeddingsPathFor(path), JSON.stringify(payload));

    const loaded = loadGraph(path, { encoder: new MockEncoder(4) });
    expect(loaded.embeddingStore.count()).toBe(2);
    expect(loaded.getEmbedding(['person', 1])!.text).toBe('Alice');
    expect(loaded.getEmbedding(['account', '007'])!.text).toBe('Bond');
    expect(loaded.similaritySearch('Bond', 1)[0].nodeRef).toEqual(['account', '007']);
  });
});
