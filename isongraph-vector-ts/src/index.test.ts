/**
 * ISONGraph Vector TypeScript Tests
 */

import { describe, it, expect } from 'vitest';
import {
  SemanticGraph,
  MockEncoder,
  EmbeddingStore,
  Direction,
  DimensionMismatchError,
  NoEncoderError,
  EMBEDDING_FORMAT
} from './index';

describe('MockEncoder', () => {
  it('should have correct dimension', () => {
    const encoder = new MockEncoder(384);
    expect(encoder.dimension).toBe(384);
  });

  it('should generate deterministic embeddings', () => {
    const encoder = new MockEncoder();
    const v1 = encoder.encode("hello");
    const v2 = encoder.encode("hello");
    expect(v1).toEqual(v2);
  });

  it('should generate different embeddings for different texts', () => {
    const encoder = new MockEncoder();
    const v1 = encoder.encode("hello");
    const v2 = encoder.encode("world");
    expect(v1).not.toEqual(v2);
  });

  it('should encode batch', () => {
    const encoder = new MockEncoder();
    const vectors = encoder.encodeBatch(["hello", "world"]);
    expect(vectors.length).toBe(2);
    expect(vectors[0].length).toBe(384);
  });
});

describe('EmbeddingStore', () => {
  it('should add and get embedding', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'Alice is an engineer');
    const record = store.get(['person', 1]);

    expect(record).not.toBeNull();
    expect(record!.text).toBe('Alice is an engineer');
    expect(record!.vector.length).toBe(384);
  });

  it('should add with pre-computed vector', () => {
    const store = new EmbeddingStore();
    const vector = new Array(384).fill(0.5);

    store.add(['person', 1], 'test', vector);
    const record = store.get(['person', 1]);

    expect(record!.vector).toEqual(vector);
  });

  it('should remove embedding', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'test');
    expect(store.remove(['person', 1])).toBe(true);
    expect(store.get(['person', 1])).toBeNull();
  });

  it('should perform similarity search', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'software engineer');
    store.add(['person', 2], 'data scientist');
    store.add(['person', 3], 'software developer');

    // Cosine similarity is signed and MockEncoder vectors are near-orthogonal,
    // so mock scores straddle zero - the default threshold of 0.0 can reject
    // every match. Accept anything.
    const results = store.similaritySearch('software', 2, undefined, -1.0);
    expect(results.length).toBe(2);
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
  });

  it('should filter by node type', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'engineer');
    store.add(['company', 1], 'tech company');

    const results = store.similaritySearch('tech', 10, 'company', -1.0);
    expect(results.length).toBe(1);
    expect(results[0].nodeRef[0]).toBe('company');
  });

  it('should count embeddings', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'test1');
    store.add(['person', 2], 'test2');

    expect(store.count()).toBe(2);
  });

  it('should clear embeddings', () => {
    const encoder = new MockEncoder();
    const store = new EmbeddingStore(encoder);

    store.add(['person', 1], 'test');
    store.clear();

    expect(store.count()).toBe(0);
  });
});

describe('SemanticGraph', () => {
  it('should create empty graph', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });
    expect(graph.nodeCount()).toBe(0);
  });

  it('should add node with embed', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });
    graph.addNodeWithEmbed('person', 1, { name: 'Alice' }, 'Alice is an engineer');

    expect(graph.nodeCount()).toBe(1);
    expect(graph.embeddingStore.count()).toBe(1);
  });

  it('should auto-embed from properties', () => {
    const graph = new SemanticGraph({
      encoder: new MockEncoder(),
      autoEmbed: true
    });

    graph.addNodeWithEmbed('person', 1, { name: 'Alice', description: 'Engineer' });
    expect(graph.embeddingStore.count()).toBe(1);
  });

  it('should remove node and embedding', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });
    graph.addNodeWithEmbed('person', 1, {}, 'test');
    graph.removeNode('person', 1);

    expect(graph.nodeCount()).toBe(0);
    expect(graph.embeddingStore.count()).toBe(0);
  });

  it('should perform similarity search', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });
    graph.addNodeWithEmbed('person', 1, { name: 'Alice' }, 'software engineer');
    graph.addNodeWithEmbed('person', 2, { name: 'Bob' }, 'data scientist');

    // Use exact text for reliable match with mock encoder
    const results = graph.similaritySearch('software engineer', 5, undefined, -1.0);
    expect(results.length).toBeGreaterThan(0);
  });

  it('should perform semantic multi-hop', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });

    graph.addNodeWithEmbed('person', 1, { name: 'Alice' }, 'software engineer Alice');
    graph.addNodeWithEmbed('person', 2, { name: 'Bob' }, 'manager Bob');
    graph.addNodeWithEmbed('person', 3, { name: 'Carol' }, 'designer Carol');

    graph.addEdge('KNOWS', ['person', 1], ['person', 2], {});
    graph.addEdge('KNOWS', ['person', 2], ['person', 3], {});

    // Use exact text and low threshold for mock encoder
    const results = graph.semanticMultiHop(
      'software engineer Alice',
      'KNOWS',
      2,  // maxHops
      5,  // topKSeeds
      10, // topKResults
      Direction.OUT,
      0.8, // decay
      0.0  // threshold (low for mock encoder)
    );

    expect(results.length).toBeGreaterThan(0);
  });

  it('should perform semantic multi-hop with decay', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });

    graph.addNodeWithEmbed('person', 1, {}, 'engineer');
    graph.addNodeWithEmbed('person', 2, {}, 'manager');

    graph.addEdge('KNOWS', ['person', 1], ['person', 2], {});

    const results = graph.semanticMultiHop('engineer', 'KNOWS', 1);

    // Seed should have higher score than traversed nodes
    const seed = results.find(r => r.hopCount === 0);
    const traversed = results.find(r => r.hopCount === 1);

    if (seed && traversed) {
      expect(seed.score).toBeGreaterThan(traversed.score);
    }
  });

  it('should find semantic path', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });

    graph.addNodeWithEmbed('person', 1, {}, 'engineer');
    graph.addNodeWithEmbed('person', 2, {}, 'manager');
    graph.addNodeWithEmbed('person', 3, {}, 'director');

    graph.addEdge('REPORTS_TO', ['person', 1], ['person', 2], {});
    graph.addEdge('REPORTS_TO', ['person', 2], ['person', 3], {});

    const result = graph.semanticPath('engineer', ['person', 3], 'REPORTS_TO');

    expect(result).not.toBeNull();
    expect(result!.hopCount).toBe(2);
  });

  it('should embed all nodes', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });

    graph.addNode('person', 1, { name: 'Alice' });
    graph.addNode('person', 2, { name: 'Bob' });

    const count = graph.embedAllNodes();
    expect(count).toBe(2);
    expect(graph.embeddingStore.count()).toBe(2);
  });

  it('should preserve graph traversal', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder() });

    graph.addNodeWithEmbed('person', 1, {}, 'Alice');
    graph.addNodeWithEmbed('person', 2, {}, 'Bob');
    graph.addEdge('KNOWS', ['person', 1], ['person', 2], {});

    const neighbors = graph.neighbors(['person', 1], 'KNOWS');
    expect(neighbors.length).toBe(1);
  });
});

describe('Integration', () => {
  it('should handle full workflow', () => {
    const encoder = new MockEncoder();
    const graph = new SemanticGraph({ encoder });

    // Add nodes
    graph.addNodeWithEmbed('user', 1, { name: 'Alice' }, 'Alice is a software engineer');
    graph.addNodeWithEmbed('user', 2, { name: 'Bob' }, 'Bob is a data scientist');
    graph.addNodeWithEmbed('user', 3, { name: 'Carol' }, 'Carol is a product manager');
    graph.addNodeWithEmbed('project', 1, { name: 'ML Platform' }, 'Machine learning infrastructure');
    graph.addNodeWithEmbed('project', 2, { name: 'Web App' }, 'Frontend web application');

    // Add edges
    graph.addEdge('WORKS_ON', ['user', 1], ['project', 1], {});
    graph.addEdge('WORKS_ON', ['user', 2], ['project', 1], {});
    graph.addEdge('WORKS_ON', ['user', 3], ['project', 2], {});
    graph.addEdge('KNOWS', ['user', 1], ['user', 2], {});
    graph.addEdge('KNOWS', ['user', 2], ['user', 3], {});

    // Similarity search - use exact text for mock encoder
    const similar = graph.similaritySearch('Machine learning infrastructure', 5, undefined, -1.0);
    expect(similar.length).toBeGreaterThan(0);

    // Semantic multi-hop with low threshold for mock encoder
    const relatedPeople = graph.semanticMultiHop(
      'Alice is a software engineer',
      undefined,
      2,
      5,
      10,
      Direction.OUT,
      0.8,
      0.0  // Low threshold for mock encoder
    );
    expect(relatedPeople.length).toBeGreaterThan(0);

    // Verify graph structure
    expect(graph.nodeCount()).toBe(5);
    expect(graph.edgeCount()).toBe(5);
    expect(graph.embeddingStore.count()).toBe(5);
  });
});


// =============================================================================
// Regression tests for fixed defects
// =============================================================================

const GOLDEN = {
  hello: [0.8381705284, -0.8255031109, 0.5899617672, 0.4615051746],
  ison: [-0.3658730984, -0.2765417099, 0.2944871187, -0.6502785683],
  empty: [-0.4520676136, -0.3797093630, 0.3339765072, 0.9023951292],
};

function seededGraph() {
  const graph = new SemanticGraph({ encoder: new MockEncoder(4), autoEmbed: false });
  graph.addNode('person', 1, { name: 'A' });
  graph.addNode('person', 2, { name: 'B' });
  graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
  // person:1 matches the query perfectly, person:2 slightly less well.
  graph.embeddingStore.add(['person', 1], 'a', [1, 0, 0, 0]);
  graph.embeddingStore.add(['person', 2], 'b', [0.9, Math.sqrt(0.19), 0, 0]);
  return graph;
}

describe('semanticMultiHop seed scoring', () => {
  it('keeps the direct score of a node that is itself a seed', () => {
    const graph = seededGraph();
    const query = [1, 0, 0, 0];

    const direct = graph.similaritySearch(query, 5, undefined, -1.0);
    expect(direct.find(r => r.nodeRef[1] === 2)!.score).toBeCloseTo(0.9, 6);

    // Seeds used to be inserted only when absent from the result map, so an
    // entry written by an earlier traversal won even when it scored lower:
    // person:2 came back at seed(1.0) * decay(0.8) = 0.8, at hop 1, down a
    // path through person:1 that it never needed.
    const results = graph.semanticMultiHop(query, {
      relType: 'KNOWS', maxHops: 2, decay: 0.8, threshold: 0.3,
    });
    const second = results.find(r => r.nodeRef[1] === 2)!;
    expect(second.score).toBeCloseTo(0.9, 6);
    expect(second.hopCount).toBe(0);
    expect(second.path).toEqual([['person', 2]]);
  });

  it('still scores nodes only reachable by traversal', () => {
    const graph = seededGraph();
    graph.addNode('person', 3, { name: 'C' });
    graph.addEdge('KNOWS', ['person', 2], ['person', 3]);
    graph.embeddingStore.add(['person', 3], 'c', [0, 0, 1, 0]);

    const results = graph.semanticMultiHop([1, 0, 0, 0], {
      relType: 'KNOWS', maxHops: 2, decay: 0.8, threshold: 0.3,
    });
    const third = results.find(r => r.nodeRef[1] === 3)!;
    expect(third.hopCount).toBeGreaterThan(0);
    expect(third.score).toBeGreaterThan(0);
  });

  it('does not let a negative seed score grow with distance', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(4), autoEmbed: false });
    graph.addNode('person', 1, {});
    graph.addNode('person', 2, {});
    graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
    graph.embeddingStore.add(['person', 1], 'a', [-1, 0, 0, 0]);
    graph.embeddingStore.add(['person', 2], 'b', [0, 0, 1, 0]);

    const results = graph.semanticMultiHop([1, 0, 0, 0], {
      relType: 'KNOWS', maxHops: 1, threshold: -1.0,
    });
    const byId = Object.fromEntries(results.map(r => [r.nodeRef[1], r.score]));
    expect(byId[1]).toBeCloseTo(-1.0, 6);
    // 0, not -1 * 0.8 = -0.8, which would have outranked the seed.
    expect(byId[2]).toBeCloseTo(0.0, 6);
  });

  it('blends the relevance of a node itself into its hop score', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(4), autoEmbed: false });
    for (const i of [1, 2, 3]) graph.addNode('person', i, {});
    graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
    graph.addEdge('KNOWS', ['person', 1], ['person', 3]);
    graph.embeddingStore.add(['person', 1], 'seed', [1, 0, 0, 0]);
    graph.embeddingStore.add(['person', 2], 'off-topic', [0, 1, 0, 0]);
    graph.embeddingStore.add(['person', 3], 'on-topic', [0.8, 0.6, 0, 0]);

    const query = [1, 0, 0, 0];
    const opts = { relType: 'KNOWS', maxHops: 1, threshold: 0.5, topKSeeds: 1 };

    const plain = Object.fromEntries(
      graph.semanticMultiHop(query, opts).map(r => [r.nodeRef[1], r.score]));
    expect(plain[2]).toBeCloseTo(plain[3], 6);

    const blended = Object.fromEntries(
      graph.semanticMultiHop(query, { ...opts, blend: 0.5 }).map(r => [r.nodeRef[1], r.score]));
    expect(blended[3]).toBeGreaterThan(blended[2]);
    expect(blended[1]).toBeCloseTo(plain[1], 6);
  });

  it('rejects a blend outside [0, 1]', () => {
    const graph = seededGraph();
    expect(() => graph.semanticMultiHop([1, 0, 0, 0], { blend: 1.5 })).toThrow();
  });

  it('accepts the original positional argument form', () => {
    const graph = seededGraph();
    const results = graph.semanticMultiHop([1, 0, 0, 0], 'KNOWS', 2, 5, 10, Direction.OUT, 0.8, 0.3);
    expect(results.length).toBe(2);
  });
});

describe('dimension validation', () => {
  it('rejects a vector of the wrong width', () => {
    const store = new EmbeddingStore(new MockEncoder(8));
    store.add(['a', 1], 'full width');
    expect(() => store.add(['a', 2], 'too short', [1, 0])).toThrow(DimensionMismatchError);
  });

  it('rejects an empty vector', () => {
    const store = new EmbeddingStore(new MockEncoder(8));
    expect(() => store.add(['a', 1], 'empty', [])).toThrow(DimensionMismatchError);
  });

  it('rejects a query of the wrong width', () => {
    const store = new EmbeddingStore(new MockEncoder(8));
    store.add(['a', 1], 'full width');
    expect(() => store.similaritySearch([1, 0], 1)).toThrow(DimensionMismatchError);
  });

  it('pins its dimension to the first vector stored', () => {
    const store = new EmbeddingStore();
    expect(store.dimension).toBeNull();
    store.add(['a', 1], 'x', [1, 0, 0]);
    expect(store.dimension).toBe(3);
  });

  it('rejects text with no encoder', () => {
    const store = new EmbeddingStore();
    expect(() => store.add(['a', 1], 'no encoder here')).toThrow(NoEncoderError);
  });
});

describe('MockEncoder quality and cross-language parity', () => {
  it('produces the same vectors as every other language port', () => {
    // The same values are asserted in the Python, Rust and C++ suites.
    const enc = new MockEncoder(4);
    const cases: Array<[string, number[]]> = [
      ['hello', GOLDEN.hello], ['ison', GOLDEN.ison], ['', GOLDEN.empty],
    ];
    for (const [text, want] of cases) {
      const got = enc.encode(text);
      got.forEach((v, i) => expect(v).toBeCloseTo(want[i], 9));
    }
  });

  it('does not collide across many distinct texts', () => {
    const enc = new MockEncoder(32);
    const seen = new Set();
    for (let i = 0; i < 5000; i++) seen.add(enc.encode(`text${i}`).join(','));
    expect(seen.size).toBe(5000);
  });

  it('stays within [-1, 1)', () => {
    for (const v of new MockEncoder(128).encode('range check')) {
      expect(v).toBeGreaterThanOrEqual(-1.0);
      expect(v).toBeLessThan(1.0);
    }
  });

  it('rejects a bad dimension', () => {
    expect(() => new MockEncoder(0)).toThrow();
  });
});

describe('embedding portability', () => {
  it('round-trips through the portable payload', () => {
    const store = new EmbeddingStore(new MockEncoder(8));
    store.add(['person', 1], 'Alice');
    store.add(['account', '007'], 'Bond');

    const payload = store.exportEmbeddings();
    expect(payload.format).toBe(EMBEDDING_FORMAT);
    expect(payload.count).toBe(2);

    const other = new EmbeddingStore(new MockEncoder(8));
    expect(other.importEmbeddings(payload)).toBe(2);
    expect(other.get(['person', 1])!.text).toBe('Alice');
    // a numeric-looking string id keeps its type across the round trip
    expect(other.get(['account', '007'])!.text).toBe('Bond');
    expect(other.get(['account', '007'])!.nodeRef[1]).toBe('007');
  });

  it('rejects a foreign payload', () => {
    const store = new EmbeddingStore(new MockEncoder(8));
    expect(() => store.importEmbeddings({ format: 'something-else' } as never)).toThrow();
  });
});

describe('graph-level conveniences', () => {
  it('auto-embeds through the plain addNode', () => {
    // autoEmbed used to apply only inside addNodeWithEmbed, so a plain
    // addNode silently produced an unembedded, unsearchable node.
    const graph = new SemanticGraph({ encoder: new MockEncoder(16) });
    graph.addNode('person', 1, { name: 'Alice', description: 'engineer' });
    expect(graph.embeddingStore.count()).toBe(1);
    expect(graph.getEmbedding(['person', 1])!.text).toBe('Alice engineer');
  });

  it('leaves nodes unembedded when autoEmbed is off', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16), autoEmbed: false });
    graph.addNode('person', 1, { name: 'Alice' });
    expect(graph.embeddingStore.count()).toBe(0);
  });

  it('finds nodes similar to an existing one, excluding itself', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16) });
    graph.addNode('person', 1, { name: 'Alice' });
    graph.addNode('person', 2, { name: 'Bob' });
    graph.addNode('person', 3, { name: 'Carol' });

    const similar = graph.similarToNode(['person', 1], 2, undefined, -1.0);
    expect(similar.length).toBe(2);
    expect(similar.every(r => r.nodeRef[1] !== 1)).toBe(true);
  });

  it('returns nothing for a node with no embedding', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16), autoEmbed: false });
    graph.addNode('person', 99, { name: 'unembedded' });
    expect(graph.similarToNode(['person', 99])).toEqual([]);
  });

  it('extracts a working semantic subgraph', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16) });
    graph.addNode('person', 1, { name: 'Alice' });
    graph.addNode('person', 2, { name: 'Bob' });
    graph.addNode('person', 3, { name: 'Carol' });
    graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
    graph.addEdge('KNOWS', ['person', 2], ['person', 3]);

    const sub = graph.semanticSubgraph('Alice', {
      relType: 'KNOWS', maxHops: 1, topKSeeds: 1, topKResults: 2,
      threshold: -1.0, name: 'slice',
    });

    expect(sub.name).toBe('slice');
    expect(sub.nodeCount()).toBeGreaterThan(0);
    expect(sub.nodeCount()).toBeLessThanOrEqual(2);
    // every kept node brought its embedding along
    expect(sub.embeddingStore.count()).toBe(sub.nodeCount());
  });

  it('embeds every node in one batched call', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16), autoEmbed: false });
    for (let i = 0; i < 10; i++) graph.addNode('doc', i, { name: `Document ${i}` });
    expect(graph.embedAllNodes()).toBe(10);
    expect(graph.embeddingStore.count()).toBe(10);
  });

  it('can skip nodes that already have an embedding', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(16), autoEmbed: false });
    graph.addNode('doc', 1, { name: 'One' });
    graph.addNode('doc', 2, { name: 'Two' });
    graph.embedNode(['doc', 1], 'already embedded');
    expect(graph.embedAllNodes(undefined, true)).toBe(1);
    expect(graph.getEmbedding(['doc', 1])!.text).toBe('already embedded');
  });
});

describe('cross-language payload', () => {
  // Captured verbatim from the Python port. Its node ids are typed, so an int
  // id arrives unquoted ("id":1) and has to come back as a number here.
  const PYTHON_PAYLOAD = '{"format":"ison-embeddings","version":1,"dimension":4,"count":2,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","model":"mock-4","vector":[-0.8610728979110718,-0.33433854579925537,0.8408418893814087,-0.2771846055984497]},{"type":"account","id":"007","id_type":"str","text":"Bond","model":"mock-4","vector":[-0.9531009197235107,0.9167747497558594,0.5073114633560181,-0.0420149564743042]}]}';

  it('imports a payload written by the Python port', () => {
    const store = new EmbeddingStore(new MockEncoder(4));
    const payload = JSON.parse(PYTHON_PAYLOAD);
    expect(store.importEmbeddings(payload)).toBe(2);

    expect(store.get(['person', 1])!.text).toBe('Alice');
    expect(store.get(['account', '007'])!.text).toBe('Bond');

    // and the vectors match what this port encodes for the same text
    const expected = new MockEncoder(4).encode('Alice');
    store.get(['person', 1])!.vector.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 9));
  });
});

describe('typed property rendering', () => {
  // The same golden string is asserted in all six suites. ISONGraph 1.4.0
  // made property values typed, which is where the ports could drift:
  // Python would otherwise render None as "None" and True as "True", while
  // JavaScript renders "null" and "true". Different text means a different
  // vector, which would break the cross-port guarantee.
  it('produces embed text identical to every other port', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(8) });
    graph.addNode('doc', '1', { name: 'Report', description: null, content: 1.5, text: true });
    expect(graph.getEmbedding(['doc', '1'])!.text).toBe('Report 1.5 true');
  });

  it('lets null contribute nothing', () => {
    const graph = new SemanticGraph({ encoder: new MockEncoder(8) });
    expect(graph.embedTextFor({ name: null, description: 'kept' })).toBe('kept');
    expect(graph.embedTextFor({ name: undefined, description: 'kept' })).toBe('kept');
  });

  it('uses the ISON spelling for booleans', () => {
    expect(SemanticGraph.propertyText(true)).toBe('true');
    expect(SemanticGraph.propertyText(false)).toBe('false');
  });
});
