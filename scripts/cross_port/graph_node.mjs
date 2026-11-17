/**
 * JavaScript and TypeScript halves of the graph-file cross-port check (see
 * graph_files.py).
 *
 *   node graph_node.mjs <js|ts=DIR> write    <dir>
 *   node graph_node.mjs <js|ts=DIR> read     <dir> <observation.json>
 *   node graph_node.mjs <js|ts=DIR> mismatch <dir>
 *
 * `js` imports the JavaScript port's sources. `ts=<dir>` imports the
 * TypeScript port's compiled output from <dir> (the orchestrator builds it
 * outside the repository).
 *
 * `write` saves the shared fixture graph as <dir>/graph.ison plus its
 * embeddings sidecar and binary snapshot. `read` loads it with onMismatch
 * 'error' and writes what it found (and the load source, snapshot or json, with
 * the reason a snapshot was skipped) to <observation.json>. `mismatch` loads a graph whose file was
 * edited after the save and exits 0 only if the load throws
 * GraphEmbeddingMismatchError.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const [implArg, command, dir, out] = process.argv.slice(2);
// `js`, or `ts=<directory holding the TypeScript port's compiled output>`.
const [impl, tsDist] = String(implArg).split('=');

async function load(impl) {
  if (impl === 'js') {
    const here = fileURLToPath(new URL('.', import.meta.url));
    const src = join(here, '..', '..', 'isongraph-vector-js', 'src');
    return {
      core: await import(pathToFileURL(join(src, 'index.js')).href),
      node: await import(pathToFileURL(join(src, 'node.js')).href),
    };
  }
  if (impl === 'ts') {
    const dist = tsDist;
    if (!dist) throw new Error('usage: ts=<compiled output directory>');
    return {
      core: await import(pathToFileURL(join(dist, 'index.js')).href),
      node: await import(pathToFileURL(join(dist, 'node.js')).href),
    };
  }
  throw new Error(`unknown implementation '${impl}' (expected js or ts)`);
}

const DIM = 384;
const BS = String.fromCharCode(92); // a backslash, spelled so no source escaping can mangle it
const DOC_TEXT = 'Résumé of 日本語 engineers, "quoted" and back' + BS + 'slash';
const QUERIES = ['Bob analyses ML models', 'Carol plans roadmaps for über-teams', DOC_TEXT];

// Keep in step with the other graph_* helpers: every port writes (and expects)
// the same graph. JavaScript ids keep their type, so 1 and 2 are numbers here.
const NODES = [
  ['person', 1, { text: 'Alice is a software engineer', age: 30, score: 0.5, active: true, nickname: null }],
  ['person', 2, { text: 'Bob analyses ML models', age: 41, score: 2.75, active: false, nickname: null }],
  ['person', 'carol', { text: 'Carol plans roadmaps for über-teams', age: 35, score: -1.25, active: true, nickname: 'Caz' }],
  ['doc', 'résumé', { text: DOC_TEXT }],
  ['topic', 'ml', { text: 'machine learning models', weight: 0.125 }],
];
const EDGES = [
  ['KNOWS', ['person', 1], ['person', 2], { since: 2020, weight: 0.75 }],
  ['WROTE', ['person', 'carol'], ['doc', 'résumé'], { note: 'first draft' }],
  ['ABOUT', ['doc', 'résumé'], ['topic', 'ml'], {}],
];

const graphPath = (d) => join(d, 'graph.ison');

function write(lib, d) {
  const graph = new lib.core.SemanticGraph({ name: 'fixture', encoder: new lib.core.MockEncoder(DIM) });
  for (const [type, id, props] of NODES) graph.addNodeWithEmbed(type, id, props, props.text);
  for (const [rel, source, target, props] of EDGES) graph.addEdge(rel, source, target, props);
  mkdirSync(d, { recursive: true });
  lib.node.saveGraph(graph, graphPath(d), { snapshot: true });
}

function observe(graph, encoder, report) {
  const nodes = [];
  const missing = [];
  let worst = 0;
  for (const node of graph.nodes()) {
    nodes.push({ type: node.type, id: node.id, props: { ...node.properties } });
    const record = graph.getEmbedding([node.type, node.id]);
    if (!record) {
      missing.push([node.type, node.id]);
      continue;
    }
    const expected = encoder.encode(node.properties.text);
    if (expected.length !== record.vector.length) worst = 1e9;
    else for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(record.vector[i] - expected[i]));
  }
  const edges = [];
  for (const rel of graph.edgeTypes()) {
    for (const edge of graph.edges(rel)) {
      edges.push({
        rel: edge.relType,
        source: [...edge.source],
        target: [...edge.target],
        props: { ...edge.properties },
      });
    }
  }
  const searches = QUERIES.map((query) => {
    const hits = graph.similaritySearch(query, 1);
    const top = hits[0];
    return { query, top: top ? [...top.nodeRef] : null, score: top ? top.score : null };
  });
  return {
    port: impl,
    nodes,
    edges,
    embedding_count: graph.embeddingStore.count(),
    missing_embeddings: missing,
    max_vec_diff: worst,
    searches,
    source: report.source,
    snapshot_skipped: report.snapshotSkipped,
  };
}

async function main() {
  const lib = await load(impl);
  if (command === 'write' && dir) {
    write(lib, dir);
  } else if (command === 'read' && dir && out) {
    const encoder = new lib.core.MockEncoder(DIM);
    const { graph, report } = lib.node.loadGraphWithReport(graphPath(dir), { encoder, onMismatch: 'error' });
    writeFileSync(out, JSON.stringify(observe(graph, encoder, report)), 'utf8');
  } else if (command === 'mismatch' && dir) {
    try {
      lib.node.loadGraph(graphPath(dir), { encoder: new lib.core.MockEncoder(DIM), onMismatch: 'error' });
    } catch (error) {
      if (error instanceof lib.core.GraphEmbeddingMismatchError) return 0;
      console.error(`expected GraphEmbeddingMismatchError, got ${error?.name}: ${error?.message}`);
      return 1;
    }
    console.error('expected GraphEmbeddingMismatchError, but the load succeeded');
    return 1;
  } else {
    console.error('usage: node graph_node.mjs <js|ts=DIR> write <dir> | read <dir> <out.json> | mismatch <dir>');
    return 2;
  }
  return 0;
}

main().then(
  (code) => process.exit(code),
  (error) => {
    console.error(error?.stack ?? String(error));
    process.exit(1);
  },
);
