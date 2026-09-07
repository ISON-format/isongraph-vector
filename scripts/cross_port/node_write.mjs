/**
 * Write a SQLite store and a portable JSON payload from the JavaScript port,
 * for another port to read back.
 *
 * The repository's central claim is that all six ports share one SQLite schema
 * and one payload format, so a store built by any of them opens in any other.
 * That was verified by hand; these scripts make CI verify it on every push.
 *
 * Usage: node node_write.mjs <db-path> <json-path>
 */
import { writeFileSync } from 'node:fs';
import { MockEncoder } from '../../isongraph-vector-js/src/index.js';
import { SqliteEmbeddingStore } from '../../isongraph-vector-js/src/sqlite-store.js';

const [dbPath, jsonPath] = process.argv.slice(2);
if (!dbPath || !jsonPath) {
  console.error('usage: node node_write.mjs <db-path> <json-path>');
  process.exit(2);
}

// Deliberately mixed: a string id, an integer id, and text with non-ASCII
// bytes, because those are the three things a shared format gets wrong.
export const FIXTURES = [
  [['person', '1'], 'Alice is a software engineer'],
  [['person', 2], 'Bob analyses ML models'],
  [['doc', 'r\u00e9sum\u00e9'], 'Carol plans roadmaps for \u00fcber-teams'],
];

const store = new SqliteEmbeddingStore(dbPath, new MockEncoder(384));
store.addBatch(FIXTURES);
writeFileSync(jsonPath, JSON.stringify(store.exportEmbeddings(), null, 2));
store.close();

console.log(`node wrote ${FIXTURES.length} embeddings to ${dbPath} and ${jsonPath}`);
