/**
 * Read a SQLite store and a payload written by another port, and check the
 * vectors match what this port encodes for the same text.
 *
 * MockEncoder is byte-identical across all six ports, so this compares floats
 * exactly. A tolerance would hide the drift this exists to catch.
 *
 * The payload is checked by importing it into a store rather than by comparing
 * the JSON text, because the text legitimately differs between ports: every
 * vector is float32, but Rust serialises the shortest form that round-trips as
 * f32 (0.33590567) while Python widens to float64 first (0.33590567111968994).
 * Those are the same number. Importing narrows back to float32, which is the
 * operation a consumer actually performs, and after that the comparison is
 * exact.
 *
 * Usage: node node_read.mjs <db-path> <json-path>
 */
import { readFileSync } from 'node:fs';
import { MockEncoder } from '../../isongraph-vector-js/src/index.js';
import { SqliteEmbeddingStore } from '../../isongraph-vector-js/src/sqlite-store.js';

const [dbPath, jsonPath] = process.argv.slice(2);
if (!dbPath || !jsonPath) {
  console.error('usage: node node_read.mjs <db-path> <json-path>');
  process.exit(2);
}

const encoder = new MockEncoder(384);
let failures = 0;

function check(label, actual, text) {
  const expected = encoder.encode(text);
  const same = actual.length === expected.length &&
               actual.every((v, i) => v === expected[i]);
  console.log(`  ${same ? 'ok  ' : 'FAIL'}  ${label}`);
  if (!same) {
    failures++;
    console.log(`        got      ${actual.slice(0, 3)}`);
    console.log(`        expected ${expected.slice(0, 3)}`);
  }
}

const store = new SqliteEmbeddingStore(dbPath, encoder);
console.log(`node opened ${dbPath}, ${store.count()} embeddings`);
for (const record of store.exportEmbeddings().embeddings) {
  check(`db     ${record.type}:${record.id}`, record.vector, record.text);
}
store.close();

const payload = JSON.parse(readFileSync(jsonPath, 'utf8'));
const headerOk = payload.format === 'ison-embeddings' && payload.version === 1;
console.log(`  ${headerOk ? 'ok  ' : 'FAIL'}  payload header: ` +
            `${payload.format} v${payload.version}`);
if (!headerOk) failures++;

const imported = new SqliteEmbeddingStore(':memory:', encoder);
const count = imported.importEmbeddings(payload);
if (count !== payload.embeddings.length) {
  console.log(`  FAIL  imported ${count} of ${payload.embeddings.length} rows`);
  failures++;
}
for (const record of imported.exportEmbeddings().embeddings) {
  check(`json   ${record.type}:${record.id}`, record.vector, record.text);
}
imported.close();

console.log(`${failures ? 'FAILED' : 'ok'} - ${failures} mismatch(es)`);
process.exit(failures ? 1 : 0);
