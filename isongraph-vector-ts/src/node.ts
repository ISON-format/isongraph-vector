/**
 * Saving and loading a SemanticGraph as files (Node.js only).
 *
 * A separate entry point (`isongraph-vector-ts/node`) rather than part of the
 * main module: it imports `node:fs`, so importing it from a browser bundle
 * would break. The main entry stays runtime-agnostic.
 *
 * The ISON graph format carries nodes and edges only, so the embeddings go to
 * a JSON sidecar at `<path>.embeddings.json` - the same file, in the same
 * `ison-embeddings` payload, that the Python port's `SemanticGraph.save` and
 * `SemanticGraph.load` write and read. A graph saved by either port loads in
 * the other.
 *
 * @example
 * ```typescript
 * import { SemanticGraph, MockEncoder } from 'isongraph-vector-ts';
 * import { saveGraph, loadGraph } from 'isongraph-vector-ts/node';
 *
 * saveGraph(graph, 'social.ison');                    // + social.ison.embeddings.json
 * const again = loadGraph('social.ison', { encoder: new MockEncoder(384) });
 * ```
 */

import {
  readFileSync,
  writeFileSync,
  existsSync,
  statSync,
  openSync,
  fsyncSync,
  closeSync,
  renameSync,
  unlinkSync,
  fstatSync,
  readSync,
  writeSync,
} from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { basename, dirname, extname, join } from 'node:path';
import { ISONGraph } from 'ison-graph-ts';
import {
  SemanticGraph,
  EmbeddingStore,
  EmbeddingError,
  GraphEmbeddingMismatchError,
  GRAPH_DIGEST_BASIS,
  EMBEDDING_SIDECAR_SUFFIX,
  graphDigest,
  type ExportedEmbeddings,
  type SemanticGraphOptions,
} from './index.js';
import {
  SNAPSHOT_SUFFIX,
  SnapshotError,
  writeSnapshot,
  readSnapshot,
  applySnapshot,
  findUnrestorableId,
  type Sha256Like,
  type SidecarId,
  type SnapshotIdentity,
  type SnapshotInfo,
  type SnapshotReadOptions,
  type SnapshotWriteOptions,
} from './snapshot.js';

export { EMBEDDING_SIDECAR_SUFFIX };

export type GraphFileFormat = 'auto' | 'ison' | 'isonl';

/** What loadGraph does when the sidecar's graph digest does not match the graph file. */
export type GraphMismatchMode = 'error' | 'warn' | 'reembed';

function resolveFormat(path: string, format: GraphFileFormat): 'ison' | 'isonl' {
  if (format !== 'auto') return format;
  return extname(path) === '.isonl' ? 'isonl' : 'ison';
}

/** Sidecar path holding the embeddings for a saved graph. */
export function embeddingsPathFor(path: string): string {
  return path + EMBEDDING_SIDECAR_SUFFIX;
}

/**
 * Write `text` to `path` so a reader sees the old file or the new one.
 *
 * The text goes to a temp file in the same directory (same filesystem, so the
 * rename is not a copy), is flushed to disk, and is then renamed into place. A
 * crash mid-write leaves the previous file intact instead of a truncated one,
 * and a failed write removes its temp file.
 */
export function writeFileAtomic(path: string, text: string): void {
  const tmp = join(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`
  );
  let fd: number | undefined;
  try {
    fd = openSync(tmp, 'w');
    writeFileSync(fd, text, 'utf8');
    try {
      fsyncSync(fd);
    } catch {
      // fsync unsupported on this filesystem; the rename still gives atomicity.
    }
    closeSync(fd);
    fd = undefined;
    renameSync(tmp, path);
  } catch (error) {
    if (fd !== undefined) {
      try { closeSync(fd); } catch { /* already failing */ }
    }
    try { unlinkSync(tmp); } catch { /* temp file never created */ }
    throw error;
  }
}

/**
 * Anything that exports and imports the portable payload: the in-memory
 * EmbeddingStore, or the SQLite-backed one from `isongraph-vector-ts/sqlite`.
 */
export type EmbeddingPayloadStore = Pick<EmbeddingStore, 'exportEmbeddings' | 'importEmbeddings'>;

/**
 * Write every embedding in a store to a JSON sidecar, atomically. Returns the
 * count. `graphDigest` ties the sidecar to one particular graph.
 */
export function saveEmbeddings(
  store: EmbeddingPayloadStore,
  path: string,
  graphDigest?: string | null
): number {
  const payload = store.exportEmbeddings(graphDigest);
  writeFileAtomic(path, JSON.stringify(payload));
  return payload.count;
}

/** Read a JSON sidecar written by saveEmbeddings (in any language port). */
export function loadEmbeddings(
  store: EmbeddingPayloadStore,
  path: string,
  replace: boolean = false
): number {
  const payload = JSON.parse(readFileSync(path, 'utf8')) as ExportedEmbeddings;
  return store.importEmbeddings(payload, replace);
}

// =============================================================================
// Binary snapshot (optional, beside the JSON sidecar)
// =============================================================================

export { SNAPSHOT_SUFFIX, SnapshotError };
export type { SidecarId, SnapshotIdentity, SnapshotInfo, SnapshotReadOptions, SnapshotWriteOptions };

/** `<graph path>.embeddings.snap`. */
export function snapshotPathFor(path: string): string {
  return path + SNAPSHOT_SUFFIX;
}

/** SHA-256 over `node:crypto`, in the shape the browser-safe snapshot module asks for. */
function sha256(): Sha256Like {
  const hash = createHash('sha256');
  return {
    update: bytes => {
      hash.update(bytes);
    },
    digest: () => new Uint8Array(hash.digest()),
  };
}

/** Bytes sampled from the head and tail of a sidecar, and per probe (see DESIGN.md 11.1). */
const FP_EDGE = 64 * 1024;
const FP_PROBE = 4 * 1024;
const FP_PROBES = 64;

/**
 * Identity of a sidecar file: its length and a SHA-256 over the length (u64 LE),
 * its first and last 64 KiB and 64 probes of 4 KiB at `floor(len * i / 65)`.
 * Any serializer rewriting the sidecar changes it; an in-place edit that keeps
 * the length and touches no sampled byte does not.
 */
export function sidecarIdentity(path: string): SidecarId {
  const fd = openSync(path, 'r');
  try {
    const len = fstatSync(fd).size;
    const hash = createHash('sha256');
    const head = Buffer.alloc(8);
    head.writeBigUInt64LE(BigInt(len));
    hash.update(head);
    const take = (at: number, want: number): void => {
      const n = Math.min(want, Math.max(0, len - at));
      const buf = Buffer.alloc(n);
      let got = 0;
      while (got < n) {
        const read = readSync(fd, buf, got, n - got, at + got);
        if (read === 0) throw new EmbeddingError(`${path} changed while its identity was being read`);
        got += read;
      }
      hash.update(buf);
    };
    take(0, FP_EDGE);
    take(Math.max(0, len - FP_EDGE), FP_EDGE);
    for (let i = 1; i <= FP_PROBES; i++) take(Number((BigInt(len) * BigInt(i)) / BigInt(FP_PROBES + 1)), FP_PROBE);
    return { len, fingerprint: new Uint8Array(hash.digest()) };
  } finally {
    closeSync(fd);
  }
}

/**
 * Write whatever `fill` streams to `write` into `path`, so a reader sees the old
 * file or the new one: a temp file in the same directory, flushed, renamed over
 * `path`. A failure removes the temp file and leaves `path` as it was.
 */
function writeStreamAtomic<T>(path: string, fill: (write: (bytes: Uint8Array) => void) => T): T {
  const tmp = join(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`
  );
  let fd: number | undefined;
  try {
    fd = openSync(tmp, 'w');
    const handle = fd;
    const value = fill(bytes => {
      let at = 0;
      while (at < bytes.length) at += writeSync(handle, bytes, at, bytes.length - at);
    });
    try {
      fsyncSync(fd);
    } catch {
      // fsync unsupported on this filesystem; the rename still gives atomicity.
    }
    closeSync(fd);
    fd = undefined;
    renameSync(tmp, path);
    return value;
  } catch (error) {
    if (fd !== undefined) {
      try { closeSync(fd); } catch { /* already failing */ }
    }
    try { unlinkSync(tmp); } catch { /* temp file never created */ }
    throw error;
  }
}

/**
 * Write a binary snapshot of every embedding in `store` to `path`, atomically (a
 * temp file, flushed, renamed over `path`: a failure or crash leaves the
 * previous file untouched). `identity` ties it to the graph and sidecar it was
 * saved with; a load that finds a different identity refuses it. Prefer
 * `saveGraph(graph, path, { snapshot: true })`, which does all of that.
 */
export function saveSnapshot(
  store: EmbeddingStore,
  path: string,
  identity: SnapshotIdentity = {},
  options: SnapshotWriteOptions = {}
): SnapshotInfo {
  return writeStreamAtomic(path, write => writeSnapshot(store, identity, sha256(), write, options));
}

function readSnapshotFile(path: string, options: SnapshotReadOptions): ReturnType<typeof readSnapshot> {
  const fd = openSync(path, 'r');
  try {
    const size = fstatSync(fd).size;
    return readSnapshot(into => readSync(fd, into, 0, into.length, null), size, sha256(), options);
  } finally {
    closeSync(fd);
  }
}

/**
 * Replace the store's embeddings with those of a snapshot file. Nothing changes
 * unless the whole file verifies (structure and checksum); throws SnapshotError
 * otherwise. Does not check identity (pass `accept` to), and ids come back as
 * strings unless `resolveId` says otherwise: use `loadGraph`, which does both.
 */
export function loadSnapshot(
  store: EmbeddingStore,
  path: string,
  options: SnapshotReadOptions = {}
): SnapshotInfo {
  const result = readSnapshotFile(path, options);
  applySnapshot(store, result);
  return result.info;
}

/**
 * How a snapshot's id strings map back to node ids, given the graph they were
 * saved with: an id that names a node of that type as a string stays a string;
 * a canonical integer string that names a node as a number becomes the number;
 * anything else stays a string. (The snapshot format carries no id type; the
 * JSON sidecar's `id_type` is what this stands in for.)
 */
function idResolverFor(graph: ISONGraph): (type: string, id: string) => string | number {
  return (type, id) => {
    if (graph.hasNode(type, id)) return id;
    const n = Number(id);
    if (id !== '' && String(n) === id && graph.hasNode(type, n)) return n;
    return id;
  };
}

function removeQuietly(path: string): void {
  try {
    unlinkSync(path);
  } catch {
    // no such file, or it cannot be removed: either way there is nothing to keep stale
  }
}

// =============================================================================
// Graph files
// =============================================================================

export interface SaveGraphOptions {
  /** 'ison', 'isonl', or 'auto' (by extension: .isonl is ISONL, else ISON). */
  format?: GraphFileFormat;
  /** Write the embeddings sidecar (set false for graph-only saves). Default true. */
  embeddings?: boolean;
  /**
   * Also write a binary snapshot of the embeddings at `<path>.embeddings.snap`,
   * after and in addition to the JSON sidecar (which stays the portable copy).
   * `loadGraph` uses it, when it provably belongs to the graph and sidecar on
   * disk, instead of parsing the JSON. Default false. A save without it deletes
   * an existing snapshot, which the new sidecar would make stale. If the
   * snapshot cannot be written the graph and sidecar are saved anyway, a stale
   * snapshot is removed, and the error is thrown (a SnapshotError when the store
   * holds something the snapshot cannot carry, such as an integer id for a node
   * the graph does not have).
   */
  snapshot?: boolean;
}

/**
 * Save a graph to a file, and its embeddings alongside it.
 *
 * Without the sidecar a save/load round trip would return a graph whose
 * searches all come back empty.
 *
 * Both files are written atomically (temp file, fsync, rename), so a crash
 * never leaves a truncated file. The pair cannot be replaced as one unit, so
 * the sidecar also records a `graphDigest` of the graph it belongs to;
 * `loadGraph` checks it and catches a crash between the two writes, a sidecar
 * copied from another save, and a graph edited afterwards. The digest is taken
 * of the graph *as it reads back from the text just written*, so a round trip
 * that normalises a value does not cause a false mismatch.
 *
 * With `embeddings: false` an existing sidecar is left in place (and will fail
 * the digest check on load, as it no longer matches).
 */
export function saveGraph(
  graph: SemanticGraph,
  path: string,
  options: SaveGraphOptions = {}
): void {
  const format = resolveFormat(path, options.format ?? 'auto');
  const text = format === 'isonl' ? graph.toIsonl() : graph.toIson();
  writeFileAtomic(path, text);
  if (options.embeddings ?? true) {
    const stem = basename(path, extname(path));
    const reread =
      format === 'isonl' ? ISONGraph.fromIsonl(text, stem) : ISONGraph.fromIson(text, stem);
    const digest = graphDigest(reread);
    const sidecar = embeddingsPathFor(path);
    const snap = snapshotPathFor(path);
    if (!options.snapshot) removeQuietly(snap);          // the sidecar is about to change: a snapshot of the old one is stale
    saveEmbeddings(graph.embeddingStore, sidecar, digest);
    if (options.snapshot) {
      try {
        const problem = findUnrestorableId(graph.embeddingStore, idResolverFor(reread));
        if (problem !== null) throw new SnapshotError(`not written: ${problem}`);
        saveSnapshot(graph.embeddingStore, snap, {
          graphDigest: digest,
          sidecar: sidecarIdentity(sidecar),
        });
      } catch (error) {
        removeQuietly(snap);
        throw error;
      }
    }
  }
}

export interface LoadGraphOptions extends SemanticGraphOptions {
  /** 'ison', 'isonl', or 'auto' (by extension). */
  format?: GraphFileFormat;
  /** Restore the sidecar written by saveGraph, if it exists. Default true. */
  embeddings?: boolean;
  /**
   * What to do when the sidecar carries a graph digest that does not match the
   * graph file (a sidecar from a different save, or a graph edited since):
   * 'error' (default) throws GraphEmbeddingMismatchError; 'warn' warns and
   * loads the sidecar's vectors anyway; 'reembed' discards the sidecar's
   * vectors and re-embeds every node with the encoder (embedAllNodes). A
   * sidecar without a digest (older releases, other ports) or with a digest
   * over an unknown basis is loaded as before and never checked.
   */
  onMismatch?: GraphMismatchMode;
}

/** Where a load's embeddings came from (see {@link loadGraphWithReport}). */
export interface LoadReport {
  /** 'snapshot' (the binary file), 'json' (the sidecar) or 'none' (embeddings off, or no sidecar). */
  source: 'snapshot' | 'json' | 'none';
  /**
   * Why a snapshot that exists was not used (and the JSON sidecar was read
   * instead); null when it was used or there is none.
   */
  snapshotSkipped: string | null;
  /** Embeddings restored. */
  embeddings: number;
  /** Wall time of the whole load. */
  seconds: number;
}

/**
 * Number literals this port reads differently from Python, so a digest taken
 * here can differ from one taken there for an unedited file: integral floats
 * (`1.0` - a JS number cannot tell it from `1`) and integers beyond 2^53.
 */
function hasInexactNumberLiteral(text: string): boolean {
  return /(?:^|[\s|])[-+]?(?:\d+\.0+|\d{16,})(?=$|[\s|])/m.test(text);
}

/** Warn when the sidecar's (or snapshot's) encoder version is not the one in use. */
function warnOnVersionSkew(graph: SemanticGraph): void {
  const storedVersion = graph.embeddingStore.lastImportMeta.encoder_version;
  const currentVersion = graph.embeddingStore.version;
  if (
    storedVersion !== undefined &&
    currentVersion !== null &&
    String(storedVersion) !== String(currentVersion)
  ) {
    console.warn(
      `embeddings were made by encoder version '${storedVersion}', ` +
        `but the encoder in use is version '${currentVersion}'`
    );
  }
}

/**
 * Load a graph from a file, restoring its embeddings from the sidecar.
 *
 * Pass the encoder the graph was saved with: without one, searching by text
 * throws NoEncoderError (stored vectors can still be searched by vector).
 * The graph is named after the file (its stem) unless `name` is given.
 * Loading never triggers auto-embedding; the saved vectors are restored as
 * they were.
 *
 * If the sidecar records a `graph_digest` it is checked against the graph just
 * read (see `onMismatch`). The comparison is skipped when the digest is absent
 * or taken over a basis this build does not know. Where the file holds number
 * literals this port cannot read exactly like Python (see graphDigest), a
 * mismatch is reported as a warning whatever `onMismatch` says, since it may
 * be a false positive. If both the encoder's version and the sidecar's
 * `encoder_version` are set and differ, a warning is issued.
 *
 * A binary snapshot beside the sidecar (`saveGraph` with `snapshot: true`, or
 * the Rust crate's `save_with_snapshot`) is used instead of the JSON when its
 * header names this graph's digest and the sidecar file as it is now and its
 * checksum verifies; the result is the same store the JSON gives. Any other
 * outcome falls back to the JSON, exactly as without a snapshot. Use
 * {@link loadGraphWithReport} to see which one was used.
 */
export function loadGraph(path: string, options: LoadGraphOptions = {}): SemanticGraph {
  return loadGraphWithReport(path, options).graph;
}

/** {@link loadGraph}, also saying where the embeddings came from. */
export function loadGraphWithReport(
  path: string,
  options: LoadGraphOptions = {}
): { graph: SemanticGraph; report: LoadReport } {
  const started = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const { format: fmt, embeddings, onMismatch = 'error', ...graphOptions } = options;
  if (onMismatch !== 'error' && onMismatch !== 'warn' && onMismatch !== 'reembed') {
    throw new EmbeddingError(
      `onMismatch must be 'error', 'warn' or 'reembed', got ${JSON.stringify(onMismatch)}`
    );
  }
  const format = resolveFormat(path, fmt ?? 'auto');
  // A graph saved on Windows by a tool that translates newlines arrives
  // CRLF-terminated, which the ISON parser rejects.
  const text = readFileSync(path, 'utf8').replace(/\r\n/g, '\n');
  const name = graphOptions.name ?? basename(path, extname(path));

  const base =
    format === 'isonl' ? ISONGraph.fromIsonl(text, name) : ISONGraph.fromIson(text, name);
  const graph = SemanticGraph.fromISONGraph(base, { ...graphOptions, name });

  const report = (
    source: LoadReport['source'],
    snapshotSkipped: string | null,
    count: number
  ): { graph: SemanticGraph; report: LoadReport } => ({
    graph,
    report: {
      source,
      snapshotSkipped,
      embeddings: count,
      seconds: ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - started) / 1000,
    },
  });

  if (!(embeddings ?? true)) return report('none', null, 0);
  const sidecar = embeddingsPathFor(path);
  if (!(existsSync(sidecar) && statSync(sidecar).isFile())) return report('none', null, 0);

  let actualDigest: string | null = null;
  const digestNow = (): string => (actualDigest ??= graphDigest(base));

  // The snapshot first: a graph digest equal to this graph's own (no leniency
  // for number literals read differently from Python), the sidecar as it is
  // now, a valid body and checksum. Anything else and the JSON is read.
  let snapshotSkipped: string | null = null;
  const snap = snapshotPathFor(path);
  if (existsSync(snap) && statSync(snap).isFile()) {
    try {
      const current = sidecarIdentity(sidecar);
      const digest = digestNow();
      const result = readSnapshotFile(snap, {
        resolveId: idResolverFor(base),
        accept: info => {
          if (info.graphDigest !== digest || info.graphDigestBasis !== GRAPH_DIGEST_BASIS) {
            return 'it was saved for a different graph (digest differs)';
          }
          if (info.sidecar === null) return 'it records no sidecar identity';
          if (
            info.sidecar.len !== current.len ||
            !info.sidecar.fingerprint.every((b, i) => b === current.fingerprint[i])
          ) {
            return 'the JSON sidecar changed since the snapshot was written (stale snapshot)';
          }
          return null;
        },
      });
      applySnapshot(graph.embeddingStore, result);
      warnOnVersionSkew(graph);
      return report('snapshot', null, graph.embeddingStore.count());
    } catch (error) {
      snapshotSkipped = error instanceof SnapshotError ? error.reason : (error as Error).message;
    }
  }

  let payload: ExportedEmbeddings;
  try {
    payload = JSON.parse(readFileSync(sidecar, 'utf8')) as ExportedEmbeddings;
  } catch (error) {
    throw new EmbeddingError(
      `unreadable embeddings sidecar ${sidecar}: ${(error as Error).message}`
    );
  }

  let problem: string | null = null;
  let unverifiable = false;
  const recorded = payload && typeof payload === 'object' ? payload.graph_digest : undefined;
  if (
    recorded !== undefined &&
    recorded !== null &&
    (payload.graph_digest_basis ?? GRAPH_DIGEST_BASIS) === GRAPH_DIGEST_BASIS
  ) {
    const actual = digestNow();
    if (recorded !== actual) {
      unverifiable = hasInexactNumberLiteral(text);
      problem =
        `embeddings sidecar ${sidecar} does not belong to graph ${path}: it was written ` +
        `for graph digest ${recorded}, but the graph reads as ${actual} (the sidecar is ` +
        `from a different save, or the graph was edited since). ` +
        (unverifiable
          ? `The graph file holds number literals (integral floats such as 1.0, or ` +
            `integers beyond 2^53) that this port reads differently from Python, so the ` +
            `digests may differ for an unedited file; using the sidecar's vectors anyway.`
          : `Pass onMismatch: 'reembed' to rebuild the vectors or onMismatch: 'warn' to ` +
            `use them anyway.`);
    }
  }

  if (problem !== null && onMismatch === 'error' && !unverifiable) {
    throw new GraphEmbeddingMismatchError(problem);
  }
  if (problem !== null && onMismatch === 'reembed' && !unverifiable) {
    graph.embedAllNodes();
  } else {
    if (problem !== null) console.warn(problem);
    graph.embeddingStore.importEmbeddings(payload);
    warnOnVersionSkew(graph);
  }
  return report('json', snapshotSkipped, graph.embeddingStore.count());
}
