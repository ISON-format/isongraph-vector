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
} from 'node:fs';
import { randomBytes } from 'node:crypto';
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

export interface SaveGraphOptions {
  /** 'ison', 'isonl', or 'auto' (by extension: .isonl is ISONL, else ISON). */
  format?: GraphFileFormat;
  /** Write the embeddings sidecar (set false for graph-only saves). Default true. */
  embeddings?: boolean;
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
    saveEmbeddings(graph.embeddingStore, embeddingsPathFor(path), graphDigest(reread));
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

/**
 * Number literals this port reads differently from Python, so a digest taken
 * here can differ from one taken there for an unedited file: integral floats
 * (`1.0` - a JS number cannot tell it from `1`) and integers beyond 2^53.
 */
function hasInexactNumberLiteral(text: string): boolean {
  return /(?:^|[\s|])[-+]?(?:\d+\.0+|\d{16,})(?=$|[\s|])/m.test(text);
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
 */
export function loadGraph(path: string, options: LoadGraphOptions = {}): SemanticGraph {
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

  if (embeddings ?? true) {
    const sidecar = embeddingsPathFor(path);
    if (existsSync(sidecar) && statSync(sidecar).isFile()) {
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
        const actual = graphDigest(base);
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
    }
  }
  return graph;
}
