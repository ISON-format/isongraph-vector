/**
 * Binary snapshot of an EmbeddingStore (optional, beside the JSON sidecar).
 *
 * The byte layout is version 1 of the snapshot format in the Rust crate's
 * DESIGN.md, section 11.2; every integer is little endian and every float an
 * IEEE-754 binary32 (read and written through a DataView with an explicit byte
 * order, with a plain byte copy as the fast path on little-endian hosts). The
 * JSON sidecar stays the portable, authoritative format; a snapshot is a cache
 * of it, taken only when it provably belongs to the graph and sidecar on disk.
 *
 * This module is browser-safe: it touches no file and no `node:` API, and it
 * does not hash anything itself. The SHA-256 trailer is computed by a
 * {@link Sha256Like} the caller supplies (`node:crypto` in `./node`, which also
 * does the file handling and the sidecar fingerprint).
 *
 * Differences from the Rust crate that are forced by this port:
 *  - rows are written in the order of this store's JSON sidecar (arrival order,
 *    live rows only), so that a snapshot load and a JSON load give the same
 *    slots; the Rust crate's sidecar happens to be sorted, ours is not;
 *  - the format has no `id_type`, so an integer node id comes back as a string
 *    unless the caller's `resolveId` says otherwise (`./node` asks the graph).
 */
import { EmbeddingError, VectorTable, GRAPH_DIGEST_BASIS, hostIsLittleEndian, } from './index.js';
/** Appended to the graph file's name: `<graph>.embeddings.snap`. */
export const SNAPSHOT_SUFFIX = '.embeddings.snap';
/** The only layout version this build reads and writes. */
export const SNAPSHOT_VERSION = 1;
const MAGIC = 'ISONVSNP';
/** Rows per block (a writer's choice, recorded in the header; readers honour it). */
const BLOCK_ROWS = 1024;
/** Length value of an absent optional string. */
const ABSENT = 0xffffffff;
/** Header flag: the header carries the sidecar's identity. */
const FLAG_SIDECAR = 1;
const KNOWN_FLAGS = FLAG_SIDECAR;
const TRAILER = 32;
const TWO_32 = 4294967296;
const DEFAULT_WINDOW = 4 << 20;
/** The snapshot is not usable (malformed, truncated, stale, or not the one asked for). */
export class SnapshotError extends EmbeddingError {
    constructor(reason) {
        super(`invalid snapshot: ${reason}`);
        this.name = 'SnapshotError';
        this.reason = reason;
    }
}
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
// =============================================================================
// Writing
// =============================================================================
/** Appends little-endian values to a growing buffer. */
class ByteWriter {
    constructor(capacity) {
        this.len = 0;
        this.buf = new Uint8Array(capacity);
        this.dv = new DataView(this.buf.buffer);
    }
    reset() {
        this.len = 0;
    }
    /** Make room for `n` more bytes. */
    reserve(n) {
        if (this.len + n <= this.buf.length)
            return;
        const grown = new Uint8Array(Math.max(this.len + n, this.buf.length * 2));
        grown.set(this.buf.subarray(0, this.len));
        this.buf = grown;
        this.dv = new DataView(grown.buffer);
    }
    u32(value) {
        this.reserve(4);
        this.dv.setUint32(this.len, value, true);
        this.len += 4;
    }
    u64(value) {
        this.reserve(8);
        this.dv.setUint32(this.len, value % TWO_32, true);
        this.dv.setUint32(this.len + 4, Math.floor(value / TWO_32), true);
        this.len += 8;
    }
    bytes(data) {
        this.reserve(data.length);
        this.buf.set(data, this.len);
        this.len += data.length;
    }
    /** A length-prefixed string, already encoded. */
    str(data) {
        this.u32(data.length);
        this.bytes(data);
    }
    /** An optional string: absent is a length of 0xFFFFFFFF. */
    opt(value) {
        if (value === null || value === undefined)
            this.u32(ABSENT);
        else
            this.str(encoder.encode(value));
    }
    view() {
        return this.buf.subarray(0, this.len);
    }
}
/** A string the format can carry: UTF-8 cannot hold a lone surrogate. */
function wellFormed(s) {
    const check = s.isWellFormed;
    if (typeof check === 'function')
        return check.call(s);
    return !/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/.test(s);
}
function encodeText(s, what) {
    if (!wellFormed(s)) {
        throw new SnapshotError(`${what} holds a lone surrogate, which UTF-8 cannot carry; the JSON sidecar can`);
    }
    return encoder.encode(s);
}
/**
 * Find a row whose id would not come back from a snapshot as it went in, given
 * how `resolveId` reads id strings. Returns a description of the first one, or
 * null. The format has no id type, so this is how a writer makes sure the
 * snapshot restores the same store the JSON sidecar would.
 */
export function findUnrestorableId(store, resolveId) {
    const table = store.snapshotState().table;
    if (table === null)
        return null;
    for (let slot = 0; slot < table.n; slot++) {
        const ref = table.refs[slot];
        if (ref === null)
            continue;
        const back = resolveId(String(ref[0]), String(ref[1]));
        if (back !== ref[1]) {
            return (`the id ${JSON.stringify(ref[1])} of a '${String(ref[0])}' embedding would read back as ` +
                `${JSON.stringify(back)}: a snapshot stores ids as text, and the graph does not say which kind this is`);
        }
    }
    return null;
}
/**
 * Write the store as a snapshot: the bytes go to `sink` in order (header, then
 * one piece per block, then the 32-byte SHA-256 trailer taken from `hasher`),
 * each piece valid only until `sink` returns. Rows are the live ones in arrival
 * order. Throws SnapshotError for a value the format cannot carry; the caller
 * decides what to do with a partial write (`./node` writes to a temp file).
 */
export function writeSnapshot(store, identity, hasher, sink, options = {}) {
    const blockRows = options.blockRows ?? BLOCK_ROWS;
    if (!Number.isInteger(blockRows) || blockRows < 1 || blockRows > 1 << 20) {
        throw new EmbeddingError(`blockRows must be an integer in 1..${1 << 20}, got ${blockRows}`);
    }
    const littleEndian = hostIsLittleEndian() && !options.portable;
    const { table, encoderModel, encoderVersion } = store.snapshotState();
    const slots = [];
    if (table !== null)
        for (let s = 0; s < table.n; s++)
            if (table.keys[s] !== null)
                slots.push(s);
    const count = slots.length;
    const dim = table === null || count === 0 ? 0 : table.dim;
    // models, in order of first appearance
    const modelIndex = new Map();
    const models = [];
    if (table !== null) {
        for (const s of slots) {
            const name = table.modelNames[table.modelIx[s]];
            if (!modelIndex.has(name)) {
                modelIndex.set(name, models.length);
                models.push(name);
            }
        }
    }
    const digest = identity.graphDigest ?? null;
    const basis = digest === null ? null : GRAPH_DIGEST_BASIS;
    const sidecar = identity.sidecar ?? null;
    if (sidecar !== null && sidecar.fingerprint.length !== 32) {
        throw new EmbeddingError(`sidecar fingerprint must be 32 bytes, got ${sidecar.fingerprint.length}`);
    }
    const out = new ByteWriter(4096);
    out.bytes(encoder.encode(MAGIC));
    out.u32(SNAPSHOT_VERSION);
    out.u32(sidecar !== null ? FLAG_SIDECAR : 0);
    out.u32(dim);
    out.u32(blockRows);
    out.u64(count);
    out.opt(digest);
    out.opt(basis);
    out.opt(encoderModel);
    out.opt(encoderVersion);
    if (sidecar !== null) {
        out.u64(sidecar.len);
        out.bytes(sidecar.fingerprint);
    }
    out.u32(models.length);
    for (const name of models)
        out.str(encodeText(name, 'a model name'));
    let written = 0;
    const emit = () => {
        const bytes = out.view();
        hasher.update(bytes);
        sink(bytes);
        written += bytes.length;
    };
    emit();
    for (let start = 0; start < count; start += blockRows) {
        const rows = Math.min(blockRows, count - start);
        out.reset();
        for (let i = 0; i < rows; i++) {
            const s = slots[start + i];
            const ref = table.refs[s];
            out.str(encodeText(String(ref[0]), 'a node type'));
            out.str(encodeText(String(ref[1]), 'a node id'));
            out.str(encodeText(table.texts[s], 'an embedding text'));
            out.u32(modelIndex.get(table.modelNames[table.modelIx[s]]));
        }
        out.reserve(rows * dim * 4);
        for (let i = 0; i < rows; i++) {
            const s = slots[start + i];
            const chunk = table.chunks[s >>> 10];
            const base = (s & 1023) * dim;
            if (littleEndian) {
                out.buf.set(new Uint8Array(chunk.buffer, chunk.byteOffset + base * 4, dim * 4), out.len);
                out.len += dim * 4;
            }
            else {
                for (let j = 0; j < dim; j++) {
                    out.dv.setFloat32(out.len, chunk[base + j], true);
                    out.len += 4;
                }
            }
        }
        emit();
    }
    const trailer = hasher.digest();
    if (trailer.length !== TRAILER)
        throw new EmbeddingError('the hasher did not return 32 bytes');
    sink(trailer);
    written += TRAILER;
    return {
        version: SNAPSHOT_VERSION,
        dimension: dim,
        count,
        graphDigest: digest,
        graphDigestBasis: basis,
        encoderModel,
        encoderVersion,
        sidecar,
        bytes: written,
    };
}
/** {@link writeSnapshot} into memory. */
export function snapshotToBytes(store, identity, hasher, options = {}) {
    const pieces = [];
    const info = writeSnapshot(store, identity, hasher, piece => pieces.push(piece.slice()), options);
    const bytes = new Uint8Array(info.bytes);
    let at = 0;
    for (const piece of pieces) {
        bytes.set(piece, at);
        at += piece.length;
    }
    return { bytes, info };
}
// =============================================================================
// Reading
// =============================================================================
/**
 * A read-ahead window over a stream of known size. It hands out the next `n`
 * bytes of the body (everything before the trailer) as a view that is valid until
 * the next call, hashes each byte of the body once as it enters the window, and
 * refuses to hand out more than the body holds.
 */
class Window {
    constructor(_fill, size, _hasher, capacity) {
        this._fill = _fill;
        this._hasher = _hasher;
        this.pos = 0;
        this.end = 0;
        this.consumed = 0;
        this._pulled = 0;
        this._body = size - TRAILER;
        this.buf = new Uint8Array(capacity);
        this.dv = new DataView(this.buf.buffer);
    }
    remaining() {
        return this._body - this.consumed;
    }
    /** Make `n` bytes available at `pos`. */
    _ensure(n) {
        if (this.end - this.pos >= n)
            return;
        if (this.pos > 0) {
            this.buf.copyWithin(0, this.pos, this.end);
            this.end -= this.pos;
            this.pos = 0;
        }
        if (this.buf.length < n) {
            const grown = new Uint8Array(Math.max(n, this.buf.length * 2));
            grown.set(this.buf.subarray(0, this.end));
            this.buf = grown;
            this.dv = new DataView(grown.buffer);
        }
        while (this.end < n) {
            const got = this._fill(this.buf.subarray(this.end));
            if (!(got > 0))
                throw new SnapshotError('truncated');
            const hashable = Math.max(0, Math.min(got, this._body - this._pulled));
            if (hashable > 0)
                this._hasher.update(this.buf.subarray(this.end, this.end + hashable));
            this._pulled += got;
            this.end += got;
        }
    }
    take(n) {
        if (n > this.remaining())
            throw new SnapshotError('truncated');
        this._ensure(n);
        const view = this.buf.subarray(this.pos, this.pos + n);
        this.pos += n;
        this.consumed += n;
        return view;
    }
    u32() {
        if (4 > this.remaining())
            throw new SnapshotError('truncated');
        this._ensure(4);
        const value = this.dv.getUint32(this.pos, true);
        this.pos += 4;
        this.consumed += 4;
        return value;
    }
    u64() {
        const low = this.u32();
        const high = this.u32();
        return high * TWO_32 + low;
    }
    /** A length-prefixed string; `optional` lets the length be ABSENT. */
    string(optional) {
        const len = this.u32();
        if (len === ABSENT) {
            if (optional)
                return null;
            throw new SnapshotError('a required string is marked absent');
        }
        if (len > this.remaining())
            throw new SnapshotError('a string is longer than the file');
        const bytes = this.take(len);
        try {
            return decoder.decode(bytes);
        }
        catch {
            throw new SnapshotError('a string is not UTF-8');
        }
    }
    /** The trailer and a check that nothing follows it. Call after the body is consumed. */
    trailer() {
        this._ensureRaw(TRAILER);
        const view = this.buf.slice(this.pos, this.pos + TRAILER);
        this.pos += TRAILER;
        return view;
    }
    _ensureRaw(n) {
        if (this.end - this.pos >= n)
            return;
        if (this.pos > 0) {
            this.buf.copyWithin(0, this.pos, this.end);
            this.end -= this.pos;
            this.pos = 0;
        }
        while (this.end < n) {
            const got = this._fill(this.buf.subarray(this.end));
            if (!(got > 0))
                throw new SnapshotError('truncated');
            this._pulled += got;
            this.end += got;
        }
    }
    /** True when the stream has more bytes than the size it was declared with. */
    hasMore() {
        if (this.end > this.pos)
            return true;
        const probe = new Uint8Array(1);
        return this._fill(probe) > 0;
    }
}
function sameBytes(a, b) {
    if (a.length !== b.length)
        return false;
    for (let i = 0; i < a.length; i++)
        if (a[i] !== b[i])
            return false;
    return true;
}
/**
 * Read and fully verify a snapshot of `size` bytes whose bytes `fill` supplies,
 * applying every rejection rule of the format: bad magic, version or flags,
 * implausible shape, a length past the end, a model index out of range, a
 * repeated key, bytes between the last block and the trailer or after it, a
 * checksum mismatch. The rows go into a fresh table; nothing else is touched, so
 * a failure (a SnapshotError, or whatever `fill` throws) leaves every store as
 * it was. Install the result with {@link applySnapshot}.
 */
export function readSnapshot(fill, size, hasher, options = {}) {
    if (size < TRAILER + MAGIC.length)
        throw new SnapshotError('file too short');
    const portable = options.portable ?? false;
    const resolveId = options.resolveId ?? ((_type, id) => id);
    const w = new Window(fill, size, hasher, Math.max(64, options.windowBytes ?? DEFAULT_WINDOW));
    if (!sameBytes(w.take(MAGIC.length), encoder.encode(MAGIC)))
        throw new SnapshotError('not a snapshot (bad magic)');
    const version = w.u32();
    if (version !== SNAPSHOT_VERSION) {
        throw new SnapshotError(`unsupported snapshot version ${version} (this build reads ${SNAPSHOT_VERSION})`);
    }
    const flags = w.u32();
    if ((flags & ~KNOWN_FLAGS) !== 0)
        throw new SnapshotError(`unknown flags 0x${flags.toString(16)}`);
    const dim = w.u32();
    const blockRows = w.u32();
    const count = w.u64();
    if (blockRows === 0 || blockRows > 1 << 20)
        throw new SnapshotError(`implausible block size ${blockRows}`);
    if (count > 0xffffffff || (count > 0 && dim === 0))
        throw new SnapshotError(`implausible shape ${count} x ${dim}`);
    if (count * dim * 4 > w.remaining())
        throw new SnapshotError('fewer bytes than the rows need');
    const graphDigest = w.string(true);
    const graphDigestBasis = w.string(true);
    const encoderModel = w.string(true);
    const encoderVersion = w.string(true);
    let sidecar = null;
    if ((flags & FLAG_SIDECAR) !== 0) {
        const len = w.u64();
        sidecar = { len, fingerprint: w.take(32).slice() };
    }
    const modelCount = w.u32();
    if (modelCount > w.remaining() / 4)
        throw new SnapshotError('more models than the file can hold');
    const models = [];
    for (let i = 0; i < modelCount; i++)
        models.push(w.string(false));
    const info = {
        version,
        dimension: dim,
        count,
        graphDigest,
        graphDigestBasis,
        encoderModel,
        encoderVersion,
        sidecar,
        bytes: size,
    };
    const refusal = options.accept ? options.accept(info) : null;
    if (refusal !== null)
        throw new SnapshotError(refusal);
    const table = count > 0 ? new VectorTable(dim) : null;
    let left = count;
    while (left > 0) {
        const rows = Math.min(left, blockRows);
        const types = new Array(rows);
        const ids = new Array(rows);
        const texts = new Array(rows);
        const modelOf = new Int32Array(rows);
        for (let i = 0; i < rows; i++) {
            types[i] = w.string(false);
            ids[i] = w.string(false);
            texts[i] = w.string(false);
            const m = w.u32();
            if (m >= models.length)
                throw new SnapshotError('a row names a model that is not listed');
            modelOf[i] = m;
        }
        const need = rows * dim * 4;
        if (need > w.remaining())
            throw new SnapshotError('truncated');
        const floats = w.take(need);
        for (let i = 0; i < rows; i++) {
            const key = `${types[i]}:${ids[i]}`;
            if (table.index.has(key))
                throw new SnapshotError(`the key ${key} appears twice`);
            const ref = [types[i], resolveId(types[i], ids[i])];
            table.putLittleEndian(key, ref, texts[i], models[modelOf[i]], floats.subarray(i * dim * 4, (i + 1) * dim * 4), portable);
        }
        left -= rows;
    }
    if (w.remaining() !== 0)
        throw new SnapshotError('bytes left over after the last block');
    const expected = hasher.digest();
    const stored = w.trailer();
    if (!sameBytes(stored, expected))
        throw new SnapshotError('checksum mismatch');
    if (w.hasMore())
        throw new SnapshotError('bytes after the checksum');
    return { table, info };
}
/** {@link readSnapshot} from a byte array. */
export function readSnapshotBytes(bytes, hasher, options = {}) {
    let at = 0;
    const fill = into => {
        const n = Math.min(into.length, bytes.length - at);
        if (n <= 0)
            return 0;
        into.set(bytes.subarray(at, at + n));
        at += n;
        return n;
    };
    return readSnapshot(fill, bytes.length, hasher, options);
}
/**
 * Replace the store's embeddings with a verified snapshot's rows (and remember
 * its header as the store's `lastImportMeta`, as a JSON import does). The
 * approximate-index settings stay; the index is rebuilt lazily.
 */
export function applySnapshot(store, result) {
    const info = result.info;
    const meta = {};
    if (info.encoderModel !== null)
        meta.encoder_model = info.encoderModel;
    if (info.encoderVersion !== null)
        meta.encoder_version = info.encoderVersion;
    if (info.graphDigest !== null)
        meta.graph_digest = info.graphDigest;
    if (info.graphDigestBasis !== null)
        meta.graph_digest_basis = info.graphDigestBasis;
    store.adoptTable(result.table ?? new VectorTable(1), meta);
}
