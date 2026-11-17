//! Optional binary snapshot of an [`EmbeddingStore`], written next to (never
//! instead of) the JSON embeddings sidecar.
//!
//! The JSON sidecar stays the portable, authoritative format. A snapshot is a
//! cache of it: the same rows in the same order (the sidecar's: sorted by node
//! type then id, live rows only), as little-endian binary, so a load is a read
//! and a copy instead of a parse. A load takes it only when it provably belongs
//! to the graph and sidecar on disk and its checksum verifies; otherwise the
//! caller falls back to the JSON sidecar and is told why ([`LoadReport`]).
//!
//! `DESIGN.md` section 12 gives the exact byte layout. In short:
//!
//! ```text
//! header | block* | sha256(everything before it)
//! block  = metadata of R rows, then R * dim f32      (R = 1024 except the last)
//! ```
//!
//! Not stored, on purpose: the IVF index (see DESIGN.md: a rebuilt index is
//! deterministic but only reproduces a *saved* index when the slot order and
//! fold history are, which a canonical-order file does not keep) and the int8
//! copy (derived data, rebuilt by `set_scan_precision`).

use crate::digest::GRAPH_DIGEST_BASIS;
use crate::{EmbeddingError, EmbeddingStore, ImportMeta};
use sha2::{Digest, Sha256};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Appended to the graph file's name: `<graph>.embeddings.snap`.
pub const SNAPSHOT_SUFFIX: &str = ".embeddings.snap";
/// The only layout version this build reads and writes.
pub const SNAPSHOT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"ISONVSNP";
/// Rows per block (a writer's choice, recorded in the header; readers honour it).
const BLOCK_ROWS: u32 = 1024;
/// Length value of an absent optional string.
const ABSENT: u32 = u32::MAX;
/// Header flag: the header carries the sidecar's identity.
const FLAG_SIDECAR: u32 = 1;
const KNOWN_FLAGS: u32 = FLAG_SIDECAR;
const TRAILER: u64 = 32;
/// Bytes sampled from the head and the tail of the sidecar, and per probe.
const FP_EDGE: u64 = 64 * 1024;
const FP_PROBE: u64 = 4 * 1024;
const FP_PROBES: u64 = 64;

// =============================================================================
// Public types
// =============================================================================

/// A cheap identity for a sidecar file: its length and a SHA-256 over its
/// length, its first and last 64 KiB and 64 probes of 4 KiB spread over the
/// rest. Any rewrite of the sidecar by a serializer (every float re-rendered)
/// changes it; a hand edit that keeps the length and avoids every sampled byte
/// does not, which is why a save that does not write a snapshot deletes the old
/// one. See DESIGN.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SidecarId {
    pub len: u64,
    pub fingerprint: [u8; 32],
}

impl SidecarId {
    /// Read the identity of the file at `path`.
    pub fn of_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let len = file.metadata()?.len();
        let mut hasher = Sha256::new();
        hasher.update(len.to_le_bytes());
        let mut buf = Vec::new();
        let mut take = |file: &mut std::fs::File, at: u64, n: u64| -> io::Result<()> {
            let n = n.min(len.saturating_sub(at));
            buf.resize(n as usize, 0);
            file.seek(SeekFrom::Start(at))?;
            file.read_exact(&mut buf)?;
            hasher.update(&buf);
            Ok(())
        };
        take(&mut file, 0, FP_EDGE)?;
        take(&mut file, len.saturating_sub(FP_EDGE), FP_EDGE)?;
        for i in 1..=FP_PROBES {
            take(
                &mut file,
                (len as u128 * i as u128 / (FP_PROBES + 1) as u128) as u64,
                FP_PROBE,
            )?;
        }
        Ok(Self {
            len,
            fingerprint: hasher.finalize().into(),
        })
    }
}

/// What a snapshot is tied to.
#[derive(Clone, Debug, Default)]
pub struct SnapshotIdentity {
    /// `graph_digest` of the graph file (the JSON sidecar records the same).
    pub graph_digest: Option<String>,
    /// The JSON sidecar written for the same save.
    pub sidecar: Option<SidecarId>,
}

/// The header of a snapshot, and its size.
#[derive(Clone, Debug)]
pub struct SnapshotInfo {
    pub version: u32,
    pub dimension: usize,
    pub count: usize,
    pub graph_digest: Option<String>,
    pub graph_digest_basis: Option<String>,
    pub encoder_model: Option<String>,
    pub encoder_version: Option<String>,
    pub sidecar: Option<SidecarId>,
    /// File size in bytes.
    pub bytes: u64,
}

/// Where a load's embeddings came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadSource {
    /// The binary snapshot.
    Snapshot,
    /// The JSON sidecar.
    Json,
    /// Nothing was restored (embeddings off, or no sidecar).
    NoEmbeddings,
}

/// What `SemanticGraph::load_with_report` did.
#[derive(Clone, Debug)]
pub struct LoadReport {
    pub source: LoadSource,
    /// Why a snapshot that exists was not used (and the JSON sidecar was read
    /// instead); `None` when it was used or there is none.
    pub snapshot_skipped: Option<String>,
    /// Embeddings restored.
    pub embeddings: usize,
    /// Wall time of the whole load.
    pub seconds: f64,
}

/// `<graph path>.embeddings.snap`.
pub fn snapshot_path_for(graph_path: impl AsRef<Path>) -> PathBuf {
    let mut name = graph_path.as_ref().as_os_str().to_os_string();
    name.push(SNAPSHOT_SUFFIX);
    PathBuf::from(name)
}

// =============================================================================
// Errors, hashing wrappers
// =============================================================================

enum Fail {
    /// The file cannot be read or written.
    Io(String),
    /// The file is not a valid snapshot of the right identity.
    Bad(String),
}

impl Fail {
    fn into_error(self) -> EmbeddingError {
        match self {
            Fail::Io(m) => EmbeddingError::Io(m),
            Fail::Bad(m) => EmbeddingError::Payload(format!("invalid snapshot: {m}")),
        }
    }

    fn message(self) -> String {
        match self {
            Fail::Io(m) | Fail::Bad(m) => m,
        }
    }
}

fn io_fail(what: &str, e: io::Error) -> Fail {
    if e.kind() == io::ErrorKind::UnexpectedEof {
        Fail::Bad("truncated".to_string())
    } else {
        Fail::Io(format!("{what}: {e}"))
    }
}

/// Hashes everything written through it.
struct HashWriter<W: Write> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Hashes everything read through it and tracks how much of the body is left.
struct HashReader<R: Read> {
    inner: R,
    hasher: Sha256,
    consumed: u64,
    /// Bytes before the trailer.
    body: u64,
}

impl<R: Read> HashReader<R> {
    fn remaining(&self) -> u64 {
        self.body.saturating_sub(self.consumed)
    }

    fn exact(&mut self, buf: &mut [u8]) -> Result<(), Fail> {
        if buf.len() as u64 > self.remaining() {
            return Err(Fail::Bad("truncated".to_string()));
        }
        self.inner
            .read_exact(buf)
            .map_err(|e| io_fail("reading snapshot", e))?;
        self.hasher.update(&*buf);
        self.consumed += buf.len() as u64;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, Fail> {
        let mut b = [0u8; 4];
        self.exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, Fail> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// A length-prefixed string; `optional` lets the length be `ABSENT`.
    fn string(&mut self, optional: bool) -> Result<Option<String>, Fail> {
        let len = self.u32()?;
        if len == ABSENT {
            return if optional {
                Ok(None)
            } else {
                Err(Fail::Bad("a required string is marked absent".to_string()))
            };
        }
        if len as u64 > self.remaining() {
            return Err(Fail::Bad("a string is longer than the file".to_string()));
        }
        let mut b = vec![0u8; len as usize];
        self.exact(&mut b)?;
        String::from_utf8(b)
            .map(Some)
            .map_err(|_| Fail::Bad("a string is not UTF-8".to_string()))
    }
}

// =============================================================================
// Writing
// =============================================================================

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn put_opt(out: &mut Vec<u8>, s: Option<&str>) {
    match s {
        Some(s) => put_str(out, s),
        None => put_u32(out, ABSENT),
    }
}

/// Write `fill` to a temp file beside `path`, flush it to disk and rename it
/// over `path`, so a reader sees the old file or the new one, never a partial
/// one. A failure removes the temp file and leaves `path` as it was.
fn write_atomic<T>(
    path: &Path,
    fill: impl FnOnce(&mut BufWriter<std::fs::File>) -> Result<T, Fail>,
) -> Result<T, Fail> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "snapshot".to_string());
    let tmp = path.with_file_name(format!(
        ".{}.{}.{}.tmp",
        name,
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<T, Fail> {
        let file = std::fs::File::create(&tmp).map_err(|e| io_fail("creating snapshot", e))?;
        let mut writer = BufWriter::with_capacity(4 << 20, file);
        let value = fill(&mut writer)?;
        writer.flush().map_err(|e| io_fail("writing snapshot", e))?;
        let file = writer
            .into_inner()
            .map_err(|e| io_fail("writing snapshot", e.into_error()))?;
        // A filesystem that cannot fsync still gets the atomic rename.
        let _ = file.sync_all();
        drop(file);
        std::fs::rename(&tmp, path).map_err(|e| io_fail("replacing snapshot", e))?;
        Ok(value)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

impl EmbeddingStore {
    /// Write a binary snapshot of every live embedding to `path`, atomically
    /// (a temp file, flushed, renamed over `path`: a failure or crash leaves
    /// the previous file untouched). Rows are written in the order the JSON
    /// sidecar uses (sorted by node type, then id), so loading a snapshot gives
    /// the same store, slot for slot, as loading the sidecar.
    ///
    /// `identity` ties it to the graph and sidecar it was saved with; a load
    /// that finds a different identity refuses the snapshot.
    pub fn save_snapshot(
        &self,
        path: impl AsRef<Path>,
        identity: &SnapshotIdentity,
    ) -> Result<SnapshotInfo, EmbeddingError> {
        write_atomic(path.as_ref(), |w| self.write_snapshot(w, identity)).map_err(Fail::into_error)
    }

    /// Replace the store's embeddings with those of a snapshot file. Nothing
    /// changes unless the whole file verifies (structure and checksum); the
    /// scan precision and approximate-index settings are kept (the index is
    /// rebuilt lazily). Does not check identity: use `SemanticGraph::load` for
    /// that.
    pub fn load_snapshot(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<SnapshotInfo, EmbeddingError> {
        let (stage, info) = read_snapshot(path.as_ref(), &|_| Ok(())).map_err(Fail::into_error)?;
        self.adopt(stage, &info);
        Ok(info)
    }

    fn write_snapshot<W: Write>(
        &self,
        w: &mut W,
        identity: &SnapshotIdentity,
    ) -> Result<SnapshotInfo, Fail> {
        let table = &self.table;
        let dim = table.dim;
        // the sidecar's order: node type, then id
        let mut rows: Vec<(&str, &str, usize)> = table
            .live_slots()
            .filter_map(|s| table.type_and_id(s).map(|(t, i)| (t, i, s)))
            .collect();
        rows.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let count = rows.len();
        // models, in order of first appearance
        let mut models: Vec<&str> = Vec::new();
        let mut model_ix: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
        for &(_, _, s) in &rows {
            let m: &str = &table.meta(s).expect("live").model;
            if !model_ix.contains_key(m) {
                model_ix.insert(m, models.len() as u32);
                models.push(m);
            }
        }

        let basis = identity.graph_digest.as_ref().map(|_| GRAPH_DIGEST_BASIS);
        let encoder_model = self.current_model();
        let encoder_version = self.version();
        let mut head = Vec::with_capacity(512);
        head.extend_from_slice(MAGIC);
        put_u32(&mut head, SNAPSHOT_VERSION);
        put_u32(
            &mut head,
            if identity.sidecar.is_some() {
                FLAG_SIDECAR
            } else {
                0
            },
        );
        put_u32(&mut head, dim as u32);
        put_u32(&mut head, BLOCK_ROWS);
        head.extend_from_slice(&(count as u64).to_le_bytes());
        put_opt(&mut head, identity.graph_digest.as_deref());
        put_opt(&mut head, basis);
        put_opt(&mut head, encoder_model.as_deref());
        put_opt(&mut head, encoder_version.as_deref());
        if let Some(id) = &identity.sidecar {
            head.extend_from_slice(&id.len.to_le_bytes());
            head.extend_from_slice(&id.fingerprint);
        }
        put_u32(&mut head, models.len() as u32);
        for m in &models {
            put_str(&mut head, m);
        }

        let mut out = HashWriter {
            inner: w,
            hasher: Sha256::new(),
            written: 0,
        };
        let werr = |e: io::Error| io_fail("writing snapshot", e);
        out.write_all(&head).map_err(werr)?;
        let mut buf: Vec<u8> = Vec::with_capacity(BLOCK_ROWS as usize * (dim * 4 + 96));
        for block in rows.chunks(BLOCK_ROWS as usize) {
            buf.clear();
            for &(ty, id, s) in block {
                let meta = table.meta(s).expect("live");
                put_str(&mut buf, ty);
                put_str(&mut buf, id);
                put_str(&mut buf, &meta.text);
                put_u32(&mut buf, model_ix[&*meta.model]);
            }
            for &(_, _, s) in block {
                let at = buf.len();
                buf.resize(at + dim * 4, 0);
                for (dst, f) in buf[at..].chunks_exact_mut(4).zip(table.row(s)) {
                    dst.copy_from_slice(&f.to_le_bytes());
                }
            }
            out.write_all(&buf).map_err(werr)?;
        }
        let digest: [u8; 32] = out.hasher.clone().finalize().into();
        let body = out.written;
        out.inner.write_all(&digest).map_err(werr)?;
        Ok(SnapshotInfo {
            version: SNAPSHOT_VERSION,
            dimension: dim,
            count,
            graph_digest: identity.graph_digest.clone(),
            graph_digest_basis: basis.map(String::from),
            encoder_model,
            encoder_version,
            sidecar: identity.sidecar,
            bytes: body + TRAILER,
        })
    }

    /// Add one row of a snapshot to a staging store. False for a repeated key.
    fn put_snapshot_row(
        &mut self,
        ty: &str,
        id: &str,
        text: &str,
        model: &str,
        vector: &[f32],
    ) -> bool {
        let key = format!("{ty}:{id}");
        let model_name = self.intern_model(model);
        let (_, replaced) = self.table.put(&key, ty, vector, text, &model_name);
        if replaced.is_some() {
            return false;
        }
        self.dimension.get_or_insert(vector.len());
        match self.model_counts.get_mut(model) {
            Some(n) => *n += 1,
            None => {
                self.model_counts.insert(model.to_string(), 1);
            }
        }
        true
    }

    /// Install a verified staging store's rows in this one.
    fn adopt(&mut self, mut stage: EmbeddingStore, info: &SnapshotInfo) {
        let quant = self.table.quant().map(|q| q.rerank_factor);
        self.table = std::mem::replace(&mut stage.table, crate::table::VectorTable::new());
        self.table.set_quant(quant);
        self.model_names = std::mem::take(&mut stage.model_names);
        self.model_counts = std::mem::take(&mut stage.model_counts);
        self.dimension = stage.dimension;
        self.last_import_meta = ImportMeta {
            encoder_model: info.encoder_model.clone(),
            encoder_version: info.encoder_version.clone(),
            graph_digest: info.graph_digest.clone(),
            graph_digest_basis: info.graph_digest_basis.clone(),
        };
        *self.ivf_mut() = None;
    }

    /// Use the snapshot at `snap` for the graph whose digest is `graph_digest`
    /// and whose sidecar is at `sidecar`, if it belongs to them. `Err` is the
    /// reason it was not used (the store is untouched).
    pub(crate) fn adopt_snapshot_for(
        &mut self,
        snap: &Path,
        sidecar: &Path,
        graph_digest: &str,
    ) -> Result<SnapshotInfo, String> {
        let current = SidecarId::of_file(sidecar)
            .map_err(|e| format!("cannot read the sidecar {}: {e}", sidecar.display()))?;
        let (stage, info) = read_snapshot(snap, &|h| {
            if h.graph_digest.as_deref() != Some(graph_digest)
                || h.graph_digest_basis.as_deref() != Some(GRAPH_DIGEST_BASIS)
            {
                return Err("it was saved for a different graph (digest differs)".to_string());
            }
            match &h.sidecar {
                None => Err("it records no sidecar identity".to_string()),
                Some(id) if *id != current => Err(
                    "the JSON sidecar changed since the snapshot was written (stale snapshot)"
                        .to_string(),
                ),
                Some(_) => Ok(()),
            }
        })
        .map_err(Fail::message)?;
        self.adopt(stage, &info);
        Ok(info)
    }
}

// =============================================================================
// Reading
// =============================================================================

/// Read and fully verify the snapshot at `path`, after `accept` has approved
/// its header. Returns the rows in a fresh staging store.
fn read_snapshot(
    path: &Path,
    accept: &dyn Fn(&SnapshotInfo) -> Result<(), String>,
) -> Result<(EmbeddingStore, SnapshotInfo), Fail> {
    let file = std::fs::File::open(path)
        .map_err(|e| Fail::Io(format!("opening snapshot {}: {e}", path.display())))?;
    let file_len = file
        .metadata()
        .map_err(|e| io_fail("reading snapshot", e))?
        .len();
    if file_len < TRAILER + MAGIC.len() as u64 {
        return Err(Fail::Bad("file too short".to_string()));
    }
    let mut r = HashReader {
        inner: BufReader::with_capacity(4 << 20, file),
        hasher: Sha256::new(),
        consumed: 0,
        body: file_len - TRAILER,
    };
    let mut magic = [0u8; 8];
    r.exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Fail::Bad("not a snapshot (bad magic)".to_string()));
    }
    let version = r.u32()?;
    if version != SNAPSHOT_VERSION {
        return Err(Fail::Bad(format!(
            "unsupported snapshot version {version} (this build reads {SNAPSHOT_VERSION})"
        )));
    }
    let flags = r.u32()?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(Fail::Bad(format!("unknown flags {flags:#x}")));
    }
    let dim = r.u32()? as usize;
    let block_rows = r.u32()? as usize;
    let count = r.u64()?;
    if block_rows == 0 || block_rows > 1 << 20 {
        return Err(Fail::Bad(format!("implausible block size {block_rows}")));
    }
    if count > u32::MAX as u64 || (count > 0 && dim == 0) {
        return Err(Fail::Bad(format!("implausible shape {count} x {dim}")));
    }
    if (count as u128) * (dim as u128) * 4 > r.remaining() as u128 {
        return Err(Fail::Bad("fewer bytes than the rows need".to_string()));
    }
    let graph_digest = r.string(true)?;
    let graph_digest_basis = r.string(true)?;
    let encoder_model = r.string(true)?;
    let encoder_version = r.string(true)?;
    let sidecar = if flags & FLAG_SIDECAR != 0 {
        let len = r.u64()?;
        let mut fingerprint = [0u8; 32];
        r.exact(&mut fingerprint)?;
        Some(SidecarId { len, fingerprint })
    } else {
        None
    };
    let model_count = r.u32()? as usize;
    if model_count as u64 > r.remaining() / 4 {
        return Err(Fail::Bad("more models than the file can hold".to_string()));
    }
    let mut models = Vec::with_capacity(model_count);
    for _ in 0..model_count {
        models.push(r.string(false)?.unwrap_or_default());
    }
    let info = SnapshotInfo {
        version,
        dimension: dim,
        count: count as usize,
        graph_digest,
        graph_digest_basis,
        encoder_model,
        encoder_version,
        sidecar,
        bytes: file_len,
    };
    accept(&info).map_err(Fail::Bad)?;

    let mut stage = EmbeddingStore::new(None);
    let mut left = count as usize;
    let (mut bytes, mut floats): (Vec<u8>, Vec<f32>) = (Vec::new(), Vec::new());
    while left > 0 {
        let rows = left.min(block_rows);
        let mut meta: Vec<(String, String, String, usize)> = Vec::with_capacity(rows);
        for _ in 0..rows {
            let ty = r.string(false)?.unwrap_or_default();
            let id = r.string(false)?.unwrap_or_default();
            let text = r.string(false)?.unwrap_or_default();
            let model = r.u32()? as usize;
            if model >= models.len() {
                return Err(Fail::Bad(
                    "a row names a model that is not listed".to_string(),
                ));
            }
            meta.push((ty, id, text, model));
        }
        let need = rows * dim * 4;
        if need as u64 > r.remaining() {
            return Err(Fail::Bad("truncated".to_string()));
        }
        bytes.resize(need, 0);
        r.exact(&mut bytes)?;
        floats.clear();
        floats.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        for (i, (ty, id, text, model)) in meta.iter().enumerate() {
            if !stage.put_snapshot_row(
                ty,
                id,
                text,
                &models[*model],
                &floats[i * dim..(i + 1) * dim],
            ) {
                return Err(Fail::Bad(format!("the key {ty}:{id} appears twice")));
            }
        }
        left -= rows;
    }
    if r.remaining() != 0 {
        return Err(Fail::Bad(
            "bytes left over after the last block".to_string(),
        ));
    }
    let expected: [u8; 32] = r.hasher.clone().finalize().into();
    let mut stored = [0u8; 32];
    r.inner
        .read_exact(&mut stored)
        .map_err(|e| io_fail("reading snapshot", e))?;
    if stored != expected {
        return Err(Fail::Bad("checksum mismatch".to_string()));
    }
    let mut extra = [0u8; 1];
    if matches!(r.inner.read(&mut extra), Ok(n) if n > 0) {
        return Err(Fail::Bad("bytes after the checksum".to_string()));
    }
    Ok((stage, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_atomic_write_keeps_the_previous_file_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.snap");
        std::fs::write(&path, b"previous").unwrap();
        let result: Result<(), Fail> = write_atomic(&path, |w| {
            w.write_all(b"half of the new contents")
                .map_err(|e| io_fail("w", e))?;
            Err(Fail::Io("disk full".to_string()))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"previous");
        let left: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(left.len(), 1, "only the previous file remains");
        write_atomic(&path, |w| w.write_all(b"new").map_err(|e| io_fail("w", e)))
            .unwrap_or_else(|_| panic!("the next write works"));
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn a_failing_writer_fails_the_snapshot_without_touching_the_file() {
        struct Failing(usize);
        impl Write for Failing {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                if self.0 < b.len() {
                    return Err(io::Error::other("injected"));
                }
                self.0 -= b.len();
                Ok(b.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut store = EmbeddingStore::new(None);
        for i in 0..50 {
            store
                .add(
                    crate::NodeId::new("doc", i.to_string()),
                    "t",
                    Some(vec![i as f32 + 1.0; 8]),
                )
                .unwrap();
        }
        let mut sink = Failing(300);
        assert!(store
            .write_snapshot(&mut sink, &SnapshotIdentity::default())
            .is_err());
    }
}
