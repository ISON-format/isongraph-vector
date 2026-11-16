/*
 * SQLite-backed embedding storage for IsonGraph.Vector, optionally indexed
 * with sqlite-vec.
 *
 * The schema is identical to the one the Python, Rust, TypeScript, JavaScript
 * and C++ ports write, so a database created by any of them can be opened by
 * any other. Vectors are little-endian float32, four bytes per dimension.
 *
 * Author: Mahesh Vaikri
 */

using System.Security.Cryptography;
using System.Text;
using Microsoft.Data.Sqlite;
using IsonGraph;
using IsonGraph.Vector;

namespace IsonGraph.Vector.Sqlite;

/// <summary>SQLite-backed embedding storage.</summary>
public sealed class SqliteEmbeddingStore : IDisposable
{
    /// <summary>Virtual table holding the vec0 index.</summary>
    public const string VecTable = "vec_embeddings";

    private readonly SqliteConnection _conn;
    private readonly object _lock = new();
    private IEmbeddingEncoder? _encoder;
    private int? _dimension;
    private readonly bool _sqliteVec;
    private bool _vecReady;
    private string? _version;
    private ModelMismatchMode _onModelMismatch;
    private Dictionary<string, int>? _modelsCache;
    private readonly HashSet<string> _warnedMismatch = new();
    private IReadOnlyDictionary<string, string> _lastImportMeta = new Dictionary<string, string>();

    /// <summary>
    /// Open (or create) a store. Use ":memory:" for a scratch store.
    /// </summary>
    /// <param name="path">Database path, or ":memory:"</param>
    /// <param name="encoder">Encoder used when a text is added without a vector</param>
    /// <param name="sqliteVec">
    /// Answer top-k searches through a sqlite-vec vec0 virtual table instead
    /// of scanning every row. Results are the same either way - vec0 runs an
    /// exact brute-force KNN in C.
    /// </param>
    /// <param name="version">
    /// Free-form version of the encoder/model that made these vectors, recorded
    /// in the sidecar. Falls back to the encoder's own Version.
    /// </param>
    /// <param name="onModelMismatch">
    /// What a text query does when the encoder's ModelName differs from the
    /// model that produced the stored vectors (see <see cref="EmbeddingStore"/>).
    /// </param>
    public SqliteEmbeddingStore(string path = ":memory:", IEmbeddingEncoder? encoder = null,
                                bool sqliteVec = false, string? version = null,
                                ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn)
    {
        if (!Enum.IsDefined(onModelMismatch))
            throw new ArgumentOutOfRangeException(nameof(onModelMismatch), onModelMismatch, "unknown ModelMismatchMode");
        _encoder = encoder;
        _sqliteVec = sqliteVec;
        _version = version;
        _onModelMismatch = onModelMismatch;

        _conn = new SqliteConnection($"Data Source={path}");
        _conn.Open();

        if (_sqliteVec) LoadVecExtension(_conn);

        CreateSchema();
        AdoptStoredDimension();
        if (_sqliteVec && _dimension is not null) EnsureVecTable(_dimension.Value);
    }

    /// <summary>
    /// Load the vec0 extension, trying each place it plausibly lives.
    ///
    /// SQLite resolves an extension name through the OS loader, which searches
    /// the working directory and PATH - not the paths .NET uses to resolve
    /// package assets. The vec0 natives ship under runtimes/{rid}/native and
    /// the package's .targets only performs a platform check, so a bare
    /// LoadExtension("vec0") finds nothing in a library or test project.
    /// Probing the package layout explicitly covers that.
    /// </summary>
    private static void LoadVecExtension(SqliteConnection conn)
    {
        conn.EnableExtensions(true);

        var rid = System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier;
        var baseDir = AppContext.BaseDirectory;
        var candidates = new List<string>
        {
            "vec0",
            System.IO.Path.Combine(baseDir, "vec0"),
            System.IO.Path.Combine(baseDir, "runtimes", rid, "native", "vec0"),
        };

        var failures = new List<string>();
        foreach (var candidate in candidates)
        {
            try
            {
                conn.LoadExtension(candidate);
                return;
            }
            catch (Exception ex)
            {
                failures.Add($"{candidate}: {ex.Message.Trim()}");
            }
        }

        throw new EmbeddingException(
            "could not load the sqlite-vec extension. Add the sqlite-vec package, " +
            "or construct the store with sqliteVec: false for the exact scan. " +
            "Tried " + string.Join("; ", failures));
    }

    /// <summary>Whether top-k search is answered from the sqlite-vec index.</summary>
    public bool SqliteVec => _sqliteVec;

    /// <summary>Vector length this store holds, or null while it is empty.</summary>
    public int? Dimension { get { lock (_lock) return _dimension; } }

    public IEmbeddingEncoder? Encoder
    {
        get => _encoder;
        set => _encoder = value;
    }

    /// <summary>Encoder/model version recorded in the sidecar: the one set explicitly, else the encoder's own.</summary>
    public string? Version
    {
        get => _version ?? _encoder?.Version;
        set => _version = value;
    }

    /// <summary>What a text query does when the encoder's model differs from the stored vectors'.</summary>
    public ModelMismatchMode OnModelMismatch
    {
        get => _onModelMismatch;
        set
        {
            if (!Enum.IsDefined(value)) throw new ArgumentOutOfRangeException(nameof(value), value, "unknown ModelMismatchMode");
            _onModelMismatch = value;
        }
    }

    /// <summary>Raised with a message when the store has something to warn about; with no subscriber it goes to standard error.</summary>
    public event Action<string>? Warning;

    /// <summary>Metadata of the most recently imported payload (only the keys it carried).</summary>
    public IReadOnlyDictionary<string, string> LastImportMeta
    {
        get { lock (_lock) return _lastImportMeta; }
    }

    private void CreateSchema()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_id_type TEXT NOT NULL DEFAULT 'str',
                vector BLOB NOT NULL,
                text TEXT NOT NULL,
                model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node_type, node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_node
                ON embeddings(node_type, node_id);";
        cmd.ExecuteNonQuery();
    }

    private void AdoptStoredDimension()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT LENGTH(vector) FROM embeddings LIMIT 1";
        var result = cmd.ExecuteScalar();
        if (result is not null && result is not DBNull)
        {
            var bytes = Convert.ToInt32(result);
            if (bytes > 0) _dimension = bytes / 4;
        }
    }

    /// <summary>
    /// Create the vec0 index once the vector width is known, and backfill it.
    /// </summary>
    private void EnsureVecTable(int dimension)
    {
        if (!_sqliteVec || _vecReady) return;

        using (var cmd = _conn.CreateCommand())
        {
            // node_type is a vec0 metadata column so a type-filtered search
            // stays exact - filtering after a top-k fetch would silently drop
            // matches whenever the first k hits were all of the wrong type.
            cmd.CommandText =
                $@"CREATE VIRTUAL TABLE IF NOT EXISTS {VecTable} USING vec0(
                       node_type text,
                       embedding float[{dimension}] distance_metric=cosine
                   )";
            cmd.ExecuteNonQuery();
        }

        long indexed = Scalar($"SELECT COUNT(*) FROM {VecTable}");
        long stored = Scalar("SELECT COUNT(*) FROM embeddings");

        if (indexed != stored)
        {
            // Opening a database written without the index, or with it off.
            using (var wipe = _conn.CreateCommand())
            {
                wipe.CommandText = $"DELETE FROM {VecTable}";
                wipe.ExecuteNonQuery();
            }

            var rows = new List<(long RowId, string NodeType, byte[] Vector)>();
            using (var read = _conn.CreateCommand())
            {
                read.CommandText = "SELECT rowid, node_type, vector FROM embeddings";
                using var reader = read.ExecuteReader();
                while (reader.Read())
                    rows.Add((reader.GetInt64(0), reader.GetString(1), (byte[])reader["vector"]));
            }
            foreach (var (rowId, nodeType, vector) in rows) InsertVecRow(rowId, nodeType, vector);
        }

        _vecReady = true;
    }

    private long Scalar(string sql)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    private void InsertVecRow(long rowId, string nodeType, byte[] vector)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText =
            $"INSERT INTO {VecTable}(rowid, node_type, embedding) VALUES ($r, $t, $e)";
        cmd.Parameters.AddWithValue("$r", rowId);
        cmd.Parameters.AddWithValue("$t", nodeType);
        cmd.Parameters.AddWithValue("$e", vector);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Surrogate primary key, matching what the other ports compute.</summary>
    private static string EmbeddingId(NodeRef nodeRef)
    {
        var digest = MD5.HashData(Encoding.UTF8.GetBytes($"{nodeRef.Type}:{nodeRef.Id}"));
        return Convert.ToHexString(digest).ToLowerInvariant()[..16];
    }

    private static byte[] Pack(float[] vector)
    {
        var bytes = new byte[vector.Length * 4];
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] Unpack(byte[] bytes)
    {
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private void CheckDimension(float[] vector, string what)
    {
        if (vector.Length == 0) throw new DimensionMismatchException($"{what} is empty");
        if (_dimension is null) _dimension = vector.Length;
        else if (vector.Length != _dimension)
            throw new DimensionMismatchException(
                $"{what} has {vector.Length} dimensions, store holds {_dimension}");
    }

    private void CheckQuery(float[] vector)
    {
        if (vector.Length == 0) throw new DimensionMismatchException("query vector is empty");
        if (_dimension is not null && vector.Length != _dimension)
            throw new DimensionMismatchException(
                $"query vector has {vector.Length} dimensions, store holds {_dimension}");
    }

    /// <summary>
    /// Cosine similarity over two vectors.
    ///
    /// Spans rather than IReadOnlyList: through the interface every element
    /// access is a dispatch, which measured roughly twenty times slower than
    /// the Rust and C++ ports scoring the same vectors. The stored vectors are
    /// float[], so a span costs nothing and the JIT indexes them directly.
    /// </summary>
    private static float CosineSimilarity(ReadOnlySpan<float> v1, ReadOnlySpan<float> v2)
    {
        float dot = 0f, norm1 = 0f, norm2 = 0f;
        int n = Math.Min(v1.Length, v2.Length);
        for (int i = 0; i < n; i++)
        {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        norm1 = MathF.Sqrt(norm1);
        norm2 = MathF.Sqrt(norm2);
        if (norm1 == 0f || norm2 == 0f) return 0f;
        return dot / (norm1 * norm2);
    }

    /// <summary>
    /// Insert or update one row, keeping the vec0 index in step.
    ///
    /// An UPSERT rather than INSERT OR REPLACE: REPLACE assigns a new rowid,
    /// which would orphan the matching vec0 entry and reset created_at.
    /// </summary>
    private string WriteRow(NodeRef nodeRef, float[] vector, string text, string model,
                            string nodeIdType = "str")
    {
        var embId = EmbeddingId(nodeRef);
        var blob = Pack(vector);
        _modelsCache = null;

        using (var cmd = _conn.CreateCommand())
        {
            cmd.CommandText = @"
                INSERT INTO embeddings
                (id, node_type, node_id, node_id_type, vector, text, model)
                VALUES ($id, $type, $nid, $idtype, $vec, $text, $model)
                ON CONFLICT(node_type, node_id) DO UPDATE SET
                    node_id_type = excluded.node_id_type,
                    vector = excluded.vector,
                    text = excluded.text,
                    model = excluded.model";
            cmd.Parameters.AddWithValue("$id", embId);
            cmd.Parameters.AddWithValue("$type", nodeRef.Type);
            cmd.Parameters.AddWithValue("$nid", nodeRef.Id);
            cmd.Parameters.AddWithValue("$idtype", nodeIdType);
            cmd.Parameters.AddWithValue("$vec", blob);
            cmd.Parameters.AddWithValue("$text", text);
            cmd.Parameters.AddWithValue("$model", model);
            cmd.ExecuteNonQuery();
        }

        if (_sqliteVec)
        {
            EnsureVecTable(vector.Length);
            long rowId;
            using (var cmd = _conn.CreateCommand())
            {
                cmd.CommandText =
                    "SELECT rowid FROM embeddings WHERE node_type = $t AND node_id = $i";
                cmd.Parameters.AddWithValue("$t", nodeRef.Type);
                cmd.Parameters.AddWithValue("$i", nodeRef.Id);
                rowId = Convert.ToInt64(cmd.ExecuteScalar());
            }
            // vec0 has no upsert; replace the entry at this rowid.
            using (var cmd = _conn.CreateCommand())
            {
                cmd.CommandText = $"DELETE FROM {VecTable} WHERE rowid = $r";
                cmd.Parameters.AddWithValue("$r", rowId);
                cmd.ExecuteNonQuery();
            }
            InsertVecRow(rowId, nodeRef.Type, blob);
        }

        return embId;
    }

    private string ModelName() => ModelCompat.Stamp(_encoder);

    /// <summary>
    /// Which models produced the stored vectors: model name to count.
    /// "unknown" and "provided" appear as-is.
    /// </summary>
    public IReadOnlyDictionary<string, int> Models()
    {
        lock (_lock)
        {
            if (_modelsCache is null)
            {
                var counts = new Dictionary<string, int>();
                using var cmd = _conn.CreateCommand();
                cmd.CommandText = "SELECT COALESCE(model, 'unknown'), COUNT(*) FROM embeddings GROUP BY 1";
                using var reader = cmd.ExecuteReader();
                while (reader.Read()) counts[reader.GetString(0)] = reader.GetInt32(1);
                _modelsCache = counts;
            }
            return new Dictionary<string, int>(_modelsCache);
        }
    }

    /// <summary>Complain when a text query's encoder is not the stored vectors' model.</summary>
    private void CheckModelCompat()
    {
        if (_onModelMismatch == ModelMismatchMode.Ignore) return;
        var current = ModelCompat.Current(_encoder);
        if (current is null) return;
        var foreign = ModelCompat.Foreign(current, Models().Keys);
        if (foreign.Count == 0) return;

        var message = ModelCompat.Message(current, foreign);
        if (_onModelMismatch == ModelMismatchMode.Error) throw new ModelMismatchException(message);

        bool first;
        lock (_lock) first = _warnedMismatch.Add(ModelCompat.WarnKey(current, foreign));
        if (!first) return;
        var handler = Warning;
        if (handler is not null) handler(message);
        else Console.Error.WriteLine("isongraph-vector: warning: " + message);
    }

    /// <summary>
    /// Node refs whose stored vector does not belong to the current encoder
    /// (see <see cref="EmbeddingStore.Stale"/>).
    /// </summary>
    public IReadOnlyList<NodeRef> Stale(string? model = null, int? dimension = null, bool includeUnknown = false)
    {
        lock (_lock)
        {
            model ??= ModelCompat.Current(_encoder);
            dimension ??= _encoder?.Dimension;

            var stale = new List<NodeRef>();
            using var cmd = _conn.CreateCommand();
            cmd.CommandText =
                "SELECT node_type, node_id, COALESCE(model, 'unknown'), LENGTH(vector) FROM embeddings ORDER BY rowid";
            using var reader = cmd.ExecuteReader();
            while (reader.Read())
            {
                if (ModelCompat.IsStale(reader.GetString(2), reader.GetInt32(3) / 4, model, dimension, includeUnknown))
                    stale.Add(new NodeRef(reader.GetString(0), reader.GetString(1)));
            }
            return stale;
        }
    }

    public string Add(NodeRef nodeRef, string text, float[]? vector = null)
    {
        lock (_lock)
        {
            if (vector is null)
            {
                if (_encoder is null)
                    throw new NoEncoderException("No encoder provided and no vector given");
                vector = _encoder.Encode(text);
            }
            CheckDimension(vector, "vector");
            return WriteRow(nodeRef, vector, text, ModelName());
        }
    }

    /// <summary>Add many embeddings in one transaction, encoding together.</summary>
    public int AddBatch(IReadOnlyList<(NodeRef NodeRef, string Text)> entries)
    {
        if (entries.Count == 0) return 0;
        if (_encoder is null)
            throw new NoEncoderException("No encoder provided and no vectors given");

        var vectors = _encoder.EncodeBatch(entries.Select(e => e.Text).ToList());
        if (vectors.Count != entries.Count)
            throw new EmbeddingException(
                $"encoder returned {vectors.Count} vectors for {entries.Count} texts");

        lock (_lock)
        {
            foreach (var v in vectors) CheckDimension(v, "vector");

            using var tx = _conn.BeginTransaction();
            var model = ModelName();
            for (int i = 0; i < entries.Count; i++)
                WriteRow(entries[i].NodeRef, vectors[i], entries[i].Text, model);
            tx.Commit();
            return entries.Count;
        }
    }

    public EmbeddingRecord? Get(NodeRef nodeRef)
    {
        lock (_lock)
        {
            using var cmd = _conn.CreateCommand();
            cmd.CommandText =
                "SELECT id, node_type, node_id, vector, text, model FROM embeddings " +
                "WHERE node_type = $t AND node_id = $i";
            cmd.Parameters.AddWithValue("$t", nodeRef.Type);
            cmd.Parameters.AddWithValue("$i", nodeRef.Id);

            using var reader = cmd.ExecuteReader();
            if (!reader.Read()) return null;
            return new EmbeddingRecord(
                reader.GetString(0),
                new NodeRef(reader.GetString(1), reader.GetString(2)),
                Unpack((byte[])reader["vector"]),
                reader.GetString(4),
                reader.IsDBNull(5) ? "unknown" : reader.GetString(5));
        }
    }

    public bool Remove(NodeRef nodeRef)
    {
        lock (_lock)
        {
            _modelsCache = null;
            if (_sqliteVec && _vecReady)
            {
                using var find = _conn.CreateCommand();
                find.CommandText =
                    "SELECT rowid FROM embeddings WHERE node_type = $t AND node_id = $i";
                find.Parameters.AddWithValue("$t", nodeRef.Type);
                find.Parameters.AddWithValue("$i", nodeRef.Id);
                var rowId = find.ExecuteScalar();
                if (rowId is not null && rowId is not DBNull)
                {
                    using var drop = _conn.CreateCommand();
                    drop.CommandText = $"DELETE FROM {VecTable} WHERE rowid = $r";
                    drop.Parameters.AddWithValue("$r", Convert.ToInt64(rowId));
                    drop.ExecuteNonQuery();
                }
            }

            using var cmd = _conn.CreateCommand();
            cmd.CommandText = "DELETE FROM embeddings WHERE node_type = $t AND node_id = $i";
            cmd.Parameters.AddWithValue("$t", nodeRef.Type);
            cmd.Parameters.AddWithValue("$i", nodeRef.Id);
            return cmd.ExecuteNonQuery() > 0;
        }
    }

    public int Count { get { lock (_lock) return (int)Scalar("SELECT COUNT(*) FROM embeddings"); } }

    /// <summary>
    /// Clear all embeddings. With <paramref name="resetDimension"/> (the
    /// default, as this method has always behaved) the pinned vector length is
    /// forgotten too, and the vec0 index - whose width is part of its DDL - is
    /// dropped so the next vector may have a different length; pass false to
    /// keep refusing vectors of the wrong length even when empty.
    /// </summary>
    public void Clear(bool resetDimension = true)
    {
        lock (_lock)
        {
            _modelsCache = null;
            using (var cmd = _conn.CreateCommand())
            {
                cmd.CommandText = "DELETE FROM embeddings";
                cmd.ExecuteNonQuery();
            }
            if (_sqliteVec && _vecReady)
            {
                using var cmd = _conn.CreateCommand();
                cmd.CommandText = $"DELETE FROM {VecTable}";
                cmd.ExecuteNonQuery();
            }
            if (!resetDimension) return;
            _dimension = null;
            if (_sqliteVec)
            {
                using var drop = _conn.CreateCommand();
                drop.CommandText = $"DROP TABLE IF EXISTS {VecTable}";
                drop.ExecuteNonQuery();
                _vecReady = false;
            }
        }
    }

    /// <summary>Score every stored vector against a query. Always exact.</summary>
    private List<(NodeRef NodeRef, float Score, string Text)> Scan(
        float[] queryVector, string? nodeType)
    {
        var results = new List<(NodeRef, float, string)>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT node_type, node_id, vector, text FROM embeddings";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var type = reader.GetString(0);
            if (nodeType is not null && type != nodeType) continue;
            // Reinterpret the blob as floats rather than Unpack-ing it into a
            // fresh array per row: one allocation per row measured 117 ms per
            // query over 20,000 rows against 22 ms through the vec0 index.
            var blob = (byte[])reader["vector"];
            var stored = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(blob);
            var score = CosineSimilarity(queryVector, stored);
            results.Add((new NodeRef(type, reader.GetString(1)), score, reader.GetString(3)));
        }
        return results;
    }

    /// <summary>Top-k from the vec0 index.</summary>
    private List<SimilarityResult> VecSearch(
        float[] queryVector, int topK, string? nodeType, float threshold)
    {
        EnsureVecTable(queryVector.Length);

        using var cmd = _conn.CreateCommand();
        var where = nodeType is null
            ? "WHERE v.embedding MATCH $q AND k = $k"
            : "WHERE v.node_type = $t AND v.embedding MATCH $q AND k = $k";
        cmd.CommandText =
            $"SELECT e.node_type, e.node_id, e.text, v.distance " +
            $"FROM {VecTable} v JOIN embeddings e ON e.rowid = v.rowid {where}";
        if (nodeType is not null) cmd.Parameters.AddWithValue("$t", nodeType);
        cmd.Parameters.AddWithValue("$q", Pack(queryVector));
        cmd.Parameters.AddWithValue("$k", topK);

        // Rows arrive ordered by distance ascending, i.e. score descending.
        var results = new List<SimilarityResult>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var score = 1f - (float)reader.GetDouble(3);
            if (score >= threshold)
                results.Add(new SimilarityResult(
                    new NodeRef(reader.GetString(0), reader.GetString(1)),
                    score, reader.GetString(2)));
        }
        return results;
    }

    public IReadOnlyList<SimilarityResult> SimilaritySearch(
        string query, int? topK = 10, string? nodeType = null, float threshold = 0.0f)
    {
        if (_encoder is null) throw new NoEncoderException("No encoder provided for text query");
        CheckModelCompat();
        return SimilaritySearchVector(_encoder.Encode(query), topK, nodeType, threshold);
    }

    public IReadOnlyList<SimilarityResult> SimilaritySearchVector(
        float[] queryVector, int? topK = 10, string? nodeType = null, float threshold = 0.0f)
    {
        lock (_lock)
        {
            CheckQuery(queryVector);

            // A zero vector has no direction for cosine to measure; vec0
            // returns NaN there, while the scan defines it as 0.0.
            if (_sqliteVec && topK is > 0 && queryVector.Any(v => v != 0f))
                return VecSearch(queryVector, topK.Value, nodeType, threshold);

            var results = Scan(queryVector, nodeType)
                .Where(r => r.Score >= threshold)
                .Select(r => new SimilarityResult(r.NodeRef, r.Score, r.Text))
                .ToList();
            if (topK is null)
            {
                results.Sort((a, b) => b.Score.CompareTo(a.Score));
                return results;
            }
            return EmbeddingStore.SelectTopK(results, topK.Value);
        }
    }

    /// <summary>
    /// Cosine score for every embedded node, keyed by "type:id".
    ///
    /// Always scans, even with the index on: the index answers top-k queries,
    /// and blending needs a score for every node.
    /// </summary>
    public IReadOnlyDictionary<string, float> ScoreMap(float[] queryVector, string? nodeType = null)
    {
        lock (_lock)
        {
            CheckQuery(queryVector);
            var scores = new Dictionary<string, float>();
            foreach (var (nodeRef, score, _) in Scan(queryVector, nodeType))
                scores[$"{nodeRef.Type}:{nodeRef.Id}"] = score;
            return scores;
        }
    }

    /// <summary>Serialize every embedding into the portable payload.</summary>
    public ExportedEmbeddings ExportEmbeddings(string? graphDigest = null)
    {
        lock (_lock)
        {
            var payload = new ExportedEmbeddings { Dimension = _dimension };
            using var cmd = _conn.CreateCommand();
            cmd.CommandText =
                "SELECT node_type, node_id, node_id_type, vector, text, model FROM embeddings " +
                "ORDER BY node_type, node_id";
            using var reader = cmd.ExecuteReader();
            while (reader.Read())
            {
                payload.Embeddings.Add(new ExportedEmbedding
                {
                    Type = reader.GetString(0),
                    Id = System.Text.Json.JsonSerializer.SerializeToElement(reader.GetString(1)),
                    IdType = reader.GetString(2),
                    Text = reader.GetString(4),
                    Model = reader.IsDBNull(5) ? "unknown" : reader.GetString(5),
                    Vector = Unpack((byte[])reader["vector"]),
                });
            }
            payload.Count = payload.Embeddings.Count;
            payload.EncoderModel = ModelCompat.Current(_encoder);
            payload.EncoderVersion = Version;
            if (graphDigest is not null)
            {
                payload.GraphDigest = graphDigest;
                payload.GraphDigestBasis = IsonGraphVectorVersion.GraphDigestBasis;
            }
            return payload;
        }
    }

    /// <summary>Load a payload produced by any language port.</summary>
    public int ImportEmbeddings(ExportedEmbeddings payload, bool replace = false)
    {
        if (payload.Format != IsonGraphVectorVersion.EmbeddingFormat)
            throw new EmbeddingException(
                $"not an {IsonGraphVectorVersion.EmbeddingFormat} payload: format={payload.Format}");
        if (payload.Version != IsonGraphVectorVersion.EmbeddingFormatVersion)
            throw new EmbeddingException(
                $"unsupported {IsonGraphVectorVersion.EmbeddingFormat} version {payload.Version} " +
                $"(this build reads version {IsonGraphVectorVersion.EmbeddingFormatVersion})");

        lock (_lock)
        {
            var meta = new Dictionary<string, string>();
            if (payload.EncoderModel is not null) meta["encoder_model"] = payload.EncoderModel;
            if (payload.EncoderVersion is not null) meta["encoder_version"] = payload.EncoderVersion;
            if (payload.GraphDigest is not null) meta["graph_digest"] = payload.GraphDigest;
            if (payload.GraphDigestBasis is not null) meta["graph_digest_basis"] = payload.GraphDigestBasis;
            _lastImportMeta = meta;

            if (replace)
            {
                Clear(resetDimension: true);
                _dimension = payload.Dimension;
            }

            using var tx = _conn.BeginTransaction();
            int written = 0;
            foreach (var item in payload.Embeddings)
            {
                CheckDimension(item.Vector, "vector");
                WriteRow(new NodeRef(item.Type, item.IdAsString()), item.Vector,
                         item.Text, item.Model ?? "unknown", item.IdType);
                written++;
            }
            tx.Commit();
            return written;
        }
    }

    public string ToJson(string? graphDigest = null) =>
        System.Text.Json.JsonSerializer.Serialize(ExportEmbeddings(graphDigest));

    public int FromJson(string json, bool replace = false)
    {
        var payload = System.Text.Json.JsonSerializer.Deserialize<ExportedEmbeddings>(json)
            ?? throw new EmbeddingException("payload did not deserialize");
        return ImportEmbeddings(payload, replace);
    }

    /// <summary>Write every embedding to a JSON sidecar, atomically. Returns the count.</summary>
    public int SaveEmbeddings(string path, string? graphDigest = null)
    {
        var payload = ExportEmbeddings(graphDigest);
        AtomicFile.WriteText(path, System.Text.Json.JsonSerializer.Serialize(payload));
        return payload.Count;
    }

    /// <summary>Read a JSON sidecar written by <see cref="SaveEmbeddings"/> (or by any port).</summary>
    public int LoadEmbeddings(string path, bool replace = false) =>
        FromJson(File.ReadAllText(path, Encoding.UTF8), replace);

    /// <summary>
    /// Close the connection and release the database file.
    ///
    /// Microsoft.Data.Sqlite pools connections, so disposing alone leaves the
    /// file handle open and a caller cannot delete or move the database. The
    /// pool is cleared explicitly.
    /// </summary>
    public void Dispose()
    {
        _conn.Close();
        SqliteConnection.ClearPool(_conn);
        _conn.Dispose();
    }
}
