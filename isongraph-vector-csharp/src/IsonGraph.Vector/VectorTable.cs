using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using IsonGraph;

namespace IsonGraph.Vector;

/// <summary>
/// The whole store as parallel arrays over "slots": a record is a slot number,
/// and nothing else owns the floats (DESIGN.md section 1).
///
/// The matrix is a list of chunks of <see cref="ChunkRows"/> rows, <c>Dim</c>
/// floats each, row-major. Chunking means growth never copies stored vectors
/// (only chunk 0, which starts small and doubles up to one full chunk, is ever
/// resized) and no single array reaches the large-object-heap sizes a
/// contiguous 1.5 GB matrix would need. A scan block never straddles chunks.
///
/// Slots are in arrival order. An update overwrites its slot in place, an
/// insert takes slot <c>n</c>, a delete tombstones (the norm becomes NaN, so a
/// scan scores it NaN and the collector drops it without a liveness branch).
/// When more than a quarter of the slots are dead the store compacts, moving
/// live rows down in order.
///
/// Not thread-safe: the owning store guards it (readers share, a writer is
/// alone).
/// </summary>
internal sealed class VectorTable
{
    public const int ChunkRows = 1024;
    private const int ChunkShift = 10;
    private const int ChunkMask = ChunkRows - 1;
    private const int FirstChunkRows = 16;

    public readonly int Dim;

    private float[]?[] _chunks = new float[]?[4];
    private int _chunkCount;
    private int _n;                      // slots in use, tombstones included
    private int _dead;

    private float[] _norms = new float[16];
    private int[] _codes = new int[16];
    private NodeRef[] _refs = new NodeRef[16];
    private string?[] _keys = new string?[16];    // null = tombstone
    private string[] _texts = new string[16];
    private string[] _models = new string[16];

    private readonly Dictionary<string, int> _index = new();          // key -> slot
    private readonly Dictionary<string, int> _codeOf = new();         // node type -> code
    private readonly List<string> _typeNames = new();
    private readonly Dictionary<string, string> _modelNames = new();  // interned model names
    private readonly Dictionary<string, int> _modelCounts = new();

    public VectorTable(int dim)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if ((long)dim * ChunkRows > Array.MaxLength)
            throw new EmbeddingException(
                $"vectors of {dim} dimensions are too wide for a {ChunkRows}-row chunk");
        Dim = dim;
    }

    public int Slots => _n;
    public int Live => _n - _dead;
    public int Dead => _dead;
    public bool ShouldCompact => _dead > 0 && (long)_dead * 4 > _n;
    public IReadOnlyDictionary<string, int> ModelCounts => _modelCounts;
    public IReadOnlyList<string> TypeNames => _typeNames;

    public float[] Norms => _norms;
    public int[] Codes => _codes;
    public bool IsLive(int slot) => (uint)slot < (uint)_n && _keys[slot] is not null;
    public NodeRef RefAt(int slot) => _refs[slot];
    public string KeyAt(int slot) => _keys[slot]!;
    public string TextAt(int slot) => _texts[slot];
    public string ModelAt(int slot) => _models[slot];
    public string TypeOf(int slot) => _typeNames[_codes[slot]];

    /// <summary>Code of a node type, or -1 when no row ever had that type.</summary>
    public int CodeOf(string nodeType) => _codeOf.TryGetValue(nodeType, out int c) ? c : -1;

    public bool TryGetSlot(string key, out int slot) => _index.TryGetValue(key, out slot);

    /// <summary>Number of chunks holding at least one slot.</summary>
    public int ChunkCount => (_n + ChunkRows - 1) >> ChunkShift;

    public float[] Chunk(int c) => _chunks[c]!;

    /// <summary>The vector of a slot (live or not), valid until the next write.</summary>
    public ReadOnlySpan<float> Row(int slot) =>
        new(_chunks[slot >> ChunkShift]!, (slot & ChunkMask) * Dim, Dim);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref float RowRef(int slot) =>
        ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_chunks[slot >> ChunkShift]!),
                       (slot & ChunkMask) * Dim);

    private Span<float> RowSpan(int slot) =>
        new(_chunks[slot >> ChunkShift]!, (slot & ChunkMask) * Dim, Dim);

    // -- writes -----------------------------------------------------------------

    private string Intern(string model)
    {
        if (!_modelNames.TryGetValue(model, out var shared)) _modelNames[model] = shared = model;
        return shared;
    }

    private int CodeFor(string nodeType)
    {
        if (!_codeOf.TryGetValue(nodeType, out int code))
        {
            code = _typeNames.Count;
            _codeOf[nodeType] = code;
            _typeNames.Add(nodeType);
        }
        return code;
    }

    private void GrowMeta()
    {
        int cap = Math.Max(_norms.Length * 2, 16);
        Array.Resize(ref _norms, cap);
        Array.Resize(ref _codes, cap);
        Array.Resize(ref _refs, cap);
        Array.Resize(ref _keys, cap);
        Array.Resize(ref _texts, cap);
        Array.Resize(ref _models, cap);
    }

    /// <summary>Make room for slot <paramref name="slot"/> in the matrix.</summary>
    private void EnsureRow(int slot)
    {
        int c = slot >> ChunkShift;
        if (c >= _chunkCount)
        {
            if (c >= _chunks.Length) Array.Resize(ref _chunks, Math.Max(_chunks.Length * 2, c + 1));
            // chunk 0 grows like a vector; every later chunk is allocated whole
            _chunks[c] = new float[(long)(c == 0 ? FirstChunkRows : ChunkRows) * Dim];
            _chunkCount = c + 1;
        }
        else if (c == 0)
        {
            var first = _chunks[0]!;
            if ((slot + 1) * Dim > first.Length)
            {
                int rows = Math.Min(ChunkRows, Math.Max(first.Length / Dim * 2, slot + 1));
                var bigger = new float[(long)rows * Dim];
                Array.Copy(first, bigger, first.Length);
                _chunks[0] = bigger;
            }
        }
        else if (_chunks[c] is null)
        {
            _chunks[c] = new float[(long)ChunkRows * Dim];
        }
    }

    private static float SumSq(ReadOnlySpan<float> v)
    {
        float s = 0f;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        return s;
    }

    /// <summary>
    /// Store a vector under <paramref name="key"/>: overwrite the slot in
    /// place if the key exists, else append. Returns the slot.
    /// </summary>
    public int Put(NodeRef nodeRef, string key, string text, ReadOnlySpan<float> vector, string model, out bool replaced)
    {
        if (vector.Length != Dim)
            throw new DimensionMismatchException($"vector has {vector.Length} dimensions, store holds {Dim}");
        model = Intern(model);
        int code = CodeFor(nodeRef.Type);
        var canonical = new NodeRef(_typeNames[code], nodeRef.Id);

        int slot;
        if (_index.TryGetValue(key, out slot))
        {
            replaced = true;
            string oldModel = _models[slot];
            if (!ReferenceEquals(oldModel, model)) { DropModel(oldModel); AddModel(model); }
        }
        else
        {
            replaced = false;
            if (_n >= _norms.Length) GrowMeta();
            slot = _n;
            EnsureRow(slot);
            _n++;
            _index[key] = slot;
            _keys[slot] = key;
            AddModel(model);
        }
        vector.CopyTo(RowSpan(slot));
        _norms[slot] = MathF.Sqrt(SumSq(vector));
        _codes[slot] = code;
        _refs[slot] = canonical;
        _texts[slot] = text;
        _models[slot] = model;
        return slot;
    }

    private void AddModel(string model) =>
        _modelCounts[model] = _modelCounts.GetValueOrDefault(model) + 1;

    private void DropModel(string model)
    {
        int c = _modelCounts[model] - 1;
        if (c == 0) _modelCounts.Remove(model); else _modelCounts[model] = c;
    }

    /// <summary>Tombstone a key's row. False if the key is not stored.</summary>
    public bool Remove(string key, out int slot)
    {
        if (!_index.Remove(key, out slot)) return false;
        DropModel(_models[slot]);
        _keys[slot] = null;
        _refs[slot] = default;
        _texts[slot] = "";
        _models[slot] = "";
        _norms[slot] = float.NaN;
        _dead++;
        return true;
    }

    public void Clear()
    {
        Array.Clear(_chunks);
        _chunkCount = 0;
        _n = 0;
        _dead = 0;
        _norms = new float[16];
        _codes = new int[16];
        _refs = new NodeRef[16];
        _keys = new string?[16];
        _texts = new string[16];
        _models = new string[16];
        _index.Clear();
        _modelCounts.Clear();
        // type and model name tables never shrink: they are tiny
    }

    /// <summary>
    /// Move live rows down over the tombstones, keeping their order. Returns
    /// <c>remap[old] = new slot</c> (-1 for a dead one) for the indexes that
    /// follow the table.
    /// </summary>
    public int[] Compact()
    {
        var remap = new int[_n];
        int kept = 0;
        for (int i = 0; i < _n; i++)
        {
            if (_keys[i] is null) { remap[i] = -1; continue; }
            remap[i] = kept;
            if (kept != i)                       // the destination never overtakes the source
            {
                Row(i).CopyTo(RowSpan(kept));
                _norms[kept] = _norms[i];
                _codes[kept] = _codes[i];
                _refs[kept] = _refs[i];
                _keys[kept] = _keys[i];
                _texts[kept] = _texts[i];
                _models[kept] = _models[i];
                _index[_keys[kept]!] = kept;
            }
            kept++;
        }
        for (int i = kept; i < _n; i++)
        {
            _keys[i] = null;
            _refs[i] = default;
            _texts[i] = "";
            _models[i] = "";
            _norms[i] = float.NaN;
        }
        _n = kept;
        _dead = 0;
        int chunks = (kept + ChunkRows - 1) >> ChunkShift;
        for (int c = chunks; c < _chunkCount; c++) _chunks[c] = null;       // give the emptied tail chunks back
        _chunkCount = chunks;
        return remap;
    }

    // -- checks -----------------------------------------------------------------

    /// <summary>Every invariant of DESIGN.md section 7 that concerns the table; null when they hold.</summary>
    public string? Verify()
    {
        if (_chunkCount < ChunkCount) return "fewer chunks than slots need";
        int dead = 0;
        for (int i = 0; i < _n; i++)
        {
            if (_keys[i] is null)
            {
                dead++;
                if (!float.IsNaN(_norms[i])) return $"dead slot {i} has a norm";
                continue;
            }
            if (!_index.TryGetValue(_keys[i]!, out int s) || s != i) return $"index disagrees about slot {i}";
            if (_keys[i] != $"{_refs[i].Type}:{_refs[i].Id}") return $"key of slot {i} is not type:id";
            if (_typeNames[_codes[i]] != _refs[i].Type) return $"type code of slot {i} is wrong";
        }
        if (dead != _dead) return $"dead counter {_dead} but {dead} tombstones";
        if (_index.Count != _n - _dead) return "index size is not the live count";
        var counts = new Dictionary<string, int>();
        for (int i = 0; i < _n; i++)
            if (_keys[i] is not null) counts[_models[i]] = counts.GetValueOrDefault(_models[i]) + 1;
        if (counts.Count != _modelCounts.Count || counts.Any(kv => _modelCounts.GetValueOrDefault(kv.Key) != kv.Value))
            return "model counts disagree with the rows";
        return null;
    }

    /// <summary>Bytes held by the matrix chunks (what the vectors cost).</summary>
    public long MatrixBytes
    {
        get
        {
            long total = 0;
            for (int c = 0; c < _chunkCount; c++) total += (long)_chunks[c]!.Length * sizeof(float);
            return total;
        }
    }
}
