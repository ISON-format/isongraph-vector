using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace IsonGraph.Vector;

/// <summary>
/// HNSW graph over a <see cref="VectorTable"/> (DESIGN.md section 12), ported
/// from the Rust reference (<c>src/hnsw.rs</c>).
///
/// The graph holds slot numbers and links only: every score is computed from
/// the one stored copy of each vector with the same kernels and the same
/// <see cref="Cosine"/> the exact scan uses, so a hit's score is bit-identical
/// to the exact scan's, and ties order by slot. Layer 0 holds every node with
/// up to <c>2 * M</c> links; a node also lives in layers <c>1..level</c>
/// (<c>P(level >= l) = M^-l</c>, drawn from a seeded integer generator) with up
/// to <c>M</c> links per layer.
///
/// Mutating members are called only under the store's write lock; searches run
/// under the shared lock and touch no shared mutable state (all scratch is per
/// thread), so any number run at once. The build and the per-write link are
/// planned in parallel against a frozen graph and applied in a fixed order, so
/// the graph depends on the seed, the data and the insertion order, never on
/// the thread count.
/// </summary>
internal sealed class HnswIndex
{
    public const int DefaultM = 16;
    public const int DefaultEfConstruction = 200;
    public const int MaxM = 64;

    private const byte Absent = byte.MaxValue;
    private const int None = -1;
    private const int MaxLevel = 24;
    private const int BuildBatch = 512;          // largest batch planned against one frozen graph
    private const int BuildRatio = 16;           // a batch is at most size / 16 nodes
    private const double RebuildFraction = 1.0;  // updates + deletes beyond this x nodes call for a rebuild
    private const int MaxCap = 2 * MaxM;
    private const int ChunkShift = 10;
    private const int ChunkRows = 1 << ChunkShift;
    private const int ChunkMask = ChunkRows - 1;

    private readonly int _m, _m0, _efC;
    private byte[] _levels = Array.Empty<byte>();        // per slot: highest layer, or Absent
    private int[] _upperAt = Array.Empty<int>();         // per slot: start of its upper blocks in _upper, or None
    private int _slots;                                  // slots covered (== table slots)
    private int[][] _l0 = new int[4][];                  // layer 0: ChunkRows nodes per chunk, m0 + 1 ints each (count, links)
    private int _l0Count;
    private int[] _upper = Array.Empty<int>();           // a node of level L owns L consecutive blocks of m + 1 ints
    private int _upperLen;
    private int _entry = None;
    private int _top;
    private LevelRng _rng;
    private int _members;                                // nodes in the graph, tombstones included

    public long Churn { get; private set; }
    public double BuildSeconds { get; private set; }
    public long Inserts { get; private set; }
    public double InsertSeconds { get; private set; }
    public int Members => _members;
    public int MaxLevelInGraph => _top;
    public bool Stale => Churn > RebuildFraction * Math.Max(1, _members);

    public long IndexBytes
    {
        get
        {
            long bytes = _levels.Length + (long)_upperAt.Length * 4 + (long)_upper.Length * 4;
            for (int c = 0; c < _l0Count; c++) bytes += (long)_l0[c].Length * 4;
            return bytes;
        }
    }

    private HnswIndex(int m, int efConstruction, ulong rngState)
    {
        _m = m;
        _m0 = 2 * m;
        _efC = Math.Max(efConstruction, m);
        _rng = new LevelRng { State = rngState };
    }

    private int Cap(int level) => level == 0 ? _m0 : _m;

    // -- the level generator: SplitMix64, integers only -------------------------

    private struct LevelRng
    {
        public ulong State;

        public ulong Next()
        {
            unchecked
            {
                State += 0x9E3779B97F4A7C15UL;
                ulong z = State;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                return z ^ (z >> 31);
            }
        }

        /// <summary>Uniform in [0, n): the high 64 bits of <c>Next() * n</c>.</summary>
        public int Below(int n) => (int)Math.BigMul(Next(), (ulong)n, out _);
    }

    /// <summary><c>level = 0; while level &lt; 24 and rng.below(m) == 0: level++</c>. No logarithm (portable).</summary>
    private int DrawLevel()
    {
        int level = 0;
        while (level < MaxLevel && _rng.Below(_m) == 0) level++;
        return level;
    }

    /// <summary>The seed of the level generator for a configured store seed (<c>seed XOR 0x4857535748575357</c>).</summary>
    internal static ulong RngStateFor(int seed) => unchecked((ulong)(long)seed) ^ 0x4857535748575357UL;

    // -- storage -----------------------------------------------------------------

    /// <summary>Cover slot <paramref name="s"/> (new slots start absent, with empty link blocks).</summary>
    private void EnsureSlot(int s)
    {
        if (s >= _slots)
        {
            if (s >= _levels.Length)
            {
                int cap = Math.Max(Math.Max(_levels.Length * 2, s + 1), 16);
                Array.Resize(ref _levels, cap);
                Array.Resize(ref _upperAt, cap);
            }
            for (int i = _slots; i <= s; i++) { _levels[i] = Absent; _upperAt[i] = None; }
            _slots = s + 1;
        }
        while (_l0Count <= s >> ChunkShift)
        {
            if (_l0Count == _l0.Length) Array.Resize(ref _l0, _l0.Length * 2);
            _l0[_l0Count++] = new int[ChunkRows * (_m0 + 1)];
        }
    }

    /// <summary>The links of <paramref name="s"/> on <paramref name="level"/> (it must have that layer).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ReadOnlySpan<int> List(int s, int level)
    {
        if (level == 0)
        {
            var chunk = _l0[s >> ChunkShift];
            int at = (s & ChunkMask) * (_m0 + 1);
            return new ReadOnlySpan<int>(chunk, at + 1, chunk[at]);
        }
        int block = _upperAt[s] + (level - 1) * (_m + 1);
        return new ReadOnlySpan<int>(_upper, block + 1, _upper[block]);
    }

    private void SetList(int s, int level, ReadOnlySpan<int> items)
    {
        Debug.Assert(items.Length <= Cap(level));
        int[] block;
        int at;
        if (level == 0)
        {
            block = _l0[s >> ChunkShift];
            at = (s & ChunkMask) * (_m0 + 1);
        }
        else
        {
            block = _upper;
            at = _upperAt[s] + (level - 1) * (_m + 1);
        }
        block[at] = items.Length;
        items.CopyTo(new Span<int>(block, at + 1, items.Length));
    }

    private void AddNode(int s, int level)
    {
        EnsureSlot(s);
        Debug.Assert(_levels[s] == Absent);
        _levels[s] = (byte)level;
        if (level >= 1)
        {
            int need = level * (_m + 1);
            if (_upperLen + need > _upper.Length)
                Array.Resize(ref _upper, Math.Max(Math.Max(_upper.Length * 2, _upperLen + need), 256));
            _upperAt[s] = _upperLen;
            _upperLen += need;
        }
        _members++;
    }

    private bool InGraph(int s) => (uint)s < (uint)_slots && _levels[s] != Absent;

    // -- scores -------------------------------------------------------------------

    /// <summary>
    /// The row's norm; for a tombstone (the table marks it NaN) the norm of its
    /// still-present floats, so a search can route through it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float RowNorm(VectorTable t, float[] norms, int slot)
    {
        float n = norms[slot];
        return float.IsNaN(n) ? MathF.Sqrt(RowScan.SumSq(t.Row(slot))) : n;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Finite(float score) => float.IsNaN(score) ? float.NegativeInfinity : score;

    /// <summary>
    /// Cosine of the query (<paramref name="q"/>, norm <paramref name="qn"/>)
    /// with each of <paramref name="slots"/>: 8, then 4, then 1 rows at a time
    /// with the scan's own kernels, so a live row scores exactly as the scan
    /// scores it. A tombstone is scored from its floats; anything that stays
    /// NaN (a corrupt row) ranks last.
    /// </summary>
    private static void ScoreInto(VectorTable t, ref float q, float qn, ReadOnlySpan<int> slots, Span<float> output)
    {
        int n = slots.Length, dim = t.Dim, i = 0;
        var norms = t.Norms;
        Span<float> d8 = stackalloc float[8];
        for (; i + 8 <= n; i += 8)
        {
            RowScan.Dot8Gather(ref q, ref t.RowRef(slots[i]), ref t.RowRef(slots[i + 1]), ref t.RowRef(slots[i + 2]),
                ref t.RowRef(slots[i + 3]), ref t.RowRef(slots[i + 4]), ref t.RowRef(slots[i + 5]),
                ref t.RowRef(slots[i + 6]), ref t.RowRef(slots[i + 7]), dim, d8);
            for (int j = 0; j < 8; j++)
                output[i + j] = Finite(Cosine.Score(d8[j], qn, RowNorm(t, norms, slots[i + j])));
        }
        for (; i + 4 <= n; i += 4)
        {
            RowScan.Dot4(ref q, ref t.RowRef(slots[i]), ref t.RowRef(slots[i + 1]), ref t.RowRef(slots[i + 2]),
                ref t.RowRef(slots[i + 3]), dim, out float d0, out float d1, out float d2, out float d3);
            output[i] = Finite(Cosine.Score(d0, qn, RowNorm(t, norms, slots[i])));
            output[i + 1] = Finite(Cosine.Score(d1, qn, RowNorm(t, norms, slots[i + 1])));
            output[i + 2] = Finite(Cosine.Score(d2, qn, RowNorm(t, norms, slots[i + 2])));
            output[i + 3] = Finite(Cosine.Score(d3, qn, RowNorm(t, norms, slots[i + 3])));
        }
        for (; i < n; i++)
        {
            int s = slots[i];
            output[i] = Finite(Cosine.Score(RowScan.Dot1(ref q, ref t.RowRef(s), dim), qn, RowNorm(t, norms, s)));
        }
    }

    private static Hit ScoreOne(VectorTable t, ref float q, float qn, int slot)
    {
        Span<int> one = stackalloc int[1];
        Span<float> score = stackalloc float[1];
        one[0] = slot;
        ScoreInto(t, ref q, qn, one, score);
        return new Hit(score[0], slot);
    }

    /// <summary>"Better is greater": higher score, then lower slot (the scan's tie-break).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Better(float sa, int ia, float sb, int ib) => sa > sb || (sa == sb && ia < ib);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Better(Hit a, Hit b) => Better(a.Score, a.Slot, b.Score, b.Slot);

    // -- per-thread scratch ---------------------------------------------------------

    /// <summary>A binary heap of scored nodes with the best (or the worst) on top.</summary>
    private sealed class ScHeap
    {
        private float[] _s = new float[256];
        private int[] _id = new int[256];
        private int _n;
        private readonly bool _bestOnTop;

        public ScHeap(bool bestOnTop) => _bestOnTop = bestOnTop;

        public int Count => _n;
        public void Clear() => _n = 0;
        public Hit Peek() => new(_s[0], _id[0]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool Above(float sa, int ia, float sb, int ib) =>
            _bestOnTop ? Better(sa, ia, sb, ib) : Better(sb, ib, sa, ia);

        public void Push(Hit h)
        {
            if (_n == _s.Length)
            {
                Array.Resize(ref _s, _n * 2);
                Array.Resize(ref _id, _n * 2);
            }
            int i = _n++;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (!Above(h.Score, h.Slot, _s[p], _id[p])) break;
                _s[i] = _s[p];
                _id[i] = _id[p];
                i = p;
            }
            _s[i] = h.Score;
            _id[i] = h.Slot;
        }

        public Hit Pop()
        {
            var top = new Hit(_s[0], _id[0]);
            int last = --_n;
            if (last > 0)
            {
                float s = _s[last];
                int id = _id[last];
                int i = 0;
                for (;;)
                {
                    int l = 2 * i + 1;
                    if (l >= last) break;
                    int c = l;
                    int r = l + 1;
                    if (r < last && Above(_s[r], _id[r], _s[l], _id[l])) c = r;
                    if (!Above(_s[c], _id[c], s, id)) break;
                    _s[i] = _s[c];
                    _id[i] = _id[c];
                    i = c;
                }
                _s[i] = s;
                _id[i] = id;
            }
            return top;
        }
    }

    /// <summary>
    /// Everything a searching thread reuses: the visited bitmap (1 bit per slot,
    /// cleared by the words it touched, so a query pays for what it visited),
    /// both heaps and the score buffers. Never shared between threads.
    /// </summary>
    private sealed class Scratch
    {
        private ulong[] _bits = new ulong[64];
        private int[] _words = new int[256];
        private int _wordCount;
        public readonly ScHeap Cand = new(bestOnTop: true);
        public readonly ScHeap Kept = new(bestOnTop: false);
        public readonly int[] Fresh = new int[MaxCap];
        public readonly float[] Scores = new float[MaxCap];

        public void EnsureBits(int slots)
        {
            ClearVisited();
            int need = (slots + 63) >> 6;
            if (_bits.Length < need) _bits = new ulong[Math.Max(need, _bits.Length * 2)];
        }

        public void ClearVisited()
        {
            for (int i = 0; i < _wordCount; i++) _bits[_words[i]] = 0;
            _wordCount = 0;
        }

        /// <summary>Mark <paramref name="slot"/>; true if it already was.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TestSet(int slot)
        {
            int w = slot >> 6;
            ulong m = 1UL << (slot & 63);
            ulong cur = _bits[w];
            if ((cur & m) != 0) return true;
            if (cur == 0)
            {
                if (_wordCount == _words.Length) Array.Resize(ref _words, _wordCount * 2);
                _words[_wordCount++] = w;
            }
            _bits[w] = cur | m;
            return false;
        }
    }

    [ThreadStatic] private static Scratch? t_scratch;

    private static Scratch GetScratch() => t_scratch ??= new Scratch();

    // -- search ---------------------------------------------------------------------

    /// <summary>Which nodes a layer-0 search may keep (the others are only routed through).</summary>
    private readonly struct Admit
    {
        private readonly float[] _norms;
        private readonly int[] _codes;
        private readonly int _code;
        private readonly float _threshold;
        private readonly int _exclude;

        public Admit(VectorTable t, int code, float threshold, int exclude)
        {
            _norms = t.Norms;
            _codes = t.Codes;
            _code = code;
            _threshold = threshold;
            _exclude = exclude;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Ok(int slot, float score) =>
            slot != _exclude && !float.IsNaN(_norms[slot]) && score >= _threshold && (_code < 0 || _codes[slot] == _code);
    }

    /// <summary>What a layer search returns: the admitted nodes best first, and whether it gave up scoring too many.</summary>
    internal readonly struct Found
    {
        public readonly Hit[] Hits;
        public readonly bool BudgetHit;
        public Found(Hit[] hits, bool budgetHit) { Hits = hits; BudgetHit = budgetHit; }
    }

    /// <summary>Walk to the best node reachable by single steps on <paramref name="level"/>.</summary>
    private Hit Greedy(Scratch sc, VectorTable t, ref float q, float qn, Hit cur, int level)
    {
        var scores = sc.Scores;
        for (;;)
        {
            var list = List(cur.Slot, level);
            if (list.IsEmpty) return cur;
            ScoreInto(t, ref q, qn, list, scores);
            var best = cur;
            bool moved = false;
            for (int i = 0; i < list.Length; i++)
            {
                var c = new Hit(scores[i], list[i]);
                if (Better(c, best)) { best = c; moved = true; }
            }
            if (!moved) return cur;
            cur = best;
        }
    }

    /// <summary>
    /// Best-first search of one layer from <paramref name="eps"/>, keeping the
    /// <paramref name="ef"/> best admitted nodes. Tombstones and non-admitted
    /// nodes are routed through but never kept. Stops when the best unexpanded
    /// node is worse than the worst kept one (once ef are kept), or after
    /// <paramref name="budget"/> nodes were scored.
    /// </summary>
    private Found SearchLayer(Scratch sc, VectorTable t, ref float q, float qn, ReadOnlySpan<Hit> eps, int ef, int level,
        in Admit admit, int budget)
    {
        sc.ClearVisited();
        var cand = sc.Cand;
        var kept = sc.Kept;
        cand.Clear();
        kept.Clear();
        foreach (var e in eps)
        {
            if (sc.TestSet(e.Slot)) continue;
            cand.Push(e);
            if (admit.Ok(e.Slot, e.Score))
            {
                kept.Push(e);
                if (kept.Count > ef) kept.Pop();
            }
        }
        var fresh = sc.Fresh;
        var scores = sc.Scores;
        int scored = 0;
        bool budgetHit = false;
        while (cand.Count > 0)
        {
            var c = cand.Pop();
            if (kept.Count >= ef && Better(kept.Peek(), c)) break;      // c is worse than the worst kept
            int nf = 0;
            foreach (int x in List(c.Slot, level))
                if (!sc.TestSet(x)) fresh[nf++] = x;
            if (nf == 0) continue;
            ScoreInto(t, ref q, qn, new ReadOnlySpan<int>(fresh, 0, nf), scores);
            scored += nf;
            for (int i = 0; i < nf; i++)
            {
                var node = new Hit(scores[i], fresh[i]);
                if (kept.Count < ef || Better(node, kept.Peek()))
                {
                    cand.Push(node);
                    if (admit.Ok(node.Slot, node.Score))
                    {
                        kept.Push(node);
                        if (kept.Count > ef) kept.Pop();
                    }
                }
            }
            if (scored > budget) { budgetHit = true; break; }
        }
        var best = new Hit[kept.Count];
        for (int i = best.Length - 1; i >= 0; i--) best[i] = kept.Pop();      // worst first, so fill from the back
        return new Found(best, budgetHit);
    }

    /// <summary>
    /// The <paramref name="ef"/> best admitted nodes for a query, best first
    /// (ties by slot). Thread-safe under the shared lock.
    /// </summary>
    public Found Search(VectorTable t, float[] query, float qn, int code, float threshold, int ef, int budget)
    {
        if (_entry == None) return new Found(Array.Empty<Hit>(), false);
        var sc = GetScratch();
        sc.EnsureBits(_slots);
        try
        {
            ref float q = ref MemoryMarshal.GetArrayDataReference(query);
            var cur = ScoreOne(t, ref q, qn, _entry);
            for (int level = _top; level >= 1; level--) cur = Greedy(sc, t, ref q, qn, cur, level);
            var admit = new Admit(t, code, threshold, None);
            return SearchLayer(sc, t, ref q, qn, MemoryMarshal.CreateReadOnlySpan(ref cur, 1), ef, 0, admit, budget);
        }
        finally { sc.ClearVisited(); }
    }

    // -- linking --------------------------------------------------------------------

    private struct Job
    {
        public int Slot;
        public int Level;
        public bool Relink;
    }

    /// <summary>
    /// The diversity heuristic: walk <paramref name="cands"/> (best first) and
    /// keep one only if it scores no better with any node already kept than
    /// with the base; stop at <paramref name="cap"/>. Fewer than cap candidates are all kept.
    /// </summary>
    private static int[] Select(VectorTable t, Hit[] cands, int cap)
    {
        if (cands.Length <= cap)
        {
            var all = new int[cands.Length];
            for (int i = 0; i < all.Length; i++) all[i] = cands[i].Slot;
            return all;
        }
        var chosen = new int[cap];
        int count = 0;
        Span<float> scores = stackalloc float[MaxCap];
        var norms = t.Norms;
        foreach (var c in cands)
        {
            if (count >= cap) break;
            ScoreInto(t, ref t.RowRef(c.Slot), RowNorm(t, norms, c.Slot), new ReadOnlySpan<int>(chosen, 0, count), scores);
            bool reject = false;
            for (int i = 0; i < count; i++)
                if (scores[i] > c.Score) { reject = true; break; }
            if (!reject) chosen[count++] = c.Slot;
        }
        return count == chosen.Length ? chosen : chosen.AsSpan(0, count).ToArray();
    }

    /// <summary>
    /// Neighbours chosen for <paramref name="job"/> on every layer, against the
    /// graph as it is now plus the other new members of its batch (the
    /// intra-batch slots with level >= the layer, scored by brute force).
    /// </summary>
    private int[][] Plan(VectorTable t, in Job job, int[] intraSlots, int[] intraLevels, int intraCount)
    {
        var sc = GetScratch();
        sc.EnsureBits(_slots);
        int x = job.Slot, level = job.Level;
        var norms = t.Norms;
        ref float q = ref t.RowRef(x);
        float qn = RowNorm(t, norms, x);
        var admit = new Admit(t, -1, float.NegativeInfinity, x);
        int top = _top;
        var lists = new int[level + 1][];
        Hit[] eps = Array.Empty<Hit>();
        if (_entry != None)
        {
            var cur = ScoreOne(t, ref q, qn, _entry);
            for (int l = top; l >= level + 1; l--) cur = Greedy(sc, t, ref q, qn, cur, l);
            eps = new[] { cur };
        }
        for (int l = level; l >= 0; l--)
        {
            Hit[] cands = Array.Empty<Hit>();
            if (l <= top && eps.Length > 0)
            {
                cands = SearchLayer(sc, t, ref q, qn, eps, _efC, l, admit, int.MaxValue).Hits;
                if (cands.Length > 0) eps = cands;
            }
            int members = 0;
            for (int i = 0; i < intraCount; i++)
                if (intraSlots[i] != x && intraLevels[i] >= l) members++;
            if (members > 0)
            {
                var slots = new int[members];
                for (int i = 0, j = 0; i < intraCount; i++)
                    if (intraSlots[i] != x && intraLevels[i] >= l) slots[j++] = intraSlots[i];
                var scores = new float[members];
                ScoreInto(t, ref q, qn, slots, scores);
                var all = new Hit[cands.Length + members];
                cands.CopyTo(all, 0);
                for (int i = 0; i < members; i++) all[cands.Length + i] = new Hit(scores[i], slots[i]);
                Array.Sort(all, Hit.BestFirst);
                cands = all;
            }
            lists[l] = Select(t, cands, _m);
        }
        return lists;
    }

    /// <summary>
    /// <paramref name="target"/>'s list on <paramref name="layer"/> with
    /// <paramref name="incoming"/> added, pruned to the layer's capacity when it
    /// overflows (tombstones are dropped then). Null if nothing changes.
    /// </summary>
    private int[]? GrownList(VectorTable t, int layer, int target, ReadOnlySpan<int> incoming)
    {
        int cap = Cap(layer);
        var cur = List(target, layer);
        var all = new int[cur.Length + incoming.Length];
        cur.CopyTo(all);
        int count = cur.Length;
        foreach (int n in incoming)
        {
            if (n == target) continue;
            bool seen = false;
            for (int i = 0; i < count; i++) if (all[i] == n) { seen = true; break; }
            if (!seen) all[count++] = n;
        }
        if (count == cur.Length) return null;
        if (count <= cap) return all.AsSpan(0, count).ToArray();
        var norms = t.Norms;
        int live = 0;
        for (int i = 0; i < count; i++)
            if (!float.IsNaN(norms[all[i]])) all[live++] = all[i];
        count = live;
        if (count <= cap) return all.AsSpan(0, count).ToArray();
        var scores = new float[count];
        ScoreInto(t, ref t.RowRef(target), RowNorm(t, norms, target), new ReadOnlySpan<int>(all, 0, count), scores);
        var cands = new Hit[count];
        for (int i = 0; i < count; i++) cands[i] = new Hit(scores[i], all[i]);
        Array.Sort(cands, Hit.BestFirst);
        return Select(t, cands, cap);
    }

    private struct Add
    {
        public int Layer;
        public int Target;
        public int Source;
        public int Seq;
    }

    private sealed class AddOrder : IComparer<Add>
    {
        public static readonly AddOrder Instance = new();

        public int Compare(Add a, Add b)
        {
            if (a.Layer != b.Layer) return a.Layer.CompareTo(b.Layer);
            if (a.Target != b.Target) return a.Target.CompareTo(b.Target);
            return a.Seq.CompareTo(b.Seq);        // the sequence number makes the order total (and the sort stable)
        }
    }

    private static int Degree(int maxThreads, int items)
    {
        if (items <= 1) return 1;
        int want = maxThreads > 0 ? maxThreads : Environment.ProcessorCount;
        return Math.Max(1, Math.Min(want, items));
    }

    /// <summary>
    /// Link <paramref name="count"/> jobs into the graph: plan every job against
    /// the current graph (in parallel, read-only), then apply the plans in job
    /// order. One routine for build, insert and update.
    /// </summary>
    private void Link(VectorTable t, Job[] jobs, int count, int maxThreads)
    {
        var intraSlots = new int[count];
        var intraLevels = new int[count];
        int intra = 0;
        for (int i = 0; i < count; i++)
            if (!jobs[i].Relink) { intraSlots[intra] = jobs[i].Slot; intraLevels[intra++] = jobs[i].Level; }

        var plans = new int[count][][];
        int degree = Degree(maxThreads, count);
        if (degree <= 1)
            for (int i = 0; i < count; i++) plans[i] = Plan(t, in jobs[i], intraSlots, intraLevels, intra);
        else
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = degree },
                i => plans[i] = Plan(t, in jobs[i], intraSlots, intraLevels, intra));

        // forward links
        for (int i = 0; i < count; i++)
        {
            if (!jobs[i].Relink) AddNode(jobs[i].Slot, jobs[i].Level);
            for (int l = 0; l < plans[i].Length; l++) SetList(jobs[i].Slot, l, plans[i][l]);
        }

        // backward links, grouped by (layer, target) in a fixed order
        var norms = t.Norms;
        int total = 0;
        for (int i = 0; i < count; i++)
            foreach (var list in plans[i]) total += list.Length;
        var adds = new Add[total];
        int na = 0;
        for (int i = 0; i < count; i++)
            for (int l = 0; l < plans[i].Length; l++)
                foreach (int nb in plans[i][l])
                    if (!float.IsNaN(norms[nb]))
                    {
                        adds[na] = new Add { Layer = l, Target = nb, Source = jobs[i].Slot, Seq = na };
                        na++;
                    }
        Array.Sort(adds, 0, na, AddOrder.Instance);
        var starts = new List<int>();
        for (int at = 0; at < na; at++)
            if (at == 0 || adds[at].Layer != adds[at - 1].Layer || adds[at].Target != adds[at - 1].Target)
                starts.Add(at);
        starts.Add(na);
        int groups = starts.Count - 1;
        var grown = new int[]?[groups];
        void One(int g)
        {
            int from = starts[g], to = starts[g + 1];
            var incoming = new int[to - from];
            for (int i = from; i < to; i++) incoming[i - from] = adds[i].Source;
            grown[g] = GrownList(t, adds[from].Layer, adds[from].Target, incoming);
        }
        int gdegree = count <= 1 ? 1 : Degree(maxThreads, groups);    // a lone insert stays serial (measured: parallel groups gave no gain)
        if (gdegree <= 1)
            for (int g = 0; g < groups; g++) One(g);
        else
            Parallel.For(0, groups, new ParallelOptions { MaxDegreeOfParallelism = gdegree }, One);
        for (int g = 0; g < groups; g++)
            if (grown[g] is { } list) SetList(adds[starts[g]].Target, adds[starts[g]].Layer, list);

        // the entry point is the first new node of the highest level
        for (int i = 0; i < count; i++)
        {
            if (jobs[i].Relink) continue;
            if (_entry == None || jobs[i].Level > _top)
            {
                _entry = jobs[i].Slot;
                _top = jobs[i].Level;
            }
        }
    }

    /// <summary>
    /// A graph over every live row of <paramref name="t"/> with a finite norm,
    /// built in slot order in batches of size/16 (1 at the start, at most 512).
    /// </summary>
    public static HnswIndex Build(VectorTable t, int m, int efConstruction, int seed, int maxThreads)
    {
        long started = Stopwatch.GetTimestamp();
        var h = new HnswIndex(m, efConstruction, RngStateFor(seed));
        int n = t.Slots;
        if (n > 0) h.EnsureSlot(n - 1);
        var norms = t.Norms;
        var order = new int[n];
        int live = 0;
        for (int s = 0; s < n; s++)
            if (float.IsFinite(norms[s])) order[live++] = s;       // a tombstone's norm is NaN, so this is "live and finite"
        var jobs = new Job[BuildBatch];
        int at = 0;
        while (at < live)
        {
            int size = Math.Clamp(at / BuildRatio, 1, BuildBatch);
            int end = Math.Min(at + size, live);
            int c = 0;
            for (int i = at; i < end; i++)
                jobs[c++] = new Job { Slot = order[i], Level = h.DrawLevel(), Relink = false };
            h.Link(t, jobs, c, maxThreads);
            at = end;
        }
        h.BuildSeconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
        return h;
    }

    /// <summary>
    /// A row was written (new, or its vector replaced): link it into the graph
    /// now. An update keeps the node's level and relinks it; other nodes' links to
    /// its old position stay (routing hints). Counts as churn.
    /// </summary>
    public void NotePut(VectorTable t, int slot, int maxThreads)
    {
        long started = Stopwatch.GetTimestamp();
        EnsureSlot(slot);
        if (!float.IsFinite(t.Norms[slot]))
        {
            if (InGraph(slot)) Churn++;
            return;
        }
        Job job;
        if (InGraph(slot))
        {
            Churn++;
            job = new Job { Slot = slot, Level = _levels[slot], Relink = true };
        }
        else
        {
            job = new Job { Slot = slot, Level = DrawLevel(), Relink = false };
        }
        Link(t, new[] { job }, 1, maxThreads);
        Inserts++;
        InsertSeconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
    }

    /// <summary>A row was deleted: the table's NaN norm hides it; the node and its links stay.</summary>
    public void NoteDrop() => Churn++;

    // -- compaction -----------------------------------------------------------------

    /// <summary>
    /// The graph after the table's compaction (<paramref name="remap"/>[old] is
    /// the new slot, or -1 for a dead one): tombstones leave, links are
    /// renumbered, and a node that lost links takes up the links of the
    /// tombstones it pointed to ("bridging"), without computing a distance.
    /// The entry point becomes the first node of the highest level.
    /// </summary>
    public HnswIndex Compacted(int[] remap, int newSlots)
    {
        int oldN = _slots;
        var fresh = new HnswIndex(_m, _efC, _rng.State)
        {
            Churn = Churn,
            BuildSeconds = BuildSeconds,
            Inserts = Inserts,
            InsertSeconds = InsertSeconds,
        };
        if (newSlots > 0) fresh.EnsureSlot(newSlots - 1);
        var lost = new List<int>();
        var outList = new int[MaxCap];
        for (int old = 0; old < remap.Length; old++)
        {
            int nw = remap[old];
            if (nw < 0) continue;
            int level = old < oldN ? _levels[old] : Absent;
            if (level == Absent) continue;
            fresh.AddNode(nw, level);
            for (int l = 0; l <= level; l++)
            {
                int cap = Cap(l), oc = 0;
                lost.Clear();
                foreach (int nb in List(old, l))
                {
                    int n2 = (uint)nb < (uint)remap.Length ? remap[nb] : -1;
                    if (n2 >= 0) outList[oc++] = n2; else lost.Add(nb);
                }
                foreach (int d in lost)
                {
                    if (oc >= cap) break;
                    if (d >= oldN || _levels[d] == Absent || _levels[d] < l) continue;
                    foreach (int w in List(d, l))
                    {
                        int nw2 = (uint)w < (uint)remap.Length ? remap[w] : -1;
                        if (nw2 < 0 || nw2 == nw) continue;
                        bool seen = false;
                        for (int i = 0; i < oc; i++) if (outList[i] == nw2) { seen = true; break; }
                        if (seen) continue;
                        outList[oc++] = nw2;
                        if (oc >= cap) break;
                    }
                }
                fresh.SetList(nw, l, new ReadOnlySpan<int>(outList, 0, oc));
            }
            if (fresh._entry == None || level > fresh._top)
            {
                fresh._entry = nw;
                fresh._top = level;
            }
        }
        return fresh;
    }

    // -- checks ---------------------------------------------------------------------

    /// <summary>DESIGN.md section 12.6 invariants 8-10 against the table; null when they hold.</summary>
    public string? Verify(VectorTable t)
    {
        int n = t.Slots;
        if (_slots != n) return $"graph covers {_slots} rows, table has {n}";
        if (_l0Count != (n + ChunkRows - 1) / ChunkRows) return $"{_l0Count} link chunks for {n} rows";
        int members = 0, maxLevel = 0;
        var norms = t.Norms;
        for (int s = 0; s < n; s++)
        {
            int level = _levels[s];
            if (level == Absent)
            {
                if (t.IsLive(s) && float.IsFinite(norms[s])) return $"live slot {s} is not in the graph";
                continue;
            }
            members++;
            maxLevel = Math.Max(maxLevel, level);
            for (int l = 0; l <= level; l++)
            {
                var list = List(s, l);
                if (list.Length > Cap(l)) return $"slot {s} layer {l} holds {list.Length} links";
                var seen = list.ToArray();
                Array.Sort(seen);
                for (int i = 1; i < seen.Length; i++)
                    if (seen[i] == seen[i - 1]) return $"slot {s} layer {l} links a node twice";
                foreach (int nb in list)
                {
                    if ((uint)nb >= (uint)n || nb == s) return $"slot {s} layer {l} links {nb}";
                    if (_levels[nb] == Absent || _levels[nb] < l)
                        return $"slot {s} layer {l} links {nb}, which is not on that layer";
                }
            }
        }
        if (members != _members) return $"member counter {_members} but {members} nodes";
        if (members == 0)
        {
            if (_entry != None) return "an empty graph has an entry point";
        }
        else if (_entry == None || _levels[_entry] != _top || maxLevel != _top)
        {
            return $"entry {_entry} (level {(_entry == None ? -1 : _levels[_entry])}) does not sit on top level {_top} (highest node {maxLevel})";
        }
        return null;
    }

    /// <summary>Nodes whose rows are tombstones.</summary>
    public int Dead(VectorTable t)
    {
        int dead = 0;
        for (int s = 0; s < _slots; s++)
            if (_levels[s] != Absent && float.IsNaN(t.Norms[s])) dead++;
        return dead;
    }

    // -- test hooks -------------------------------------------------------------------

    /// <summary>A comparable digest of the whole structure (levels, links, entry), to tell two graphs apart.</summary>
    internal ulong StructureHash()
    {
        ulong h = 1469598103934665603UL;
        void Mix(ulong v) { unchecked { h = (h ^ v) * 1099511628211UL; } }
        Mix((ulong)_slots); Mix((ulong)(uint)_entry); Mix((ulong)_top); Mix((ulong)_members);
        for (int s = 0; s < _slots; s++)
        {
            Mix(_levels[s]);
            if (_levels[s] == Absent) continue;
            for (int l = 0; l <= _levels[s]; l++)
            {
                var list = List(s, l);
                Mix((ulong)list.Length);
                foreach (int x in list) Mix((ulong)(uint)x);
            }
        }
        return h;
    }

    internal int LevelOf(int slot) => (uint)slot < (uint)_slots && _levels[slot] != Absent ? _levels[slot] : -1;
    internal int Entry => _entry;
    internal ReadOnlySpan<int> Links(int slot, int layer) => List(slot, layer);

    /// <summary>The first <paramref name="count"/> levels the generator draws for a store seed and M (see <c>DrawLevel</c>).</summary>
    internal static int[] LevelSequence(int seed, int m, int count)
    {
        var h = new HnswIndex(m, DefaultEfConstruction, RngStateFor(seed));
        var levels = new int[count];
        for (int i = 0; i < count; i++) levels[i] = h.DrawLevel();
        return levels;
    }
}
