using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace IsonGraph.Vector;

/// <summary>A scored row: the unit every search ranks, ordered by (score desc, slot asc).</summary>
internal readonly struct Hit
{
    public readonly float Score;
    public readonly int Slot;
    public Hit(float score, int slot) { Score = score; Slot = slot; }

    /// <summary>Best first: higher score, then lower slot (the one total order every search uses).</summary>
    public static int BestFirst(Hit a, Hit b)
    {
        if (a.Score > b.Score) return -1;
        if (a.Score < b.Score) return 1;
        return a.Slot.CompareTo(b.Slot);
    }
}

/// <summary>
/// Bounded collector for the best k hits: a min-heap whose root is the worst
/// kept, so most rows are rejected by one compare against the root.
/// Order is (score desc, slot asc), a total order, so the answer does not
/// depend on the order rows are offered in (threads, tiling, cell order).
/// </summary>
internal sealed class TopKHeap
{
    private readonly float[] _scores;
    private readonly int[] _slots;
    private int _count;

    public TopKHeap(int k)
    {
        _scores = new float[k];
        _slots = new int[k];
    }

    public int Count => _count;
    public bool Full => _count == _scores.Length;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Worse(float s1, int r1, float s2, int r2) => s1 < s2 || (s1 == s2 && r1 > r2);

    public void Offer(float score, int slot)
    {
        if (_count == _scores.Length)
        {
            if (score < _scores[0] || !Worse(_scores[0], _slots[0], score, slot)) return;
            _scores[0] = score;
            _slots[0] = slot;
            SiftDown(0);
            return;
        }
        int i = _count++;
        _scores[i] = score;
        _slots[i] = slot;
        while (i > 0)
        {
            int parent = (i - 1) >> 1;
            if (!Worse(_scores[i], _slots[i], _scores[parent], _slots[parent])) break;
            Swap(i, parent);
            i = parent;
        }
    }

    private void SiftDown(int i)
    {
        for (;;)
        {
            int l = 2 * i + 1, r = l + 1, worst = i;
            if (l < _count && Worse(_scores[l], _slots[l], _scores[worst], _slots[worst])) worst = l;
            if (r < _count && Worse(_scores[r], _slots[r], _scores[worst], _slots[worst])) worst = r;
            if (worst == i) return;
            Swap(i, worst);
            i = worst;
        }
    }

    private void Swap(int a, int b)
    {
        (_scores[a], _scores[b]) = (_scores[b], _scores[a]);
        (_slots[a], _slots[b]) = (_slots[b], _slots[a]);
    }

    public void MergeFrom(TopKHeap other)
    {
        for (int i = 0; i < other._count; i++) Offer(other._scores[i], other._slots[i]);
    }

    /// <summary>The kept hits, best first.</summary>
    public Hit[] ToSortedArray()
    {
        var hits = new Hit[_count];
        for (int i = 0; i < _count; i++) hits[i] = new Hit(_scores[i], _slots[i]);
        Array.Sort(hits, Hit.BestFirst);
        return hits;
    }
}

/// <summary>Receives (slot, dot) for every row a scan reaches; turns it into a score and keeps or drops it.</summary>
internal interface IRowVisitor
{
    void Visit(int slot, float dot);
}

/// <summary>Cosine score of a dot product: 0 when the norms' product is 0, NaN for a tombstone (NaN norm).</summary>
internal static class Cosine
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Score(float dot, float queryNorm, float rowNorm)
    {
        float denom = queryNorm * rowNorm;
        return denom == 0f ? 0f : dot / denom;
    }
}

/// <summary>Visitor that keeps the best k rows with score >= threshold.</summary>
internal struct TopKVisitor : IRowVisitor
{
    public TopKHeap Heap;
    public float[] Norms;
    public float QueryNorm;
    public float Threshold;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Visit(int slot, float dot)
    {
        float score = Cosine.Score(dot, QueryNorm, Norms[slot]);
        if (score >= Threshold) Heap.Offer(score, slot);
    }
}

/// <summary>Visitor that keeps every row with score >= threshold.</summary>
internal sealed class HitList
{
    public Hit[] Items = new Hit[256];
    public int Count;

    public void Add(float score, int slot)
    {
        if (Count == Items.Length) Array.Resize(ref Items, Count * 2);
        Items[Count++] = new Hit(score, slot);
    }

    public Hit[] ToSortedArray()
    {
        var hits = new Hit[Count];
        Array.Copy(Items, hits, Count);
        Array.Sort(hits, Hit.BestFirst);
        return hits;
    }
}

internal struct CollectVisitor : IRowVisitor
{
    public HitList List;
    public float[] Norms;
    public float QueryNorm;
    public float Threshold;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Visit(int slot, float dot)
    {
        float score = Cosine.Score(dot, QueryNorm, Norms[slot]);
        if (score >= Threshold) List.Add(score, slot);
    }
}

/// <summary>Visitor that records every score, NaN ones included (the score map).</summary>
internal struct ScoreAllVisitor : IRowVisitor
{
    public float[] Scores;
    public float[] Norms;
    public float QueryNorm;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Visit(int slot, float dot) => Scores[slot] = Cosine.Score(dot, QueryNorm, Norms[slot]);
}

/// <summary>
/// The dot-product kernels and the loops that feed them.
///
/// One summation order everywhere: every (query, row) dot is accumulated
/// sequentially in ascending dimension, one multiply then one add per element,
/// never fused. The kernels differ only in how many rows they walk at once
/// (independent accumulators, so the adds overlap in the pipeline instead of
/// each waiting on the previous one), so a single query, a batch and any thread
/// count produce the same bits, and they are the bits the store has always
/// produced.
/// </summary>
internal static class RowScan
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot1(ref float q, ref float r, int dim)
    {
        float s = 0f;
        for (int i = 0; i < dim; i++) s += Unsafe.Add(ref q, i) * Unsafe.Add(ref r, i);
        return s;
    }

    public static void Dot4(ref float q, ref float r0, ref float r1, ref float r2, ref float r3, int dim,
        out float d0, out float d1, out float d2, out float d3)
    {
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        for (int i = 0; i < dim; i++)
        {
            float x = Unsafe.Add(ref q, i);
            s0 += x * Unsafe.Add(ref r0, i);
            s1 += x * Unsafe.Add(ref r1, i);
            s2 += x * Unsafe.Add(ref r2, i);
            s3 += x * Unsafe.Add(ref r3, i);
        }
        d0 = s0; d1 = s1; d2 = s2; d3 = s3;
    }

    /// <summary>Eight consecutive rows (stride <paramref name="dim"/>) at once.</summary>
    public static void Dot8(ref float q, ref float rows, int dim, Span<float> d)
    {
        ref float r0 = ref rows;
        ref float r1 = ref Unsafe.Add(ref rows, dim);
        ref float r2 = ref Unsafe.Add(ref rows, 2 * dim);
        ref float r3 = ref Unsafe.Add(ref rows, 3 * dim);
        ref float r4 = ref Unsafe.Add(ref rows, 4 * dim);
        ref float r5 = ref Unsafe.Add(ref rows, 5 * dim);
        ref float r6 = ref Unsafe.Add(ref rows, 6 * dim);
        ref float r7 = ref Unsafe.Add(ref rows, 7 * dim);
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
        for (int i = 0; i < dim; i++)
        {
            float x = Unsafe.Add(ref q, i);
            s0 += x * Unsafe.Add(ref r0, i);
            s1 += x * Unsafe.Add(ref r1, i);
            s2 += x * Unsafe.Add(ref r2, i);
            s3 += x * Unsafe.Add(ref r3, i);
            s4 += x * Unsafe.Add(ref r4, i);
            s5 += x * Unsafe.Add(ref r5, i);
            s6 += x * Unsafe.Add(ref r6, i);
            s7 += x * Unsafe.Add(ref r7, i);
        }
        d[0] = s0; d[1] = s1; d[2] = s2; d[3] = s3; d[4] = s4; d[5] = s5; d[6] = s6; d[7] = s7;
    }

    /// <summary>Eight rows anywhere in memory (a cell's rows are scattered across chunks) at once.</summary>
    public static void Dot8Gather(ref float q, ref float r0, ref float r1, ref float r2, ref float r3,
        ref float r4, ref float r5, ref float r6, ref float r7, int dim, Span<float> d)
    {
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
        for (int i = 0; i < dim; i++)
        {
            float x = Unsafe.Add(ref q, i);
            s0 += x * Unsafe.Add(ref r0, i);
            s1 += x * Unsafe.Add(ref r1, i);
            s2 += x * Unsafe.Add(ref r2, i);
            s3 += x * Unsafe.Add(ref r3, i);
            s4 += x * Unsafe.Add(ref r4, i);
            s5 += x * Unsafe.Add(ref r5, i);
            s6 += x * Unsafe.Add(ref r6, i);
            s7 += x * Unsafe.Add(ref r7, i);
        }
        d[0] = s0; d[1] = s1; d[2] = s2; d[3] = s3; d[4] = s4; d[5] = s5; d[6] = s6; d[7] = s7;
    }

    /// <summary>Sequential sum of squares (the norm before the square root).</summary>
    public static float SumSq(ReadOnlySpan<float> v)
    {
        float s = 0f;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        return s;
    }

    /// <summary>
    /// Visit every row of chunks [fromChunk, toChunk) whose type code matches
    /// (<paramref name="code"/> &lt; 0 = any).
    /// </summary>
    public static void Chunks<TV>(VectorTable t, float[] query, int code, int fromChunk, int toChunk, ref TV visitor)
        where TV : struct, IRowVisitor
    {
        int dim = t.Dim, n = t.Slots;
        var codes = t.Codes;
        ref float q = ref MemoryMarshal.GetArrayDataReference(query);
        Span<float> d = stackalloc float[8];
        for (int c = fromChunk; c < toChunk; c++)
        {
            var data = t.Chunk(c);
            int baseSlot = c << 10;
            int rows = Math.Min(VectorTable.ChunkRows, n - baseSlot);
            ref float chunk = ref MemoryMarshal.GetArrayDataReference(data);
            int i = 0;
            for (; i + 8 <= rows; i += 8)
            {
                if (code >= 0 && !AnyCode(codes, baseSlot + i, 8, code)) continue;
                Dot8(ref q, ref Unsafe.Add(ref chunk, i * dim), dim, d);
                for (int j = 0; j < 8; j++)
                {
                    int slot = baseSlot + i + j;
                    if (code >= 0 && codes[slot] != code) continue;
                    visitor.Visit(slot, d[j]);
                }
            }
            for (; i < rows; i++)
            {
                int slot = baseSlot + i;
                if (code >= 0 && codes[slot] != code) continue;
                visitor.Visit(slot, Dot1(ref q, ref Unsafe.Add(ref chunk, i * dim), dim));
            }
        }
    }

    private static bool AnyCode(int[] codes, int start, int count, int code)
    {
        for (int i = 0; i < count; i++) if (codes[start + i] == code) return true;
        return false;
    }

    /// <summary>Visit the rows named by <paramref name="slots"/>[from, to) (a cell, or the pending rows).</summary>
    public static void Slots<TV>(VectorTable t, float[] query, int code, int[] slots, int from, int to, ref TV visitor)
        where TV : struct, IRowVisitor
    {
        int dim = t.Dim;
        var codes = t.Codes;
        ref float q = ref MemoryMarshal.GetArrayDataReference(query);
        int i = from;
        Span<float> d8 = stackalloc float[8];
        for (; i + 8 <= to; i += 8)
        {
            int s0 = slots[i], s1 = slots[i + 1], s2 = slots[i + 2], s3 = slots[i + 3];
            int s4 = slots[i + 4], s5 = slots[i + 5], s6 = slots[i + 6], s7 = slots[i + 7];
            if (code >= 0 && codes[s0] != code && codes[s1] != code && codes[s2] != code && codes[s3] != code &&
                codes[s4] != code && codes[s5] != code && codes[s6] != code && codes[s7] != code)
                continue;
            Dot8Gather(ref q, ref t.RowRef(s0), ref t.RowRef(s1), ref t.RowRef(s2), ref t.RowRef(s3),
                       ref t.RowRef(s4), ref t.RowRef(s5), ref t.RowRef(s6), ref t.RowRef(s7), dim, d8);
            if (code < 0 || codes[s0] == code) visitor.Visit(s0, d8[0]);
            if (code < 0 || codes[s1] == code) visitor.Visit(s1, d8[1]);
            if (code < 0 || codes[s2] == code) visitor.Visit(s2, d8[2]);
            if (code < 0 || codes[s3] == code) visitor.Visit(s3, d8[3]);
            if (code < 0 || codes[s4] == code) visitor.Visit(s4, d8[4]);
            if (code < 0 || codes[s5] == code) visitor.Visit(s5, d8[5]);
            if (code < 0 || codes[s6] == code) visitor.Visit(s6, d8[6]);
            if (code < 0 || codes[s7] == code) visitor.Visit(s7, d8[7]);
        }
        for (; i + 4 <= to; i += 4)
        {
            int s0 = slots[i], s1 = slots[i + 1], s2 = slots[i + 2], s3 = slots[i + 3];
            if (code >= 0 && codes[s0] != code && codes[s1] != code && codes[s2] != code && codes[s3] != code)
                continue;
            Dot4(ref q, ref t.RowRef(s0), ref t.RowRef(s1), ref t.RowRef(s2), ref t.RowRef(s3), dim,
                 out float d0, out float d1, out float d2, out float d3);
            if (code < 0 || codes[s0] == code) visitor.Visit(s0, d0);
            if (code < 0 || codes[s1] == code) visitor.Visit(s1, d1);
            if (code < 0 || codes[s2] == code) visitor.Visit(s2, d2);
            if (code < 0 || codes[s3] == code) visitor.Visit(s3, d3);
        }
        for (; i < to; i++)
        {
            int s = slots[i];
            if (code >= 0 && codes[s] != code) continue;
            visitor.Visit(s, Dot1(ref q, ref t.RowRef(s), dim));
        }
    }
}
