/*
 * Content digest of a graph and atomic file writes - the pieces that keep a
 * saved graph and its embeddings sidecar a matched pair.
 *
 * The digest reproduces the Python port's graph_digest byte for byte
 * (basis "nodes-edges-json-v1"), so a graph saved by one port verifies when
 * loaded by another.
 *
 * Author: Mahesh Vaikri
 */

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using IsonGraph;

namespace IsonGraph.Vector;

/// <summary>
/// Content digest of a graph's nodes and edges: <c>"sha256:&lt;hex&gt;"</c>.
///
/// Hashes a canonical JSON rendering (<see cref="Basis"/>): the directed flag,
/// every node as <c>[[type, id-kind, id], properties]</c> and every edge as
/// <c>[rel_type, source, target, properties]</c>, each list sorted and every
/// object key-sorted, compact, with non-ASCII escaped as <c>\uXXXX</c> and
/// numbers formatted the way Python's <c>json.dumps</c> prints them. The same
/// graph therefore hashes the same regardless of insertion order, and the same
/// in every port. The graph name is excluded (it comes from the file name on
/// load), and so are embeddings.
///
/// An id-kind of <c>i</c> marks an integer id and <c>s</c> a string one. C#
/// node ids are always strings, so an in-memory graph cannot say which a given
/// id "meant"; see <see cref="Compute(IReadOnlyGraph, Func{NodeRef, bool})"/>. <see cref="OfText"/> reads the kind
/// from the file itself, where an integer id is written unquoted.
/// </summary>
public static class GraphDigest
{
    /// <summary>Name of the serialization that is hashed; written to the sidecar.</summary>
    public const string Basis = IsonGraphVectorVersion.GraphDigestBasis;

    /// <summary>
    /// Digest of an in-memory graph.
    /// </summary>
    /// <param name="graph">The graph to digest.</param>
    /// <param name="isIntegerId">
    /// Decides which node ids count as integers (kind <c>i</c>). By default an
    /// id spelled as a plain decimal integer (<c>-?[0-9]+</c>) does, the way an
    /// unquoted id reads back from an ISON file. Pass a predicate when you know
    /// better (for example, from the source data's JSON types).
    /// </param>
    public static string Compute(ISONGraph graph, Func<NodeRef, bool>? isIntegerId = null) =>
        Compute(graph.Nodes(), graph.Edges(), graph.Directed, isIntegerId);

    /// <summary>
    /// Digest of a <see cref="SemanticGraph.Graph"/> view; same result as
    /// the <see cref="ISONGraph"/> overload for the same graph.
    /// </summary>
    public static string Compute(IReadOnlyGraph graph, Func<NodeRef, bool>? isIntegerId = null) =>
        Compute(graph.Nodes(), graph.Edges(), graph.Directed, isIntegerId);

    private static string Compute(
        IEnumerable<Node> graphNodes, IEnumerable<Edge> graphEdges, bool directed,
        Func<NodeRef, bool>? isIntegerId)
    {
        isIntegerId ??= r => IsIntegerText(r.Id);

        var nodes = new List<string>();
        foreach (var node in graphNodes)
            nodes.Add(NodeJson(Key(node.Ref, isIntegerId), node.Properties));

        var edges = new List<string>();
        foreach (var edge in graphEdges)
            edges.Add(EdgeJson(edge.RelType, Key(edge.Source, isIntegerId),
                               Key(edge.Target, isIntegerId), edge.Properties));

        return Finish(directed, nodes, edges);
    }

    /// <summary>
    /// Digest of the graph a piece of ISON (or ISONL) text reads back as.
    ///
    /// This is what <see cref="SemanticGraph.Save"/> and
    /// <see cref="SemanticGraph.Load"/> compare, and it mirrors how the Python
    /// port re-reads the file: the id kind comes from the value type in the
    /// file, every column of a block is kept (a null padding a row that lacks
    /// the column included), and an undirected graph's missing mirror edges
    /// are supplied.
    /// </summary>
    public static string OfText(string text, bool isonl = false)
    {
        var doc = isonl ? IsonParser.Ison.LoadsIsonl(text) : IsonParser.Ison.Loads(text);

        bool directed = true;
        foreach (var block in doc.Blocks)
        {
            if (block.Kind == "meta" && block.Name == "graph")
                foreach (var row in block.Rows)
                    if (row.TryGetValue("directed", out var d) && d is bool b) directed = b;
        }

        // (type, id text) -> id kind, as the node rows declare it.
        var kinds = new Dictionary<(string, string), char>();
        var nodes = new List<string>();
        foreach (var block in doc.Blocks)
        {
            if (block.Kind != "nodes") continue;
            foreach (var row in block.Rows)
            {
                row.TryGetValue("id", out var rawId);
                var (kind, idText) = IdOf(rawId);
                kinds[(block.Name, idText)] = kind;

                var props = new Dictionary<string, object?>();
                foreach (var kv in row)
                    if (kv.Key != "id") props[kv.Key] = kv.Value;
                nodes.Add(NodeJson((block.Name, kind, idText), props));
            }
        }

        (string, char, string) Resolve(object? raw)
        {
            if (raw is IsonParser.Reference reference)
            {
                var type = reference.Type ?? string.Empty;
                if (kinds.TryGetValue((type, reference.Id), out var known))
                    return (type, known, reference.Id);
                return (type, IsIntegerText(reference.Id) ? 'i' : 's', reference.Id);
            }
            var (k, t) = IdOf(raw);
            return (string.Empty, k, t);
        }

        // Edges keyed by (rel, source, target) so the first row wins; an
        // undirected graph also gets the mirror a row did not list.
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var edges = new List<string>();
        void AddEdge(string rel, (string, char, string) s, (string, char, string) t,
                     IDictionary<string, object?> props)
        {
            var json = EdgeJson(rel, s, t, props);
            var key = rel + "\u0000" + RefJson(s) + "\u0000" + RefJson(t);
            if (seen.Add(key)) edges.Add(json);
        }

        foreach (var block in doc.Blocks)
        {
            if (block.Kind != "edges") continue;
            foreach (var row in block.Rows)
            {
                row.TryGetValue("source", out var rawSource);
                row.TryGetValue("target", out var rawTarget);
                var source = Resolve(rawSource);
                var target = Resolve(rawTarget);

                var props = new Dictionary<string, object?>();
                foreach (var kv in row)
                    if (kv.Key != "source" && kv.Key != "target") props[kv.Key] = kv.Value;

                AddEdge(block.Name, source, target, props);
                if (!directed) AddEdge(block.Name, target, source, props);
            }
        }

        return Finish(directed, nodes, edges);
    }

    // -- assembling ----------------------------------------------------------

    private static string Finish(bool directed, List<string> nodes, List<string> edges)
    {
        nodes.Sort(StringComparer.Ordinal);   // ASCII only, so ordinal == code point order
        edges.Sort(StringComparer.Ordinal);
        var body = new StringBuilder();
        body.Append("{\"directed\":").Append(directed ? "true" : "false");
        body.Append(",\"edges\":[").AppendJoin(',', edges).Append(']');
        body.Append(",\"nodes\":[").AppendJoin(',', nodes).Append("]}");
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(body.ToString()));
        return "sha256:" + Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static (string, char, string) Key(NodeRef r, Func<NodeRef, bool> isIntegerId) =>
        (r.Type, isIntegerId(r) ? 'i' : 's', r.Id);

    /// <summary>True for text Python reads back as an integer id: <c>-?[0-9]+</c>.</summary>
    internal static bool IsIntegerText(string text)
    {
        int i = text.Length > 0 && text[0] == '-' ? 1 : 0;
        if (i >= text.Length) return false;
        for (; i < text.Length; i++)
            if (text[i] < '0' || text[i] > '9') return false;
        return true;
    }

    /// <summary>Kind and text of a node id value as parsed from a file.</summary>
    private static (char Kind, string Text) IdOf(object? id) => id switch
    {
        long l => ('i', l.ToString(CultureInfo.InvariantCulture)),
        int i => ('i', i.ToString(CultureInfo.InvariantCulture)),
        string s => ('s', s),
        double d => ('s', PyFloat(d)),
        bool b => ('s', b ? "True" : "False"),
        null => ('s', "None"),
        _ => ('s', id.ToString() ?? string.Empty),
    };

    private static string RefJson((string Type, char Kind, string Id) r)
    {
        var sb = new StringBuilder();
        sb.Append('[');
        WriteString(sb, r.Type);
        sb.Append(',');
        WriteString(sb, r.Kind == 'i' ? "i" : "s");
        sb.Append(',');
        WriteString(sb, r.Id);
        sb.Append(']');
        return sb.ToString();
    }

    private static string NodeJson((string, char, string) r, IDictionary<string, object?> props)
    {
        var sb = new StringBuilder();
        sb.Append('[').Append(RefJson(r)).Append(',');
        WriteObject(sb, props);
        sb.Append(']');
        return sb.ToString();
    }

    private static string EdgeJson(string rel, (string, char, string) s, (string, char, string) t,
                                   IDictionary<string, object?> props)
    {
        var sb = new StringBuilder();
        sb.Append('[');
        WriteString(sb, rel);
        sb.Append(',').Append(RefJson(s)).Append(',').Append(RefJson(t)).Append(',');
        WriteObject(sb, props);
        sb.Append(']');
        return sb.ToString();
    }

    // -- JSON the way json.dumps(sort_keys, compact, ensure_ascii) writes it --

    private static void WriteObject(StringBuilder sb, IDictionary<string, object?> props)
    {
        var keys = props.Keys.ToList();
        keys.Sort(CompareCodePoints);
        sb.Append('{');
        for (int i = 0; i < keys.Count; i++)
        {
            if (i > 0) sb.Append(',');
            WriteString(sb, keys[i]);
            sb.Append(':');
            WriteValue(sb, props[keys[i]]);
        }
        sb.Append('}');
    }

    private static void WriteValue(StringBuilder sb, object? value)
    {
        switch (value)
        {
            case null: sb.Append("null"); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case string s: WriteString(sb, s); break;
            case long or int or short or sbyte or byte or ushort or uint or ulong:
                sb.Append(Convert.ToString(value, CultureInfo.InvariantCulture)); break;
            case double d: sb.Append(PyFloat(d)); break;
            case float f: sb.Append(PyFloat(f)); break;
            case decimal m: sb.Append(PyFloat((double)m)); break;
            case System.Numerics.BigInteger big: sb.Append(big.ToString(CultureInfo.InvariantCulture)); break;
            // Anything else (a Reference, say) is stringified, as json.dumps(default=str) does.
            default: WriteString(sb, value.ToString() ?? string.Empty); break;
        }
    }

    /// <summary>A JSON string with every non-printable-ASCII character escaped.</summary>
    internal static void WriteString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c >= ' ' && c <= '~') sb.Append(c);
                    else sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Order by Unicode code point, as Python orders strings.</summary>
    private static int CompareCodePoints(string a, string b)
    {
        // UTF-16 ordinal order differs from code point order only for
        // surrogates (non-BMP) against U+E000..U+FFFF; fold those.
        static int Fix(char c) => c >= 0xE000 ? c - 0x800 : c >= 0xD800 ? c + 0x2000 : c;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            if (a[i] == b[i]) continue;
            return Fix(a[i]).CompareTo(Fix(b[i]));
        }
        return a.Length.CompareTo(b.Length);
    }

    /// <summary>
    /// Format a double the way Python's <c>repr(float)</c> (and so
    /// <c>json.dumps</c>) does: shortest round-trip digits, positional between
    /// 1e-4 and 1e16, otherwise <c>d.ddde+XX</c>, and an integral value keeps
    /// its <c>.0</c>.
    /// </summary>
    internal static string PyFloat(double value)
    {
        if (double.IsNaN(value)) return "NaN";
        if (double.IsPositiveInfinity(value)) return "Infinity";
        if (double.IsNegativeInfinity(value)) return "-Infinity";
        if (value == 0) return double.IsNegative(value) ? "-0.0" : "0.0";

        // "R" is the shortest string that round-trips on .NET Core 3.0+.
        var r = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
        int exp10 = 0;
        int e = r.IndexOfAny(new[] { 'E', 'e' });
        if (e >= 0)
        {
            exp10 = int.Parse(r[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            r = r[..e];
        }
        int dot = r.IndexOf('.');
        string intPart = dot < 0 ? r : r[..dot];
        string fracPart = dot < 0 ? "" : r[(dot + 1)..];
        string digits = intPart + fracPart;
        int decpt = intPart.Length + exp10;   // value = 0.DIGITS x 10^decpt

        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0') lead++;
        digits = digits[lead..];
        decpt -= lead;
        digits = digits.TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        var sb = new StringBuilder();
        if (value < 0) sb.Append('-');
        if (decpt <= -4 || decpt > 16)
        {
            sb.Append(digits[0]);
            if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
            int x = decpt - 1;
            sb.Append('e').Append(x < 0 ? '-' : '+');
            sb.Append(Math.Abs(x).ToString("00", CultureInfo.InvariantCulture));
        }
        else if (decpt <= 0)
        {
            sb.Append("0.").Append('0', -decpt).Append(digits);
        }
        else if (decpt >= digits.Length)
        {
            sb.Append(digits).Append('0', decpt - digits.Length).Append(".0");
        }
        else
        {
            sb.Append(digits, 0, decpt).Append('.').Append(digits, decpt, digits.Length - decpt);
        }
        return sb.ToString();
    }
}

/// <summary>File helpers shared by the save paths.</summary>
internal static class AtomicFile
{
    /// <summary>
    /// Write <paramref name="text"/> to <paramref name="path"/> so a reader
    /// sees the old file or the new one, never a truncated one.
    ///
    /// The text goes to a temp file in the same directory (same filesystem, so
    /// the rename is not a copy), is flushed to disk, and is then moved over
    /// the target. UTF-8 without a BOM, and no line-ending translation, so a
    /// file's bytes do not depend on the OS that wrote it.
    /// </summary>
    public static void WriteText(string path, string text)
    {
        var full = System.IO.Path.GetFullPath(path);
        var dir = System.IO.Path.GetDirectoryName(full) ?? ".";
        var tmp = System.IO.Path.Combine(dir,
            $".{System.IO.Path.GetFileName(full)}.{Environment.ProcessId}.{Guid.NewGuid().ToString("N")[..8]}.tmp");
        try
        {
            using (var stream = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                var bytes = new UTF8Encoding(false).GetBytes(text);
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush(flushToDisk: true);
            }
            File.Move(tmp, full, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            throw;
        }
    }
}
