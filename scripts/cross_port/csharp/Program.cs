// C# half of the graph-file cross-port check (see ../graph_files.py).
//
//     CrossPortGraph write    <dir>
//     CrossPortGraph read     <dir> <observation.json>
//     CrossPortGraph mismatch <dir>
//
// `write` saves the shared fixture graph as <dir>/graph.ison plus its
// embeddings sidecar. `read` loads it with GraphMismatchMode.Error and writes
// what it found to <observation.json>. `mismatch` loads a graph whose file was
// edited after the save and exits 0 only if the load throws
// GraphEmbeddingMismatchException.
//
// Node ids are strings in this port, so ids 1 and 2 are written as "1" and "2"
// (the other typed-id ports write integers); the orchestrator allows for that.

using System.Globalization;
using System.Text;
using System.Text.Json;
using IsonGraph;
using IsonGraph.Vector;

const int Dim = 384;
string Bs = ((char)92).ToString(); // a backslash, spelled so no source escaping can mangle it
string DocText = "Résumé of 日本語 engineers, \"quoted\" and back" + Bs + "slash";
string[] Queries = { "Bob analyses ML models", "Carol plans roadmaps for über-teams", DocText };

// Keep in step with the other graph_* helpers.
(string Type, string Id, Dictionary<string, object?> Props)[] Nodes =
{
    ("person", "1", new() { ["text"] = "Alice is a software engineer", ["age"] = 30, ["score"] = 0.5, ["active"] = true, ["nickname"] = null }),
    ("person", "2", new() { ["text"] = "Bob analyses ML models", ["age"] = 41, ["score"] = 2.75, ["active"] = false, ["nickname"] = null }),
    ("person", "carol", new() { ["text"] = "Carol plans roadmaps for über-teams", ["age"] = 35, ["score"] = -1.25, ["active"] = true, ["nickname"] = "Caz" }),
    ("doc", "résumé", new() { ["text"] = DocText }),
    ("topic", "ml", new() { ["text"] = "machine learning models", ["weight"] = 0.125 }),
};
(string Rel, NodeRef Source, NodeRef Target, Dictionary<string, object?> Props)[] Edges =
{
    ("KNOWS", new("person", "1"), new("person", "2"), new() { ["since"] = 2020, ["weight"] = 0.75 }),
    ("WROTE", new("person", "carol"), new("doc", "résumé"), new() { ["note"] = "first draft" }),
    ("ABOUT", new("doc", "résumé"), new("topic", "ml"), new()),
};

string GraphPath(string dir) => System.IO.Path.Combine(dir, "graph.ison");

void Write(string dir)
{
    var graph = new SemanticGraph("fixture", new MockEncoder(Dim));
    foreach (var (type, id, props) in Nodes)
        graph.AddNode(type, id, props, (string)props["text"]!);
    foreach (var (rel, source, target, props) in Edges)
        graph.AddEdge(rel, source, target, props);
    Directory.CreateDirectory(dir);
    graph.Save(GraphPath(dir));
}

void WriteValue(Utf8JsonWriter w, object? value)
{
    switch (value)
    {
        case null: w.WriteNullValue(); break;
        case bool b: w.WriteBooleanValue(b); break;
        case int i: w.WriteNumberValue(i); break;
        case long l: w.WriteNumberValue(l); break;
        case float f: w.WriteNumberValue((double)f); break;
        case double d: w.WriteNumberValue(d); break;
        case string s: w.WriteStringValue(s); break;
        default: w.WriteStringValue(Convert.ToString(value, CultureInfo.InvariantCulture)); break;
    }
}

void WriteProps(Utf8JsonWriter w, IDictionary<string, object?> props)
{
    w.WriteStartObject();
    foreach (var key in props.Keys.OrderBy(k => k, StringComparer.Ordinal))
    {
        w.WritePropertyName(key);
        WriteValue(w, props[key]);
    }
    w.WriteEndObject();
}

void WriteRef(Utf8JsonWriter w, NodeRef r)
{
    w.WriteStartArray();
    w.WriteStringValue(r.Type);
    w.WriteStringValue(r.Id);
    w.WriteEndArray();
}

void Read(string dir, string output)
{
    var encoder = new MockEncoder(Dim);
    var graph = SemanticGraph.Load(GraphPath(dir), encoder, onMismatch: GraphMismatchMode.Error);

    using var stream = new MemoryStream();
    using (var w = new Utf8JsonWriter(stream))
    {
        w.WriteStartObject();
        w.WriteString("port", "csharp");

        var missing = new List<NodeRef>();
        double worst = 0;
        w.WritePropertyName("nodes");
        w.WriteStartArray();
        foreach (var node in graph.Graph.Nodes())
        {
            w.WriteStartObject();
            w.WriteString("type", node.Type);
            w.WriteString("id", node.Id);
            w.WritePropertyName("props");
            WriteProps(w, node.Properties);
            w.WriteEndObject();

            var record = graph.GetEmbedding(new NodeRef(node.Type, node.Id));
            if (record is null) { missing.Add(new NodeRef(node.Type, node.Id)); continue; }
            var expected = encoder.Encode(Convert.ToString(node.Properties["text"], CultureInfo.InvariantCulture)!);
            if (expected.Length != record.Vector.Length) worst = 1e9;
            else for (int i = 0; i < expected.Length; i++)
                worst = Math.Max(worst, Math.Abs((double)record.Vector[i] - (double)expected[i]));
        }
        w.WriteEndArray();

        w.WritePropertyName("edges");
        w.WriteStartArray();
        foreach (var edge in graph.Graph.Edges())
        {
            w.WriteStartObject();
            w.WriteString("rel", edge.RelType);
            w.WritePropertyName("source"); WriteRef(w, edge.Source);
            w.WritePropertyName("target"); WriteRef(w, edge.Target);
            w.WritePropertyName("props");
            WriteProps(w, edge.Properties);
            w.WriteEndObject();
        }
        w.WriteEndArray();

        w.WriteNumber("embedding_count", graph.EmbeddingStore.Count);
        w.WritePropertyName("missing_embeddings");
        w.WriteStartArray();
        foreach (var r in missing) WriteRef(w, r);
        w.WriteEndArray();
        w.WriteNumber("max_vec_diff", worst);

        w.WritePropertyName("searches");
        w.WriteStartArray();
        foreach (var query in Queries)
        {
            var hits = graph.SimilaritySearch(query, 1);
            w.WriteStartObject();
            w.WriteString("query", query);
            w.WritePropertyName("top");
            if (hits.Count > 0) WriteRef(w, hits[0].NodeRef); else w.WriteNullValue();
            w.WritePropertyName("score");
            if (hits.Count > 0) w.WriteNumberValue((double)hits[0].Score); else w.WriteNullValue();
            w.WriteEndObject();
        }
        w.WriteEndArray();
        w.WriteEndObject();
    }
    File.WriteAllBytes(output, stream.ToArray());
}

int Mismatch(string dir)
{
    try
    {
        SemanticGraph.Load(GraphPath(dir), new MockEncoder(Dim), onMismatch: GraphMismatchMode.Error);
    }
    catch (GraphEmbeddingMismatchException)
    {
        return 0;
    }
    catch (Exception e)
    {
        Console.Error.WriteLine($"expected GraphEmbeddingMismatchException, got {e.GetType().Name}: {e.Message}");
        return 1;
    }
    Console.Error.WriteLine("expected GraphEmbeddingMismatchException, but the load succeeded");
    return 1;
}

try
{
    switch (args)
    {
        case ["write", var dir]: Write(dir); return 0;
        case ["read", var dir, var output]: Read(dir, output); return 0;
        case ["mismatch", var dir]: return Mismatch(dir);
        default:
            Console.Error.WriteLine("usage: CrossPortGraph write <dir> | read <dir> <out.json> | mismatch <dir>");
            return 2;
    }
}
catch (Exception e)
{
    Console.Error.WriteLine($"error: {e.GetType().Name}: {e.Message}");
    return 1;
}
