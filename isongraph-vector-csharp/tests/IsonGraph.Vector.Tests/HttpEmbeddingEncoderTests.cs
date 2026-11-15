using System.Net;
using System.Text;
using System.Text.Json;
using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>A scripted HttpMessageHandler: no sockets are opened.</summary>
internal sealed class FakeHandler : HttpMessageHandler
{
    public sealed record Seen(HttpRequestMessage Request, string Body, JsonElement Json);

    private readonly Func<Seen, int, Task<HttpResponseMessage>> _respond;
    public List<Seen> Requests { get; } = new();

    public FakeHandler(Func<Seen, int, Task<HttpResponseMessage>> respond) => _respond = respond;
    public FakeHandler(Func<Seen, int, HttpResponseMessage> respond)
        : this((s, n) => Task.FromResult(respond(s, n))) { }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(ct);
        var json = body.Length > 0 ? JsonDocument.Parse(body).RootElement.Clone() : default;
        var seen = new Seen(request, body, json);
        Requests.Add(seen);
        return await _respond(seen, Requests.Count);
    }

    public static HttpResponseMessage Json(string json, HttpStatusCode code = HttpStatusCode.OK) =>
        new(code) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

    /// <summary>OpenAI-style reply: one vector [len(text), i, 0...] per input, shuffled when asked.</summary>
    public static HttpResponseMessage OpenAi(Seen seen, int dim = 4, bool reverse = false)
    {
        var inputs = seen.Json.GetProperty("input").EnumerateArray().Select(e => e.GetString()!).ToList();
        var items = inputs.Select((t, i) => new
        {
            index = i,
            embedding = Enumerable.Range(0, dim).Select(d => d == 0 ? (float)t.Length : d == 1 ? i : 0f).ToArray(),
        }).ToList();
        if (reverse) items.Reverse();
        return Json(JsonSerializer.Serialize(new { data = items }));
    }
}

public class HttpEmbeddingEncoderTests
{
    private const string Url = "https://embeddings.example.test/v1/embeddings?token=URLSECRET";

    private static HttpEmbeddingOptions Opts(Action<HttpEmbeddingOptions>? tweak = null)
    {
        var o = new HttpEmbeddingOptions
        {
            Url = Url,
            Model = "test-model",
            Dimension = 4,
            RetryDelay = TimeSpan.Zero,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static HttpResponseMessage Status(HttpStatusCode code, string body = "") =>
        new(code) { Content = new StringContent(body) };

    // -- request ----------------------------------------------------------------------

    [Fact]
    public void PostsModelAndInputsAsJsonWithHeaders()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o =>
            {
                o.ApiKey = "sk-secret-key";
                o.Headers["X-Tenant"] = "acme";
                o.ExtraBody["dimensions"] = 4;
            }), handler);

        var vectors = encoder.EncodeBatch(new[] { "hello", "hi" });

        var req = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, req.Request.Method);
        Assert.Equal(Url, req.Request.RequestUri!.ToString());
        Assert.Equal("Bearer sk-secret-key", req.Request.Headers.Authorization!.ToString());
        Assert.Equal("acme", req.Request.Headers.GetValues("X-Tenant").Single());
        Assert.Equal("application/json", req.Request.Content!.Headers.ContentType!.MediaType);
        Assert.Equal("test-model", req.Json.GetProperty("model").GetString());
        Assert.Equal(new[] { "hello", "hi" },
            req.Json.GetProperty("input").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(4, req.Json.GetProperty("dimensions").GetInt32());
        Assert.Equal(2, vectors.Count);
        Assert.Equal(5f, vectors[0][0]);
        Assert.Equal(2f, vectors[1][0]);
    }

    [Fact]
    public void NonAuthorizationKeyHeaderIsSentRaw()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o => { o.ApiKey = "azure-key"; o.ApiKeyHeader = "api-key"; }), handler);
        encoder.Encode("x");
        var req = handler.Requests.Single().Request;
        Assert.Equal("azure-key", req.Headers.GetValues("api-key").Single());
        Assert.Null(req.Headers.Authorization);
    }

    [Fact]
    public void NoKeyMeansNoAuthorizationHeader()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        encoder.Encode("x");
        Assert.Null(handler.Requests.Single().Request.Headers.Authorization);
    }

    [Fact]
    public void ModelNameAndVersionAreReported()
    {
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o => o.Version = "2024-01"), new FakeHandler((s, _) => FakeHandler.OpenAi(s)));
        Assert.Equal("test-model", encoder.ModelName);
        Assert.Equal("2024-01", encoder.Version);
        Assert.Equal(4, encoder.Dimension);
    }

    // -- batching ------------------------------------------------------------------------

    [Fact]
    public void SplitsIntoBatchesAndKeepsOrder()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(o => o.BatchSize = 2), handler);

        var texts = new[] { "a", "bb", "ccc", "dddd", "eeeee" };
        var vectors = encoder.EncodeBatch(texts);

        Assert.Equal(3, handler.Requests.Count);
        Assert.Equal(new[] { 2, 2, 1 },
            handler.Requests.Select(r => r.Json.GetProperty("input").GetArrayLength()));
        Assert.Equal(new float[] { 1, 2, 3, 4, 5 }, vectors.Select(v => v[0]));
    }

    [Fact]
    public void EmptyBatchMakesNoRequest()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        Assert.Empty(encoder.EncodeBatch(Array.Empty<string>()));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task AsyncApiWorks()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var one = await encoder.EncodeAsync("abc");
        var many = await encoder.EncodeBatchAsync(new[] { "a", "bb" });
        Assert.Equal(3f, one[0]);
        Assert.Equal(2, many.Count);
    }

    // -- response shapes ---------------------------------------------------------------------

    [Fact]
    public void OpenAiDataIsSortedByIndex()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s, reverse: true));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var vectors = encoder.EncodeBatch(new[] { "a", "bb", "ccc" });
        Assert.Equal(new float[] { 1, 2, 3 }, vectors.Select(v => v[0]));
        Assert.Equal(new float[] { 0, 1, 2 }, vectors.Select(v => v[1]));
    }

    [Fact]
    public void ReadsOllamaEmbeddings()
    {
        var handler = new FakeHandler((_, _) =>
            FakeHandler.Json("""{"model":"m","embeddings":[[1,2,3,4],[5,6,7,8]]}"""));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var vectors = encoder.EncodeBatch(new[] { "x", "y" });
        Assert.Equal(new float[] { 1, 2, 3, 4 }, vectors[0]);
        Assert.Equal(new float[] { 5, 6, 7, 8 }, vectors[1]);
    }

    [Theory]
    [InlineData("""{"data":[{"index":0,"embedding":[1,2,3,4]}]}""", 2)]       // too few
    [InlineData("""{"embeddings":[[1,2,3,4],[1,2,3,4],[1,2,3,4]]}""", 2)]    // too many
    [InlineData("""{"hello":"world"}""", 1)]                                  // unknown shape
    [InlineData("""not json""", 1)]
    [InlineData("""[1,2,3]""", 1)]
    [InlineData("""{"embeddings":[["a","b"]]}""", 1)]                         // non-numbers
    [InlineData("""{"embeddings":[[]]}""", 1)]                                // empty vector
    [InlineData("""{"data":[{"index":1,"embedding":[1,2,3,4]}]}""", 1)]       // bad index
    [InlineData("""{"data":[{"index":0}]}""", 1)]                             // no embedding
    public void UnusableResponsesThrow(string body, int inputs)
    {
        var handler = new FakeHandler((_, _) => FakeHandler.Json(body));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var texts = Enumerable.Range(0, inputs).Select(i => $"t{i}").ToArray();
        Assert.Throws<HttpEmbeddingException>(() => encoder.EncodeBatch(texts));
        Assert.Single(handler.Requests); // a bad body is not retried
    }

    [Fact]
    public void WrongDimensionThrows()
    {
        var handler = new FakeHandler((_, _) => FakeHandler.Json("""{"embeddings":[[1,2,3]]}"""));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        Assert.Throws<DimensionMismatchException>(() => encoder.Encode("x"));
    }

    [Fact]
    public void DimensionIsLearnedFromTheFirstResponse()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s, dim: 6));
        using var encoder = new HttpEmbeddingEncoder(Opts(o => o.Dimension = null), handler);
        encoder.Encode("x");
        Assert.Equal(6, encoder.Dimension);
        Assert.Single(handler.Requests); // no probe needed

        // later responses of a different length are rejected
        using var drifting = new HttpEmbeddingEncoder(Opts(o => o.Dimension = null),
            new FakeHandler((s, n) => FakeHandler.OpenAi(s, n == 1 ? 6 : 7)));
        drifting.Encode("a");
        Assert.Throws<DimensionMismatchException>(() => drifting.Encode("b"));
    }

    [Fact]
    public void UnknownDimensionIsProbedOnce()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s, dim: 5));
        using var encoder = new HttpEmbeddingEncoder(Opts(o => o.Dimension = null), handler);
        Assert.Equal(5, encoder.Dimension);
        Assert.Equal(5, encoder.Dimension);
        Assert.Single(handler.Requests);
        Assert.Equal("dimension probe",
            handler.Requests[0].Json.GetProperty("input")[0].GetString());
    }

    // -- retry ---------------------------------------------------------------------------------

    [Theory]
    [InlineData(HttpStatusCode.TooManyRequests)]
    [InlineData(HttpStatusCode.InternalServerError)]
    [InlineData(HttpStatusCode.BadGateway)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    public void RetriesTransientStatusesThenSucceeds(HttpStatusCode code)
    {
        var handler = new FakeHandler((s, n) => n < 3 ? Status(code, "busy") : FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        Assert.Equal(1f, encoder.Encode("a")[0]);
        Assert.Equal(3, handler.Requests.Count);
    }

    [Fact]
    public void RetriesAreBounded()
    {
        var handler = new FakeHandler((_, _) => Status(HttpStatusCode.ServiceUnavailable, "down"));
        using var encoder = new HttpEmbeddingEncoder(Opts(o => o.MaxRetries = 3), handler);
        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.Equal(503, ex.StatusCode);
        Assert.Equal(4, handler.Requests.Count);
        Assert.Contains("503", ex.Message);
        Assert.Contains("down", ex.Message);
    }

    [Fact]
    public void ZeroRetriesMeansOneAttempt()
    {
        var handler = new FakeHandler((_, _) => Status(HttpStatusCode.TooManyRequests));
        using var encoder = new HttpEmbeddingEncoder(Opts(o => o.MaxRetries = 0), handler);
        Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest)]
    [InlineData(HttpStatusCode.Unauthorized)]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.NotImplemented)]
    public void ClientErrorsAreNotRetried(HttpStatusCode code)
    {
        var handler = new FakeHandler((_, _) => Status(code, "nope"));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.Equal((int)code, ex.StatusCode);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public void RetriesNetworkErrors()
    {
        var handler = new FakeHandler((s, n) =>
            n == 1 ? throw new HttpRequestException("connection reset") : FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        Assert.Equal(1f, encoder.Encode("a")[0]);
        Assert.Equal(2, handler.Requests.Count);
    }

    [Fact]
    public void HonoursRetryAfterButCapsIt()
    {
        var handler = new FakeHandler((s, n) =>
        {
            if (n > 1) return FakeHandler.OpenAi(s);
            var r = Status(HttpStatusCode.TooManyRequests);
            r.Headers.RetryAfter = new System.Net.Http.Headers.RetryConditionHeaderValue(TimeSpan.FromHours(1));
            return r;
        });
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o => o.MaxRetryDelay = TimeSpan.FromMilliseconds(30)), handler);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        encoder.Encode("a");
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(20), "the one-hour Retry-After must be capped");
        Assert.Equal(2, handler.Requests.Count);
    }

    [Fact]
    public void TimesOut()
    {
        var hang = new HangingHandler();
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o => { o.Timeout = TimeSpan.FromMilliseconds(80); o.MaxRetries = 1; }), hang);
        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.Contains("timed out", ex.Message);
        Assert.Equal(2, hang.Calls);
    }

    [Fact]
    public async Task CallerCancellationIsNotRetriedOrWrapped()
    {
        var hang = new HangingHandler();
        using var encoder = new HttpEmbeddingEncoder(Opts(), hang);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => encoder.EncodeBatchAsync(new[] { "a" }, cts.Token));
        Assert.Equal(1, hang.Calls);
    }

    private sealed class HangingHandler : HttpMessageHandler
    {
        public int Calls;
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            Interlocked.Increment(ref Calls);
            await Task.Delay(Timeout.Infinite, ct);
            throw new InvalidOperationException("unreachable");
        }
    }

    // -- secrets ---------------------------------------------------------------------------------

    [Fact]
    public void ErrorsNeverContainTheKeyHeaderValuesOrQueryString()
    {
        var handler = new FakeHandler((_, _) =>
            Status(HttpStatusCode.Unauthorized, "bad key sk-secret-key / tenant-secret-value"));
        using var encoder = new HttpEmbeddingEncoder(Opts(o =>
        {
            o.ApiKey = "sk-secret-key";
            o.Headers["X-Tenant"] = "tenant-secret-value";
        }), handler);

        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));

        Assert.DoesNotContain("sk-secret-key", ex.Message);
        Assert.DoesNotContain("tenant-secret-value", ex.Message);
        Assert.DoesNotContain("URLSECRET", ex.Message);
        Assert.Contains("embeddings.example.test", ex.Message);
        Assert.Contains("401", ex.Message);
        Assert.DoesNotContain("sk-secret-key", ex.ToString());
    }

    [Fact]
    public void NetworkErrorMessagesAreRedactedToo()
    {
        var handler = new FakeHandler((Func<FakeHandler.Seen, int, HttpResponseMessage>)((_, _) => throw new HttpRequestException("failed for sk-secret-key")));
        using var encoder = new HttpEmbeddingEncoder(
            Opts(o => { o.ApiKey = "sk-secret-key"; o.MaxRetries = 0; }), handler);
        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.DoesNotContain("sk-secret-key", ex.Message);
        Assert.Null(ex.StatusCode);
    }

    [Fact]
    public void LongErrorBodiesAreTruncated()
    {
        var handler = new FakeHandler((_, _) => Status(HttpStatusCode.BadRequest, new string('x', 5000)));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var ex = Assert.Throws<HttpEmbeddingException>(() => encoder.Encode("a"));
        Assert.True(ex.Message.Length < 600);
    }

    [Fact]
    public void KeyOverPlainHttpToARemoteHostIsRefused()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        Assert.Throws<ArgumentException>(() => new HttpEmbeddingEncoder(
            Opts(o => { o.Url = "http://embeddings.example.test/embed"; o.ApiKey = "k"; }), handler));

        // loopback is fine, as is an explicit opt-in, as is no key at all
        new HttpEmbeddingEncoder(
            Opts(o => { o.Url = "http://localhost:11434/api/embed"; o.ApiKey = "k"; }), handler).Dispose();
        new HttpEmbeddingEncoder(
            Opts(o => { o.Url = "http://embeddings.example.test/embed"; o.ApiKey = "k"; o.AllowInsecureHttp = true; }),
            handler).Dispose();
        new HttpEmbeddingEncoder(
            Opts(o => o.Url = "http://embeddings.example.test/embed"), handler).Dispose();
    }

    // -- construction ---------------------------------------------------------------------------------

    [Theory]
    [InlineData("url-null")]
    [InlineData("url-relative")]
    [InlineData("url-ftp")]
    [InlineData("model-empty")]
    [InlineData("batch-zero")]
    [InlineData("retries-negative")]
    [InlineData("timeout-zero")]
    [InlineData("dimension-zero")]
    [InlineData("extra-model")]
    [InlineData("extra-input")]
    public void InvalidOptionsAreRejected(string which)
    {
        var options = Opts(o =>
        {
            switch (which)
            {
                case "url-null": o.Url = null; break;
                case "url-relative": o.Url = "/v1/embeddings"; break;
                case "url-ftp": o.Url = "ftp://example.test/x"; break;
                case "model-empty": o.Model = " "; break;
                case "batch-zero": o.BatchSize = 0; break;
                case "retries-negative": o.MaxRetries = -1; break;
                case "timeout-zero": o.Timeout = TimeSpan.Zero; break;
                case "dimension-zero": o.Dimension = 0; break;
                case "extra-model": o.ExtraBody["model"] = "other"; break;
                case "extra-input": o.ExtraBody["input"] = "other"; break;
            }
        });
        Assert.Throws<ArgumentException>(() =>
            new HttpEmbeddingEncoder(options, new FakeHandler((s, _) => FakeHandler.OpenAi(s))));
    }

    [Fact]
    public void AnInjectedClientIsNotDisposedOrModified()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var client = new HttpClient(handler);
        var encoder = new HttpEmbeddingEncoder(
            Opts(o => { o.ApiKey = "k"; o.Headers["X-A"] = "1"; }), client);
        encoder.Encode("a");
        encoder.Dispose();
        Assert.Empty(client.DefaultRequestHeaders);
        // still usable
        using var again = new HttpEmbeddingEncoder(Opts(), client);
        again.Encode("b");
        Assert.Equal(2, handler.Requests.Count);
    }

    // -- through SemanticGraph ----------------------------------------------------------------------------

    [Fact]
    public void WorksAsASemanticGraphEncoder()
    {
        var handler = new FakeHandler((s, _) => FakeHandler.OpenAi(s));
        using var encoder = new HttpEmbeddingEncoder(Opts(), handler);
        var graph = new SemanticGraph("g", encoder);

        graph.AddNode("person", "1", new() { ["name"] = "Alice" });
        graph.AddNode("person", "2", new() { ["name"] = "Bo" });

        var record = graph.GetEmbedding(new NodeRef("person", "1"))!;
        Assert.Equal("test-model", record.Model);
        Assert.Equal(5f, record.Vector[0]);

        var hits = graph.SimilaritySearch("Alice", topK: 2, threshold: -1.0f);
        Assert.Equal(2, hits.Count);
        Assert.Empty(graph.StaleEmbeddings());
        // the stored texts, then the query, were what went over the wire
        Assert.Contains(handler.Requests, r => r.Body.Contains("Alice"));
    }
}
