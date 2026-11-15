// An IEmbeddingEncoder that calls an HTTP embeddings endpoint.
//
// Depends on System.Net.Http and System.Text.Json only: no SDK, no new package.
// It speaks the two request/response shapes nearly every embeddings service
// uses:
//
//   POST {"model": "...", "input": ["text", ...]}
//   OpenAI-style  <- {"data": [{"index": 0, "embedding": [..]}, ...]}
//   Ollama-style  <- {"embeddings": [[..], ...]}          (POST /api/embed)
//
// That covers OpenAI, Azure OpenAI, Ollama, vLLM, LiteLLM, LM Studio,
// text-embeddings-inference, Mistral, Voyage and any other compatible server.

using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace IsonGraph.Vector;

/// <summary>An embeddings request failed (HTTP error status, timeout, network failure or unusable response).</summary>
public sealed class HttpEmbeddingException : EmbeddingException
{
    /// <summary>HTTP status of the final failed attempt, or null when there was no response.</summary>
    public int? StatusCode { get; }

    public HttpEmbeddingException(string message, int? statusCode = null) : base(message) =>
        StatusCode = statusCode;
}

/// <summary>Settings for <see cref="HttpEmbeddingEncoder"/>.</summary>
public sealed class HttpEmbeddingOptions
{
    /// <summary>Full endpoint URL, e.g. <c>https://api.openai.com/v1/embeddings</c> or <c>http://localhost:11434/api/embed</c>. Required.</summary>
    public string? Url { get; set; }

    /// <summary>Model name sent as <c>model</c> and reported as <see cref="IEmbeddingEncoder.ModelName"/>. Required.</summary>
    public string? Model { get; set; }

    /// <summary>
    /// Vector length. When null the encoder learns it from the first response;
    /// reading <see cref="IEmbeddingEncoder.Dimension"/> before any text has
    /// been encoded then makes one probe request (sending the fixed text
    /// "dimension probe"). Set it to avoid that, and to have every response
    /// checked against it.
    /// </summary>
    public int? Dimension { get; set; }

    /// <summary>API key. Sent in <see cref="ApiKeyHeader"/>; never included in an exception message.</summary>
    public string? ApiKey { get; set; }

    /// <summary>
    /// Header carrying the key. The default, <c>Authorization</c>, sends
    /// <c>Bearer &lt;key&gt;</c>; any other name (for example Azure's
    /// <c>api-key</c>) sends the key as is.
    /// </summary>
    public string ApiKeyHeader { get; set; } = "Authorization";

    /// <summary>Texts per request (default 64). Larger batches are split into sequential requests.</summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>Time allowed for each attempt, including reading the body (default 60 s).</summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(60);

    /// <summary>Retries after the first attempt on HTTP 429, 5xx, timeouts and network errors (default 2; 0 disables).</summary>
    public int MaxRetries { get; set; } = 2;

    /// <summary>First backoff (default 500 ms), doubled for each further retry. A <c>Retry-After</c> header takes precedence.</summary>
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Upper bound on any single wait between retries (default 30 s).</summary>
    public TimeSpan MaxRetryDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>Extra request headers (for example an organization or tenant header). Their values are also kept out of exception messages.</summary>
    public IDictionary<string, string> Headers { get; set; } = new Dictionary<string, string>();

    /// <summary>Extra top-level JSON fields for the request body, e.g. <c>dimensions</c>, <c>truncate</c> or <c>encoding_format</c>. They cannot replace <c>model</c> or <c>input</c>.</summary>
    public IDictionary<string, object?> ExtraBody { get; set; } = new Dictionary<string, object?>();

    /// <summary>Recorded in the embeddings sidecar as <see cref="IEmbeddingEncoder.Version"/> (a revision or weights hash). Null when unknown.</summary>
    public string? Version { get; set; }

    /// <summary>
    /// Allow an API key to be sent over plain <c>http://</c> to a host that is
    /// not loopback. Off by default: such a key travels in the clear.
    /// </summary>
    public bool AllowInsecureHttp { get; set; }
}

/// <summary>
/// Embeds text by POSTing it to an HTTP embeddings endpoint.
///
/// <b>Privacy: every text you embed - node text on <c>AddNode</c>,
/// <c>EmbedAllNodes</c>, <c>ReembedStale</c>, and every search query - is sent
/// to the configured URL.</b> Point it only at a service you are willing to
/// give that text to.
///
/// The request is <c>{"model": ..., "input": [texts]}</c>. The response is read
/// as OpenAI-style (<c>data[i].embedding</c>, ordered by <c>index</c>) or
/// Ollama-style (<c>embeddings</c>); the count and (when known) the dimension
/// are checked, so a short or malformed reply is an error rather than silently
/// misaligned vectors.
///
/// <see cref="IEmbeddingEncoder"/> is synchronous, so <see cref="Encode"/> and
/// <see cref="EncodeBatch"/> block on the request; prefer
/// <see cref="EncodeBatchAsync"/> where you can await. The blocking calls use
/// no synchronization context, so they do not deadlock under one, but they do
/// occupy a thread for the duration of the call.
///
/// Failures (non-success status after the retries, timeout, network error,
/// unparseable body) throw <see cref="HttpEmbeddingException"/>. Messages carry
/// the status, the URL without its query string, and a short body excerpt, with
/// the API key and extra header values removed.
///
/// Thread-safe; share one instance.
/// </summary>
public sealed class HttpEmbeddingEncoder : IEmbeddingEncoder, IDisposable
{
    private const string ProbeText = "dimension probe";
    private const int BodyExcerptChars = 200;

    private readonly HttpClient _client;
    private readonly bool _ownsClient;
    private readonly Uri _uri;
    private readonly string _model;
    private readonly string? _version;
    private readonly string? _apiKey;
    private readonly string _apiKeyHeader;
    private readonly int _batchSize;
    private readonly TimeSpan _timeout;
    private readonly int _maxRetries;
    private readonly TimeSpan _retryDelay;
    private readonly TimeSpan _maxRetryDelay;
    private readonly List<KeyValuePair<string, string>> _headers;
    private readonly List<KeyValuePair<string, object?>> _extraBody;
    private readonly string[] _secrets;
    private volatile int _dimension;

    /// <summary>Create an encoder with its own <see cref="HttpClient"/>.</summary>
    public HttpEmbeddingEncoder(HttpEmbeddingOptions options)
        : this(options, new HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan }, ownsClient: true) { }

    /// <summary>
    /// Create an encoder over a handler (a fake one, in tests). The encoder
    /// owns the resulting client; the handler is disposed with it.
    /// </summary>
    public HttpEmbeddingEncoder(HttpEmbeddingOptions options, HttpMessageHandler handler)
        : this(options, new HttpClient(handler ?? throw new ArgumentNullException(nameof(handler)))
               { Timeout = System.Threading.Timeout.InfiniteTimeSpan }, ownsClient: true) { }

    /// <summary>
    /// Create an encoder over an existing client, which is not disposed with
    /// the encoder. The client's own <c>Timeout</c> still applies on top of
    /// <see cref="HttpEmbeddingOptions.Timeout"/>; the encoder adds its headers
    /// to each request, never to the client's defaults.
    /// </summary>
    public HttpEmbeddingEncoder(HttpEmbeddingOptions options, HttpClient client)
        : this(options, client ?? throw new ArgumentNullException(nameof(client)), ownsClient: false) { }

    private HttpEmbeddingEncoder(HttpEmbeddingOptions options, HttpClient client, bool ownsClient)
    {
        try
        {
            if (options is null) throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrWhiteSpace(options.Url))
                throw new ArgumentException("Url is required.", nameof(options));
            if (string.IsNullOrWhiteSpace(options.Model))
                throw new ArgumentException("Model is required.", nameof(options));
            if (!Uri.TryCreate(options.Url, UriKind.Absolute, out var uri) ||
                (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
                throw new ArgumentException("Url must be an absolute http or https URL.", nameof(options));
            if (options.BatchSize < 1)
                throw new ArgumentException("BatchSize must be at least 1.", nameof(options));
            if (options.MaxRetries < 0)
                throw new ArgumentException("MaxRetries cannot be negative.", nameof(options));
            if (options.Timeout <= TimeSpan.Zero)
                throw new ArgumentException("Timeout must be positive.", nameof(options));
            if (options.RetryDelay < TimeSpan.Zero || options.MaxRetryDelay < TimeSpan.Zero)
                throw new ArgumentException("RetryDelay and MaxRetryDelay cannot be negative.", nameof(options));
            if (options.Dimension is <= 0)
                throw new ArgumentException("Dimension must be positive.", nameof(options));
            if (string.IsNullOrWhiteSpace(options.ApiKeyHeader))
                throw new ArgumentException("ApiKeyHeader cannot be empty.", nameof(options));

            var hasKey = !string.IsNullOrEmpty(options.ApiKey);
            if (hasKey && uri.Scheme == Uri.UriSchemeHttp && !uri.IsLoopback && !options.AllowInsecureHttp)
                throw new ArgumentException(
                    "Refusing to send an API key over plain http to a non-loopback host; " +
                    "use https, or set AllowInsecureHttp.", nameof(options));

            _uri = uri;
            _model = options.Model;
            _version = options.Version;
            _apiKey = hasKey ? options.ApiKey : null;
            _apiKeyHeader = options.ApiKeyHeader;
            _batchSize = options.BatchSize;
            _timeout = options.Timeout;
            _maxRetries = options.MaxRetries;
            _retryDelay = options.RetryDelay;
            _maxRetryDelay = options.MaxRetryDelay;
            _headers = (options.Headers ?? new Dictionary<string, string>()).ToList();
            _extraBody = (options.ExtraBody ?? new Dictionary<string, object?>()).ToList();
            foreach (var extra in _extraBody)
                if (extra.Key is "model" or "input")
                    throw new ArgumentException($"ExtraBody cannot set '{extra.Key}'.", nameof(options));
            _dimension = options.Dimension ?? 0;

            _secrets = _headers.Select(h => h.Value)
                .Concat(hasKey ? new[] { options.ApiKey! } : Array.Empty<string>())
                .Where(v => !string.IsNullOrEmpty(v))
                .Distinct()
                .OrderByDescending(v => v.Length)
                .ToArray();

            _client = client;
            _ownsClient = ownsClient;
        }
        catch
        {
            if (ownsClient) client.Dispose();
            throw;
        }
    }

    /// <summary>The model name sent with every request.</summary>
    public string? ModelName => _model;

    /// <summary>The <see cref="HttpEmbeddingOptions.Version"/>, or null.</summary>
    public string? Version => _version;

    /// <summary>
    /// Vector length. If it was not configured and nothing has been encoded
    /// yet, this makes one probe request (see <see cref="HttpEmbeddingOptions.Dimension"/>).
    /// </summary>
    public int Dimension
    {
        get
        {
            if (_dimension == 0) EncodeBatchAsync(new[] { ProbeText }).GetAwaiter().GetResult();
            return _dimension;
        }
    }

    /// <summary>Embed one text (blocking).</summary>
    public float[] Encode(string text) => EncodeBatch(new[] { text })[0];

    /// <summary>Embed many texts (blocking); see <see cref="EncodeBatchAsync"/>.</summary>
    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts) =>
        EncodeBatchAsync(texts).GetAwaiter().GetResult();

    /// <summary>Embed one text.</summary>
    public async Task<float[]> EncodeAsync(string text, CancellationToken cancellationToken = default) =>
        (await EncodeBatchAsync(new[] { text }, cancellationToken).ConfigureAwait(false))[0];

    /// <summary>
    /// Embed many texts, in <see cref="HttpEmbeddingOptions.BatchSize"/>-sized
    /// requests sent one after another. Results are in input order. An empty
    /// list makes no request.
    /// </summary>
    public async Task<IReadOnlyList<float[]>> EncodeBatchAsync(
        IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        if (texts is null) throw new ArgumentNullException(nameof(texts));
        var results = new List<float[]>(texts.Count);
        for (int start = 0; start < texts.Count; start += _batchSize)
        {
            var chunk = new string[Math.Min(_batchSize, texts.Count - start)];
            for (int i = 0; i < chunk.Length; i++) chunk[i] = texts[start + i] ?? "";
            results.AddRange(await EncodeChunkAsync(chunk, cancellationToken).ConfigureAwait(false));
        }
        return results;
    }

    public void Dispose()
    {
        if (_ownsClient) _client.Dispose();
    }

    // -- one request, with bounded retry ---------------------------------------------

    private async Task<List<float[]>> EncodeChunkAsync(string[] chunk, CancellationToken ct)
    {
        var body = BuildBody(chunk);
        HttpEmbeddingException? last = null;

        for (int attempt = 0; attempt <= _maxRetries; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            TimeSpan? serverWait = null;
            try
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(_timeout);

                using var request = BuildRequest(body);
                using var response = await _client.SendAsync(request, cts.Token).ConfigureAwait(false);
                var text = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
                var status = (int)response.StatusCode;

                if (response.IsSuccessStatusCode)
                    return Parse(text, chunk.Length);

                last = new HttpEmbeddingException(
                    $"Embeddings request to {Where()} failed: HTTP {status} {response.ReasonPhrase}. {Excerpt(text)}".TrimEnd(),
                    status);
                if (!IsRetriable(response.StatusCode)) throw last;
                serverWait = RetryAfter(response);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (OperationCanceledException)
            {
                last = new HttpEmbeddingException(
                    $"Embeddings request to {Where()} timed out after {_timeout.TotalSeconds:0.##}s.");
            }
            catch (HttpRequestException ex)
            {
                last = new HttpEmbeddingException(
                    $"Embeddings request to {Where()} failed: {ex.GetType().Name}: {Redact(ex.Message)}");
            }

            if (attempt == _maxRetries) break;
            var wait = serverWait ?? TimeSpan.FromTicks(_retryDelay.Ticks * (1L << Math.Min(attempt, 20)));
            if (wait > _maxRetryDelay) wait = _maxRetryDelay;
            if (wait > TimeSpan.Zero)
                await Task.Delay(wait, ct).ConfigureAwait(false);
        }

        throw last ?? new HttpEmbeddingException($"Embeddings request to {Where()} failed.");
    }

    private static bool IsRetriable(HttpStatusCode code) =>
        code == HttpStatusCode.TooManyRequests ||
        ((int)code >= 500 && code != HttpStatusCode.NotImplemented && code != HttpStatusCode.HttpVersionNotSupported);

    private static TimeSpan? RetryAfter(HttpResponseMessage response)
    {
        var header = response.Headers.RetryAfter;
        if (header is null) return null;
        if (header.Delta is { } delta) return delta < TimeSpan.Zero ? TimeSpan.Zero : delta;
        if (header.Date is { } date)
        {
            var wait = date - DateTimeOffset.UtcNow;
            return wait < TimeSpan.Zero ? TimeSpan.Zero : wait;
        }
        return null;
    }

    private byte[] BuildBody(string[] chunk)
    {
        var fields = new Dictionary<string, object?>();
        foreach (var extra in _extraBody) fields[extra.Key] = extra.Value;
        fields["model"] = _model;
        fields["input"] = chunk;
        return JsonSerializer.SerializeToUtf8Bytes(fields);
    }

    private HttpRequestMessage BuildRequest(byte[] body)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, _uri) { Content = new ByteArrayContent(body) };
        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        foreach (var (name, value) in _headers)
            request.Headers.TryAddWithoutValidation(name, value);
        if (_apiKey is not null)
        {
            var value = string.Equals(_apiKeyHeader, "Authorization", StringComparison.OrdinalIgnoreCase)
                ? "Bearer " + _apiKey
                : _apiKey;
            request.Headers.Remove(_apiKeyHeader);
            request.Headers.TryAddWithoutValidation(_apiKeyHeader, value);
        }
        return request;
    }

    // -- response ----------------------------------------------------------------------

    private List<float[]> Parse(string json, int expected)
    {
        List<float[]> vectors;
        try
        {
            using var doc = JsonDocument.Parse(json);
            vectors = ReadVectors(doc.RootElement);
        }
        catch (JsonException)
        {
            throw Bad("the response is not valid JSON");
        }

        if (vectors.Count != expected)
            throw Bad($"expected {expected} embeddings, got {vectors.Count}");

        var dimension = _dimension;
        foreach (var vector in vectors)
        {
            if (vector.Length == 0) throw Bad("an embedding is empty");
            if (dimension == 0) dimension = vector.Length;
            if (vector.Length != dimension)
                throw new DimensionMismatchException(
                    $"Embeddings from {Where()} have length {vector.Length}, expected {dimension}.");
        }
        if (_dimension == 0) _dimension = dimension;
        return vectors;
    }

    private List<float[]> ReadVectors(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw Bad("the response is not a JSON object");

        // OpenAI-style: data[i].embedding, ordered by data[i].index.
        if (root.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
        {
            var items = new List<(int Index, float[] Vector)>();
            int position = 0;
            bool anyIndex = false;
            foreach (var item in data.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object ||
                    !item.TryGetProperty("embedding", out var embedding))
                    throw Bad("a data entry has no 'embedding'");
                int index = position;
                if (item.TryGetProperty("index", out var idx) && idx.ValueKind == JsonValueKind.Number &&
                    idx.TryGetInt32(out var parsed))
                {
                    index = parsed;
                    anyIndex = true;
                }
                items.Add((index, ReadVector(embedding)));
                position++;
            }
            if (anyIndex)
            {
                items = items.OrderBy(i => i.Index).ToList();
                for (int i = 0; i < items.Count; i++)
                    if (items[i].Index != i)
                        throw Bad("the response 'index' values are not 0..n-1");
            }
            return items.Select(i => i.Vector).ToList();
        }

        // Ollama-style: embeddings: [[...], ...].
        if (root.TryGetProperty("embeddings", out var embeddings) && embeddings.ValueKind == JsonValueKind.Array)
            return embeddings.EnumerateArray().Select(ReadVector).ToList();

        throw Bad("the response has neither 'data' (OpenAI) nor 'embeddings' (Ollama)");
    }

    private float[] ReadVector(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Array)
            throw Bad("an embedding is not an array of numbers");
        var values = new float[element.GetArrayLength()];
        int i = 0;
        foreach (var number in element.EnumerateArray())
        {
            if (number.ValueKind != JsonValueKind.Number)
                throw Bad("an embedding holds a non-number");
            var value = number.GetSingle();
            if (!float.IsFinite(value)) throw Bad("an embedding holds a value that is not a finite float");
            values[i++] = value;
        }
        return values;
    }

    private HttpEmbeddingException Bad(string what) =>
        new($"Unusable embeddings response from {Where()}: {what}.");

    // -- messages never carry the key --------------------------------------------------

    /// <summary>The URL without query string or credentials, for messages.</summary>
    private string Where() => _uri.GetLeftPart(UriPartial.Path);

    private string Redact(string text)
    {
        foreach (var secret in _secrets)
            text = text.Replace(secret, "***", StringComparison.Ordinal);
        return text;
    }

    private string Excerpt(string body)
    {
        if (string.IsNullOrWhiteSpace(body)) return "";
        var sb = new StringBuilder();
        bool space = false;
        foreach (var c in body)
        {
            if (char.IsWhiteSpace(c) || char.IsControl(c)) { space = sb.Length > 0; continue; }
            if (space) { sb.Append(' '); space = false; }
            sb.Append(c);
        }
        var flat = Redact(sb.ToString());
        return flat.Length <= BodyExcerptChars ? flat : flat[..BodyExcerptChars] + "...";
    }
}
