/**
 * HTTP embedding encoder for isongraph-vector.
 *
 * `HttpEncoder` implements EmbeddingEncoder over a remote embeddings endpoint:
 * it POSTs `{"model": ..., "input": [texts]}` as JSON and reads the vectors
 * back, in either of the two shapes in common use -
 *
 *  - OpenAI-compatible (`/v1/embeddings`, and the many servers that copy it):
 *    `{"data": [{"index": 0, "embedding": [...]}, ...]}`, ordered by `index`;
 *  - Ollama (`/api/embed`): `{"embeddings": [[...], ...]}`.
 *
 * Like the rest of the library this header is dependency-free: the transport
 * is a `std::function<HttpResponse(const HttpRequest&)>` you supply, so it
 * works with whatever HTTP stack your application already has (and with a fake
 * one in tests). An optional libcurl transport, `CurlTransport`, is compiled in
 * when ISONGRAPH_VECTOR_CURL is defined (CMake: `-DISONGRAPH_VECTOR_CURL=ON`,
 * which links libcurl); without it an HttpEncoder needs a transport argument.
 *
 * PRIVACY: every text you embed through this encoder is sent to the endpoint
 * you configure. With a hosted service that means the contents of your nodes
 * leave the machine. Point it at a local server (Ollama on localhost) if they
 * must not.
 *
 * @example
 * ```cpp
 * #include "isongraph_vector_http.hpp"
 *
 * using namespace isongraph_vector;
 *
 * HttpEncoderOptions options;
 * options.url = "https://api.openai.com/v1/embeddings";
 * options.model = "text-embedding-3-small";
 * options.apiKey = std::getenv("OPENAI_API_KEY");
 * options.dimension = 1536;          // saves one probe request, see dimension()
 *
 * auto encoder = std::make_shared<HttpEncoder>(options);   // needs ISONGRAPH_VECTOR_CURL
 * SemanticGraph graph("kb", encoder);
 * ```
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

#ifndef ISONGRAPH_VECTOR_HTTP_HPP
#define ISONGRAPH_VECTOR_HTTP_HPP

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "isongraph_vector.hpp"

#ifdef ISONGRAPH_VECTOR_CURL
#include <curl/curl.h>
#endif

namespace isongraph_vector {

// =============================================================================
// Transport types
// =============================================================================

/**
 * An HTTP request failed in a way worth reporting (the connection broke, the
 * endpoint answered with an error status). `status()` is the HTTP status, or 0
 * when there was no response. A transport throws this for a network-level
 * failure; HttpEncoder retries those like a 5xx.
 */
class HttpError : public EmbeddingError {
public:
    explicit HttpError(const std::string& what, int status = 0)
        : EmbeddingError(what), status_(status) {}
    int status() const { return status_; }

private:
    int status_;
};

/** What HttpEncoder hands to the transport. Always a POST with a JSON body. */
struct HttpRequest {
    std::string method = "POST";
    std::string url;
    /** Header name -> value. Includes the API key header when one is configured. */
    std::map<std::string, std::string> headers;
    std::string body;
    /** Budget for the whole request, in milliseconds. */
    int timeoutMs = 30000;
};

/** What the transport returns. Any HTTP status is a response, not an error. */
struct HttpResponse {
    int status = 0;
    std::string body;
    /** Optional. Only `Retry-After` is read; names may be in any case. */
    std::map<std::string, std::string> headers;
};

/**
 * Sends one request and returns the response. Throw HttpError (or any
 * std::exception) when no response could be obtained - a timeout, a refused
 * connection. Must not log or otherwise expose request.headers: they carry the
 * API key.
 */
using HttpTransport = std::function<HttpResponse(const HttpRequest&)>;

// =============================================================================
// libcurl transport (optional)
// =============================================================================

#ifdef ISONGRAPH_VECTOR_CURL

/**
 * HttpTransport over libcurl. Compiled only with ISONGRAPH_VECTOR_CURL.
 *
 * One easy handle per request (no connection reuse - an embeddings call is
 * long next to a TLS handshake). Redirects are not followed, so the API key is
 * never forwarded to another host; only http and https are allowed; the
 * response body is capped at 64 MiB. The first use initialises libcurl
 * globally (curl_global_init is not thread-safe before that point, so make a
 * first call from one thread before going concurrent).
 */
class CurlTransport {
public:
    HttpResponse operator()(const HttpRequest& request) const {
        static const GlobalInit global;
        (void)global;

        if (request.method != "POST") {
            throw HttpError("CurlTransport only supports POST");
        }

        CURL* raw = curl_easy_init();
        if (raw == nullptr) throw HttpError("could not initialise libcurl");
        std::unique_ptr<CURL, void (*)(CURL*)> easy(raw, curl_easy_cleanup);

        curl_slist* rawList = nullptr;
        for (const auto& [name, value] : request.headers) {
            std::string line = name + ": " + value;
            curl_slist* next = curl_slist_append(rawList, line.c_str());
            if (next == nullptr) {
                curl_slist_free_all(rawList);
                throw HttpError("out of memory building request headers");
            }
            rawList = next;
        }
        std::unique_ptr<curl_slist, void (*)(curl_slist*)> list(rawList, curl_slist_free_all);

        HttpResponse response;
        char errorBuffer[CURL_ERROR_SIZE] = {0};

        auto set = [&](CURLoption option, auto value) {
            CURLcode rc = curl_easy_setopt(easy.get(), option, value);
            if (rc != CURLE_OK) {
                throw HttpError(std::string("could not configure the HTTP request: ") +
                                curl_easy_strerror(rc));
            }
        };

        const long timeoutMs = request.timeoutMs > 0 ? request.timeoutMs : 30000L;
        set(CURLOPT_URL, request.url.c_str());
        set(CURLOPT_POST, 1L);
        set(CURLOPT_POSTFIELDS, request.body.c_str());
        set(CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body.size()));
        set(CURLOPT_HTTPHEADER, list.get());
        set(CURLOPT_TIMEOUT_MS, timeoutMs);
        set(CURLOPT_CONNECTTIMEOUT_MS, std::min<long>(timeoutMs, 10000L));
        set(CURLOPT_NOSIGNAL, 1L);
        set(CURLOPT_FOLLOWLOCATION, 0L);
#if LIBCURL_VERSION_NUM >= 0x075500
        set(CURLOPT_PROTOCOLS_STR, "http,https");
#else
        set(CURLOPT_PROTOCOLS, static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
        set(CURLOPT_ACCEPT_ENCODING, "");
        set(CURLOPT_ERRORBUFFER, errorBuffer);
        set(CURLOPT_WRITEFUNCTION, &CurlTransport::onBody);
        set(CURLOPT_WRITEDATA, &response.body);
        set(CURLOPT_HEADERFUNCTION, &CurlTransport::onHeader);
        set(CURLOPT_HEADERDATA, &response.headers);

        CURLcode rc = curl_easy_perform(easy.get());
        if (rc != CURLE_OK) {
            // libcurl's own text: names the failure and the host, never a header.
            throw HttpError(std::string("HTTP request failed: ") +
                            (errorBuffer[0] != '\0' ? errorBuffer : curl_easy_strerror(rc)));
        }

        long status = 0;
        curl_easy_getinfo(easy.get(), CURLINFO_RESPONSE_CODE, &status);
        response.status = static_cast<int>(status);
        return response;
    }

private:
    static constexpr size_t MAX_BODY_BYTES = 64u * 1024u * 1024u;

    struct GlobalInit {
        GlobalInit() {
            if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
                throw HttpError("could not initialise libcurl");
            }
        }
        ~GlobalInit() { curl_global_cleanup(); }
    };

    static size_t onBody(char* data, size_t size, size_t count, void* userdata) {
        auto* body = static_cast<std::string*>(userdata);
        size_t bytes = size * count;
        if (body->size() + bytes > MAX_BODY_BYTES) return 0;  // aborts the transfer
        body->append(data, bytes);
        return bytes;
    }

    static size_t onHeader(char* data, size_t size, size_t count, void* userdata) {
        auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);
        size_t bytes = size * count;
        std::string line(data, bytes);
        size_t colon = line.find(':');
        if (colon != std::string::npos && colon > 0) {
            std::string name = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
                value.erase(value.begin());
            }
            while (!value.empty() && (value.back() == '\r' || value.back() == '\n' ||
                                      value.back() == ' ')) {
                value.pop_back();
            }
            (*headers)[name] = value;
        }
        return bytes;
    }
};

#endif  // ISONGRAPH_VECTOR_CURL

// =============================================================================
// HttpEncoder
// =============================================================================

/** Settings for an HttpEncoder. `url` and `model` are required. */
struct HttpEncoderOptions {
    /** Full endpoint URL, http:// or https://. */
    std::string url;
    /** Sent as `model`, and reported as the encoder's modelName(). */
    std::string model;
    /**
     * Vector length the endpoint returns. Optional: when unset, dimension()
     * learns it from the first response (making one small probe request if it
     * is asked for before anything has been encoded). When set, every response
     * is checked against it.
     */
    std::optional<size_t> dimension;

    /** Credential, sent as `<apiKeyHeader>: <apiKeyPrefix><apiKey>`. Empty = none (e.g. local Ollama). */
    std::string apiKey;
    /** Header carrying the key. "api-key" for Azure OpenAI. */
    std::string apiKeyHeader = "Authorization";
    /** Text placed before the key. Set to "" for a header that takes the bare key. */
    std::string apiKeyPrefix = "Bearer ";
    /** Extra request headers. */
    std::map<std::string, std::string> headers;

    /** Texts per request; encodeBatch splits larger batches. Must be at least 1. */
    size_t batchSize = 64;
    /** Per-request timeout in milliseconds. */
    int timeoutMs = 30000;

    /** Retries after the first attempt, on HTTP 429, 5xx and transport failures. 0 = none. */
    int maxRetries = 3;
    /** Wait before the first retry; doubles each time. A Retry-After header (seconds) overrides it. */
    int retryBackoffMs = 500;
    /** Ceiling for any single wait. */
    int maxBackoffMs = 10000;

    /** Encoder/model version recorded in the sidecar (a model revision, a weights hash). */
    std::optional<std::string> version;

    /** Waits between retries. Default: std::this_thread::sleep_for. Replace it to test without delay. */
    std::function<void(std::chrono::milliseconds)> sleeper;
};

/**
 * EmbeddingEncoder backed by an HTTP embeddings endpoint (OpenAI-compatible or
 * Ollama `/api/embed`; the response shape is detected, not configured).
 *
 * Errors are EmbeddingError (HttpError for HTTP/transport failures,
 * DimensionMismatchError for a vector of the wrong width). Messages carry the
 * HTTP status and the server's own error text, never the request headers or
 * the API key (the key is also scrubbed from the server's text, in case an
 * error page echoes it). Not safe to call from several threads at once.
 *
 * Every text is sent to `options.url`; see the privacy note at the top of this
 * file.
 */
class HttpEncoder : public EmbeddingEncoder {
public:
    /**
     * @param options    endpoint and behaviour
     * @param transport  how to talk HTTP; omitted, CurlTransport is used when
     *                   the library was built with ISONGRAPH_VECTOR_CURL
     * @throws EmbeddingError for a missing url/model, a non-http(s) url, a
     *         zero batchSize, a header containing a line break, or no
     *         transport being available
     */
    explicit HttpEncoder(HttpEncoderOptions options, HttpTransport transport = nullptr)
        : options_(std::move(options)), transport_(std::move(transport)),
          dimension_(options_.dimension) {
        if (options_.url.empty()) throw EmbeddingError("HttpEncoder needs a url");
        if (options_.model.empty()) throw EmbeddingError("HttpEncoder needs a model");
        if (!startsWithIgnoreCase(options_.url, "http://") &&
            !startsWithIgnoreCase(options_.url, "https://")) {
            throw EmbeddingError("HttpEncoder url must start with http:// or https://");
        }
        if (options_.batchSize == 0) throw EmbeddingError("batchSize must be at least 1");
        if (options_.dimension.has_value() && *options_.dimension == 0) {
            throw EmbeddingError("dimension must be positive");
        }
        if (options_.maxRetries < 0) options_.maxRetries = 0;
        if (options_.retryBackoffMs < 0) options_.retryBackoffMs = 0;
        if (options_.maxBackoffMs < options_.retryBackoffMs) {
            options_.maxBackoffMs = options_.retryBackoffMs;
        }

        // A header value with a line break would let a value smuggle in more headers.
        for (const auto& [name, value] : options_.headers) checkHeader(name, value);
        checkHeader(options_.apiKeyHeader, options_.apiKeyPrefix + options_.apiKey);

        if (!transport_) {
#ifdef ISONGRAPH_VECTOR_CURL
            transport_ = CurlTransport();
#else
            throw EmbeddingError(
                "HttpEncoder has no transport: pass one, or build with ISONGRAPH_VECTOR_CURL "
                "to use libcurl");
#endif
        }

        if (!options_.apiKey.empty() && startsWithIgnoreCase(options_.url, "http://") &&
            !isLoopbackHost(options_.url)) {
            emitWarning("HttpEncoder will send its API key over plain http to a non-local host; "
                        "use https");
        }
    }

    /** The configured model name. */
    std::string modelName() const override { return options_.model; }

    std::optional<std::string> version() const override { return options_.version; }

    /**
     * Vector length. Known up front when HttpEncoderOptions::dimension is set
     * or once anything has been encoded; otherwise this sends one request with
     * a fixed probe text (none of your data) to find out.
     */
    size_t dimension() const override {
        if (!dimension_.has_value()) embedChunk({"dimension probe"});
        return *dimension_;
    }

    std::vector<float> encode(const std::string& text) const override {
        return std::move(embedChunk({text}).front());
    }

    std::vector<std::vector<float>> encodeBatch(const std::vector<std::string>& texts) const override {
        std::vector<std::vector<float>> all;
        all.reserve(texts.size());
        for (size_t start = 0; start < texts.size(); start += options_.batchSize) {
            size_t end = std::min(start + options_.batchSize, texts.size());
            auto part = embedChunk(std::vector<std::string>(texts.begin() + start,
                                                            texts.begin() + end));
            for (auto& vector : part) all.push_back(std::move(vector));
        }
        return all;
    }

private:
    HttpEncoderOptions options_;
    HttpTransport transport_;
    mutable std::optional<size_t> dimension_;

    // -- request ---------------------------------------------------------------

    static bool startsWithIgnoreCase(const std::string& text, const std::string& prefix) {
        if (text.size() < prefix.size()) return false;
        for (size_t i = 0; i < prefix.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(text[i])) !=
                std::tolower(static_cast<unsigned char>(prefix[i]))) {
                return false;
            }
        }
        return true;
    }

    static bool equalsIgnoreCase(const std::string& a, const std::string& b) {
        return a.size() == b.size() && startsWithIgnoreCase(a, b);
    }

    static bool isLoopbackHost(const std::string& url) {
        size_t start = url.find("://") + 3;
        size_t end = url.find_first_of("/?#", start);
        std::string authority = url.substr(start, end == std::string::npos ? end : end - start);
        size_t at = authority.rfind('@');
        if (at != std::string::npos) authority = authority.substr(at + 1);
        std::string host = authority;
        if (!host.empty() && host.front() == '[') {
            size_t close = host.find(']');
            host = host.substr(0, close == std::string::npos ? host.size() : close + 1);
        } else {
            host = host.substr(0, host.find(':'));
        }
        return equalsIgnoreCase(host, "localhost") || host == "127.0.0.1" || host == "[::1]";
    }

    static void checkHeader(const std::string& name, const std::string& value) {
        auto bad = [](const std::string& s) {
            return s.find_first_of("\r\n") != std::string::npos;
        };
        if (name.empty() || bad(name) || bad(value)) {
            throw EmbeddingError("HTTP header names and values must not be empty or contain "
                                 "line breaks");
        }
    }

    static void setHeader(std::map<std::string, std::string>& headers, const std::string& name,
                          const std::string& value) {
        for (auto it = headers.begin(); it != headers.end();) {
            it = equalsIgnoreCase(it->first, name) ? headers.erase(it) : std::next(it);
        }
        headers[name] = value;
    }

    static std::string quote(const std::string& value) {
        static const char* const HEX = "0123456789abcdef";
        std::string out = "\"";
        for (char c : value) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        out += "\\u00";
                        out += HEX[(static_cast<unsigned char>(c) >> 4) & 0xF];
                        out += HEX[static_cast<unsigned char>(c) & 0xF];
                    } else {
                        out += c;
                    }
            }
        }
        return out + "\"";
    }

    HttpRequest buildRequest(const std::vector<std::string>& texts) const {
        HttpRequest request;
        request.url = options_.url;
        request.timeoutMs = options_.timeoutMs;

        request.body = "{\"model\":" + quote(options_.model) + ",\"input\":[";
        for (size_t i = 0; i < texts.size(); ++i) {
            if (i > 0) request.body += ",";
            request.body += quote(texts[i]);
        }
        request.body += "]}";

        // defaults first, so a configured header of the same name (any case) wins
        setHeader(request.headers, "Content-Type", "application/json");
        setHeader(request.headers, "Accept", "application/json");
        for (const auto& [name, value] : options_.headers) setHeader(request.headers, name, value);
        if (!options_.apiKey.empty()) {
            setHeader(request.headers, options_.apiKeyHeader,
                      options_.apiKeyPrefix + options_.apiKey);
        }
        return request;
    }

    // -- sending, with bounded retry ------------------------------------------

    /** Replace every occurrence of the API key in `text`. */
    std::string redact(std::string text) const {
        const std::string& key = options_.apiKey;
        if (key.empty()) return text;
        size_t at = 0;
        while ((at = text.find(key, at)) != std::string::npos) {
            text.replace(at, key.size(), "[redacted]");
            at += 10;
        }
        return text;
    }

    static bool retryableStatus(int status) {
        return status == 429 || (status >= 500 && status <= 599);
    }

    /** Seconds from a Retry-After header, as milliseconds; nullopt if absent or not a number. */
    static std::optional<long long> retryAfterMs(const HttpResponse& response) {
        for (const auto& [name, value] : response.headers) {
            if (!equalsIgnoreCase(name, "retry-after")) continue;
            if (value.empty() || value.size() > 6) return std::nullopt;
            long long seconds = 0;
            for (char c : value) {
                if (c < '0' || c > '9') return std::nullopt;  // an HTTP-date: not honoured
                seconds = seconds * 10 + (c - '0');
            }
            return seconds * 1000;
        }
        return std::nullopt;
    }

    void pause(long long ms) const {
        if (ms <= 0) return;
        auto duration = std::chrono::milliseconds(ms);
        if (options_.sleeper) options_.sleeper(duration);
        else std::this_thread::sleep_for(duration);
    }

    /** Send with up to maxRetries retries; returns the final response (any status). */
    HttpResponse send(const HttpRequest& request) const {
        long long backoff = options_.retryBackoffMs;
        for (int attempt = 0;; ++attempt) {
            long long wait = backoff;
            try {
                HttpResponse response = transport_(request);
                if (!retryableStatus(response.status) || attempt >= options_.maxRetries) {
                    return response;
                }
                if (auto asked = retryAfterMs(response)) wait = *asked;
            } catch (const HttpError& error) {
                if (attempt >= options_.maxRetries) {
                    throw HttpError(redact(error.what()), error.status());
                }
            } catch (const std::exception& error) {
                throw EmbeddingError("HTTP transport failed: " + redact(error.what()));
            }
            pause(std::min<long long>(wait, options_.maxBackoffMs));
            backoff = std::min<long long>(backoff * 2, options_.maxBackoffMs);
        }
    }

    // -- response ---------------------------------------------------------------

    /** The server's own explanation of a failure, shortened and scrubbed. */
    std::string serverMessage(const std::string& body) const {
        using detail::JsonValue;
        std::string message;
        try {
            JsonValue root = detail::JsonParser(body).parse();
            if (root.kind == JsonValue::Kind::Object) {
                const JsonValue* error = root.find("error");
                if (error != nullptr && error->kind == JsonValue::Kind::String) {
                    message = error->text;
                } else if (error != nullptr && error->kind == JsonValue::Kind::Object) {
                    const JsonValue* inner = error->find("message");
                    if (inner != nullptr && inner->kind == JsonValue::Kind::String) {
                        message = inner->text;
                    }
                } else {
                    const JsonValue* top = root.find("message");
                    if (top != nullptr && top->kind == JsonValue::Kind::String) message = top->text;
                }
            }
        } catch (const EmbeddingError&) {
            message = body;  // not JSON: show the start of whatever it is
        }
        for (char& c : message) {
            if (static_cast<unsigned char>(c) < 0x20) c = ' ';
        }
        message = redact(message);
        if (message.size() > 200) message = message.substr(0, 200) + "...";
        return message;
    }

    static std::vector<float> numbersOf(const detail::JsonValue& array, const std::string& what) {
        using detail::JsonValue;
        if (array.kind != JsonValue::Kind::Array || !array.items.empty() ||
            array.numbers.empty()) {
            throw EmbeddingError("embedding endpoint returned " + what +
                                 " that is not a non-empty array of numbers");
        }
        std::vector<float> vector;
        vector.reserve(array.numbers.size());
        for (double number : array.numbers) {
            if (!std::isfinite(number)) {
                throw EmbeddingError("embedding endpoint returned a non-finite number in " + what);
            }
            vector.push_back(static_cast<float>(number));
        }
        return vector;
    }

    /** Vectors in input order from an OpenAI-style or Ollama-style body. */
    std::vector<std::vector<float>> parseVectors(const std::string& body, size_t expected) const {
        using detail::JsonValue;
        JsonValue root;
        try {
            root = detail::JsonParser(body).parse();
        } catch (const EmbeddingError& error) {
            throw EmbeddingError(std::string("embedding endpoint returned malformed JSON: ") +
                                 error.what());
        }
        if (root.kind != JsonValue::Kind::Object) {
            throw EmbeddingError("embedding endpoint returned JSON that is not an object");
        }

        std::vector<std::vector<float>> vectors;
        if (const JsonValue* data = root.find("data");
            data != nullptr && data->kind == JsonValue::Kind::Array) {
            // OpenAI: ordered by `index`, which need not match the array order.
            if (data->items.size() != expected) {
                throw EmbeddingError("embedding endpoint returned " +
                                     std::to_string(data->items.size()) + " vectors for " +
                                     std::to_string(expected) + " texts");
            }
            std::vector<std::optional<std::vector<float>>> slots(expected);
            for (size_t position = 0; position < data->items.size(); ++position) {
                const JsonValue& entry = data->items[position];
                if (entry.kind != JsonValue::Kind::Object) {
                    throw EmbeddingError("embedding endpoint returned a data entry that is not "
                                         "an object");
                }
                size_t index = position;
                if (const JsonValue* given = entry.find("index"); given != nullptr) {
                    if (given->kind != JsonValue::Kind::Number || given->number < 0 ||
                        given->number != std::floor(given->number) ||
                        given->number >= static_cast<double>(expected)) {
                        throw EmbeddingError("embedding endpoint returned an invalid index");
                    }
                    index = static_cast<size_t>(given->number);
                }
                if (slots[index].has_value()) {
                    throw EmbeddingError("embedding endpoint returned index " +
                                         std::to_string(index) + " twice");
                }
                const JsonValue* embedding = entry.find("embedding");
                if (embedding == nullptr) {
                    throw EmbeddingError("embedding endpoint returned a data entry with no "
                                         "embedding");
                }
                slots[index] = numbersOf(*embedding, "an embedding");
            }
            for (auto& slot : slots) vectors.push_back(std::move(*slot));
        } else if (const JsonValue* list = root.find("embeddings");
                   list != nullptr && list->kind == JsonValue::Kind::Array) {
            // Ollama: plain array of arrays, already in input order.
            if (list->items.size() != expected) {
                throw EmbeddingError("embedding endpoint returned " +
                                     std::to_string(list->items.size()) + " vectors for " +
                                     std::to_string(expected) + " texts");
            }
            for (const JsonValue& entry : list->items) {
                vectors.push_back(numbersOf(entry, "an embedding"));
            }
        } else if (const JsonValue* error = root.find("error"); error != nullptr) {
            throw HttpError("embedding endpoint reported an error: " + serverMessage(body));
        } else {
            throw EmbeddingError("embedding endpoint returned an unrecognised response "
                                 "(expected an OpenAI-style 'data' or an Ollama-style "
                                 "'embeddings' array)");
        }

        // One width for the whole response, and the one the endpoint is known for.
        size_t width = dimension_.value_or(vectors.front().size());
        for (const auto& vector : vectors) {
            if (vector.size() != width) {
                throw DimensionMismatchError(
                    "embedding endpoint returned a vector of " + std::to_string(vector.size()) +
                    " dimensions, expected " + std::to_string(width));
            }
        }
        dimension_ = width;
        return vectors;
    }

    /** One request for 1..batchSize texts. */
    std::vector<std::vector<float>> embedChunk(const std::vector<std::string>& texts) const {
        HttpResponse response = send(buildRequest(texts));
        if (response.status < 200 || response.status > 299) {
            std::string detail = serverMessage(response.body);
            throw HttpError("embedding request failed with HTTP " +
                                std::to_string(response.status) +
                                (detail.empty() ? std::string() : ": " + detail),
                            response.status);
        }
        return parseVectors(response.body, texts.size());
    }
};

}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_HTTP_HPP
