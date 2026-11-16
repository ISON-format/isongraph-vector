/**
 * libcurl transport tests. Built only with -DISONGRAPH_VECTOR_CURL=ON.
 *
 * No internet: every request goes to a tiny HTTP/1.1 server started inside the
 * test process, listening on 127.0.0.1 on an ephemeral port. It answers with
 * what OpenAI-compatible and Ollama endpoints answer, and misbehaves on demand
 * (error statuses, a stall, a redirect, an empty reply, a chunked or enormous
 * body), so what is checked is real libcurl talking real HTTP: the request the
 * server sees, the response the transport hands back, and the errors it
 * raises - including that the API key never ends up in one.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// winsock2.h pulls in windows.h, whose min/max macros break std::min/std::max
// (and numeric_limits<T>::max()) in every header that follows.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static const socket_t BAD_SOCKET = INVALID_SOCKET;
static void closeSocket(socket_t s) { closesocket(s); }
static const int SEND_FLAGS = 0;
#else
#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static const socket_t BAD_SOCKET = -1;
static void closeSocket(socket_t s) { close(s); }
#ifdef MSG_NOSIGNAL
static const int SEND_FLAGS = MSG_NOSIGNAL;
#else
static const int SEND_FLAGS = 0;  // macOS: SO_NOSIGPIPE is set on each socket instead
#endif
#endif

#include "../include/isongraph_vector_http.hpp"

using namespace isongraph_vector;

int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            tests_run++; \
            std::cout << "Running: " << #name << "... " << std::flush; \
            try { \
                test_##name(); \
                tests_passed++; \
                std::cout << "PASSED" << std::endl; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << std::endl; \
            } catch (...) { \
                std::cout << "FAILED: Unknown exception" << std::endl; \
            } \
        } \
    } runner_##name; \
    void test_##name()

#define ASSERT_TRUE(expr) \
    if (!(expr)) throw std::runtime_error("Assertion failed: " #expr);
#define ASSERT_FALSE(expr) \
    if (expr) throw std::runtime_error("Assertion failed: " #expr " should be false");
#define ASSERT_EQ(a, b) \
    if (!((a) == (b))) throw std::runtime_error("Assertion failed: " #a " != " #b);
#define ASSERT_THROWS(expr, exception_type) \
    { \
        bool threw = false; \
        try { expr; } catch (const exception_type&) { threw = true; } \
        if (!threw) throw std::runtime_error("Assertion failed: " #expr " did not throw " #exception_type); \
    }

namespace {

// =============================================================================
// A small in-process HTTP/1.1 server
// =============================================================================

struct Received {
    std::string method;
    std::string target;
    std::map<std::string, std::string> headers;  // names lower-cased
    std::string body;

    std::string header(const std::string& name) const {
        auto it = headers.find(name);
        return it == headers.end() ? std::string() : it->second;
    }
};

struct Reply {
    enum class Mode {
        Normal,           // status line, headers, Content-Length body
        Chunked,          // same, Transfer-Encoding: chunked in three pieces
        NoReply,          // close the connection without answering
        Stall,            // read the request, then say nothing until the test ends
        Huge,             // announce and stream `hugeBytes` of body
    };
    Mode mode = Mode::Normal;
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    size_t hugeBytes = 0;
};

Reply json(int status, const std::string& body) {
    Reply r;
    r.status = status;
    r.headers.emplace_back("Content-Type", "application/json");
    r.body = body;
    return r;
}

const char* reasonFor(int status) {
    switch (status) {
        case 200: return "OK";
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Status";
    }
}

class LoopbackServer {
public:
    using Handler = std::function<Reply(const Received&, size_t requestNumber)>;

    explicit LoopbackServer(Handler handler) : handler_(std::move(handler)) {
#ifdef _WIN32
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
#else
        std::signal(SIGPIPE, SIG_IGN);
#endif
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ == BAD_SOCKET) throw std::runtime_error("cannot create a socket");
        int yes = 1;
        setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof yes);
        sockaddr_in address;
        std::memset(&address, 0, sizeof address);
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 ||
            ::listen(listener_, 8) != 0) {
            closeSocket(listener_);
            throw std::runtime_error("cannot listen on the loopback interface");
        }
        socklen_t length = sizeof address;
        getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length);
        port_ = ntohs(address.sin_port);
        thread_ = std::thread([this] { run(); });
    }

    ~LoopbackServer() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
        closeSocket(listener_);
    }

    int port() const { return port_; }
    std::string url(const std::string& path = "/v1/embeddings") const {
        return "http://127.0.0.1:" + std::to_string(port_) + path;
    }

    size_t requestCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.size();
    }
    Received request(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.at(index);
    }

private:
    Handler handler_;
    socket_t listener_ = BAD_SOCKET;
    int port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<Received> requests_;

    void run() {
        while (!stop_) {
            fd_set readable;
            FD_ZERO(&readable);
            FD_SET(listener_, &readable);
            timeval wait;
            wait.tv_sec = 0;
            wait.tv_usec = 50 * 1000;
            int ready = select(static_cast<int>(listener_) + 1, &readable, nullptr, nullptr, &wait);
            if (ready <= 0) continue;
            socket_t client = ::accept(listener_, nullptr, nullptr);
            if (client == BAD_SOCKET) continue;
#ifdef SO_NOSIGPIPE
            int one = 1;
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            try {
                serve(client);
            } catch (...) {
                // a client that gave up mid-exchange is not a server failure
            }
            closeSocket(client);
        }
    }

    static bool sendAll(socket_t s, const char* data, size_t size) {
        while (size > 0) {
            int sent = static_cast<int>(::send(s, data, static_cast<int>(std::min<size_t>(size, 1 << 20)),
                                               SEND_FLAGS));
            if (sent <= 0) return false;
            data += sent;
            size -= static_cast<size_t>(sent);
        }
        return true;
    }

    /** Reads one request, or false when the client closed first. */
    bool readRequest(socket_t s, Received& out) {
        std::string buffer;
        size_t headerEnd = std::string::npos;
        char chunk[16384];
        while ((headerEnd = buffer.find("\r\n\r\n")) == std::string::npos) {
            int got = static_cast<int>(::recv(s, chunk, sizeof chunk, 0));
            if (got <= 0) return false;
            buffer.append(chunk, static_cast<size_t>(got));
            if (buffer.size() > (1u << 20)) return false;
        }
        std::istringstream head(buffer.substr(0, headerEnd));
        std::string line;
        std::getline(head, line);
        {
            std::istringstream request(line);
            std::string version;
            request >> out.method >> out.target >> version;
        }
        while (std::getline(head, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string name = line.substr(0, colon);
            for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            std::string value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            out.headers[name] = value;
        }
        if (out.header("expect") == "100-continue") {
            const char* go = "HTTP/1.1 100 Continue\r\n\r\n";
            sendAll(s, go, std::strlen(go));
        }
        size_t length = 0;
        if (!out.header("content-length").empty()) length = std::strtoull(out.header("content-length").c_str(), nullptr, 10);
        out.body = buffer.substr(headerEnd + 4);
        while (out.body.size() < length) {
            int got = static_cast<int>(::recv(s, chunk, sizeof chunk, 0));
            if (got <= 0) return false;
            out.body.append(chunk, static_cast<size_t>(got));
        }
        out.body.resize(length);
        return true;
    }

    void serve(socket_t s) {
        Received received;
        if (!readRequest(s, received)) return;
        size_t number;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.push_back(received);
            number = requests_.size() - 1;
        }
        Reply reply = handler_(received, number);

        if (reply.mode == Reply::Mode::NoReply) return;
        if (reply.mode == Reply::Mode::Stall) {
            // Say nothing until the client gives up (so the next connection is
            // served at once) or the test ends.
            for (int i = 0; i < 400 && !stop_; ++i) {
                fd_set readable;
                FD_ZERO(&readable);
                FD_SET(s, &readable);
                timeval wait;
                wait.tv_sec = 0;
                wait.tv_usec = 10 * 1000;
                if (select(static_cast<int>(s) + 1, &readable, nullptr, nullptr, &wait) > 0) {
                    char probe;
                    if (::recv(s, &probe, 1, MSG_PEEK) <= 0) return;  // closed by the client
                }
            }
            return;
        }

        std::string head = "HTTP/1.1 " + std::to_string(reply.status) + " " + reasonFor(reply.status) + "\r\n";
        for (const auto& [name, value] : reply.headers) head += name + ": " + value + "\r\n";
        head += "Connection: close\r\n";
        if (reply.mode == Reply::Mode::Huge) {
            head += "Content-Length: " + std::to_string(reply.hugeBytes) + "\r\n\r\n";
            if (!sendAll(s, head.data(), head.size())) return;
            std::string block(1 << 20, 'a');
            for (size_t sent = 0; sent < reply.hugeBytes;) {
                size_t n = std::min(block.size(), reply.hugeBytes - sent);
                if (!sendAll(s, block.data(), n)) return;  // the client stopped reading, as it should
                sent += n;
            }
        } else if (reply.mode == Reply::Mode::Chunked) {
            head += "Transfer-Encoding: chunked\r\n\r\n";
            sendAll(s, head.data(), head.size());
            size_t third = reply.body.size() / 3;
            std::string parts[3] = {reply.body.substr(0, third), reply.body.substr(third, third),
                                    reply.body.substr(2 * third)};
            for (const auto& part : parts) {
                char size[32];
                std::snprintf(size, sizeof size, "%zx\r\n", part.size());
                std::string piece = std::string(size) + part + "\r\n";
                sendAll(s, piece.data(), piece.size());
            }
            sendAll(s, "0\r\n\r\n", 5);
        } else {
            head += "Content-Length: " + std::to_string(reply.body.size()) + "\r\n\r\n";
            std::string whole = head + reply.body;
            sendAll(s, whole.data(), whole.size());
        }
#ifdef _WIN32
        shutdown(s, SD_SEND);
#else
        shutdown(s, SHUT_WR);
#endif
    }
};

// =============================================================================
// Canned responses and helpers
// =============================================================================

const char* const KEY = "sk-test-0123456789abcdef";

/** OpenAI-shaped answer for two inputs, deliberately out of order (index says which is which). */
const char* const OPENAI_TWO =
    "{\"object\":\"list\",\"data\":["
    "{\"object\":\"embedding\",\"index\":1,\"embedding\":[0.0,1.0,0.0,0.0]},"
    "{\"object\":\"embedding\",\"index\":0,\"embedding\":[1.0,0.0,0.0,0.0]}],"
    "\"model\":\"text-embedding-3-small\",\"usage\":{\"prompt_tokens\":4,\"total_tokens\":4}}";

const char* const OLLAMA_TWO =
    "{\"model\":\"nomic-embed-text\",\"embeddings\":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],"
    "\"total_duration\":123}";

HttpEncoderOptions optionsFor(const LoopbackServer& server, const std::string& path = "/v1/embeddings") {
    HttpEncoderOptions options;
    options.url = server.url(path);
    options.model = "test-embed";
    options.dimension = 4;
    options.maxRetries = 0;
    options.timeoutMs = 5000;
    options.sleeper = [](std::chrono::milliseconds) {};
    return options;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

// =============================================================================
// Shapes: OpenAI and Ollama, through the default (libcurl) transport
// =============================================================================

TEST(openai_shaped_response_is_decoded_in_index_order_and_the_request_is_well_formed) {
    LoopbackServer server([](const Received&, size_t) { return json(200, OPENAI_TWO); });
    auto options = optionsFor(server);
    options.apiKey = KEY;
    HttpEncoder encoder(options);  // no transport given: CurlTransport

    auto vectors = encoder.encodeBatch({"first", "second"});
    ASSERT_EQ(vectors.size(), 2u);
    ASSERT_EQ(vectors[0], (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}));  // index 0, though listed second
    ASSERT_EQ(vectors[1], (std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f}));

    ASSERT_EQ(server.requestCount(), 1u);
    Received seen = server.request(0);
    ASSERT_EQ(seen.method, "POST");
    ASSERT_EQ(seen.target, "/v1/embeddings");
    ASSERT_EQ(seen.header("content-type"), "application/json");
    ASSERT_EQ(seen.header("accept"), "application/json");
    ASSERT_EQ(seen.header("authorization"), std::string("Bearer ") + KEY);
    ASSERT_EQ(seen.body, "{\"model\":\"test-embed\",\"input\":[\"first\",\"second\"]}");
    ASSERT_EQ(seen.header("content-length"), std::to_string(seen.body.size()));
    ASSERT_EQ(seen.header("expect"), "");  // no 100-continue round trip
}

TEST(ollama_shaped_response_is_decoded_and_no_credential_is_sent_without_a_key) {
    LoopbackServer server([](const Received&, size_t) { return json(200, OLLAMA_TWO); });
    HttpEncoder encoder(optionsFor(server, "/api/embed"));

    auto vectors = encoder.encodeBatch({"a", "b"});
    ASSERT_EQ(vectors.size(), 2u);
    ASSERT_EQ(vectors[1], (std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f}));
    Received seen = server.request(0);
    ASSERT_EQ(seen.target, "/api/embed");
    ASSERT_EQ(seen.header("authorization"), "");
}

TEST(a_custom_key_header_and_extra_headers_reach_the_server) {
    LoopbackServer server([](const Received&, size_t) { return json(200, OPENAI_TWO); });
    auto options = optionsFor(server);
    options.apiKey = KEY;
    options.apiKeyHeader = "api-key";
    options.apiKeyPrefix = "";
    options.headers["X-Tenant"] = "acme";
    HttpEncoder(options).encodeBatch({"a", "b"});
    Received seen = server.request(0);
    ASSERT_EQ(seen.header("api-key"), KEY);
    ASSERT_EQ(seen.header("x-tenant"), "acme");
    ASSERT_EQ(seen.header("authorization"), "");
}

TEST(a_chunked_response_is_reassembled) {
    LoopbackServer server([](const Received&, size_t) {
        Reply r = json(200, OLLAMA_TWO);
        r.mode = Reply::Mode::Chunked;
        return r;
    });
    auto vectors = HttpEncoder(optionsFor(server)).encodeBatch({"a", "b"});
    ASSERT_EQ(vectors.size(), 2u);
    ASSERT_EQ(vectors[0], (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(a_large_request_body_arrives_whole_and_without_waiting_for_100_continue) {
    LoopbackServer server([](const Received&, size_t) { return json(200, OLLAMA_TWO); });
    auto options = optionsFor(server);
    options.batchSize = 2;
    std::string big(3 * 1024 * 1024, 'x');
    auto started = std::chrono::steady_clock::now();
    HttpEncoder(options).encodeBatch({big, "y"});
    auto elapsed = std::chrono::steady_clock::now() - started;
    Received seen = server.request(0);
    ASSERT_EQ(seen.body.size(), big.size() + std::string("{\"model\":\"test-embed\",\"input\":[\"\",\"y\"]}").size());
    ASSERT_EQ(seen.header("expect"), "");
    ASSERT_TRUE(elapsed < std::chrono::milliseconds(900));  // libcurl would wait 1 s for a 100-continue
}

TEST(the_transport_reports_status_body_and_lower_cased_headers) {
    LoopbackServer server([](const Received&, size_t) {
        Reply r = json(429, "{\"error\":\"slow down\"}");
        r.headers.emplace_back("Retry-After", "7");
        r.headers.emplace_back("X-Request-Id", "abc");
        return r;
    });
    HttpRequest request;
    request.url = server.url();
    request.body = "{}";
    request.headers["Content-Type"] = "application/json";
    HttpResponse response = CurlTransport()(request);  // any status is a response, not an error
    ASSERT_EQ(response.status, 429);
    ASSERT_EQ(response.body, "{\"error\":\"slow down\"}");
    ASSERT_EQ(response.headers.at("retry-after"), "7");
    ASSERT_EQ(response.headers.at("x-request-id"), "abc");
}

// =============================================================================
// Failures: statuses, retries, timeouts, refused connections, empty replies
// =============================================================================

TEST(a_client_error_is_an_http_error_with_the_server_text_and_is_not_retried) {
    LoopbackServer server([](const Received&, size_t) {
        return json(404, "{\"error\":{\"message\":\"model not found\"}}");
    });
    auto options = optionsFor(server);
    options.maxRetries = 3;
    HttpEncoder encoder(options);
    try {
        encoder.encode("x");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 404);
        ASSERT_TRUE(contains(error.what(), "HTTP 404"));
        ASSERT_TRUE(contains(error.what(), "model not found"));
    }
    ASSERT_EQ(server.requestCount(), 1u);
}

TEST(a_server_error_is_retried_and_a_later_success_is_returned) {
    LoopbackServer server([](const Received&, size_t number) {
        return number < 2 ? json(503, "{\"error\":\"overloaded\"}") : json(200, OLLAMA_TWO);
    });
    auto options = optionsFor(server);
    options.maxRetries = 3;
    HttpEncoder encoder(options);
    auto vectors = encoder.encodeBatch({"a", "b"});
    ASSERT_EQ(vectors.size(), 2u);
    ASSERT_EQ(server.requestCount(), 3u);
}

TEST(a_persistent_server_error_gives_up_after_the_configured_retries) {
    LoopbackServer server([](const Received&, size_t) { return json(500, "{\"error\":\"boom\"}"); });
    auto options = optionsFor(server);
    options.maxRetries = 2;
    HttpEncoder encoder(options);
    ASSERT_THROWS(encoder.encode("x"), HttpError);
    ASSERT_EQ(server.requestCount(), 3u);
}

TEST(retry_after_from_a_real_response_sets_the_wait) {
    LoopbackServer server([](const Received&, size_t number) {
        if (number > 0) return json(200, OLLAMA_TWO);
        Reply r = json(429, "{\"error\":\"rate limited\"}");
        r.headers.emplace_back("Retry-After", "3");
        return r;
    });
    auto options = optionsFor(server);
    options.maxRetries = 1;
    options.maxBackoffMs = 60000;
    std::vector<long long> waits;
    options.sleeper = [&](std::chrono::milliseconds ms) { waits.push_back(ms.count()); };
    HttpEncoder(options).encodeBatch({"a", "b"});
    ASSERT_EQ(waits.size(), 1u);
    ASSERT_EQ(waits[0], 3000);
}

TEST(a_stalled_server_times_out_within_the_budget) {
    LoopbackServer server([](const Received&, size_t) {
        Reply r;
        r.mode = Reply::Mode::Stall;
        return r;
    });
    auto options = optionsFor(server);
    options.timeoutMs = 300;
    HttpEncoder encoder(options);
    auto started = std::chrono::steady_clock::now();
    try {
        encoder.encode("x");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 0);  // no response at all
        ASSERT_TRUE(contains(lower(error.what()), "time"));
    }
    auto elapsed = std::chrono::steady_clock::now() - started;
    ASSERT_TRUE(elapsed < std::chrono::milliseconds(3000));
}

TEST(a_timeout_is_retried_like_any_transport_failure) {
    LoopbackServer server([](const Received&, size_t number) {
        if (number == 0) {
            Reply r;
            r.mode = Reply::Mode::Stall;
            return r;
        }
        return json(200, OLLAMA_TWO);
    });
    auto options = optionsFor(server);
    options.timeoutMs = 300;
    options.maxRetries = 1;
    auto vectors = HttpEncoder(options).encodeBatch({"a", "b"});
    ASSERT_EQ(vectors.size(), 2u);
    ASSERT_EQ(server.requestCount(), 2u);
}

TEST(a_refused_connection_is_an_http_error) {
    int deadPort;
    {
        LoopbackServer gone([](const Received&, size_t) { return json(200, "{}"); });
        deadPort = gone.port();
    }  // nothing listens there any more
    HttpEncoderOptions options;
    options.url = "http://127.0.0.1:" + std::to_string(deadPort) + "/v1/embeddings";
    options.model = "m";
    options.dimension = 4;
    options.maxRetries = 0;
    options.timeoutMs = 2000;
    HttpEncoder encoder(options);
    try {
        encoder.encode("hello");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 0);
        ASSERT_TRUE(contains(error.what(), "HTTP request failed"));
    }
}

TEST(a_connection_closed_without_a_reply_is_an_http_error) {
    LoopbackServer server([](const Received&, size_t) {
        Reply r;
        r.mode = Reply::Mode::NoReply;
        return r;
    });
    HttpEncoder encoder(optionsFor(server));
    try {
        encoder.encode("x");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 0);
    }
}

TEST(https_to_a_plain_http_port_is_a_transport_error_that_does_not_leak_the_key) {
    LoopbackServer server([](const Received&, size_t) { return json(200, OLLAMA_TWO); });
    auto options = optionsFor(server);
    options.url = "https://127.0.0.1:" + std::to_string(server.port()) + "/v1/embeddings";
    options.apiKey = KEY;
    options.timeoutMs = 1500;  // the plain server never answers a TLS hello; some TLS stacks wait
    try {
        HttpEncoder(options).encode("x");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 0);
        ASSERT_FALSE(contains(error.what(), KEY));
    }
}

TEST(a_malformed_success_body_is_an_embedding_error_not_a_crash) {
    LoopbackServer server([](const Received&, size_t) { return json(200, "{\"data\": [ not json"); });
    HttpEncoder encoder(optionsFor(server));
    ASSERT_THROWS(encoder.encode("x"), EmbeddingError);
    LoopbackServer other([](const Received&, size_t) { return json(200, "{\"unexpected\":true}"); });
    ASSERT_THROWS(HttpEncoder(optionsFor(other)).encode("x"), EmbeddingError);
}

TEST(a_response_body_past_the_cap_is_refused_with_a_clear_message) {
    LoopbackServer server([](const Received&, size_t) {
        Reply r = json(200, "");
        r.mode = Reply::Mode::Huge;
        r.hugeBytes = 65u * 1024u * 1024u;  // one MiB over the transport's 64 MiB limit
        return r;
    });
    HttpRequest request;
    request.url = server.url();
    request.body = "{}";
    request.timeoutMs = 20000;
    try {
        CurlTransport()(request);
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_TRUE(contains(error.what(), "64 MiB"));
    }
}

// =============================================================================
// Safety: redirects, protocols, and the API key
// =============================================================================

TEST(a_redirect_is_not_followed_and_the_key_is_never_forwarded) {
    LoopbackServer elsewhere([](const Received&, size_t) { return json(200, OLLAMA_TWO); });
    std::string target = elsewhere.url("/stolen");
    LoopbackServer origin([&](const Received&, size_t) {
        Reply r = json(302, "");
        r.headers.emplace_back("Location", target);
        return r;
    });
    auto options = optionsFor(origin);
    options.apiKey = KEY;
    HttpEncoder encoder(options);
    try {
        encoder.encode("x");
        throw std::runtime_error("expected an HttpError");
    } catch (const HttpError& error) {
        ASSERT_EQ(error.status(), 302);
        ASSERT_FALSE(contains(error.what(), KEY));
    }
    ASSERT_EQ(origin.requestCount(), 1u);
    ASSERT_EQ(elsewhere.requestCount(), 0u);  // never contacted

    // and at the transport level the 302 is simply returned
    HttpRequest request;
    request.url = origin.url();
    request.body = "{}";
    request.headers["Authorization"] = std::string("Bearer ") + KEY;
    HttpResponse response = CurlTransport()(request);
    ASSERT_EQ(response.status, 302);
    ASSERT_EQ(elsewhere.requestCount(), 0u);
}

TEST(only_http_and_https_are_allowed) {
    HttpRequest request;
    request.url = "file:///etc/hostname";
    request.body = "{}";
    ASSERT_THROWS(CurlTransport()(request), HttpError);
    request.url = "ftp://127.0.0.1:1/x";
    ASSERT_THROWS(CurlTransport()(request), HttpError);
    request.url = "gopher://127.0.0.1:1/x";
    ASSERT_THROWS(CurlTransport()(request), HttpError);
    request.method = "GET";
    request.url = "http://127.0.0.1:1/";
    ASSERT_THROWS(CurlTransport()(request), HttpError);
}

TEST(the_api_key_never_appears_in_any_error) {
    // 1. an error page that echoes the credential back, in each shape the encoder reads
    const std::string echoed = std::string("invalid api key ") + KEY;
    std::vector<std::string> bodies = {
        "{\"error\":\"" + echoed + "\"}",
        "{\"error\":{\"message\":\"" + echoed + "\"}}",
        "{\"message\":\"" + echoed + "\"}",
        "<html>" + echoed + "</html>",
    };
    for (const auto& body : bodies) {
        LoopbackServer server([&](const Received&, size_t) { return json(401, body); });
        auto options = optionsFor(server);
        options.apiKey = KEY;
        try {
            HttpEncoder(options).encode("x");
            throw std::runtime_error("expected an HttpError");
        } catch (const HttpError& error) {
            ASSERT_EQ(error.status(), 401);
            ASSERT_FALSE(contains(error.what(), KEY));
            ASSERT_TRUE(contains(error.what(), "[redacted]"));
        }
    }

    // 2. a 200 whose error object echoes it
    {
        LoopbackServer server([&](const Received&, size_t) {
            return json(200, "{\"error\":{\"message\":\"" + echoed + "\"}}");
        });
        auto options = optionsFor(server);
        options.apiKey = KEY;
        try {
            HttpEncoder(options).encode("x");
            throw std::runtime_error("expected an error");
        } catch (const EmbeddingError& error) {
            ASSERT_FALSE(contains(error.what(), KEY));
        }
    }

    // 3. transport-level failures: the key travels in a header and a header is
    //    never part of libcurl's error text
    {
        LoopbackServer server([](const Received&, size_t) {
            Reply r;
            r.mode = Reply::Mode::NoReply;
            return r;
        });
        auto options = optionsFor(server);
        options.apiKey = KEY;
        try {
            HttpEncoder(options).encode("x");
            throw std::runtime_error("expected an HttpError");
        } catch (const HttpError& error) {
            ASSERT_FALSE(contains(error.what(), KEY));
        }
        ASSERT_EQ(server.request(0).header("authorization"), std::string("Bearer ") + KEY);  // it was sent...
    }
    {
        LoopbackServer stall([](const Received&, size_t) {
            Reply r;
            r.mode = Reply::Mode::Stall;
            return r;
        });
        auto options = optionsFor(stall);
        options.apiKey = KEY;
        options.timeoutMs = 200;
        try {
            HttpEncoder(options).encode("x");
            throw std::runtime_error("expected an HttpError");
        } catch (const HttpError& error) {
            ASSERT_FALSE(contains(error.what(), KEY));  // ...and still not in the timeout text
        }
    }

    // 4. a key that is also part of the URL (a query-string credential) is scrubbed from curl's text
    {
        int deadPort;
        {
            LoopbackServer gone([](const Received&, size_t) { return json(200, "{}"); });
            deadPort = gone.port();
        }
        HttpEncoderOptions options;
        options.url = "http://127.0.0.1:" + std::to_string(deadPort) + "/v1/embeddings?key=" + KEY;
        options.model = "m";
        options.apiKey = KEY;
        options.maxRetries = 0;
        options.timeoutMs = 1000;
        try {
            HttpEncoder(options).encode("x");
            throw std::runtime_error("expected an HttpError");
        } catch (const HttpError& error) {
            ASSERT_FALSE(contains(error.what(), KEY));
        }
    }
}

TEST(the_encoder_can_drive_a_semantic_graph_over_real_http) {
    LoopbackServer server([](const Received& request, size_t) {
        // one vector per input: inputs are the strings of the "input" array
        // (none of these texts contains a comma)
        size_t start = request.body.find("\"input\":[");
        size_t count = 1;
        for (size_t at = request.body.find("\",\"", start); at != std::string::npos;
             at = request.body.find("\",\"", at + 1)) {
            ++count;
        }
        std::string reply = "{\"data\":[";
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) reply += ",";
            reply += "{\"index\":" + std::to_string(i) + ",\"embedding\":[" +
                     (i % 2 == 0 ? "1.0,0.0,0.0,0.0" : "0.0,1.0,0.0,0.0") + "]}";
        }
        return json(200, reply + "]}");
    });
    auto options = optionsFor(server);
    auto encoder = std::make_shared<HttpEncoder>(options);
    SemanticGraph graph("remote", encoder);
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.addNode("person", "2", {{"name", "Bob"}});
    ASSERT_EQ(graph.embeddingStore().count(), 2u);
    ASSERT_EQ(graph.getEmbedding(NodeRef("person", "1"))->model, "test-embed");
}

int main() {
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
