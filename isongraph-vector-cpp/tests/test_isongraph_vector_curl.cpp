/**
 * libcurl transport smoke test. Built only with -DISONGRAPH_VECTOR_CURL=ON.
 *
 * No network: the request goes to a port on the loopback interface where
 * nothing listens, so what is checked is that the transport is wired up and
 * reports a refused connection as an HttpError rather than crashing.
 */

#include <iostream>
#include <string>

#include "../include/isongraph_vector_http.hpp"

using namespace isongraph_vector;

int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            tests_run++; \
            std::cout << "Running: " << #name << "... "; \
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
#define ASSERT_THROWS(expr, exception_type) \
    { \
        bool threw = false; \
        try { expr; } catch (const exception_type&) { threw = true; } \
        if (!threw) throw std::runtime_error("Assertion failed: " #expr " did not throw"); \
    }

TEST(refused_connection_is_an_http_error) {
    HttpRequest request;
    request.url = "http://127.0.0.1:1/v1/embeddings";
    request.body = "{}";
    request.headers["Content-Type"] = "application/json";
    request.timeoutMs = 2000;

    CurlTransport transport;
    ASSERT_THROWS(transport(request), HttpError);
}

TEST(encoder_uses_curl_when_no_transport_is_given) {
    HttpEncoderOptions options;
    options.url = "http://127.0.0.1:1/v1/embeddings";
    options.model = "m";
    options.dimension = 4;
    options.maxRetries = 0;
    options.timeoutMs = 2000;

    HttpEncoder encoder(options);  // would throw "no transport" without ISONGRAPH_VECTOR_CURL
    ASSERT_THROWS(encoder.encode("hello"), HttpError);
}

TEST(rejects_a_non_post_request) {
    HttpRequest request;
    request.method = "GET";
    request.url = "http://127.0.0.1:1/";
    CurlTransport transport;
    ASSERT_THROWS(transport(request), HttpError);
}

int main() {
    std::cout << "\n=== isongraph-vector libcurl transport tests ===" << std::endl << std::endl;
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
