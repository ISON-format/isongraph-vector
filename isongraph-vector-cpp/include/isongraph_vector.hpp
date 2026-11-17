/**
 * ISONGraph Vector - Semantic Graph Extension for C++
 *
 * Extends ISONGraph with embedding-based similarity search and semantic traversal.
 *
 * @example
 * ```cpp
 * #include "isongraph_vector.hpp"
 *
 * using namespace isongraph_vector;
 *
 * auto encoder = std::make_shared<MockEncoder>(384);
 * SemanticGraph graph("semantic", encoder);
 *
 * graph.addNode("person", "1", {{"name", "Alice"}});   // auto-embedded
 * auto results = graph.similaritySearch("engineer", 5, "", -1.0f);
 * ```
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

#ifndef ISONGRAPH_VECTOR_HPP
#define ISONGRAPH_VECTOR_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <limits>
#include <filesystem>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <random>
#include <cstdio>
#include <mutex>
#include "ison_graph.hpp"
#include "isongraph_vector_ivf.hpp"

namespace isongraph_vector {

constexpr const char* VERSION = "1.0.0";

/**
 * Portable embedding payload shared by every language port, so a store built
 * in one can be searched by another.
 */
constexpr const char* EMBEDDING_FORMAT = "ison-embeddings";
constexpr int EMBEDDING_FORMAT_VERSION = 1;

/** Suffix of the file a graph's embeddings are saved to, next to the graph file. */
constexpr const char* EMBEDDING_SIDECAR_SUFFIX = ".embeddings.json";

/**
 * Name of the serialization graphDigest() hashes. Written next to the digest so
 * a reader that computes a different digest (or none) can tell it is not
 * comparable and skip the check instead of reporting a false mismatch.
 */
constexpr const char* GRAPH_DIGEST_BASIS = "nodes-edges-json-v1";

/**
 * Model names that carry no information about which encoder made a vector:
 * "provided" (vector handed in, store has no encoder) and "unknown" (sidecars
 * from ports that do not record a model, encoders that do not name theirs).
 */
inline bool isUnknownModel(const std::string& model) {
    return model == "unknown" || model == "provided";
}

using namespace ison_graph;

// =============================================================================
// Errors
// =============================================================================

/** Base class for embedding-specific failures. */
class EmbeddingError : public std::runtime_error {
public:
    explicit EmbeddingError(const std::string& what) : std::runtime_error(what) {}
};

/**
 * A vector was stored or queried with the wrong number of dimensions.
 *
 * Cosine similarity over a mismatched pair returns a plausible-looking number
 * rather than an error, so a store that quietly accepted mixed widths would
 * degrade silently: a 2-dim vector used to score 0.06 against a 384-dim query
 * instead of failing. A store pins its dimension to the first vector it sees.
 */
class DimensionMismatchError : public EmbeddingError {
public:
    explicit DimensionMismatchError(const std::string& what) : EmbeddingError(what) {}
};

/** Text was handed to a store with no encoder to embed it with. */
class NoEncoderError : public EmbeddingError {
public:
    explicit NoEncoderError(const std::string& what) : EmbeddingError(what) {}
};

/**
 * A text query was encoded by a different model than the stored vectors.
 *
 * Two models of the same dimension give a plausible-looking cosine score
 * between vectors that live in unrelated spaces, so the ranking is noise
 * rather than an error. Thrown only with ModelMismatchMode::Error; the default
 * is a warning.
 */
class ModelMismatchError : public EmbeddingError {
public:
    explicit ModelMismatchError(const std::string& what) : EmbeddingError(what) {}
};

/**
 * A graph file and its embeddings sidecar are not from the same save.
 *
 * Thrown by SemanticGraph::load when the sidecar's recorded graph_digest does
 * not match the graph that was read - the sidecar came from a different save,
 * or the graph file was edited afterwards.
 */
class GraphEmbeddingMismatchError : public EmbeddingError {
public:
    explicit GraphEmbeddingMismatchError(const std::string& what) : EmbeddingError(what) {}
};

/** What a text query does when the encoder's model differs from the stored vectors'. */
enum class ModelMismatchMode {
    Warn,    ///< one warning per distinct (encoder model, stored models) pair per store
    Error,   ///< throw ModelMismatchError
    Ignore   ///< say nothing
};

/** What SemanticGraph::load does when the sidecar's graph digest does not match the graph. */
enum class GraphMismatchMode {
    Error,    ///< throw GraphEmbeddingMismatchError (default)
    Warn,     ///< warn, then load the sidecar's vectors anyway
    Reembed   ///< discard the sidecar's vectors and re-embed every node with the encoder
};

// =============================================================================
// Warnings
// =============================================================================

/** Receives the text of a warning. */
using WarningHandler = std::function<void(const std::string&)>;

namespace detail {
inline WarningHandler& warningHandlerSlot() {
    static WarningHandler handler;
    return handler;
}
}  // namespace detail

/**
 * Route this library's warnings (model mismatch, sidecar digest mismatch with
 * GraphMismatchMode::Warn, encoder version drift) to `handler`. Pass an empty
 * function to restore the default, which writes one line to stderr. Process
 * wide; not synchronised, so set it before the library is used concurrently.
 */
inline void setWarningHandler(WarningHandler handler) {
    detail::warningHandlerSlot() = std::move(handler);
}

/** Deliver a warning to the configured handler (stderr by default). */
inline void emitWarning(const std::string& message) {
    const WarningHandler& handler = detail::warningHandlerSlot();
    if (handler) {
        handler(message);
    } else {
        std::cerr << "isongraph_vector warning: " << message << std::endl;
    }
}

// =============================================================================
// Embedding Encoder Interface
// =============================================================================

class EmbeddingEncoder {
public:
    virtual ~EmbeddingEncoder() = default;
    virtual size_t dimension() const = 0;
    virtual std::vector<float> encode(const std::string& text) const = 0;

    /**
     * Name of the model behind this encoder, stamped on every vector it
     * writes. It decides whether stored vectors are *stale* for this encoder
     * and whether a text query warns about a model mismatch. The default,
     * "unknown", opts out of both.
     */
    virtual std::string modelName() const { return "unknown"; }

    /**
     * Free-form encoder/model version (a model revision, a weights hash),
     * recorded in the sidecar. nullopt when unknown.
     */
    virtual std::optional<std::string> version() const { return std::nullopt; }

    virtual std::vector<std::vector<float>> encodeBatch(const std::vector<std::string>& texts) const {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(encode(text));
        }
        return results;
    }
};

// =============================================================================
// Mock Encoder
// =============================================================================

/**
 * Deterministic encoder with no dependencies, for tests and offline work.
 *
 * The same text produces the same vector in *every* language port
 * (Python/TypeScript/JavaScript/Rust/C++), so a store built by one can be
 * searched by another: a 32-bit FNV-1a hash over the UTF-8 bytes seeds an
 * xorshift32 generator, and the top 24 bits of each state become a value in
 * [-1, 1).
 *
 * This replaces a scheme that emitted the *low* 16 bits of an LCG. The low
 * bits of an LCG evolve independently of the high bits, so every vector was
 * determined by `seed mod 65536` - with exact uint32_t arithmetic that meant
 * 280 outright collisions in 5,000 distinct texts, with colliding texts
 * scoring a perfect 1.0 against each other.
 *
 * These are pseudo-embeddings with no semantic structure. Use them to test
 * plumbing, never to judge relevance quality.
 */
class MockEncoder : public EmbeddingEncoder {
private:
    static constexpr uint32_t FNV_OFFSET_BASIS = 0x811c9dc5u;
    static constexpr uint32_t FNV_PRIME = 0x01000193u;

    size_t dim_;
    std::string modelName_;
    std::optional<std::string> version_;

public:
    explicit MockEncoder(size_t dimension = 384,
                         std::optional<std::string> version = std::nullopt)
        : dim_(dimension), modelName_("mock-" + std::to_string(dimension)),
          version_(std::move(version)) {
        if (dimension == 0) {
            throw EmbeddingError("dimension must be positive");
        }
    }

    size_t dimension() const override {
        return dim_;
    }

    /** "mock-<dimension>" unless renamed with setModelName. */
    std::string modelName() const override { return modelName_; }
    std::optional<std::string> version() const override { return version_; }

    /** Rename the model (to simulate switching models of the same dimension). */
    void setModelName(std::string name) { modelName_ = std::move(name); }
    void setVersion(std::optional<std::string> version) { version_ = std::move(version); }

    std::vector<float> encode(const std::string& text) const override {
        uint32_t state = FNV_OFFSET_BASIS;
        for (unsigned char byte : text) {
            state = (state ^ static_cast<uint32_t>(byte)) * FNV_PRIME;
        }
        // xorshift32 has no way out of a zero state; steer off it.
        if (state == 0) {
            state = 0x9e3779b9u;
        }

        std::vector<float> values;
        values.reserve(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Top 24 bits -> [-1, 1)
            values.push_back(static_cast<float>(state >> 8) / 8388608.0f - 1.0f);
        }

        return values;
    }
};

// =============================================================================
// Embedding Record
// =============================================================================

struct EmbeddingRecord {
    std::string id;
    NodeRef node_ref;
    std::vector<float> vector;
    std::string text;
    /** Model that produced the vector: the encoder's modelName(), "provided" or "unknown". */
    std::string model = "unknown";
    /**
     * Internal: this record's row in the approximate index, valid only while
     * an index exists (see EmbeddingStore::enableAnn). Not part of the data;
     * never saved, copied between stores or compared.
     */
    mutable uint32_t ann_slot = detail::IvfIndex::kNoSlot;
};

// =============================================================================
// Similarity Result
// =============================================================================

struct SimilarityResult {
    NodeRef node_ref;
    float score;
    std::string text;

    bool operator<(const SimilarityResult& other) const {
        return score > other.score;  // Higher score first
    }
};

// =============================================================================
// Portable payload
// =============================================================================

/** One entry of the portable embedding payload. */
struct ExportedEmbedding {
    std::string type;
    std::string id;
    std::string id_type = "str";
    std::string text;
    std::vector<float> vector;
    /** Model that produced the vector ("unknown" when the source records none). */
    std::string model = "unknown";
};

/**
 * Optional metadata appended after `embeddings` in the payload. Absent values
 * are omitted, never written as null, and readers ignore fields they do not
 * know, so the payload stays version 1.
 */
struct EmbeddingMeta {
    std::optional<std::string> encoderModel;
    std::optional<std::string> encoderVersion;
    std::optional<std::string> graphDigest;
    std::optional<std::string> graphDigestBasis;
};

/** Payload written by exportEmbeddings, readable by every language port. */
struct ExportedEmbeddings {
    std::string format = EMBEDDING_FORMAT;
    int version = EMBEDDING_FORMAT_VERSION;
    std::optional<size_t> dimension;
    size_t count = 0;
    std::vector<ExportedEmbedding> embeddings;
    EmbeddingMeta meta;
};

// =============================================================================
// Internal helpers: JSON, SHA-256, atomic writes, graph digest
// =============================================================================

namespace detail {

/**
 * Parsed JSON, just enough for the embeddings payload and sidecar metadata. An array made only
 * of numbers keeps them in `numbers` rather than as one node each: a
 * store of 100k 384-dim vectors would otherwise build tens of millions of
 * tree nodes to read what is a flat run of doubles.
 */
struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;  // string value, or the literal text of a number
    std::vector<JsonValue> items;
    std::vector<double> numbers;
    std::vector<std::string> keys;
    std::vector<JsonValue> values;

    const JsonValue* find(const std::string& key) const {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == key) return &values[i];
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& source) : src_(source) {}

    JsonValue parse() {
        // tolerate the UTF-8 byte order mark some Windows tools prepend
        if (src_.compare(0, 3, "\xEF\xBB\xBF") == 0) pos_ = 3;
        JsonValue value = parseValue(0);
        skipSpace();
        if (pos_ != src_.size()) fail("unexpected trailing content");
        return value;
    }

private:
    static constexpr int MAX_DEPTH = 64;
    const std::string& src_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw EmbeddingError("malformed JSON at offset " +
                             std::to_string(pos_) + ": " + what);
    }

    void skipSpace() {
        while (pos_ < src_.size() &&
               (src_[pos_] == ' ' || src_[pos_] == '\t' ||
                src_[pos_] == '\n' || src_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool consume(char c) {
        skipSpace();
        if (pos_ < src_.size() && src_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expectLiteral(const char* literal) {
        size_t length = std::char_traits<char>::length(literal);
        if (src_.compare(pos_, length, literal) != 0) fail("invalid literal");
        pos_ += length;
    }

    double parseNumber(std::string* literal) {
        size_t start = pos_;
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
                c == 'e' || c == 'E') {
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) fail("expected a value");
        std::string token = src_.substr(start, pos_ - start);
        char* end = nullptr;
        double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) fail("invalid number");
        if (literal != nullptr) *literal = std::move(token);
        return value;
    }

    static void appendUtf8(std::string& out, unsigned long cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    unsigned long parseHex4() {
        if (pos_ + 4 > src_.size()) fail("truncated unicode escape");
        unsigned long value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = src_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned long>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned long>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned long>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }

    std::string parseString() {
        // the opening quote has been consumed
        std::string out;
        while (true) {
            if (pos_ >= src_.size()) fail("unterminated string");
            char c = src_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= src_.size()) fail("unterminated escape");
            char esc = src_[pos_++];
            switch (esc) {
                case '"':  out += '"'; break;
                case '\\': out += '\\'; break;
                case '/':  out += '/'; break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    unsigned long cp = parseHex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // a high surrogate pairs with the escape after it
                        if (pos_ + 1 < src_.size() && src_[pos_] == '\\' &&
                            src_[pos_ + 1] == 'u') {
                            size_t save = pos_;
                            pos_ += 2;
                            unsigned long low = parseHex4();
                            if (low >= 0xDC00 && low <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            } else {
                                pos_ = save;
                                cp = 0xFFFD;
                            }
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("invalid escape");
            }
        }
    }

    JsonValue parseValue(int depth) {
        if (depth > MAX_DEPTH) fail("nesting too deep");
        skipSpace();
        if (pos_ >= src_.size()) fail("unexpected end of input");

        JsonValue value;
        char c = src_[pos_];
        if (c == '{') {
            ++pos_;
            value.kind = JsonValue::Kind::Object;
            if (consume('}')) return value;
            while (true) {
                if (!consume('"')) fail("expected a string key");
                value.keys.push_back(parseString());
                if (!consume(':')) fail("expected ':'");
                value.values.push_back(parseValue(depth + 1));
                if (consume(',')) continue;
                if (consume('}')) return value;
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++pos_;
            value.kind = JsonValue::Kind::Array;
            if (consume(']')) return value;
            while (true) {
                skipSpace();
                bool numeric = pos_ < src_.size() &&
                               ((src_[pos_] >= '0' && src_[pos_] <= '9') || src_[pos_] == '-');
                if (numeric && value.items.empty()) {
                    value.numbers.push_back(parseNumber(nullptr));
                } else {
                    if (!value.numbers.empty()) {
                        // a mixed array: fall back to one node per element
                        for (double n : value.numbers) {
                            JsonValue number;
                            number.kind = JsonValue::Kind::Number;
                            number.number = n;
                            value.items.push_back(std::move(number));
                        }
                        value.numbers.clear();
                    }
                    value.items.push_back(parseValue(depth + 1));
                }
                if (consume(',')) continue;
                if (consume(']')) return value;
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            ++pos_;
            value.kind = JsonValue::Kind::String;
            value.text = parseString();
            return value;
        }
        if (c == 't') { expectLiteral("true");  value.kind = JsonValue::Kind::Bool; value.boolean = true; return value; }
        if (c == 'f') { expectLiteral("false"); value.kind = JsonValue::Kind::Bool; return value; }
        if (c == 'n') { expectLiteral("null"); return value; }

        value.kind = JsonValue::Kind::Number;
        value.number = parseNumber(&value.text);
        return value;
    }
};


// -----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained so the graph digest needs no dependency.
// -----------------------------------------------------------------------------

inline std::string sha256Hex(const std::string& data) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    std::string padded = data;
    padded.push_back(static_cast<char>(0x80));
    while (padded.size() % 64 != 56) padded.push_back('\0');
    const uint64_t bitLength = static_cast<uint64_t>(data.size()) * 8u;
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<char>((bitLength >> shift) & 0xFFu));
    }

    auto rotr = [](uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); };

    for (size_t offset = 0; offset < padded.size(); offset += 64) {
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(padded.data()) + offset + i * 4;
            w[i] = (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
                   (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (size_t i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    static const char* const HEX = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : h) {
        for (int shift = 28; shift >= 0; shift -= 4) out.push_back(HEX[(word >> shift) & 0xFu]);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Atomic file writes
// -----------------------------------------------------------------------------

/**
 * Write `text` to `path` so a reader sees the old file or the new one.
 *
 * The bytes go to a temp file in the same directory (same filesystem, so the
 * rename is not a copy), are flushed, and are then renamed over the target
 * (std::filesystem::rename replaces an existing file on Windows as well as on
 * POSIX). A failure leaves the previous file intact and removes the temp file.
 * The text is written byte for byte: no newline translation on any platform.
 * `Err` is the exception type thrown on failure.
 */
template <typename Err>
inline void atomicWriteText(const std::string& path, const std::string& text) {
    namespace fs = std::filesystem;
    static std::atomic<unsigned long long> counter{0};

    fs::path target(path);
    fs::path dir = target.parent_path();
    std::random_device rd;
    std::string tmpName = "." + target.filename().string() + "." +
        std::to_string(static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count())) + "." +
        std::to_string(rd()) + "." + std::to_string(counter++) + ".tmp";
    fs::path tmp = dir.empty() ? fs::path(tmpName) : dir / tmpName;

    try {
        {
            std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
            if (!file) throw Err("cannot open file for writing: " + path);
            file.write(text.data(), static_cast<std::streamsize>(text.size()));
            file.flush();
            if (!file) throw Err("failed writing: " + path);
        }
        std::error_code ec;
        fs::rename(tmp, target, ec);
        if (ec) throw Err("cannot replace " + path + ": " + ec.message());
    } catch (...) {
        std::error_code ignored;
        fs::remove(tmp, ignored);
        throw;
    }
}

// -----------------------------------------------------------------------------
// Graph digest: canonical JSON rendering, Python json.dumps compatible
// -----------------------------------------------------------------------------

/** Append `value` as a JSON string the way Python's json.dumps(ensure_ascii=True) writes it. */
inline void appendPyString(std::string& out, const std::string& value) {
    static const char* const HEX = "0123456789abcdef";
    auto unicodeEscape = [&](uint32_t unit) {
        out += "\\u";
        out += HEX[(unit >> 12) & 0xF];
        out += HEX[(unit >> 8) & 0xF];
        out += HEX[(unit >> 4) & 0xF];
        out += HEX[unit & 0xF];
    };

    out += '"';
    const size_t n = value.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        auto cont = [&](size_t k) {
            return i + k < n && (static_cast<unsigned char>(value[i + k]) & 0xC0) == 0x80;
        };
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0 && cont(1)) {
            uint32_t v = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(value[i + 1]) & 0x3Fu);
            if (v >= 0x80) { cp = v; len = 2; }
        } else if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
            uint32_t v = ((c & 0x0Fu) << 12) |
                         ((static_cast<unsigned char>(value[i + 1]) & 0x3Fu) << 6) |
                         (static_cast<unsigned char>(value[i + 2]) & 0x3Fu);
            if (v >= 0x800) { cp = v; len = 3; }
        } else if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
            uint32_t v = ((c & 0x07u) << 18) |
                         ((static_cast<unsigned char>(value[i + 1]) & 0x3Fu) << 12) |
                         ((static_cast<unsigned char>(value[i + 2]) & 0x3Fu) << 6) |
                         (static_cast<unsigned char>(value[i + 3]) & 0x3Fu);
            if (v >= 0x10000 && v <= 0x10FFFF) { cp = v; len = 4; }
        }
        i += len;

        if (cp == '"') out += "\\\"";
        else if (cp == '\\') out += "\\\\";
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (cp == '\t') out += "\\t";
        else if (cp == '\b') out += "\\b";
        else if (cp == '\f') out += "\\f";
        else if (cp >= 0x20 && cp <= 0x7E) out += static_cast<char>(cp);
        else if (cp < 0x10000) unicodeEscape(cp);
        else {
            uint32_t v = cp - 0x10000;
            unicodeEscape(0xD800 + (v >> 10));
            unicodeEscape(0xDC00 + (v & 0x3FF));
        }
    }
    out += '"';
}

/** Python's repr(float), which is what json.dumps writes for a float. */
inline std::string pyFloatRepr(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value > 0 ? "Infinity" : "-Infinity";

    char buffer[64];
    auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                   std::chars_format::scientific);
    if (ec != std::errc()) return "0.0";
    std::string sci(buffer, end);  // shortest round-trip digits, e.g. "-1.2345e+02"

    bool negative = false;
    if (!sci.empty() && sci[0] == '-') { negative = true; sci.erase(0, 1); }
    const size_t epos = sci.find('e');
    const int exp10 = std::atoi(sci.c_str() + epos + 1);
    std::string digits;
    for (size_t i = 0; i < epos; ++i) {
        if (sci[i] != '.') digits.push_back(sci[i]);
    }

    std::string out = negative ? "-" : "";
    if (exp10 < -4 || exp10 >= 16) {
        out += digits[0];
        if (digits.size() > 1) { out += '.'; out.append(digits, 1, std::string::npos); }
        out += 'e';
        out += exp10 < 0 ? '-' : '+';
        std::string magnitude = std::to_string(exp10 < 0 ? -exp10 : exp10);
        if (magnitude.size() < 2) magnitude.insert(0, "0");
        out += magnitude;
    } else if (exp10 >= 0) {
        const size_t intDigits = static_cast<size_t>(exp10) + 1;
        if (digits.size() <= intDigits) {
            out += digits;
            out.append(intDigits - digits.size(), '0');
            out += ".0";
        } else {
            out.append(digits, 0, intDigits);
            out += '.';
            out.append(digits, intDigits, std::string::npos);
        }
    } else {
        out += "0.";
        out.append(static_cast<size_t>(-exp10 - 1), '0');
        out += digits;
    }
    return out;
}

/** Append one ISON value as Python's json.dumps(..., default=str) writes the equivalent value. */
inline void appendPyValue(std::string& out, const ison::Value& value) {
    switch (value.type()) {
        case ison::ValueType::Null: out += "null"; break;
        case ison::ValueType::Bool: out += value.as_bool() ? "true" : "false"; break;
        case ison::ValueType::Int: out += std::to_string(value.as_int()); break;
        case ison::ValueType::Float: out += pyFloatRepr(value.as_float()); break;
        case ison::ValueType::String: appendPyString(out, value.as_string()); break;
        case ison::ValueType::Reference: {
            // json.dumps(default=str) on the Python parser's Reference object.
            const ison::Reference& ref = *value.as_reference_ptr();
            std::string text = "Reference(";
            if (ref.type.has_value() && !ref.type.value().empty()) text += ref.type.value() + ":";
            text += ref.id + ")";
            appendPyString(out, text);
            break;
        }
    }
}

/** Append a property bag as a key-sorted compact JSON object. */
inline void appendPyProperties(std::string& out, const Properties& properties) {
    out += '{';
    bool first = true;
    for (const auto& [key, value] : properties) {  // std::map: already key-sorted
        if (!first) out += ',';
        first = false;
        appendPyString(out, key);
        out += ':';
        appendPyValue(out, value);
    }
    out += '}';
}

/** `["type","i"|"s","id"]` - ids are tagged by kind so int 7 and string "7" hash differently. */
inline std::string digestRef(const std::string& type, bool integerId, const std::string& id) {
    std::string out = "[";
    appendPyString(out, type);
    out += integerId ? ",\"i\"," : ",\"s\",";
    appendPyString(out, id);
    out += ']';
    return out;
}

inline std::string digestNode(const std::string& type, bool integerId, const std::string& id,
                              const Properties& properties) {
    std::string out = "[" + digestRef(type, integerId, id) + ",";
    appendPyProperties(out, properties);
    out += ']';
    return out;
}

inline std::string digestEdge(const std::string& relType, const std::string& source,
                              const std::string& target, const Properties& properties) {
    std::string out = "[";
    appendPyString(out, relType);
    out += "," + source + "," + target + ",";
    appendPyProperties(out, properties);
    out += ']';
    return out;
}

inline std::string finishDigest(bool directed, std::vector<std::string>& nodes,
                                std::vector<std::string>& edges) {
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    std::string body = std::string("{\"directed\":") + (directed ? "true" : "false") + ",\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i > 0) body += ',';
        body += edges[i];
    }
    body += "],\"nodes\":[";
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i > 0) body += ',';
        body += nodes[i];
    }
    body += "]}";
    return "sha256:" + sha256Hex(body);
}

inline bool allAsciiDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

/**
 * Digest of a graph as it reads back from ISON text, i.e. what the Python
 * port's graph_digest(ISONGraph.from_ison(text)) computes.
 *
 * Computed from the parsed document rather than from an ISONGraph for two
 * reasons the text knows and the graph has lost: whether a node id was an
 * integer (`7`) or a string (`"7"`), and which cells were `null` (Python keeps
 * those as None properties, ISONGraph drops them). Edge endpoints are tagged
 * the way Python resolves a reference: against the loaded nodes, preferring
 * the string form.
 */
inline std::string digestOfDocument(const ison::Document& doc) {
    std::set<std::pair<std::string, std::string>> intNodes, strNodes;
    std::vector<std::string> nodes, edges;

    for (const ison::Block& block : doc.blocks) {
        if (block.kind != "nodes") continue;
        for (const ison::Row& row : block.rows) {
            auto idIt = row.find("id");
            if (idIt == row.end()) continue;
            const ison::Value& idValue = idIt->second;
            bool isInt = idValue.is_int();
            std::string id;
            if (!ison_graph::ISONGraph::idText(idValue, id)) continue;
            (isInt ? intNodes : strNodes).insert({block.name, id});

            Properties props;
            for (const auto& [key, value] : row) {
                if (key != "id") props[key] = value;
            }
            nodes.push_back(digestNode(block.name, isInt, id, props));
        }
    }

    auto resolve = [&](const ison::Reference& ref) {
        const std::string type = ref.type.value_or("");
        bool integer;
        if (strNodes.count({type, ref.id})) integer = false;
        else if (allAsciiDigits(ref.id) && intNodes.count({type, ref.id})) integer = true;
        else integer = allAsciiDigits(ref.id);
        return digestRef(type, integer, ref.id);
    };

    for (const ison::Block& block : doc.blocks) {
        if (block.kind != "edges") continue;
        for (const ison::Row& row : block.rows) {
            auto s = row.find("source");
            auto t = row.find("target");
            if (s == row.end() || t == row.end() || !s->second.is_reference() ||
                !t->second.is_reference()) {
                continue;
            }
            Properties props;
            for (const auto& [key, value] : row) {
                if (key != "source" && key != "target") props[key] = value;
            }
            edges.push_back(digestEdge(block.name, resolve(*s->second.as_reference_ptr()),
                                       resolve(*t->second.as_reference_ptr()), props));
        }
    }
    return finishDigest(true, nodes, edges);
}

}  // namespace detail

/**
 * Content digest of a graph's nodes and edges: "sha256:<hex>".
 *
 * Hashes a canonical JSON rendering (GRAPH_DIGEST_BASIS): the `directed` flag,
 * every node as `[[type, id-kind, id], properties]` and every edge as
 * `[relType, source, target, properties]`, each list sorted and every object
 * key-sorted, written exactly as Python's
 * `json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True)`
 * would. The same graph therefore hashes the same regardless of insertion
 * order, and the same as the Python port's graph_digest() of the same graph
 * (see tests/fixtures/graph_digest_golden.json). The graph's name and the
 * embeddings are excluded.
 *
 * Node ids are tagged "i" (integer) or "s" (string). Ids are always strings in
 * this port, so by default every id is tagged "s"; pass `integerId` to mark
 * the ids that are integers in the graph you are comparing against (what a
 * Python graph with `add_node('person', 1)` has). SemanticGraph::save and load
 * do not need it: they digest the ISON text, which records which it was.
 */
inline std::string graphDigest(
    const ISONGraph& graph,
    const std::function<bool(const NodeRef&)>& integerId = nullptr
) {
    auto isInt = [&](const NodeRef& ref) { return integerId && integerId(ref); };

    std::vector<std::string> nodes, edges;
    graph.forEachNode([&](const Node& node) {
        nodes.push_back(detail::digestNode(node.type, isInt(node.ref()), node.id, node.properties));
    });
    for (const Edge& edge : graph.getAllEdges()) {
        edges.push_back(detail::digestEdge(
            edge.relType,
            detail::digestRef(edge.source.type, isInt(edge.source), edge.source.id),
            detail::digestRef(edge.target.type, isInt(edge.target), edge.target.id),
            edge.properties));
    }
    return detail::finishDigest(graph.directed, nodes, edges);
}

// =============================================================================
// Approximate search (opt-in)
// =============================================================================

/**
 * Configuration of the opt-in IVF-flat approximate index (see
 * EmbeddingStore::enableAnn). Defaults match the Python port.
 */
struct AnnOptions {
    /** Cells in the index; 0 = automatic, about 4 * sqrt(N) (see autoNlist) when the index is built. */
    size_t nlist = 0;
    /** Cells scanned per query; 0 = automatic, about 5% of nlist. >= nlist is the exact scan. */
    size_t nprobe = 0;
    /** Below this many vectors the index is not used and search stays exact. */
    size_t minSize = 50000;
    /** Seed of the k-means initialisation: the same data gives the same index. */
    uint64_t seed = 0;
    /**
     * Threads used to train and fill the index (0 = hardware concurrency, 1 =
     * none). Queries are always single-threaded. The thread count never
     * changes the resulting index.
     */
    size_t threads = 0;

    /** The automatic nlist for a store of n vectors: round(4 * sqrt(n)) clamped to [16, 65536]. */
    static size_t autoNlist(size_t n) {
        double wanted = std::nearbyint(4.0 * std::sqrt(static_cast<double>(n)));
        return static_cast<size_t>(std::min(std::max(wanted, 16.0), 65536.0));
    }
};

/** What the approximate index looks like right now (EmbeddingStore::annStats). Never builds anything. */
struct AnnStats {
    bool enabled = false;
    /** An index exists and will answer searches. */
    bool active = false;
    size_t nlist = 0;
    size_t nprobe = 0;
    /** Vectors present when the centroids were trained. */
    size_t trainedN = 0;
    /** Rows scored exactly on every query until folded into a cell. */
    size_t pending = 0;
    /** Writes + deletes since training. */
    size_t churn = 0;
    double buildSeconds = 0.0;
    /** Memory held by the index on top of the vectors it scans. */
    size_t indexBytes = 0;
    /** Memory of the contiguous copy of the vectors the index scans. */
    size_t vectorBytes = 0;
    size_t minSize = 0;
    /** Searches answered from the index, from a widened probe, by the full scan, and index builds, since enableAnn. */
    size_t approximateSearches = 0;
    size_t widenedSearches = 0;
    size_t exactSearches = 0;
    size_t builds = 0;
};

/**
 * Recall of the approximate index against the exact scan, over some queries
 * (EmbeddingStore::recallCheck). `recall` is the mean over queries of
 * |approximate top-k & exact top-k| / |exact top-k|; `minRecall` the worst
 * query. `approximate` is false when the index was never consulted (it is
 * off, the store is under minSize, or nprobe >= nlist), in which case recall
 * is 1.0 by construction. Times are per-query medians in milliseconds.
 */
struct RecallReport {
    double recall = 1.0;
    double minRecall = 1.0;
    size_t queries = 0;
    size_t k = 0;
    size_t nlist = 0;
    size_t nprobe = 0;
    bool approximate = false;
    double annMs = 0.0;
    double exactMs = 0.0;
};

namespace detail {

/**
 * Everything the store keeps for approximate search. Copying drops the index
 * (its slot table points into the source's records) and the counters; moving
 * carries the index along, since moving the record map keeps its nodes.
 */
struct AnnState {
    mutable std::mutex mutex;
    std::optional<AnnOptions> options;
    std::unique_ptr<IvfIndex> index;
    std::vector<const EmbeddingRecord*> recs;            // slot -> record, null once dropped
    std::unordered_map<std::string, uint32_t> codes;     // node type -> small integer
    std::vector<IvfHit> hits;                            // per-query scratch
    size_t approximate = 0, widened = 0, exact = 0, builds = 0;

    AnnState() = default;
    AnnState(const AnnState& other) : options(other.options) {}
    AnnState(AnnState&& other) noexcept { takeFrom(other); }
    AnnState& operator=(const AnnState& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(mutex);
            reset();
            options = other.options;
            approximate = widened = exact = builds = 0;
        }
        return *this;
    }
    AnnState& operator=(AnnState&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(mutex);
            takeFrom(other);
        }
        return *this;
    }

    void reset() {
        index.reset();
        recs.clear();
        recs.shrink_to_fit();
        codes.clear();
        hits.clear();
    }

private:
    void takeFrom(AnnState& other) {
        options = std::move(other.options);
        index = std::move(other.index);
        recs = std::move(other.recs);
        codes = std::move(other.codes);
        hits = std::move(other.hits);
        approximate = other.approximate; widened = other.widened;
        exact = other.exact; builds = other.builds;
        other.options.reset();
        other.reset();
    }
};

}  // namespace detail

// =============================================================================
// Embedding Store
// =============================================================================

class EmbeddingStore {
private:
    std::unordered_map<std::string, EmbeddingRecord> embeddings_;
    std::shared_ptr<EmbeddingEncoder> encoder_;
    std::optional<size_t> dimension_;
    std::optional<std::string> version_;
    ModelMismatchMode onModelMismatch_ = ModelMismatchMode::Warn;
    mutable std::set<std::string> warnedMismatch_;
    EmbeddingMeta lastImportMeta_;
    mutable detail::AnnState ann_;

    std::string nodeRefToKey(const NodeRef& ref) const {
        return ref.type + ":" + ref.id;
    }

    /** Model stamped on vectors written now: the encoder's, "provided" with none. */
    std::string currentModelStamp() const {
        return encoder_ ? encoder_->modelName() : "provided";
    }

    void checkDimension(const std::vector<float>& vec, const std::string& what) {
        if (vec.empty()) {
            throw DimensionMismatchError(what + " is empty");
        }
        if (!dimension_.has_value()) {
            dimension_ = vec.size();
        } else if (vec.size() != *dimension_) {
            throw DimensionMismatchError(
                what + " has " + std::to_string(vec.size()) + " dimensions, store holds " +
                std::to_string(*dimension_));
        }
    }

    void checkQuery(const std::vector<float>& vec) const {
        if (vec.empty()) {
            throw DimensionMismatchError("query vector is empty");
        }
        if (dimension_.has_value() && vec.size() != *dimension_) {
            throw DimensionMismatchError(
                "query vector has " + std::to_string(vec.size()) + " dimensions, store holds " +
                std::to_string(*dimension_));
        }
    }

    /** Complain when a text query's encoder is not the stored vectors' model. */
    void checkModelCompat() const {
        if (onModelMismatch_ == ModelMismatchMode::Ignore) return;
        std::optional<std::string> current = currentModel();
        if (!current.has_value()) return;

        std::set<std::string> foreign;  // sorted, which also orders the message
        for (const auto& [key, record] : embeddings_) {
            if (!isUnknownModel(record.model) && record.model != *current) {
                foreign.insert(record.model);
            }
        }
        if (foreign.empty()) return;

        std::string listed;
        for (const std::string& model : foreign) {
            if (!listed.empty()) listed += ", ";
            listed += "'" + model + "'";
        }
        std::string message =
            "encoder model '" + *current + "' differs from the model that produced the "
            "stored vectors (" + listed + "); scores compare vectors from different spaces. "
            "Re-embed with reembedStale() or search with the original encoder.";

        if (onModelMismatch_ == ModelMismatchMode::Error) {
            throw ModelMismatchError(message);
        }
        std::string key = *current + "\n";
        for (const std::string& model : foreign) key += model + "\n";
        if (warnedMismatch_.insert(key).second) {
            emitWarning(message);
        }
    }

    // One out-of-line copy on purpose: the exact scan, the approximate path's
    // re-score and the tests all compare these scores for bit equality, and
    // two inlined copies are free to be contracted or vectorised differently
    // by an optimiser (observed as last-bit differences on GCC 15).
#if defined(__clang__)
    __attribute__((noinline))
#elif defined(__GNUC__)
    __attribute__((noinline, noclone))
#elif defined(_MSC_VER)
    __declspec(noinline)
#endif
    float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) const {
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);

        if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
        return dot / (norm1 * norm2);
    }

    // -- approximate index upkeep (all of it under ann_.mutex) -----------------

    using Ivf = detail::IvfIndex;

    uint32_t annCodeLocked(const std::string& type) const {
        auto it = ann_.codes.find(type);
        if (it != ann_.codes.end()) return it->second;
        uint32_t code = static_cast<uint32_t>(ann_.codes.size());
        ann_.codes.emplace(type, code);
        return code;
    }

    size_t annThreads() const { return ann_.options ? ann_.options->threads : 1; }

    /** Cells probed at the configured nprobe. */
    size_t annProbeLocked(const Ivf& ix) const {
        size_t wanted = ann_.options->nprobe;
        if (wanted == 0) {
            wanted = std::max<size_t>(1, static_cast<size_t>(std::nearbyint(0.05 * static_cast<double>(ix.nlist))));
        }
        return std::min(wanted, ix.nlist);
    }

    /**
     * Cells to probe at widening `scale`; 0 = widened enough that the full scan
     * costs about the same. Scale 1 honours nprobe as configured (nprobe >=
     * nlist is the exact scan); later steps multiply it by 4 and give up on the
     * index once they would touch half the cells.
     */
    size_t annListsLocked(const Ivf& ix, size_t scale) const {
        size_t lists = annProbeLocked(ix) * scale;
        if (scale == 1) return lists >= ix.nlist ? 0 : lists;
        return lists * 2 >= ix.nlist ? 0 : lists;
    }

    void annDropIndexLocked() const { ann_.reset(); }

    void annBuildLocked() const {
        std::vector<const EmbeddingRecord*> recs;
        recs.reserve(embeddings_.size());
        for (const auto& entry : embeddings_) recs.push_back(&entry.second);
        // Slots are numbered by key, not by hash-table order, so the same data
        // gives the same index however (and on whichever standard library) it
        // was inserted.
        std::sort(recs.begin(), recs.end(),
                  [](const EmbeddingRecord* a, const EmbeddingRecord* b) { return a->id < b->id; });

        const size_t n = recs.size();
        const size_t dim = *dimension_;
        std::vector<float> data(n * dim);
        std::vector<uint32_t> codes(n);
        ann_.codes.clear();
        for (size_t i = 0; i < n; ++i) {
            const EmbeddingRecord& r = *recs[i];
            if (r.vector.size() != dim) {
                throw EmbeddingError("cannot index " + r.id + ": it has " +
                                     std::to_string(r.vector.size()) + " dimensions, store holds " +
                                     std::to_string(dim));
            }
            std::copy(r.vector.begin(), r.vector.end(), data.begin() + static_cast<std::ptrdiff_t>(i * dim));
            codes[i] = annCodeLocked(r.node_ref.type);
            r.ann_slot = static_cast<uint32_t>(i);
        }
        size_t nlist = ann_.options->nlist ? ann_.options->nlist : AnnOptions::autoNlist(n);
        ann_.index = std::make_unique<Ivf>(Ivf::build(dim, std::move(data), std::move(codes), nlist,
                                                      ann_.options->seed, annThreads()));
        ann_.recs = std::move(recs);
        ++ann_.builds;
    }

    /**
     * The index for the store as it is now, built or maintained as the policy
     * says; null means "answer exactly". Policy (as in Python):
     *  - built lazily by the first eligible search, never persisted;
     *  - new and updated rows are scored exactly until max(512, 2%) of the
     *    indexed rows are pending, then folded into their nearest cell;
     *  - deletes tombstone; a quarter dead and the arrays are compacted;
     *  - once writes + deletes since training exceed 50% of the trained size
     *    the index is retrained from the current data;
     *  - a store under minSize, a clear() or a dimension change drops it.
     */
    Ivf* annIndexLocked() const {
        if (!ann_.options) return nullptr;
        const size_t live = embeddings_.size();
        if (live < std::max<size_t>(2, ann_.options->minSize) || !dimension_.has_value()) {
            annDropIndexLocked();
            return nullptr;
        }
        if (ann_.index && static_cast<double>(ann_.index->churn) > 0.5 * static_cast<double>(ann_.index->trainedN)) {
            annDropIndexLocked();
        }
        if (!ann_.index) {
            annBuildLocked();
        } else if (static_cast<double>(ann_.index->pendingLive) >=
                   std::max(512.0, 0.02 * static_cast<double>(ann_.index->trainedN))) {
            ann_.index->fold(annThreads());
        }
        return ann_.index.get();
    }

    /** A record was written (inserted or replaced). */
    void annAfterPut(const EmbeddingRecord& record) {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        if (!ann_.index) return;
        uint32_t code = annCodeLocked(record.node_ref.type);
        if (record.ann_slot != Ivf::kNoSlot) {
            ann_.index->overwrite(record.ann_slot, record.vector.data(), code);
            ann_.recs[record.ann_slot] = &record;
        } else {
            record.ann_slot = ann_.index->append(record.vector.data(), code);
            ann_.recs.push_back(&record);
        }
    }

    /** A record is about to be erased. */
    void annBeforeErase(const EmbeddingRecord& record) {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        if (!ann_.index || record.ann_slot == Ivf::kNoSlot) return;
        ann_.index->drop(record.ann_slot);
        ann_.recs[record.ann_slot] = nullptr;
        if (ann_.index->needsCompaction()) {
            std::vector<uint32_t> remap = ann_.index->compact();
            std::vector<const EmbeddingRecord*> kept(ann_.index->slots(), nullptr);
            for (size_t old = 0; old < remap.size(); ++old) {
                if (remap[old] == Ivf::kNoSlot) continue;
                kept[remap[old]] = ann_.recs[old];
                if (kept[remap[old]] != nullptr) kept[remap[old]]->ann_slot = remap[old];
            }
            ann_.recs = std::move(kept);
        }
    }

    /** Insert or replace the record for `key`, keeping the approximate index in step. */
    void putRecord(const std::string& key, EmbeddingRecord&& record) {
        auto it = embeddings_.find(key);
        uint32_t slot = Ivf::kNoSlot;
        if (it == embeddings_.end()) {
            it = embeddings_.emplace(key, std::move(record)).first;
        } else {
            slot = it->second.ann_slot;
            it->second = std::move(record);
        }
        it->second.ann_slot = slot;
        annAfterPut(it->second);
    }

    /**
     * Top-k from the index, or nullopt to use the exact scan (counted as an
     * exact search here). Filters (type, threshold) apply to the probed rows;
     * if fewer than topK survive, the probe is widened 4x and tried again, and
     * past half the cells the exact scan answers instead, so a rare type or a
     * high threshold costs time rather than silently returning a short list.
     * Scores in the result are recomputed with the exact cosine formula, so
     * they equal the exact scan's bit for bit.
     */
    std::optional<std::vector<SimilarityResult>> annSearch(
        const std::vector<float>& query, size_t topK, const std::string& nodeType,
        float threshold) const {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        auto fallBack = [&]() -> std::optional<std::vector<SimilarityResult>> {
            ++ann_.exact;
            return std::nullopt;
        };
        Ivf* ix = annIndexLocked();
        if (ix == nullptr) return fallBack();
        const float norm = std::sqrt(detail::simd::dot(query.data(), query.data(), query.size()));
        if (!(norm > 0.0f)) return fallBack();      // a zero query has no direction to rank cells by
        if (topK >= ix->live()) return fallBack();  // nothing to narrow down

        uint32_t code = Ivf::kAnyCode;
        if (!nodeType.empty()) {
            auto found = ann_.codes.find(nodeType);
            if (found == ann_.codes.end()) return std::vector<SimilarityResult>();  // no such type at all
            code = found->second;
        }
        // The fast scores differ from the exact ones in the last bits: admit
        // rows that close to the threshold and re-check them below.
        const float slack = threshold - 1e-5f;
        for (size_t scale = 1;; scale *= 4) {
            size_t lists = annListsLocked(*ix, scale);
            if (lists == 0) return fallBack();
            ix->search(query.data(), 1.0f / norm, code, slack, lists, topK, ann_.hits);
            if (ann_.hits.size() < topK) continue;
            std::vector<SimilarityResult> results;
            results.reserve(ann_.hits.size());
            for (const detail::IvfHit& hit : ann_.hits) {
                const EmbeddingRecord& r = *ann_.recs[hit.slot];
                float score = cosineSimilarity(query, r.vector);
                if (score >= threshold) results.push_back(SimilarityResult{r.node_ref, score, r.text});
            }
            if (results.size() >= topK) {
                std::stable_sort(results.begin(), results.end());
                ++ann_.approximate;
                if (scale > 1) ++ann_.widened;
                return results;
            }
        }
    }

public:
    /**
     * @param encoder          encoder for text; may be null when only vectors are added
     * @param onModelMismatch  what a *text* query does when the encoder's model differs
     *                         from the model that produced the stored vectors (vector
     *                         queries are never checked; "unknown"/"provided" vectors
     *                         never trigger it)
     */
    explicit EmbeddingStore(std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
                            ModelMismatchMode onModelMismatch = ModelMismatchMode::Warn)
        : encoder_(std::move(encoder)), onModelMismatch_(onModelMismatch) {}

    /** Vector length this store holds, or nullopt while it is still empty. */
    std::optional<size_t> dimension() const { return dimension_; }

    void setEncoder(std::shared_ptr<EmbeddingEncoder> encoder) { encoder_ = std::move(encoder); }
    const std::shared_ptr<EmbeddingEncoder>& encoder() const { return encoder_; }

    ModelMismatchMode onModelMismatch() const { return onModelMismatch_; }
    void setOnModelMismatch(ModelMismatchMode mode) { onModelMismatch_ = mode; }

    /**
     * Encoder/model version recorded in the sidecar: the one set with
     * setVersion, else the encoder's own, else nullopt.
     */
    std::optional<std::string> version() const {
        if (version_.has_value()) return version_;
        return encoder_ ? encoder_->version() : std::nullopt;
    }
    void setVersion(std::optional<std::string> version) { version_ = std::move(version); }

    /** Metadata of the most recently imported payload (all unset before any). */
    const EmbeddingMeta& lastImportMeta() const { return lastImportMeta_; }

    // -- model lifecycle -------------------------------------------------

    /**
     * Which models produced the stored vectors: {modelName: count}.
     * "unknown" / "provided" appear as-is (vectors imported from a port that
     * records no model, or handed in with no encoder attached).
     */
    std::map<std::string, size_t> models() const {
        std::map<std::string, size_t> counts;
        for (const auto& [key, record] : embeddings_) ++counts[record.model];
        return counts;
    }

    /** The encoder's model name, or nullopt with no encoder or an unnamed ("unknown") one. */
    std::optional<std::string> currentModel() const {
        if (!encoder_) return std::nullopt;
        std::string name = encoder_->modelName();
        if (name.empty() || isUnknownModel(name)) return std::nullopt;
        return name;
    }

    /**
     * Refs whose stored vector does not belong to the current encoder, sorted.
     *
     * A vector is stale when its recorded model differs from `model` (default:
     * the encoder's modelName) or its length differs from `dimension` (default:
     * the encoder's dimension). Vectors whose model is "unknown"/"provided"
     * cannot be judged by model and are skipped unless `includeUnknown` (a
     * dimension mismatch still counts). With no encoder and no arguments
     * nothing can be judged and the result is empty.
     */
    std::vector<NodeRef> stale(std::optional<std::string> model = std::nullopt,
                               std::optional<size_t> dimension = std::nullopt,
                               bool includeUnknown = false) const {
        if (!model.has_value()) model = currentModel();
        if (!dimension.has_value() && encoder_) dimension = encoder_->dimension();

        std::vector<NodeRef> out;
        for (const auto& [key, record] : embeddings_) {
            bool byModel = model.has_value() && record.model != *model &&
                           (includeUnknown || !isUnknownModel(record.model));
            bool byDim = dimension.has_value() && record.vector.size() != *dimension;
            if (byModel || byDim) out.push_back(record.node_ref);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    /**
     * Encode a text query, first checking that the encoder's model is the one
     * that made the stored vectors (per the store's ModelMismatchMode).
     * Only text queries go through here; vector queries are never checked.
     *
     * @throws NoEncoderError with no encoder
     * @throws ModelMismatchError with ModelMismatchMode::Error on a mismatch
     */
    std::vector<float> encodeQuery(const std::string& query) const {
        if (!encoder_) {
            throw NoEncoderError("No encoder provided for text query");
        }
        checkModelCompat();
        return encoder_->encode(query);
    }

    std::string add(const NodeRef& node_ref, const std::string& text,
                    const std::vector<float>& vector = {}) {
        std::vector<float> vec = vector;
        if (vec.empty()) {
            if (!encoder_) {
                throw NoEncoderError("No encoder provided and no vector given");
            }
            vec = encoder_->encode(text);
        }
        checkDimension(vec, "vector");

        std::string key = nodeRefToKey(node_ref);
        putRecord(key, EmbeddingRecord{key, node_ref, std::move(vec), text, currentModelStamp()});
        return key;
    }

    /** Add many embeddings, encoding their texts in one batched call. */
    size_t addBatch(const std::vector<std::pair<NodeRef, std::string>>& entries) {
        if (entries.empty()) return 0;
        if (!encoder_) {
            throw NoEncoderError("No encoder provided and no vectors given");
        }

        std::vector<std::string> texts;
        texts.reserve(entries.size());
        for (const auto& entry : entries) {
            texts.push_back(entry.second);
        }

        auto vectors = encoder_->encodeBatch(texts);
        if (vectors.size() != entries.size()) {
            throw EmbeddingError("encoder returned " + std::to_string(vectors.size()) +
                                 " vectors for " + std::to_string(entries.size()) + " texts");
        }

        for (size_t i = 0; i < entries.size(); ++i) {
            add(entries[i].first, entries[i].second, vectors[i]);
        }
        return entries.size();
    }

    const EmbeddingRecord* get(const NodeRef& node_ref) const {
        std::string key = nodeRefToKey(node_ref);
        auto it = embeddings_.find(key);
        if (it != embeddings_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    bool remove(const NodeRef& node_ref) {
        std::string key = nodeRefToKey(node_ref);
        auto it = embeddings_.find(key);
        if (it == embeddings_.end()) return false;
        annBeforeErase(it->second);
        embeddings_.erase(it);
        return true;
    }

    /**
     * @param exact skip the approximate index and scan every vector (no effect
     *              unless enableAnn() was called)
     */
    std::vector<SimilarityResult> similaritySearch(
        const std::string& query,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f,
        bool exact = false
    ) const {
        auto queryVector = encodeQuery(query);
        return similaritySearchVector(queryVector, topK, nodeType, threshold, exact);
    }

    /**
     * Cosine top-k of a query vector. With enableAnn() a bounded topK on a
     * large enough store is answered from the approximate index (see there);
     * `exact` forces the full scan.
     */
    std::vector<SimilarityResult> similaritySearchVector(
        const std::vector<float>& queryVector,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f,
        bool exact = false
    ) const {
        checkQuery(queryVector);

        if (topK > 0) {
            // A store without enableAnn() only reads its (empty) options under a
            // brief lock here; the scan below is what it always was.
            bool enabled;
            {
                std::lock_guard<std::mutex> lock(ann_.mutex);
                enabled = ann_.options.has_value();
                if (enabled && exact) ++ann_.exact;
            }
            if (enabled && !exact) {
                if (auto answer = annSearch(queryVector, topK, nodeType, threshold)) {
                    return std::move(*answer);
                }
            }
        }

        std::vector<SimilarityResult> results;

        for (const auto& [key, record] : embeddings_) {
            if (!nodeType.empty() && record.node_ref.type != nodeType) {
                continue;
            }

            float score = cosineSimilarity(queryVector, record.vector);
            if (score >= threshold) {
                results.push_back(SimilarityResult{record.node_ref, score, record.text});
            }
        }

        // Selecting the top k rather than ordering every candidate: this is
        // O(n log k) where a full sort is O(n log n), and returns the same
        // rows in the same order.
        if (results.size() > topK) {
            std::partial_sort(results.begin(), results.begin() + topK, results.end());
            results.resize(topK);
        } else {
            std::sort(results.begin(), results.end());
        }

        return results;
    }

    /**
     * Cosine score for every embedded node, keyed by "type:id".
     *
     * One scan serving both seed selection and the per-node relevance that
     * semanticMultiHop blends in, instead of two.
     */
    std::map<std::string, float> scoreMap(const std::vector<float>& queryVector,
                                          const std::string& nodeType = "") const {
        checkQuery(queryVector);

        std::map<std::string, float> scores;
        for (const auto& [key, record] : embeddings_) {
            if (!nodeType.empty() && record.node_ref.type != nodeType) continue;
            scores[key] = cosineSimilarity(queryVector, record.vector);
        }
        return scores;
    }

    // -- approximate (IVF) search ------------------------------------------------

    /**
     * Opt in to approximate search. Off by default, and then search is the
     * exact scan, unchanged.
     *
     * With it on, a bounded top-k search on a store of at least
     * `options.minSize` vectors is answered from an in-memory IVF-flat index:
     * the vectors are clustered into `nlist` cells and only the `nprobe` cells
     * nearest the query are scored. It is approximate - a true neighbour in an
     * unprobed cell is missed - so measure it with recallCheck() on your own
     * queries before relying on a setting, and raise nprobe until the recall
     * you need holds (nprobe >= nlist is the exact scan).
     *
     * The index is built lazily by the first eligible search (or by
     * buildAnnIndex()), kept current through add/update/remove, and never
     * written to disk: the graph file, the embeddings sidecar and the graph
     * digest are exactly what they are without it. Scores are the same cosine
     * values the exact scan computes. searches with `exact = true`, an
     * unbounded topK (one that asks for every row), a zero query vector, or a
     * store under minSize use the exact scan.
     *
     * The index keeps its own contiguous copy of the vectors (see
     * AnnStats::vectorBytes), so an indexed store holds them twice.
     * Concurrent searches are safe; they take turns on the index. Calling this
     * again replaces the options and drops the current index.
     */
    void enableAnn(AnnOptions options = AnnOptions()) {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        ann_.reset();
        ann_.options = options;
    }

    /** Turn approximate search off and free the index. */
    void disableAnn() {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        ann_.reset();
        ann_.options.reset();
    }

    bool annEnabled() const {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        return ann_.options.has_value();
    }

    /** Cells scanned per query from now on (0 = automatic). Takes effect immediately; no rebuild. */
    void setAnnNprobe(size_t nprobe) {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        if (!ann_.options) throw EmbeddingError("this store has no approximate index; call enableAnn() first");
        ann_.options->nprobe = nprobe;
    }

    /**
     * Build the index now instead of on the first search; with `force`,
     * retrain even if one exists. Does nothing (and says so in the stats) while
     * the store is under minSize.
     *
     * @throws EmbeddingError when enableAnn() was not called
     */
    AnnStats buildAnnIndex(bool force = false) {
        {
            std::lock_guard<std::mutex> lock(ann_.mutex);
            if (!ann_.options) throw EmbeddingError("this store has no approximate index; call enableAnn() first");
            if (force) ann_.reset();
            Ivf* ix = annIndexLocked();
            if (ix != nullptr && ix->pendingLive > 0) ix->fold(annThreads());
        }
        return annStats();
    }

    /** What the approximate index looks like right now. Never builds anything. */
    AnnStats annStats() const {
        std::lock_guard<std::mutex> lock(ann_.mutex);
        AnnStats stats;
        stats.enabled = ann_.options.has_value();
        stats.minSize = ann_.options ? ann_.options->minSize : 0;
        stats.approximateSearches = ann_.approximate;
        stats.widenedSearches = ann_.widened;
        stats.exactSearches = ann_.exact;
        stats.builds = ann_.builds;
        const Ivf* ix = ann_.index.get();
        stats.active = ix != nullptr;
        if (ix != nullptr) {
            stats.nlist = ix->nlist;
            stats.nprobe = annProbeLocked(*ix);
            stats.trainedN = ix->trainedN;
            stats.pending = ix->pendingLive;
            stats.churn = ix->churn;
            stats.buildSeconds = ix->buildSeconds;
            stats.indexBytes = ix->overheadBytes();
            stats.vectorBytes = ix->vectorBytes();
        } else if (ann_.options) {
            stats.nprobe = ann_.options->nprobe;
        }
        return stats;
    }

    /**
     * Recall@k of the approximate index against the exact scan: runs every
     * query both ways and compares the sets of top-k nodes. Use real queries
     * drawn from your own workload - recall depends on how the data clusters,
     * and a score measured on unrelated queries or data does not transfer.
     *
     * Without an index in use (disabled, under minSize, or nprobe >= nlist)
     * both answers are the exact scan, recall is 1.0 and `approximate` false.
     *
     * @throws EmbeddingError when k is 0
     */
    RecallReport recallCheck(const std::vector<std::vector<float>>& queries, size_t k = 10,
                             const std::string& nodeType = "", float threshold = 0.0f) const {
        if (k == 0) throw EmbeddingError("k must be positive");
        using Clock = std::chrono::steady_clock;
        std::vector<double> recalls, annTimes, exactTimes;
        bool used = false;
        for (const auto& query : queries) {
            size_t before = annStats().approximateSearches;
            auto t0 = Clock::now();
            auto found = similaritySearchVector(query, k, nodeType, threshold);
            auto t1 = Clock::now();
            used = used || annStats().approximateSearches > before;
            auto truth = similaritySearchVector(query, k, nodeType, threshold, /*exact=*/true);
            auto t2 = Clock::now();
            annTimes.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            exactTimes.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            if (truth.empty()) continue;
            std::unordered_set<std::string> wanted;
            for (const auto& r : truth) wanted.insert(nodeRefToKey(r.node_ref));
            size_t hit = 0;
            for (const auto& r : found) hit += wanted.count(nodeRefToKey(r.node_ref));
            recalls.push_back(static_cast<double>(hit) / static_cast<double>(wanted.size()));
        }
        auto median = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            size_t mid = v.size() / 2;
            return v.size() % 2 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
        };
        RecallReport report;
        report.queries = queries.size();
        report.k = k;
        report.approximate = used;
        report.annMs = median(annTimes);
        report.exactMs = median(exactTimes);
        if (!recalls.empty()) {
            report.recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / static_cast<double>(recalls.size());
            report.minRecall = *std::min_element(recalls.begin(), recalls.end());
        }
        AnnStats stats = annStats();
        report.nlist = stats.nlist;
        report.nprobe = stats.nprobe;
        return report;
    }

    /** recallCheck() with text queries (needs an encoder). */
    RecallReport recallCheckTexts(const std::vector<std::string>& queries, size_t k = 10,
                                  const std::string& nodeType = "", float threshold = 0.0f) const {
        std::vector<std::vector<float>> vectors;
        vectors.reserve(queries.size());
        for (const auto& query : queries) vectors.push_back(encodeQuery(query));
        return recallCheck(vectors, k, nodeType, threshold);
    }

    /**
     * Visit the matches of the probed cells best first, for a caller that
     * filters further (SemanticGraph's node filter). `visit` returns false to
     * stop. Returns false, without visiting anything, when the index cannot
     * answer at this widening `scale` (1, 4, 16, ...) and the caller should use
     * the exact scan. Matches are ordered by the index's fast scores; the
     * score handed to `visit` is the exact cosine.
     */
    bool annVisit(const std::vector<float>& query, const std::string& nodeType, float threshold,
                  size_t scale, const std::function<bool(SimilarityResult&&)>& visit) const {
        checkQuery(query);
        std::vector<const EmbeddingRecord*> order;
        {
            std::lock_guard<std::mutex> lock(ann_.mutex);
            Ivf* ix = annIndexLocked();
            if (ix == nullptr) return false;
            const float norm = std::sqrt(detail::simd::dot(query.data(), query.data(), query.size()));
            if (!(norm > 0.0f)) return false;
            uint32_t code = Ivf::kAnyCode;
            if (!nodeType.empty()) {
                auto found = ann_.codes.find(nodeType);
                if (found == ann_.codes.end()) return true;  // nothing of that type exists
                code = found->second;
            }
            size_t lists = annListsLocked(*ix, scale);
            if (lists == 0) return false;
            ix->candidates(query.data(), 1.0f / norm, code, threshold - 1e-5f, lists, ann_.hits);
            order.reserve(ann_.hits.size());
            for (const detail::IvfHit& hit : ann_.hits) order.push_back(ann_.recs[hit.slot]);
            ++ann_.approximate;
            if (scale > 1) ++ann_.widened;
        }
        // Outside the lock: `visit` is user code and may search this store again.
        for (const EmbeddingRecord* r : order) {
            float score = cosineSimilarity(query, r->vector);
            if (!(score >= threshold)) continue;
            if (!visit(SimilarityResult{r->node_ref, score, r->text})) break;
        }
        return true;
    }

    size_t count() const {
        return embeddings_.size();
    }

    /**
     * Clear all embeddings.
     *
     * @param resetDimension also forget the pinned vector length, so the next
     *        vector may have a different one (switching to an encoder of a
     *        different dimension). Off by default: a store normally refuses
     *        vectors of the wrong length even when empty. (Before this
     *        parameter existed, clear() always reset it.)
     */
    void clear(bool resetDimension = false) {
        {
            std::lock_guard<std::mutex> lock(ann_.mutex);
            ann_.reset();
        }
        embeddings_.clear();
        if (resetDimension) dimension_.reset();
    }

    /**
     * Serialize every embedding into the portable cross-language payload.
     *
     * Besides the embeddings it carries the optional metadata: the encoder's
     * model and the store's version when known, and `graphDigest` (with its
     * basis) when given - see graphDigest().
     */
    ExportedEmbeddings exportEmbeddings(
        const std::optional<std::string>& graphDigest = std::nullopt) const {
        ExportedEmbeddings payload;
        payload.dimension = dimension_;
        payload.embeddings.reserve(embeddings_.size());

        for (const auto& [key, record] : embeddings_) {
            payload.embeddings.push_back(ExportedEmbedding{
                record.node_ref.type,
                record.node_ref.id,
                // C++ node ids are always strings; the other ports use this to
                // tell a numeric id from a numeric-looking string one.
                "str",
                record.text,
                record.vector,
                record.model
            });
        }
        std::sort(payload.embeddings.begin(), payload.embeddings.end(),
                  [](const ExportedEmbedding& a, const ExportedEmbedding& b) {
                      return std::tie(a.type, a.id) < std::tie(b.type, b.id);
                  });
        payload.count = payload.embeddings.size();

        payload.meta.encoderModel = currentModel();
        payload.meta.encoderVersion = version();
        if (graphDigest.has_value()) {
            payload.meta.graphDigest = graphDigest;
            payload.meta.graphDigestBasis = GRAPH_DIGEST_BASIS;
        }
        return payload;
    }

    /** Load a payload produced by exportEmbeddings (in any language port). */
    size_t importEmbeddings(const ExportedEmbeddings& payload, bool replace = false) {
        if (payload.format != std::string(EMBEDDING_FORMAT)) {
            throw EmbeddingError(std::string("not an ") + EMBEDDING_FORMAT +
                                 " payload: format=" + payload.format);
        }
        if (payload.version != EMBEDDING_FORMAT_VERSION) {
            throw EmbeddingError(std::string("unsupported ") + EMBEDDING_FORMAT + " version " +
                                 std::to_string(payload.version) + " (this build reads version " +
                                 std::to_string(EMBEDDING_FORMAT_VERSION) + ")");
        }

        lastImportMeta_ = payload.meta;
        if (replace) {
            clear(true);
            if (payload.dimension.has_value() && *payload.dimension > 0) {
                dimension_ = payload.dimension;
            }
        }

        size_t written = 0;
        for (const auto& item : payload.embeddings) {
            checkDimension(item.vector, "embedding for :" + item.type + ":" + item.id);
            std::string key = nodeRefToKey(NodeRef(item.type, item.id));
            putRecord(key, EmbeddingRecord{key, NodeRef(item.type, item.id), item.vector,
                                           item.text, item.model});
            ++written;
        }
        return written;
    }

    /**
     * Serialize the portable payload to a JSON string.
     *
     * The optional metadata (`encoder_model`, `encoder_version`, and
     * `graph_digest` + `graph_digest_basis` when `graphDigest` is given) is
     * appended after `embeddings`, so the leading keys stay `format`,
     * `version`, `dimension`, `count`, `embeddings`. Absent values are omitted.
     */
    std::string toJson(const std::optional<std::string>& graphDigest = std::nullopt) const {
        return serializePayload(exportEmbeddings(graphDigest));
    }

    /**
     * Serialize an already-built payload to the portable JSON text (what
     * toJson writes). Shared with the SQLite-backed store, so both emit
     * byte-identical documents.
     */
    static std::string serializePayload(const ExportedEmbeddings& payload) {
        std::ostringstream out;
        out.precision(9);
        out << "{\"format\":\"" << EMBEDDING_FORMAT << "\",\"version\":" << EMBEDDING_FORMAT_VERSION
            << ",\"dimension\":";
        if (payload.dimension.has_value()) {
            out << *payload.dimension;
        } else {
            out << "null";
        }
        out << ",\"count\":" << payload.count << ",\"embeddings\":[";

        for (size_t i = 0; i < payload.embeddings.size(); ++i) {
            const auto& item = payload.embeddings[i];
            if (i > 0) out << ",";
            out << "{\"type\":" << jsonString(item.type)
                << ",\"id\":" << jsonString(item.id)
                << ",\"id_type\":" << jsonString(item.id_type)
                << ",\"text\":" << jsonString(item.text)
                << ",\"model\":" << jsonString(item.model)
                << ",\"vector\":[";
            for (size_t j = 0; j < item.vector.size(); ++j) {
                if (j > 0) out << ",";
                out << item.vector[j];
            }
            out << "]}";
        }
        out << "]";
        if (payload.meta.encoderModel.has_value()) {
            out << ",\"encoder_model\":" << jsonString(*payload.meta.encoderModel);
        }
        if (payload.meta.encoderVersion.has_value()) {
            out << ",\"encoder_version\":" << jsonString(*payload.meta.encoderVersion);
        }
        if (payload.meta.graphDigest.has_value()) {
            out << ",\"graph_digest\":" << jsonString(*payload.meta.graphDigest)
                << ",\"graph_digest_basis\":"
                << jsonString(payload.meta.graphDigestBasis.value_or(GRAPH_DIGEST_BASIS));
        }
        out << "}";
        return out.str();
    }

    /**
     * Load embeddings from a JSON payload written by any language port.
     *
     * Reads real JSON, so it accepts what the other ports emit: Python's
     * json.dumps escapes every non-ASCII character (including surrogate
     * pairs), and a "}" or "]" can legitimately appear inside text. Anything
     * that is not an ison-embeddings payload of a version this build
     * understands is rejected. Fields it does not know (a newer writer's
     * extras) are ignored; the optional metadata (encoder model/version, graph
     * digest) is kept and readable afterwards through lastImportMeta().
     */
    size_t fromJson(const std::string& json, bool replace = false) {
        return importEmbeddings(parseJson(json), replace);
    }

    /**
     * Parse and validate a JSON payload without touching the store - for a
     * caller that wants to look at the metadata before deciding whether to
     * import it (SemanticGraph::load checks the graph digest this way).
     */
    static ExportedEmbeddings parseJson(const std::string& json) {
        using detail::JsonValue;
        using detail::JsonParser;
        JsonValue root = JsonParser(json).parse();
        if (root.kind != JsonValue::Kind::Object) {
            throw EmbeddingError("embeddings payload is not a JSON object");
        }

        ExportedEmbeddings payload;
        const JsonValue* format = root.find("format");
        payload.format = (format && format->kind == JsonValue::Kind::String) ? format->text : "";
        const JsonValue* version = root.find("version");
        payload.version = (version && version->kind == JsonValue::Kind::Number)
                              ? static_cast<int>(version->number) : -1;

        if (payload.format != std::string(EMBEDDING_FORMAT)) {
            throw EmbeddingError(std::string("not an ") + EMBEDDING_FORMAT +
                                 " payload: format=" + payload.format);
        }
        if (payload.version != EMBEDDING_FORMAT_VERSION) {
            throw EmbeddingError(std::string("unsupported ") + EMBEDDING_FORMAT + " version " +
                                 std::to_string(payload.version) + " (this build reads version " +
                                 std::to_string(EMBEDDING_FORMAT_VERSION) + ")");
        }

        const JsonValue* list = root.find("embeddings");
        if (list == nullptr || list->kind != JsonValue::Kind::Array) {
            throw EmbeddingError("payload has no embeddings array");
        }

        for (const JsonValue& entry : list->items) {
            if (entry.kind != JsonValue::Kind::Object) continue;

            ExportedEmbedding item;
            const JsonValue* type = entry.find("type");
            if (type == nullptr || type->kind != JsonValue::Kind::String) continue;
            item.type = type->text;

            // Ports with typed node ids write an int id unquoted ("id":1), so
            // accept either form and keep the exact text of the id either way.
            const JsonValue* id = entry.find("id");
            if (id != nullptr && (id->kind == JsonValue::Kind::String ||
                                  id->kind == JsonValue::Kind::Number)) {
                item.id = id->text;
            }
            const JsonValue* idType = entry.find("id_type");
            if (idType != nullptr && idType->kind == JsonValue::Kind::String) {
                item.id_type = idType->text;
            }
            const JsonValue* text = entry.find("text");
            if (text != nullptr && text->kind == JsonValue::Kind::String) {
                item.text = text->text;
            }
            // Ports that record no model omit the key; "unknown" then.
            const JsonValue* model = entry.find("model");
            if (model != nullptr && model->kind == JsonValue::Kind::String && !model->text.empty()) {
                item.model = model->text;
            }

            const JsonValue* vector = entry.find("vector");
            if (vector == nullptr || vector->kind != JsonValue::Kind::Array ||
                !vector->items.empty()) {
                // not an array of plain numbers: the vector is untrustworthy
                continue;
            }
            item.vector.reserve(vector->numbers.size());
            for (double number : vector->numbers) {
                item.vector.push_back(static_cast<float>(number));
            }
            if (item.vector.empty()) continue;

            payload.embeddings.push_back(std::move(item));
        }

        payload.count = payload.embeddings.size();

        const JsonValue* dimension = root.find("dimension");
        if (dimension != nullptr && dimension->kind == JsonValue::Kind::Number &&
            dimension->number >= 1) {
            payload.dimension = static_cast<size_t>(dimension->number);
        }

        auto optionalString = [&](const char* key) -> std::optional<std::string> {
            const JsonValue* value = root.find(key);
            if (value != nullptr && value->kind == JsonValue::Kind::String) return value->text;
            return std::nullopt;
        };
        payload.meta.encoderModel = optionalString("encoder_model");
        payload.meta.encoderVersion = optionalString("encoder_version");
        payload.meta.graphDigest = optionalString("graph_digest");
        payload.meta.graphDigestBasis = optionalString("graph_digest_basis");
        return payload;
    }

    /**
     * Write the portable payload to `path` (the format Python's save_embeddings
     * writes), atomically: a temp file is written and renamed into place, so a
     * crash never leaves a half-written sidecar. `graphDigest` ties the sidecar
     * to one particular graph (see graphDigest()).
     */
    size_t saveEmbeddings(const std::string& path,
                          const std::optional<std::string>& graphDigest = std::nullopt) const {
        detail::atomicWriteText<EmbeddingError>(path, toJson(graphDigest));
        return embeddings_.size();
    }

    /** Read a payload file written by saveEmbeddings (or any port's equivalent). */
    size_t loadEmbeddings(const std::string& path, bool replace = false) {
        return importEmbeddings(parseJson(readFile(path)), replace);
    }

    /** Whole file as bytes; throws EmbeddingError when it cannot be opened. */
    static std::string readFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw EmbeddingError("cannot open file for reading: " + path);
        }
        std::ostringstream content;
        content << file.rdbuf();
        return content.str();
    }

private:
    static std::string jsonString(const std::string& value) {
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
                        // other control characters are not legal bare in JSON
                        out += '\\';
                        out += 'u';
                        out += "00";
                        out += HEX[(static_cast<unsigned char>(c) >> 4) & 0xF];
                        out += HEX[static_cast<unsigned char>(c) & 0xF];
                    } else {
                        out += c;
                    }
            }
        }
        return out + "\"";
    }
};

// =============================================================================
// Semantic Search Result
// =============================================================================

struct SemanticSearchResult {
    NodeRef node_ref;
    float score;
    size_t hop_count;
    std::vector<NodeRef> path;
};

/**
 * Restrict a search to some of the nodes. All set criteria must hold (AND).
 *
 * The filter looks at the node in the graph: an embedding whose node is not in
 * the graph fails `nodeFilter` (and passes a `nodeTypes`-only filter, which
 * only needs the ref). Default semantics for the multi-hop family:
 *  - seeds must pass, and the top-k seeds are chosen *among* passing nodes;
 *  - only passing nodes are returned;
 *  - a node that fails can still be walked *through* (it costs a hop and its
 *    decay, and a returned node's path may list it) unless `strictFilter` is
 *    set, which makes the filter a hard wall for traversal as well.
 */
struct NodeFilterOptions {
    /** Predicate on the node; empty = no predicate. */
    std::function<bool(const Node&)> nodeFilter;
    /**
     * Node types allowed. nullopt = any type; an empty vector passes nothing.
     * Combines with the `nodeType` argument of similaritySearch by AND.
     */
    std::optional<std::vector<std::string>> nodeTypes;
    /** Never enter a failing node while traversing (semanticMultiHop, semanticPath, semanticSubgraph). */
    bool strictFilter = false;
};

/** Options for semanticMultiHop and semanticSubgraph. */
struct MultiHopOptions : NodeFilterOptions {
    std::string relType = "";
    size_t maxHops = 2;
    size_t topKSeeds = 5;
    size_t topKResults = 10;
    Direction direction = Direction::Out;
    float decay = 0.8f;
    float threshold = 0.3f;
    /**
     * How much of the relevance of a node itself to mix into its traversal
     * score, in [0, 1]. 0 (the default) keeps pure seed-and-decay scoring.
     * Raise it so a node found two hops out but highly relevant in its own
     * right is not buried under a closer, unrelated neighbour. Hop-0 seeds
     * score the same either way.
     */
    float blend = 0.0f;
    /**
     * Name of an edge property whose numeric value multiplies the decay of
     * every hop that crosses that edge (clamped to [0, 1]; a missing,
     * non-numeric, boolean or NaN value counts as 1.0, so partly weighted
     * graphs work). A weight of 0 scores the node 0 but still reaches it. A
     * node is scored by its *best* route, which with weights can be a longer
     * one than the fewest-hops route. Empty (the default) ignores edge
     * properties entirely, with scores identical to releases without weights.
     */
    std::string weightField = "";
    /**
     * semanticSubgraph only: also add *connector* nodes so the selected nodes
     * fall into as few connected components as possible. Components are weak
     * (edge direction ignored when judging the result) but connector routes
     * honour relType and direction. Each round adds the interior of the
     * shortest route between two components (greedy; not a minimum Steiner
     * tree) until one component remains or no route fits connectMaxHops.
     * Connectors are extra: the result can hold more than topKResults nodes.
     * Without strictFilter a connector may fail the node filter (it is only a
     * bridge); with it, connectors must pass too.
     */
    bool connect = false;
    /**
     * semanticSubgraph only: longest route, counted in edges, that may join
     * two components. The default 2 allows one connector node per join; 3
     * allows two. Must be at least 1.
     */
    size_t connectMaxHops = 2;
};

// =============================================================================
// Semantic Graph
// =============================================================================

class SemanticGraph {
private:
    ISONGraph graph_;
    EmbeddingStore embedding_store_;
    bool auto_embed_;
    std::vector<std::string> embed_fields_;

    /** ref -> does this node pass the filter? Empty function = no filter. */
    using NodeGate = std::function<bool(const NodeRef&)>;

    /**
     * Build the predicate behind a NodeFilterOptions, or an empty function when
     * nothing is set so unfiltered calls take exactly the pre-existing path.
     * Answers are memoized for the one call; a ref with no node in the graph
     * fails whenever a nodeFilter is set (it cannot be shown to the predicate).
     */
    NodeGate makeGate(const NodeFilterOptions& filter) const {
        if (!filter.nodeFilter && !filter.nodeTypes.has_value()) return NodeGate();

        std::shared_ptr<std::unordered_set<std::string>> types;
        if (filter.nodeTypes.has_value()) {
            types = std::make_shared<std::unordered_set<std::string>>(
                filter.nodeTypes->begin(), filter.nodeTypes->end());
        }
        auto memo = std::make_shared<std::unordered_map<std::string, bool>>();
        std::function<bool(const Node&)> predicate = filter.nodeFilter;
        const ISONGraph* graph = &graph_;

        return [graph, types, memo, predicate](const NodeRef& ref) -> bool {
            std::string key = ref.type;
            key += '\n';
            key += ref.id;
            auto cached = memo->find(key);
            if (cached != memo->end()) return cached->second;

            bool ok = !types || types->count(ref.type) > 0;
            if (ok && predicate) {
                ok = graph->hasNode(ref) &&
                     predicate(graph->getNode(ref.type, ref.id));
            }
            (*memo)[key] = ok;
            return ok;
        };
    }

    /**
     * (neighbor, connecting edge) pairs of a node, in the order neighbors()
     * yields them. Without edge data (the common case) it defers to the graph's
     * own adjacency and the edge pointer is null; with it, an index of the
     * edges of `relType` is built once up front.
     */
    class Adjacency {
    public:
        using Pair = std::pair<NodeRef, const Edge*>;

        Adjacency(const ISONGraph& graph, std::string relType, Direction direction, bool withEdges)
            : graph_(graph), relType_(std::move(relType)), direction_(direction),
              withEdges_(withEdges) {
            if (!withEdges_) return;
            std::vector<std::string> relTypes = graph_.edgeTypes();
            std::sort(relTypes.begin(), relTypes.end());
            for (const std::string& rel : relTypes) {
                if (!relType_.empty() && rel != relType_) continue;
                for (const Edge& edge : graph_.getEdges(rel)) edges_.push_back(edge);
            }
            for (size_t i = 0; i < edges_.size(); ++i) {
                out_[edges_[i].source.key()].push_back(i);
                in_[edges_[i].target.key()].push_back(i);
            }
        }

        std::vector<Pair> adjacent(const NodeRef& ref) const {
            std::vector<Pair> pairs;
            if (!withEdges_) {
                for (const NodeRef& neighbor : graph_.neighbors(ref, relType_, direction_)) {
                    pairs.emplace_back(neighbor, nullptr);
                }
                return pairs;
            }
            if (direction_ == Direction::Out || direction_ == Direction::Both) {
                auto it = out_.find(ref.key());
                if (it != out_.end()) {
                    for (size_t i : it->second) pairs.emplace_back(edges_[i].target, &edges_[i]);
                }
            }
            if (direction_ == Direction::In || direction_ == Direction::Both) {
                auto it = in_.find(ref.key());
                if (it != in_.end()) {
                    for (size_t i : it->second) pairs.emplace_back(edges_[i].source, &edges_[i]);
                }
            }
            return pairs;
        }

        /** Every indexed edge (only when built with edge data). */
        const std::vector<Edge>& edges() const { return edges_; }

    private:
        const ISONGraph& graph_;
        std::string relType_;
        Direction direction_;
        bool withEdges_;
        std::vector<Edge> edges_;
        std::unordered_map<std::string, std::vector<size_t>> out_, in_;
    };

    /**
     * An edge's numeric `field` property clamped to [0, 1]. A missing,
     * non-numeric (including boolean) or NaN value counts as 1.0, so an
     * unweighted edge never penalises a hop.
     */
    static float edgeWeight(const Edge& edge, const std::string& field) {
        auto it = edge.properties.find(field);
        if (it == edge.properties.end()) return 1.0f;
        const ison::Value& value = it->second;
        double weight;
        if (value.is_int()) weight = static_cast<double>(value.as_int());
        else if (value.is_float()) weight = value.as_float();
        else return 1.0f;
        if (std::isnan(weight)) return 1.0f;
        return static_cast<float>(std::min(1.0, std::max(0.0, weight)));
    }

    /**
     * Top-k similarity among the nodes that pass `gate`.
     *
     * With a gate the store cannot truncate first: the best k *unfiltered*
     * matches may all be filtered out, so every match is scored, filtered, and
     * only then cut to k.
     *
     * With the store's approximate index the candidates are the probed cells
     * only. If fewer than k of them pass the gate, the probe is widened (4x
     * each step) and the search retried, ending in the exact scan, so a
     * selective filter costs time rather than results.
     */
    std::vector<SimilarityResult> seedSearchVector(
        const std::vector<float>& queryVector, size_t topK, const std::string& nodeType,
        float threshold, const NodeGate& gate, bool exact = false
    ) const {
        if (!gate) {
            return embedding_store_.similaritySearchVector(queryVector, topK, nodeType, threshold, exact);
        }
        if (!exact && topK > 0 && topK < embedding_store_.count() && embedding_store_.annEnabled()) {
            for (size_t scale = 1;; scale *= 4) {
                std::vector<SimilarityResult> kept;
                bool answered = embedding_store_.annVisit(
                    queryVector, nodeType, threshold, scale, [&](SimilarityResult&& result) {
                        if (gate(result.node_ref)) kept.push_back(std::move(result));
                        return kept.size() < topK;
                    });
                if (!answered) break;
                if (kept.size() >= topK) return kept;
            }
        }
        auto all = embedding_store_.similaritySearchVector(
            queryVector, std::numeric_limits<size_t>::max(), nodeType, threshold, exact);
        std::vector<SimilarityResult> kept;
        for (auto& result : all) {
            if (kept.size() >= topK) break;
            if (gate(result.node_ref)) kept.push_back(std::move(result));
        }
        return kept;
    }

public:
    /**
     * @param onModelMismatch what a text query does when the encoder's model
     *        differs from the model that produced the stored vectors
     * @param version         encoder/model version recorded in the sidecar
     *        (default: the encoder's own version(), if any)
     * @param ann             opt in to approximate search for the graph's
     *        embedding store (see EmbeddingStore::enableAnn); nullopt (the
     *        default) keeps search exact
     */
    SemanticGraph(
        const std::string& name = "semantic_graph",
        std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
        bool autoEmbed = true,
        std::vector<std::string> embedFields = {"name", "description", "content", "text"},
        ModelMismatchMode onModelMismatch = ModelMismatchMode::Warn,
        std::optional<std::string> version = std::nullopt,
        std::optional<AnnOptions> ann = std::nullopt
    )
        : graph_(name)
        , embedding_store_(encoder, onModelMismatch)
        , auto_embed_(autoEmbed)
        , embed_fields_(std::move(embedFields))
    {
        if (version.has_value()) embedding_store_.setVersion(std::move(version));
        if (ann.has_value()) embedding_store_.enableAnn(*ann);
    }

    /**
     * Read-only view of the underlying graph: counts, lookups, neighbours,
     * paths, queries, serialization.
     *
     * Deliberately const. Every way of changing a node - adding, updating,
     * removing - has to go through this class, because each one must keep the
     * node's embedding in step; a mutable graph() let
     * graph().updateNode(...) and graph().removeNode(...) leave vectors for
     * text the node no longer carries. Edges carry no embedding, so
     * addEdge/removeEdge below forward to the graph.
     */
    const ISONGraph& graph() const { return graph_; }

    /**
     * Add an edge between two existing nodes.
     *
     * @throws NodeNotFoundError if either endpoint is missing
     * @throws DuplicateEdgeError if the edge already exists
     * @return the forward edge (source -> target), by value
     */
    Edge addEdge(const std::string& relType, const NodeRef& source, const NodeRef& target,
                 const Properties& properties = {}) {
        return graph_.addEdge(relType, source, target, properties);
    }

    /**
     * Remove an edge (both directions on an undirected graph).
     *
     * @throws EdgeNotFoundError if there is no such edge
     */
    void removeEdge(const std::string& relType, const NodeRef& source, const NodeRef& target) {
        graph_.removeEdge(relType, source, target);
    }

    EmbeddingStore& embeddingStore() { return embedding_store_; }
    const EmbeddingStore& embeddingStore() const { return embedding_store_; }

    /** The encoder in use, or null. */
    const std::shared_ptr<EmbeddingEncoder>& encoder() const { return embedding_store_.encoder(); }

    /** Install an encoder, replacing any current one. */
    void setEncoder(std::shared_ptr<EmbeddingEncoder> encoder) {
        embedding_store_.setEncoder(std::move(encoder));
    }

    /**
     * Render one ISON property value as embedding text.
     *
     * Property values are typed since ISONGraph 1.4.0, so a property can be a
     * number, a bool or null. Strings come through unquoted; nulls and
     * references contribute nothing to the text.
     */
    static std::string propertyText(const ison::Value& value) {
        switch (value.type()) {
            case ison::ValueType::String: return value.as_string();
            case ison::ValueType::Bool:   return value.as_bool() ? "true" : "false";
            case ison::ValueType::Int:    return std::to_string(value.as_int());
            case ison::ValueType::Float: {
                // Shortest round-trip, matching what Python, JavaScript, Rust
                // and C# produce. The default ostringstream format is six
                // significant figures, which rendered 1/3 as "0.333333" and
                // put this port out of step with the other five.
                char buffer[32];
                auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer),
                                               value.as_float());
                if (ec != std::errc()) return "";
                return std::string(buffer, end);
            }
            default: return "";
        }
    }

    /** Concatenate the configured embed fields present in a property bag. */
    std::string embedTextFor(const Properties& properties) const {
        std::string text;
        for (const auto& field : embed_fields_) {
            auto it = properties.find(field);
            if (it == properties.end()) continue;
            std::string part = propertyText(it->second);
            if (part.empty()) continue;
            if (!text.empty()) text += " ";
            text += part;
        }
        return text;
    }

    /**
     * Add a node, embedding it from its properties when autoEmbed is on.
     *
     * Previously only addNodeWithEmbed considered autoEmbed, so a node added
     * through the plain graph API came back unembedded and unsearchable.
     *
     * All-or-nothing: if embedding fails (an encoder that throws, a vector of
     * the wrong width), the node is taken out of the graph again and the
     * error propagates, so a failed add leaves nothing half-done. The node is
     * returned read-only - changing it goes through updateNode, which keeps
     * the embedding in step.
     */
    const Node& addNode(
        const std::string& nodeType,
        const std::string& nodeId,
        const Properties& properties = {}
    ) {
        const Node& node = graph_.addNode(nodeType, nodeId, properties);
        if (auto_embed_ && embedding_store_.encoder()) {
            std::string text = embedTextFor(properties);
            if (!text.empty()) {
                try {
                    embedNode(NodeRef(nodeType, nodeId), text);
                } catch (...) {
                    graph_.removeNode(nodeType, nodeId);
                    throw;
                }
            }
        }
        return node;
    }

    /** Add a node with an explicit embedding text (all-or-nothing, like addNode). */
    const Node& addNodeWithEmbed(
        const std::string& nodeType,
        const std::string& nodeId,
        const Properties& properties = {},
        const std::string& embedText = ""
    ) {
        if (embedText.empty()) {
            return addNode(nodeType, nodeId, properties);
        }
        const Node& node = graph_.addNode(nodeType, nodeId, properties);
        try {
            embedNode(NodeRef(nodeType, nodeId), embedText);
        } catch (...) {
            graph_.removeNode(nodeType, nodeId);
            throw;
        }
        return node;
    }

    std::string embedNode(const NodeRef& nodeRef, const std::string& text) {
        return embedding_store_.add(nodeRef, text);
    }

    /** Embed many nodes in a single batched encoder call. */
    size_t embedNodes(const std::vector<std::pair<NodeRef, std::string>>& items) {
        return embedding_store_.addBatch(items);
    }

    const EmbeddingRecord* getEmbedding(const NodeRef& nodeRef) const {
        return embedding_store_.get(nodeRef);
    }

    /**
     * Remove a node, its edges and its embedding.
     *
     * @throws NodeNotFoundError if the node does not exist
     */
    void removeNode(const std::string& nodeType, const std::string& nodeId) {
        embedding_store_.remove(NodeRef(nodeType, nodeId));
        graph_.removeNode(nodeType, nodeId);
    }

    /**
     * Update node properties, keeping the embedding in step with them.
     *
     * With autoEmbed on, a change to any of the embed fields re-embeds the
     * node from its new properties, so searches never rank it by text it no
     * longer carries. If the new properties leave nothing to embed, the stale
     * vector is dropped. An update that touches no embed field, and a graph
     * with autoEmbed off (whose vectors the caller manages), leave the
     * embedding alone. As in addNode, a graph with no encoder cannot embed, so
     * a changed node keeps whatever vector it had unless its text is now empty.
     *
     * This is the only way to change a node's properties: graph() is a const
     * view, so there is no route that updates them without the embedding.
     * To manage a vector by hand, construct the graph with autoEmbed off and
     * use embeddingStore().
     *
     * If re-embedding fails (encoder error, wrong vector width) the node's
     * properties are restored and the error propagates.
     *
     * @throws NodeNotFoundError if the node does not exist
     */
    const Node& updateNode(
        const std::string& nodeType,
        const std::string& nodeId,
        const Properties& properties
    ) {
        Node& node = graph_.getNode(nodeType, nodeId);
        const Properties previous = node.properties;
        graph_.updateNode(nodeType, nodeId, properties);

        if (auto_embed_) {
            bool touchesEmbedField = false;
            for (const auto& field : embed_fields_) {
                if (properties.find(field) != properties.end()) {
                    touchesEmbedField = true;
                    break;
                }
            }

            if (touchesEmbedField) {
                NodeRef ref(nodeType, nodeId);
                std::string text = embedTextFor(node.properties);
                if (text.empty()) {
                    embedding_store_.remove(ref);
                } else if (embedding_store_.encoder()) {
                    try {
                        embedNode(ref, text);
                    } catch (...) {
                        node.properties = previous;
                        throw;
                    }
                }
            }
        }

        return node;
    }

    /**
     * Find similar nodes by embedding.
     *
     * `filter` (nodeFilter / nodeTypes; strictFilter does not apply here) is
     * applied *before* topK is taken, so you still get topK matching nodes when
     * that many exist. A text query is checked against the model that made the
     * stored vectors (see ModelMismatchMode).
     *
     * With approximate search enabled on the embedding store the candidates
     * come from the probed cells; `exact` forces the full scan (no effect
     * otherwise). A filter that few nodes pass widens the probe, ending in the
     * full scan, so you still get topK matching nodes when that many exist.
     */
    std::vector<SimilarityResult> similaritySearch(
        const std::string& query,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f,
        const NodeFilterOptions& filter = NodeFilterOptions(),
        bool exact = false
    ) const {
        NodeGate gate = makeGate(filter);
        if (!gate) {
            return embedding_store_.similaritySearch(query, topK, nodeType, threshold, exact);
        }
        return seedSearchVector(embedding_store_.encodeQuery(query), topK, nodeType,
                                threshold, gate, exact);
    }

    /** similaritySearch() with a ready-made query vector (never checked against the encoder's model). */
    std::vector<SimilarityResult> similaritySearchVector(
        const std::vector<float>& queryVector,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f,
        const NodeFilterOptions& filter = NodeFilterOptions(),
        bool exact = false
    ) const {
        NodeGate gate = makeGate(filter);
        return seedSearchVector(queryVector, topK, nodeType, threshold, gate, exact);
    }

    /** Recall@k of the approximate index against the exact scan (see EmbeddingStore::recallCheck). */
    RecallReport recallCheck(const std::vector<std::vector<float>>& queries, size_t k = 10,
                             const std::string& nodeType = "", float threshold = 0.0f) const {
        return embedding_store_.recallCheck(queries, k, nodeType, threshold);
    }

    /** recallCheck() with text queries (needs an encoder). */
    RecallReport recallCheckTexts(const std::vector<std::string>& queries, size_t k = 10,
                                  const std::string& nodeType = "", float threshold = 0.0f) const {
        return embedding_store_.recallCheckTexts(queries, k, nodeType, threshold);
    }

    /**
     * Find nodes semantically closest to one already in the graph.
     *
     * The "more like this" query - no text and no encoder call: it searches
     * with the stored vector of the node and drops the node itself from the
     * results.
     */
    std::vector<SimilarityResult> similarToNode(
        const NodeRef& nodeRef,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f
    ) const {
        const EmbeddingRecord* record = embedding_store_.get(nodeRef);
        if (record == nullptr) return {};

        auto results = embedding_store_.similaritySearchVector(
            record->vector, topK + 1, nodeType, threshold);
        results.erase(
            std::remove_if(results.begin(), results.end(),
                           [&](const SimilarityResult& r) { return r.node_ref == nodeRef; }),
            results.end());
        if (results.size() > topK) {
            results.resize(topK);
        }
        return results;
    }

    /**
     * Semantic multi-hop search.
     *
     * 1. Find top-k similar nodes as seeds (hop 0)
     * 2. Traverse the graph following relationships
     * 3. Score by max(seedSimilarity, 0) * prod(decay * weight) over the hops
     *    (weight is 1.0 unless options.weightField is set), optionally blended
     *    with the relevance of each node itself
     *
     * Node filtering (options.nodeFilter / nodeTypes / strictFilter) is
     * described on NodeFilterOptions.
     */
    std::vector<SemanticSearchResult> semanticMultiHop(
        const std::string& query,
        const MultiHopOptions& options
    ) const {
        if (options.blend < 0.0f || options.blend > 1.0f) {
            throw EmbeddingError("blend must be in [0, 1], got " + std::to_string(options.blend));
        }

        auto queryVector = embedding_store_.encodeQuery(query);
        NodeGate gate = makeGate(options);

        // Blending needs a score for every node; otherwise only the seeds matter.
        std::map<std::string, float> scores;
        if (options.blend > 0.0f) {
            scores = embedding_store_.scoreMap(queryVector);
            if (scores.empty()) return {};
        }

        auto seeds = seedSearchVector(queryVector, options.topKSeeds, "", options.threshold, gate);
        if (seeds.empty()) {
            return {};
        }

        auto nodeRefToKey = [](const NodeRef& ref) { return ref.type + ":" + ref.id; };

        auto combined = [&](float base, const std::string& key) -> float {
            if (options.blend <= 0.0f) return base;
            auto it = scores.find(key);
            float own = (it == scores.end()) ? 0.0f : it->second;
            return (1.0f - options.blend) * base + options.blend * own;
        };

        const bool weighted = !options.weightField.empty();
        Adjacency adjacency(graph_, options.relType, options.direction, weighted);

        struct Walk {
            NodeRef ref;
            std::vector<NodeRef> path;
            float factor;
            size_t hop;
        };

        std::map<std::string, SemanticSearchResult> results;

        for (const auto& seed : seeds) {
            std::string seedKey = nodeRefToKey(seed.node_ref);

            // Cosine similarity is signed, and `base * decay ** hops` only
            // decays a non-negative base: multiplying a negative seed score by
            // 0.8 moves it toward zero, so everything it reached outranked the
            // seed itself and relevance grew with distance.
            float traversalBase = std::max(seed.score, 0.0f);

            // A seed is a hop-0 match on its own merit. An earlier traversal
            // may already have recorded it with a decayed score; that must not
            // win over the higher, direct similarity of the node. (Only the
            // neighbour branch below used to compare scores, so a direct match
            // reachable from a better seed came back at seed.score * decay, at
            // hop 1, down a path it never needed.)
            auto seeded = results.find(seedKey);
            if (seeded == results.end() || seeded->second.score < seed.score) {
                results[seedKey] = SemanticSearchResult{
                    seed.node_ref, seed.score, 0, {seed.node_ref}
                };
            }

            // Breadth-first traversal. Each node keeps its best decay factor
            // and its fewest hops *separately*: with edge weights the route
            // with the best factor is not always the shortest one, and a
            // shorter route must still be expanded because it has hops left.
            // Without weights the first (shortest) arrival is both, so this
            // reduces to the plain visited-set BFS.
            std::unordered_map<std::string, float> bestFactor{{seedKey, 1.0f}};
            std::unordered_map<std::string, size_t> bestHops{{seedKey, 0}};
            std::queue<Walk> queue;
            queue.push(Walk{seed.node_ref, {seed.node_ref}, 1.0f, 0});

            while (!queue.empty()) {
                Walk current = std::move(queue.front());
                queue.pop();

                if (current.hop >= options.maxHops) continue;

                for (const auto& [neighbor, edge] : adjacency.adjacent(current.ref)) {
                    if (std::find(current.path.begin(), current.path.end(), neighbor) !=
                        current.path.end()) {
                        continue;
                    }
                    if (options.strictFilter && gate && !gate(neighbor)) continue;

                    size_t newHop = current.hop + 1;
                    // Unweighted: decay^hops exactly, not a running product, so
                    // scores are bit-identical to releases without weights.
                    float newFactor = weighted
                        ? current.factor * options.decay * edgeWeight(*edge, options.weightField)
                        : std::pow(options.decay, static_cast<float>(newHop));

                    std::string neighborKey = nodeRefToKey(neighbor);
                    auto factorIt = bestFactor.find(neighborKey);
                    auto hopsIt = bestHops.find(neighborKey);
                    float knownFactor = factorIt == bestFactor.end() ? -1.0f : factorIt->second;
                    size_t knownHops = hopsIt == bestHops.end() ? newHop + 1 : hopsIt->second;
                    if (newFactor <= knownFactor && newHop >= knownHops) continue;
                    bestFactor[neighborKey] = std::max(newFactor, knownFactor);
                    bestHops[neighborKey] = std::min(newHop, knownHops);

                    std::vector<NodeRef> newPath = current.path;
                    newPath.push_back(neighbor);

                    if (!gate || gate(neighbor)) {
                        float newScore = combined(traversalBase * newFactor, neighborKey);
                        auto it = results.find(neighborKey);
                        if (it == results.end() || it->second.score < newScore) {
                            results[neighborKey] = SemanticSearchResult{
                                neighbor, newScore, newHop, newPath
                            };
                        }
                    }

                    queue.push(Walk{neighbor, std::move(newPath), newFactor, newHop});
                }
            }
        }

        std::vector<SemanticSearchResult> sortedResults;
        sortedResults.reserve(results.size());
        for (const auto& [key, result] : results) {
            sortedResults.push_back(result);
        }

        std::sort(sortedResults.begin(), sortedResults.end(),
            [](const SemanticSearchResult& a, const SemanticSearchResult& b) {
                return a.score > b.score;
            });

        if (sortedResults.size() > options.topKResults) {
            sortedResults.resize(options.topKResults);
        }

        return sortedResults;
    }

    /** Semantic multi-hop search, original positional form. */
    std::vector<SemanticSearchResult> semanticMultiHop(
        const std::string& query,
        const std::string& relType = "",
        size_t maxHops = 2,
        size_t topKSeeds = 5,
        size_t topKResults = 10,
        Direction direction = Direction::Out,
        float decay = 0.8f,
        float threshold = 0.3f
    ) const {
        MultiHopOptions options;
        options.relType = relType;
        options.maxHops = maxHops;
        options.topKSeeds = topKSeeds;
        options.topKResults = topKResults;
        options.direction = direction;
        options.decay = decay;
        options.threshold = threshold;
        return semanticMultiHop(query, options);
    }

    /**
     * Find a semantically-rooted path to a target node.
     *
     * Starts from the most similar nodes and tries every seed that reaches the
     * target, returning the highest-scoring path: a slightly less similar seed
     * one hop from the target beats a better seed five hops away. Seeds arrive
     * best-first, so a tie goes to the more similar one. The score is
     * max(seedSimilarity, 0) * decay^hops; as in semanticMultiHop, a negative
     * seed passes no relevance along rather than "decaying" toward zero and
     * outscoring itself. Lower `threshold` (to -1) to search from any seed.
     *
     * Filtering: seeds are chosen among passing nodes, and the *target must
     * pass* or the result is nullopt. By default the path may run through
     * failing nodes (it is the fewest-hops path, as without a filter); with
     * filter.strictFilter it is the fewest-hops path that uses passing nodes
     * only, which can be longer or not exist.
     */
    std::optional<SemanticSearchResult> semanticPath(
        const std::string& query,
        const NodeRef& targetRef,
        const std::string& relType = "",
        size_t maxHops = 5,
        size_t topKSeeds = 3,
        Direction direction = Direction::Out,
        float decay = 0.9f,
        float threshold = 0.0f,
        const NodeFilterOptions& filter = NodeFilterOptions()
    ) const {
        NodeGate gate = makeGate(filter);
        if (gate && !gate(targetRef)) return std::nullopt;

        auto seeds = seedSearchVector(embedding_store_.encodeQuery(query), topKSeeds, "",
                                      threshold, gate);

        std::optional<SemanticSearchResult> best;
        for (const auto& seed : seeds) {
            std::vector<NodeRef> nodes;
            if (filter.strictFilter && gate) {
                auto gated = gatedShortestPath(seed.node_ref, targetRef, relType, maxHops,
                                               direction, gate);
                if (!gated.has_value()) continue;
                nodes = std::move(*gated);
            } else {
                auto path = graph_.shortestPath(seed.node_ref, targetRef, relType, maxHops, direction);
                if (!path.has_value()) continue;
                nodes = path->nodes;
            }

            size_t hops = nodes.size() - 1;
            float score = std::max(seed.score, 0.0f) * std::pow(decay, static_cast<float>(hops));
            if (!best.has_value() || score > best->score) {
                best = SemanticSearchResult{targetRef, score, hops, std::move(nodes)};
            }
        }

        return best;
    }

private:
    /** Fewest-hops path from start to end whose every node passes `gate`. */
    std::optional<std::vector<NodeRef>> gatedShortestPath(
        const NodeRef& start, const NodeRef& end, const std::string& relType,
        size_t maxHops, Direction direction, const NodeGate& gate
    ) const {
        if (!gate(start) || !gate(end)) return std::nullopt;
        if (start == end) return std::vector<NodeRef>{start};

        Adjacency adjacency(graph_, relType, direction, false);
        std::set<NodeRef> visited{start};
        std::queue<std::vector<NodeRef>> queue;
        queue.push({start});
        while (!queue.empty()) {
            std::vector<NodeRef> path = std::move(queue.front());
            queue.pop();
            if (path.size() - 1 >= maxHops) continue;
            for (const auto& [neighbor, edge] : adjacency.adjacent(path.back())) {
                if (visited.count(neighbor) || !gate(neighbor)) continue;
                std::vector<NodeRef> next = path;
                next.push_back(neighbor);
                if (neighbor == end) return next;
                visited.insert(neighbor);
                queue.push(std::move(next));
            }
        }
        return std::nullopt;
    }

    /** Shortest route from any of `members` to a kept node outside them. */
    std::optional<std::vector<NodeRef>> shortestLink(
        const std::vector<NodeRef>& members, const std::set<NodeRef>& keep,
        const Adjacency& adjacency, size_t maxHops, const NodeGate& gate
    ) const {
        std::set<NodeRef> visited(members.begin(), members.end());
        std::queue<std::vector<NodeRef>> queue;
        for (const NodeRef& member : members) queue.push({member});
        while (!queue.empty()) {
            std::vector<NodeRef> path = std::move(queue.front());
            queue.pop();
            if (path.size() - 1 >= maxHops) continue;
            for (const auto& [neighbor, edge] : adjacency.adjacent(path.back())) {
                if (visited.count(neighbor)) continue;
                std::vector<NodeRef> next = path;
                next.push_back(neighbor);
                if (keep.count(neighbor)) return next;
                if (gate && !gate(neighbor)) continue;
                if (!graph_.hasNode(neighbor)) continue;
                visited.insert(neighbor);
                queue.push(std::move(next));
            }
        }
        return std::nullopt;
    }

    /**
     * Grow `keep` with connector nodes until it is as connected as the hop
     * budget allows; the connectors added are appended to `added`, in order.
     *
     * Greedy: each round finds, over all pairs of components, the shortest
     * route (at most connectMaxHops edges, honouring relType/direction) whose
     * interior nodes lie outside `keep`, adds its interior, and recomputes
     * components. Each round merges at least two, so it terminates; it stops
     * when one component remains or no route fits the budget. `gate` (set for
     * strictFilter) bars failing nodes from serving as connectors.
     */
    void connectComponents(std::set<NodeRef>& keep, std::vector<NodeRef>& added,
                           const MultiHopOptions& options, const NodeGate& gate) const {
        Adjacency adjacency(graph_, options.relType, options.direction, true);

        while (keep.size() > 1) {
            // Weakly connected components of the induced subgraph.
            std::map<NodeRef, NodeRef> parent;
            for (const NodeRef& ref : keep) parent[ref] = ref;
            auto find = [&](NodeRef ref) {
                while (parent[ref] != ref) {
                    parent[ref] = parent[parent[ref]];
                    ref = parent[ref];
                }
                return ref;
            };
            for (const Edge& edge : adjacency.edges()) {
                if (keep.count(edge.source) && keep.count(edge.target)) {
                    parent[find(edge.source)] = find(edge.target);
                }
            }
            std::map<NodeRef, std::vector<NodeRef>> groups;
            for (const NodeRef& ref : keep) groups[find(ref)].push_back(ref);  // members arrive sorted
            if (groups.size() <= 1) break;

            std::vector<std::vector<NodeRef>> ordered;
            for (auto& [root, members] : groups) ordered.push_back(std::move(members));
            std::sort(ordered.begin(), ordered.end(),
                      [](const std::vector<NodeRef>& a, const std::vector<NodeRef>& b) {
                          return a.front() < b.front();
                      });

            std::optional<std::vector<NodeRef>> bestRoute;
            for (const auto& members : ordered) {
                auto route = shortestLink(members, keep, adjacency, options.connectMaxHops, gate);
                if (route.has_value() && (!bestRoute.has_value() || route->size() < bestRoute->size())) {
                    bestRoute = std::move(route);
                }
            }
            if (!bestRoute.has_value()) break;
            for (size_t i = 1; i + 1 < bestRoute->size(); ++i) {
                keep.insert((*bestRoute)[i]);
                added.push_back((*bestRoute)[i]);
            }
        }
    }

public:
    /**
     * Extract the slice of the graph a query is actually about.
     *
     * Runs semanticMultiHop, then returns a new SemanticGraph holding exactly
     * those nodes, every edge induced between them, and their embeddings - so
     * the result can be traversed, searched or serialized on its own. This is
     * what makes an ISON graph useful as LLM context: a small, on-topic graph
     * to inject rather than the whole store.
     *
     * options.connect (and connectMaxHops) additionally bridges the selected
     * nodes with connector nodes; see MultiHopOptions.
     *
     * @throws EmbeddingError if options.connectMaxHops is 0
     */
    SemanticGraph semanticSubgraph(
        const std::string& query,
        const MultiHopOptions& options,
        const std::string& name = ""
    ) const {
        if (options.connectMaxHops < 1) {
            throw EmbeddingError("connectMaxHops must be >= 1, got " +
                                 std::to_string(options.connectMaxHops));
        }

        auto results = semanticMultiHop(query, options);

        std::set<NodeRef> keep;
        std::vector<NodeRef> order;  // selected nodes best-first, then connectors
        for (const auto& result : results) {
            if (keep.insert(result.node_ref).second) order.push_back(result.node_ref);
        }
        if (options.connect && !keep.empty()) {
            NodeGate gate = options.strictFilter ? makeGate(options) : NodeGate();
            connectComponents(keep, order, options, gate);
        }

        SemanticGraph sub(
            name.empty() ? graph_.name + "_subgraph" : name,
            embedding_store_.encoder(),
            false,
            embed_fields_
        );

        for (const NodeRef& ref : order) {
            if (!graph_.hasNode(ref)) continue;
            const Node& node = graph_.getNode(ref.type, ref.id);
            sub.graph_.addNode(node.type, node.id, node.properties);

            const EmbeddingRecord* record = embedding_store_.get(ref);
            if (record != nullptr) {
                sub.embedding_store_.add(ref, record->text, record->vector);
            }
        }

        for (const auto& relType : graph_.edgeTypes()) {
            if (!options.relType.empty() && relType != options.relType) continue;
            for (const auto& edge : graph_.getEdges(relType)) {
                if (keep.count(edge.source) && keep.count(edge.target) &&
                    sub.graph_.hasNode(edge.source) && sub.graph_.hasNode(edge.target)) {
                    sub.graph_.addEdge(edge.relType, edge.source, edge.target, edge.properties);
                }
            }
        }

        return sub;
    }

    /** Embed all nodes in the graph, in one batched encoder call. */
    template<typename F>
    size_t embedAllNodes(F textFn, bool skipExisting = false) {
        std::vector<std::pair<NodeRef, std::string>> batch;

        graph_.forEachNode([&](const Node& node) {
            NodeRef ref(node.type, node.id);
            if (skipExisting && embedding_store_.get(ref) != nullptr) return;
            std::string text = textFn(node);
            if (!text.empty()) {
                batch.emplace_back(ref, text);
            }
        });

        return embedNodes(batch);
    }

    size_t embedAllNodes(bool skipExisting = false) {
        return embedAllNodes([this](const Node& node) -> std::string {
            std::string text = embedTextFor(node.properties);
            return text.empty() ? node.type + ":" + node.id : text;
        }, skipExisting);
    }

    // =========================================================================
    // Model lifecycle
    // =========================================================================

    /**
     * Nodes whose stored vector was not made by the current encoder.
     *
     * Stale means the vector's recorded model differs from the encoder's
     * modelName(), or its length differs from the encoder's dimension().
     * Vectors recorded as "unknown"/"provided" (from ports that store no
     * model, or supplied without an encoder) cannot be judged by model and are
     * left out unless `includeUnknown`; a dimension mismatch always counts.
     * Sorted by (type, id).
     *
     * @throws NoEncoderError if there is no encoder to compare against
     */
    std::vector<NodeRef> staleEmbeddings(bool includeUnknown = false) const {
        if (!embedding_store_.encoder()) {
            throw NoEncoderError("No encoder provided: cannot tell which embeddings are stale");
        }
        return embedding_store_.stale(std::nullopt, std::nullopt, includeUnknown);
    }

    /**
     * Re-encode every stale embedding with the current encoder.
     *
     * Each vector is rebuilt from the text it was originally embedded from (so
     * a custom embed text survives); a vector stored with no text falls back
     * to the node's embed fields, then to "type:id". If the encoder's
     * dimension differs from the store's, every stored vector is stale, so the
     * store is emptied, its pinned dimension reset, and everything re-embedded.
     *
     * @param includeUnknown also re-embed vectors of unknown provenance
     * @param batchSize      texts per encoder call
     * @return number of embeddings rewritten
     */
    size_t reembedStale(bool includeUnknown = false, size_t batchSize = 256) {
        std::vector<NodeRef> stale = staleEmbeddings(includeUnknown);
        if (stale.empty()) return 0;

        std::vector<std::pair<NodeRef, std::string>> pairs;
        for (const NodeRef& ref : stale) {
            const EmbeddingRecord* record = embedding_store_.get(ref);
            if (record == nullptr) continue;
            std::string text = record->text;
            if (text.empty()) {
                if (graph_.hasNode(ref)) {
                    text = embedTextFor(graph_.getNode(ref.type, ref.id).properties);
                }
                if (text.empty()) text = ref.type + ":" + ref.id;
            }
            pairs.emplace_back(ref, std::move(text));
        }

        const auto& encoder = embedding_store_.encoder();
        if (embedding_store_.dimension().has_value() &&
            *embedding_store_.dimension() != encoder->dimension()) {
            embedding_store_.clear(true);
        }

        if (batchSize == 0) batchSize = pairs.size() ? pairs.size() : 1;
        size_t count = 0;
        for (size_t start = 0; start < pairs.size(); start += batchSize) {
            size_t end = std::min(start + batchSize, pairs.size());
            count += embedNodes(std::vector<std::pair<NodeRef, std::string>>(
                pairs.begin() + start, pairs.begin() + end));
        }
        return count;
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    /** Sidecar path holding the embeddings for a graph saved at `path`. */
    static std::string embeddingsPathFor(const std::string& path) {
        return path + EMBEDDING_SIDECAR_SUFFIX;
    }

    /**
     * Save the graph to a file, and its embeddings alongside it.
     *
     * The ISON graph format carries nodes and edges only, so embeddings go to
     * a JSON sidecar at `<path>.embeddings.json` (an ison-embeddings payload,
     * the same file the Python port writes) that load() picks up
     * automatically. Without it a save/load round trip returns a graph whose
     * searches all come back empty.
     *
     * Both files are written atomically (temp file in the same directory,
     * flush, rename over the target), so a crash never leaves a truncated
     * file. The pair cannot be replaced as one unit, so the sidecar also
     * records a graphDigest() of the graph it belongs to; load() checks it and
     * catches a crash between the two writes, a sidecar copied from another
     * save, and a graph edited afterwards. The digest is taken of the graph
     * *as it reads back from the text just written*, so a round trip that
     * normalises a value does not cause a false mismatch. The sidecar also
     * records the encoder's model and version when they are known.
     *
     * With `embeddings` false an existing sidecar is left in place (and will
     * fail the digest check on load, as it no longer matches).
     *
     * @param format "auto" (by extension), "ison" or "isonl"
     * @param embeddings write the sidecar (false for a graph-only save)
     */
    void save(const std::string& path, const std::string& format = "auto",
              bool embeddings = true) const {
        std::string actualFormat = format;
        if (format == "auto") {
            actualFormat = (path.size() >= 6 && path.compare(path.size() - 6, 6, ".isonl") == 0)
                           ? "isonl" : "ison";
        }
        const bool isonl = (actualFormat == "isonl");

        std::string text = isonl ? graph_.toIsonl() : graph_.toIson();
        detail::atomicWriteText<GraphError>(path, text);

        if (embeddings) {
            ison::Document reread = isonl ? ison::loads_isonl(text) : ison::loads(text);
            embedding_store_.saveEmbeddings(embeddingsPathFor(path),
                                            detail::digestOfDocument(reread));
        }
    }

    /**
     * Load a graph from a file, restoring its embeddings from the sidecar.
     *
     * Loading never auto-embeds: the vectors come from the sidecar, already
     * computed, so no encoder is needed to load and none overwrites what was
     * saved. Pass the encoder the graph was saved with so text searches and
     * later auto-embedding stay in the same vector space. A missing sidecar is
     * not an error (the graph loads with no embeddings), but a sidecar that is
     * present and unreadable is.
     *
     * @param encoder encoder for the loaded graph (may be null)
     * @param format "auto" (by extension), "ison" or "isonl"
     * @param embeddings restore the sidecar written by save()
     * @param onMismatch what to do when the sidecar carries a graph digest that
     *        does not match the graph file (a sidecar from a different save, or
     *        a graph edited since): Error (default) throws
     *        GraphEmbeddingMismatchError; Warn warns and loads the sidecar's
     *        vectors anyway; Reembed discards them and re-embeds every node
     *        with the encoder (embedAllNodes). A sidecar without a digest
     *        (older releases, other ports), or with a digest of a basis this
     *        build does not compute, is loaded as before and never checked.
     * @param version encoder/model version string for the loaded graph's
     *        store. If both it (or the encoder's own version()) and the
     *        sidecar's encoder_version are set and differ, a warning is issued.
     * @param onModelMismatch for text searches whose encoder model differs from
     *        the stored vectors' model (see EmbeddingStore)
     * @param ann opt in to approximate search (see EmbeddingStore::enableAnn).
     *        The index is never saved; it is rebuilt lazily from the loaded
     *        vectors on the first search.
     */
    static SemanticGraph load(
        const std::string& path,
        std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
        const std::string& format = "auto",
        bool embeddings = true,
        bool autoEmbed = true,
        std::vector<std::string> embedFields = {"name", "description", "content", "text"},
        GraphMismatchMode onMismatch = GraphMismatchMode::Error,
        std::optional<std::string> version = std::nullopt,
        ModelMismatchMode onModelMismatch = ModelMismatchMode::Warn,
        std::optional<AnnOptions> ann = std::nullopt
    ) {
        // Read and parse the text here rather than through ISONGraph::load: the
        // digest has to be taken from the document (it records which ids were
        // integers and which cells were null, and the graph does not).
        std::string content;
        {
            std::ifstream file(path);
            if (!file) {
                throw GraphError("Cannot open file for reading: " + path);
            }
            std::ostringstream ss;
            ss << file.rdbuf();
            content = ss.str();
        }

        std::string actualFormat = format;
        if (format == "auto") {
            actualFormat = (path.size() >= 6 && path.compare(path.size() - 6, 6, ".isonl") == 0)
                           ? "isonl" : "ison";
        }

        size_t lastSlash = path.find_last_of("/\\");
        size_t lastDot = path.find_last_of('.');
        size_t nameStart = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;
        std::string graphName = path.substr(
            nameStart, lastDot == std::string::npos ? std::string::npos : lastDot - nameStart);

        ison::Document doc = (actualFormat == "isonl") ? ison::loads_isonl(content)
                                                       : ison::loads(content);
        ISONGraph base = ISONGraph::buildFromDocument(doc, graphName);

        SemanticGraph result(base.name, std::move(encoder), autoEmbed, std::move(embedFields),
                             onModelMismatch, std::move(version), std::move(ann));
        result.graph_ = std::move(base);

        if (embeddings) {
            std::string sidecar = embeddingsPathFor(path);
            std::ifstream probe(sidecar, std::ios::binary);
            if (probe.is_open()) {
                probe.close();

                ExportedEmbeddings payload;
                try {
                    payload = EmbeddingStore::parseJson(EmbeddingStore::readFile(sidecar));
                } catch (const EmbeddingError& error) {
                    throw EmbeddingError("unreadable embeddings sidecar " + sidecar + ": " +
                                         error.what());
                }

                std::optional<std::string> problem;
                if (payload.meta.graphDigest.has_value() &&
                    payload.meta.graphDigestBasis.value_or(GRAPH_DIGEST_BASIS) ==
                        std::string(GRAPH_DIGEST_BASIS)) {
                    std::string actual = detail::digestOfDocument(doc);
                    if (*payload.meta.graphDigest != actual) {
                        problem = "embeddings sidecar " + sidecar + " does not belong to graph " +
                                  path + ": it was written for graph digest " +
                                  *payload.meta.graphDigest + ", but the graph reads as " +
                                  actual + " (the sidecar is from a different save, or the "
                                  "graph was edited since). Pass GraphMismatchMode::Reembed to "
                                  "rebuild the vectors or GraphMismatchMode::Warn to use them "
                                  "anyway.";
                    }
                }

                if (problem.has_value() && onMismatch == GraphMismatchMode::Error) {
                    throw GraphEmbeddingMismatchError(*problem);
                }
                if (problem.has_value() && onMismatch == GraphMismatchMode::Reembed) {
                    result.embedAllNodes();
                } else {
                    if (problem.has_value()) emitWarning(*problem);
                    result.embedding_store_.importEmbeddings(payload);

                    const auto& storedVersion = result.embedding_store_.lastImportMeta().encoderVersion;
                    auto currentVersion = result.embedding_store_.version();
                    if (storedVersion.has_value() && currentVersion.has_value() &&
                        *storedVersion != *currentVersion) {
                        emitWarning("embeddings were made by encoder version '" + *storedVersion +
                                    "', but the encoder in use is version '" + *currentVersion + "'");
                    }
                }
            }
        }

        return result;
    }
};

} // namespace isongraph_vector

#endif // ISONGRAPH_VECTOR_HPP
