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
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <functional>
#include "ison_graph.hpp"

namespace isongraph_vector {

constexpr const char* VERSION = "1.0.0";

/**
 * Portable embedding payload shared by every language port, so a store built
 * in one can be searched by another.
 */
constexpr const char* EMBEDDING_FORMAT = "ison-embeddings";
constexpr int EMBEDDING_FORMAT_VERSION = 1;

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

// =============================================================================
// Embedding Encoder Interface
// =============================================================================

class EmbeddingEncoder {
public:
    virtual ~EmbeddingEncoder() = default;
    virtual size_t dimension() const = 0;
    virtual std::vector<float> encode(const std::string& text) const = 0;

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

public:
    explicit MockEncoder(size_t dimension = 384) : dim_(dimension) {
        if (dimension == 0) {
            throw EmbeddingError("dimension must be positive");
        }
    }

    size_t dimension() const override {
        return dim_;
    }

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
};

/** Payload written by exportEmbeddings, readable by every language port. */
struct ExportedEmbeddings {
    std::string format = EMBEDDING_FORMAT;
    int version = EMBEDDING_FORMAT_VERSION;
    std::optional<size_t> dimension;
    size_t count = 0;
    std::vector<ExportedEmbedding> embeddings;
};

// =============================================================================
// Embedding Store
// =============================================================================

class EmbeddingStore {
private:
    std::unordered_map<std::string, EmbeddingRecord> embeddings_;
    std::shared_ptr<EmbeddingEncoder> encoder_;
    std::optional<size_t> dimension_;

    std::string nodeRefToKey(const NodeRef& ref) const {
        return ref.type + ":" + ref.id;
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

public:
    explicit EmbeddingStore(std::shared_ptr<EmbeddingEncoder> encoder = nullptr)
        : encoder_(std::move(encoder)) {}

    /** Vector length this store holds, or nullopt while it is still empty. */
    std::optional<size_t> dimension() const { return dimension_; }

    void setEncoder(std::shared_ptr<EmbeddingEncoder> encoder) { encoder_ = std::move(encoder); }
    const std::shared_ptr<EmbeddingEncoder>& encoder() const { return encoder_; }

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
        embeddings_[key] = EmbeddingRecord{key, node_ref, std::move(vec), text};
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
        return embeddings_.erase(key) > 0;
    }

    std::vector<SimilarityResult> similaritySearch(
        const std::string& query,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f
    ) const {
        if (!encoder_) {
            throw NoEncoderError("No encoder provided for text query");
        }

        auto queryVector = encoder_->encode(query);
        return similaritySearchVector(queryVector, topK, nodeType, threshold);
    }

    std::vector<SimilarityResult> similaritySearchVector(
        const std::vector<float>& queryVector,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f
    ) const {
        checkQuery(queryVector);

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

        std::sort(results.begin(), results.end());
        if (results.size() > topK) {
            results.resize(topK);
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

    size_t count() const {
        return embeddings_.size();
    }

    void clear() {
        embeddings_.clear();
        dimension_.reset();
    }

    /** Serialize every embedding into the portable cross-language payload. */
    ExportedEmbeddings exportEmbeddings() const {
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
                record.vector
            });
        }
        std::sort(payload.embeddings.begin(), payload.embeddings.end(),
                  [](const ExportedEmbedding& a, const ExportedEmbedding& b) {
                      return std::tie(a.type, a.id) < std::tie(b.type, b.id);
                  });
        payload.count = payload.embeddings.size();
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
        if (replace) {
            clear();
        }

        size_t written = 0;
        for (const auto& item : payload.embeddings) {
            add(NodeRef(item.type, item.id), item.text, item.vector);
            ++written;
        }
        return written;
    }

    /** Serialize the portable payload to a JSON string. */
    std::string toJson() const {
        auto payload = exportEmbeddings();
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
                << ",\"vector\":[";
            for (size_t j = 0; j < item.vector.size(); ++j) {
                if (j > 0) out << ",";
                out << item.vector[j];
            }
            out << "]}";
        }
        out << "]}";
        return out.str();
    }

    /**
     * Load embeddings from a JSON payload written by any language port.
     *
     * A focused reader for this one schema rather than a general JSON parser:
     * it pulls the fields it knows and rejects anything that is not an
     * ison-embeddings payload of a version it understands.
     */
    size_t fromJson(const std::string& json, bool replace = false) {
        ExportedEmbeddings payload;
        payload.format = jsonFindString(json, "format", 0).value_or("");
        payload.version = static_cast<int>(jsonFindNumber(json, "version", 0).value_or(-1));

        if (payload.format != std::string(EMBEDDING_FORMAT)) {
            throw EmbeddingError(std::string("not an ") + EMBEDDING_FORMAT +
                                 " payload: format=" + payload.format);
        }
        if (payload.version != EMBEDDING_FORMAT_VERSION) {
            throw EmbeddingError(std::string("unsupported ") + EMBEDDING_FORMAT + " version " +
                                 std::to_string(payload.version) + " (this build reads version " +
                                 std::to_string(EMBEDDING_FORMAT_VERSION) + ")");
        }

        size_t pos = json.find("\"embeddings\"");
        if (pos == std::string::npos) {
            throw EmbeddingError("payload has no embeddings array");
        }
        pos = json.find('[', pos);
        if (pos == std::string::npos) {
            throw EmbeddingError("payload has no embeddings array");
        }

        while (true) {
            size_t objStart = json.find('{', pos);
            if (objStart == std::string::npos) break;
            size_t objEnd = json.find('}', objStart);
            // The vector array closes with ']' before the object's '}', so the
            // first '}' after the object start is genuinely its end.
            if (objEnd == std::string::npos) break;

            std::string object = json.substr(objStart, objEnd - objStart + 1);
            ExportedEmbedding item;
            item.type = jsonFindString(object, "type", 0).value_or("");
            // Ports with typed node ids write an int id unquoted ("id":1), so
            // accept either form and keep the exact text of the id either way.
            item.id = jsonFindScalar(object, "id").value_or("");
            item.id_type = jsonFindString(object, "id_type", 0).value_or("str");
            item.text = jsonFindString(object, "text", 0).value_or("");
            item.vector = jsonFindFloatArray(object, "vector");
            if (!item.type.empty() && !item.vector.empty()) {
                payload.embeddings.push_back(std::move(item));
            }

            pos = objEnd + 1;
            size_t next = json.find_first_not_of(" \t\r\n", pos);
            if (next == std::string::npos || json[next] != ',') break;
            pos = next + 1;
        }

        payload.count = payload.embeddings.size();
        return importEmbeddings(payload, replace);
    }

private:
    static std::string jsonString(const std::string& value) {
        std::string out = "\"";
        for (char c : value) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:   out += c;
            }
        }
        return out + "\"";
    }

    static std::optional<size_t> jsonFindKey(const std::string& json, const std::string& key,
                                             size_t from) {
        std::string needle = "\"" + key + "\"";
        size_t pos = json.find(needle, from);
        if (pos == std::string::npos) return std::nullopt;
        pos = json.find(':', pos + needle.size());
        if (pos == std::string::npos) return std::nullopt;
        return json.find_first_not_of(" \t\r\n", pos + 1);
    }

    static std::optional<std::string> jsonFindString(const std::string& json,
                                                     const std::string& key, size_t from) {
        auto start = jsonFindKey(json, key, from);
        if (!start.has_value() || *start >= json.size() || json[*start] != '"') {
            return std::nullopt;
        }
        std::string out;
        for (size_t i = *start + 1; i < json.size(); ++i) {
            if (json[i] == '\\' && i + 1 < json.size()) {
                char next = json[++i];
                switch (next) {
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    default:  out += next;
                }
                continue;
            }
            if (json[i] == '"') return out;
            out += json[i];
        }
        return std::nullopt;
    }

    /**
     * A value that may be quoted or bare, returned as its literal text.
     *
     * Node ids cross the wire as `"id":"007"` from the ports whose ids are
     * always strings and as `"id":1` from those that keep an integer id, and
     * both have to land on the same C++ string id.
     */
    static std::optional<std::string> jsonFindScalar(const std::string& json,
                                                     const std::string& key) {
        auto start = jsonFindKey(json, key, 0);
        if (!start.has_value() || *start >= json.size()) return std::nullopt;
        if (json[*start] == '"') {
            return jsonFindString(json, key, 0);
        }
        size_t end = json.find_first_of(",}]", *start);
        if (end == std::string::npos) return std::nullopt;
        std::string raw = json.substr(*start, end - *start);
        while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) {
            raw.pop_back();
        }
        // Integers arrive as "1", not "1.0"; leave anything else as written.
        return raw.empty() ? std::nullopt : std::optional<std::string>(raw);
    }

    static std::optional<double> jsonFindNumber(const std::string& json, const std::string& key,
                                                size_t from) {
        auto start = jsonFindKey(json, key, from);
        if (!start.has_value()) return std::nullopt;
        try {
            size_t used = 0;
            double value = std::stod(json.substr(*start), &used);
            if (used == 0) return std::nullopt;
            return value;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    static std::vector<float> jsonFindFloatArray(const std::string& json, const std::string& key) {
        std::vector<float> values;
        auto start = jsonFindKey(json, key, 0);
        if (!start.has_value() || *start >= json.size() || json[*start] != '[') return values;

        size_t end = json.find(']', *start);
        if (end == std::string::npos) return values;

        std::string body = json.substr(*start + 1, end - *start - 1);
        std::istringstream in(body);
        std::string token;
        while (std::getline(in, token, ',')) {
            try {
                values.push_back(std::stof(token));
            } catch (const std::exception&) {
                // a malformed element makes the whole vector untrustworthy
                return {};
            }
        }
        return values;
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

/** Options for semanticMultiHop. */
struct MultiHopOptions {
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

public:
    SemanticGraph(
        const std::string& name = "semantic_graph",
        std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
        bool autoEmbed = true,
        std::vector<std::string> embedFields = {"name", "description", "content", "text"}
    )
        : graph_(name)
        , embedding_store_(encoder)
        , auto_embed_(autoEmbed)
        , embed_fields_(std::move(embedFields))
    {}

    ISONGraph& graph() { return graph_; }
    const ISONGraph& graph() const { return graph_; }
    EmbeddingStore& embeddingStore() { return embedding_store_; }
    const EmbeddingStore& embeddingStore() const { return embedding_store_; }

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
     */
    Node& addNode(
        const std::string& nodeType,
        const std::string& nodeId,
        const Properties& properties = {}
    ) {
        Node& node = graph_.addNode(nodeType, nodeId, properties);
        if (auto_embed_ && embedding_store_.encoder()) {
            std::string text = embedTextFor(properties);
            if (!text.empty()) {
                embedNode(NodeRef(nodeType, nodeId), text);
            }
        }
        return node;
    }

    /** Add a node with an explicit embedding text. */
    Node& addNodeWithEmbed(
        const std::string& nodeType,
        const std::string& nodeId,
        const Properties& properties = {},
        const std::string& embedText = ""
    ) {
        if (embedText.empty()) {
            return addNode(nodeType, nodeId, properties);
        }
        Node& node = graph_.addNode(nodeType, nodeId, properties);
        embedNode(NodeRef(nodeType, nodeId), embedText);
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

    void removeNode(const std::string& nodeType, const std::string& nodeId) {
        embedding_store_.remove(NodeRef(nodeType, nodeId));
        graph_.removeNode(nodeType, nodeId);
    }

    std::vector<SimilarityResult> similaritySearch(
        const std::string& query,
        size_t topK = 10,
        const std::string& nodeType = "",
        float threshold = 0.0f
    ) const {
        return embedding_store_.similaritySearch(query, topK, nodeType, threshold);
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
     * 3. Score by max(seedSimilarity, 0) * decay^hopCount, optionally blended
     *    with the relevance of each node itself
     */
    std::vector<SemanticSearchResult> semanticMultiHop(
        const std::string& query,
        const MultiHopOptions& options
    ) const {
        if (options.blend < 0.0f || options.blend > 1.0f) {
            throw EmbeddingError("blend must be in [0, 1], got " + std::to_string(options.blend));
        }
        if (!embedding_store_.encoder()) {
            throw NoEncoderError("No encoder provided for text query");
        }

        auto queryVector = embedding_store_.encoder()->encode(query);

        // One scan serves both seed selection and per-node relevance.
        auto scores = embedding_store_.scoreMap(queryVector);
        if (scores.empty()) {
            return {};
        }

        auto seeds = embedding_store_.similaritySearchVector(
            queryVector, options.topKSeeds, "", options.threshold);
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

            // BFS traversal
            std::queue<std::tuple<NodeRef, std::vector<NodeRef>, size_t>> queue;
            queue.push({seed.node_ref, {seed.node_ref}, 0});

            std::set<std::string> visited;
            visited.insert(seedKey);

            while (!queue.empty()) {
                auto [nodeRef, path, hop] = queue.front();
                queue.pop();

                if (hop >= options.maxHops) continue;

                auto neighbors = graph_.neighbors(nodeRef, options.relType, options.direction);

                for (const auto& neighbor : neighbors) {
                    std::string neighborKey = nodeRefToKey(neighbor);

                    if (visited.count(neighborKey)) continue;
                    visited.insert(neighborKey);

                    std::vector<NodeRef> newPath = path;
                    newPath.push_back(neighbor);
                    size_t newHop = hop + 1;
                    float newScore = combined(
                        traversalBase * std::pow(options.decay, static_cast<float>(newHop)),
                        neighborKey);

                    auto it = results.find(neighborKey);
                    if (it == results.end() || it->second.score < newScore) {
                        results[neighborKey] = SemanticSearchResult{
                            neighbor, newScore, newHop, newPath
                        };
                    }

                    queue.push({neighbor, newPath, newHop});
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
     * Starts from the most similar nodes and returns the first path found to
     * the target, scored by the similarity of the seed decayed per hop.
     */
    std::optional<SemanticSearchResult> semanticPath(
        const std::string& query,
        const NodeRef& targetRef,
        const std::string& relType = "",
        size_t maxHops = 5,
        size_t topKSeeds = 3,
        Direction direction = Direction::Out,
        float decay = 0.9f,
        float threshold = 0.0f
    ) const {
        auto seeds = embedding_store_.similaritySearch(query, topKSeeds, "", threshold);

        for (const auto& seed : seeds) {
            auto path = graph_.shortestPath(seed.node_ref, targetRef, relType, maxHops, direction);
            if (path.has_value()) {
                size_t hops = path->length();
                return SemanticSearchResult{
                    targetRef,
                    seed.score * std::pow(decay, static_cast<float>(hops)),
                    hops,
                    path->nodes
                };
            }
        }

        return std::nullopt;
    }

    /**
     * Extract the slice of the graph a query is actually about.
     *
     * Runs semanticMultiHop, then returns a new SemanticGraph holding exactly
     * those nodes, every edge induced between them, and their embeddings - so
     * the result can be traversed, searched or serialized on its own. This is
     * what makes an ISON graph useful as LLM context: a small, on-topic graph
     * to inject rather than the whole store.
     */
    SemanticGraph semanticSubgraph(
        const std::string& query,
        const MultiHopOptions& options,
        const std::string& name = ""
    ) const {
        auto results = semanticMultiHop(query, options);

        std::set<std::string> keep;
        for (const auto& result : results) {
            keep.insert(result.node_ref.type + ":" + result.node_ref.id);
        }

        SemanticGraph sub(
            name.empty() ? graph_.name + "_subgraph" : name,
            embedding_store_.encoder(),
            false,
            embed_fields_
        );

        for (const auto& result : results) {
            if (!graph_.hasNode(result.node_ref)) continue;
            const Node& node = graph_.getNode(result.node_ref.type, result.node_ref.id);
            sub.graph_.addNode(node.type, node.id, node.properties);

            const EmbeddingRecord* record = embedding_store_.get(result.node_ref);
            if (record != nullptr) {
                sub.embedding_store_.add(result.node_ref, record->text, record->vector);
            }
        }

        for (const auto& relType : graph_.edgeTypes()) {
            if (!options.relType.empty() && relType != options.relType) continue;
            for (const auto& edge : graph_.getEdges(relType)) {
                std::string sourceKey = edge.source.type + ":" + edge.source.id;
                std::string targetKey = edge.target.type + ":" + edge.target.id;
                if (keep.count(sourceKey) && keep.count(targetKey)) {
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
};

} // namespace isongraph_vector

#endif // ISONGRAPH_VECTOR_HPP
