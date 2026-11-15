/**
 * ISONGraph Vector C++ Tests
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <random>
#include <set>
#include "../include/isongraph_vector.hpp"

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

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " != " #b); \
    }

#define ASSERT_NE(a, b) \
    if ((a) == (b)) { \
        throw std::runtime_error("Assertion failed: " #a " == " #b); \
    }

#define ASSERT_TRUE(expr) \
    if (!(expr)) { \
        throw std::runtime_error("Assertion failed: " #expr); \
    }

#define ASSERT_FALSE(expr) \
    if (expr) { \
        throw std::runtime_error("Assertion failed: " #expr " should be false"); \
    }

#define ASSERT_GE(a, b) \
    if ((a) < (b)) { \
        throw std::runtime_error("Assertion failed: " #a " < " #b); \
    }

#define ASSERT_GT(a, b) \
    if ((a) <= (b)) { \
        throw std::runtime_error("Assertion failed: " #a " <= " #b); \
    }

#define ASSERT_NULL(ptr) \
    if ((ptr) != nullptr) { \
        throw std::runtime_error("Assertion failed: " #ptr " is not null"); \
    }

#define ASSERT_NOT_NULL(ptr) \
    if ((ptr) == nullptr) { \
        throw std::runtime_error("Assertion failed: " #ptr " is null"); \
    }

// =============================================================================
// MockEncoder Tests
// =============================================================================

TEST(mock_encoder_dimension) {
    MockEncoder encoder(384);
    ASSERT_EQ(encoder.dimension(), 384);
}

TEST(mock_encoder_custom_dimension) {
    MockEncoder encoder(128);
    ASSERT_EQ(encoder.dimension(), 128);
}

TEST(mock_encoder_deterministic) {
    MockEncoder encoder;
    auto v1 = encoder.encode("hello");
    auto v2 = encoder.encode("hello");

    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(v1[i], v2[i]);
    }
}

TEST(mock_encoder_different_texts) {
    MockEncoder encoder;
    auto v1 = encoder.encode("hello");
    auto v2 = encoder.encode("world");

    bool different = false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > 0.001f) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different);
}

TEST(mock_encoder_batch) {
    MockEncoder encoder;
    std::vector<std::string> texts = {"hello", "world"};
    auto vectors = encoder.encodeBatch(texts);

    ASSERT_EQ(vectors.size(), 2);
    ASSERT_EQ(vectors[0].size(), 384);
    ASSERT_EQ(vectors[1].size(), 384);
}

// =============================================================================
// EmbeddingStore Tests
// =============================================================================

TEST(embedding_store_add_and_get) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    NodeRef ref("person", "1");
    store.add(ref, "Alice is an engineer");

    auto record = store.get(ref);
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, "Alice is an engineer");
    ASSERT_EQ(record->vector.size(), 384);
}

TEST(embedding_store_precomputed_vector) {
    EmbeddingStore store;

    std::vector<float> vector(384, 0.5f);
    NodeRef ref("person", "1");
    store.add(ref, "test", vector);

    auto record = store.get(ref);
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->vector.size(), 384);
    ASSERT_EQ(record->vector[0], 0.5f);
}

TEST(embedding_store_remove) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    NodeRef ref("person", "1");
    store.add(ref, "test");

    ASSERT_TRUE(store.remove(ref));
    ASSERT_NULL(store.get(ref));
}

TEST(embedding_store_similarity_search) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    store.add(NodeRef("person", "1"), "software engineer");
    store.add(NodeRef("person", "2"), "data scientist");
    store.add(NodeRef("person", "3"), "software developer");

    // Search with no threshold to get all results
    auto results = store.similaritySearchVector(encoder->encode("software engineer"), 10, "", -1.0f);
    ASSERT_EQ(results.size(), 3);  // Returns all entries
    ASSERT_GE(results[0].score, results[1].score);
}

TEST(embedding_store_filter_by_type) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    store.add(NodeRef("person", "1"), "engineer");
    store.add(NodeRef("company", "1"), "tech company");

    // Cosine similarity is signed and MockEncoder vectors are near-orthogonal,
    // so mock scores straddle zero - the default threshold of 0.0 can reject
    // every match. Accept anything.
    auto results = store.similaritySearch("tech", 10, "company", -1.0f);
    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].node_ref.type, "company");
}

TEST(top_k_selection_matches_a_full_sort) {
    // Selecting the best k must return what ordering everything would.
    auto encoder = std::make_shared<MockEncoder>(64);
    EmbeddingStore store(encoder);
    for (int i = 0; i < 500; ++i) {
        store.add(NodeRef("n", std::to_string(i)), "text " + std::to_string(i));
    }

    for (size_t k : {size_t(1), size_t(5), size_t(10), size_t(50), size_t(200)}) {
        for (int qi = 0; qi < 5; ++qi) {
            auto query = encoder->encode("text " + std::to_string(qi * 17));
            auto got = store.similaritySearchVector(query, k, "", -1.0f);
            auto all = store.similaritySearchVector(query, 100000, "", -1.0f);
            size_t want = std::min(k, all.size());
            ASSERT_EQ(got.size(), want);
            // Scores, not identities: std::sort and std::partial_sort are
            // both unstable, so the order among equal scores is arbitrary.
            for (size_t i = 0; i < want; ++i) {
                ASSERT_EQ(got[i].score, all[i].score);
            }
        }
    }
}

TEST(top_k_edges) {
    auto encoder = std::make_shared<MockEncoder>(16);
    EmbeddingStore store(encoder);
    store.add(NodeRef("n", "1"), "one");
    store.add(NodeRef("n", "2"), "two");
    ASSERT_EQ(store.similaritySearch("anything", 100, "", -1.0f).size(), 2u);
    ASSERT_EQ(store.similaritySearch("anything", 0, "", -1.0f).size(), 0u);
}

TEST(embedding_store_threshold) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    store.add(NodeRef("person", "1"), "software engineer");
    store.add(NodeRef("person", "2"), "completely unrelated topic");

    auto results = store.similaritySearch("software", 10, "", 0.5f);
    // With high threshold, may return fewer results
    ASSERT_TRUE(results.size() <= 2);
}

TEST(embedding_store_count) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    store.add(NodeRef("person", "1"), "test1");
    store.add(NodeRef("person", "2"), "test2");

    ASSERT_EQ(store.count(), 2);
}

TEST(embedding_store_clear) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    store.add(NodeRef("person", "1"), "test");
    store.clear();

    ASSERT_EQ(store.count(), 0);
}

// =============================================================================
// SemanticGraph Tests
// =============================================================================

TEST(semantic_graph_create_empty) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    ASSERT_EQ(graph.graph().nodeCount(), 0);
}

TEST(semantic_graph_add_node_with_embed) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {{"name", "Alice"}}, "Alice is an engineer");

    ASSERT_EQ(graph.graph().nodeCount(), 1);
    ASSERT_EQ(graph.embeddingStore().count(), 1);
}

TEST(semantic_graph_auto_embed) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder, true, {"name", "description"});

    graph.addNodeWithEmbed("person", "1", {{"name", "Alice"}, {"description", "Engineer"}});

    ASSERT_EQ(graph.embeddingStore().count(), 1);
}

TEST(semantic_graph_remove_node) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {}, "test");
    graph.removeNode("person", "1");

    ASSERT_EQ(graph.graph().nodeCount(), 0);
    ASSERT_EQ(graph.embeddingStore().count(), 0);
}

TEST(semantic_graph_similarity_search) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {{"name", "Alice"}}, "software engineer");
    graph.addNodeWithEmbed("person", "2", {{"name", "Bob"}}, "data scientist");

    auto results = graph.similaritySearch("software engineer", 5);
    ASSERT_GT(results.size(), 0);
}

TEST(semantic_graph_multi_hop) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {{"name", "Alice"}}, "software engineer Alice");
    graph.addNodeWithEmbed("person", "2", {{"name", "Bob"}}, "manager Bob");
    graph.addNodeWithEmbed("person", "3", {{"name", "Carol"}}, "designer Carol");

    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("person", "2"), NodeRef("person", "3"));

    auto results = graph.semanticMultiHop(
        "software",  // query
        "KNOWS",     // relType
        2,           // maxHops
        5,           // topKSeeds
        10,          // topKResults
        Direction::Out,
        0.8f,        // decay
        0.0f         // threshold
    );

    ASSERT_GT(results.size(), 0);
}

TEST(semantic_graph_multi_hop_decay) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {}, "engineer");
    graph.addNodeWithEmbed("person", "2", {}, "manager");

    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));

    auto results = graph.semanticMultiHop("engineer", "KNOWS", 1);

    // Find seed and traversed nodes
    const SemanticSearchResult* seed = nullptr;
    const SemanticSearchResult* traversed = nullptr;

    for (const auto& r : results) {
        if (r.hop_count == 0) seed = &r;
        if (r.hop_count == 1) traversed = &r;
    }

    if (seed && traversed) {
        ASSERT_GT(seed->score, traversed->score);
    }
}

TEST(semantic_graph_embed_all_nodes) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.graph().addNode("person", "1", {{"name", "Alice"}});
    graph.graph().addNode("person", "2", {{"name", "Bob"}});

    size_t count = graph.embedAllNodes();

    ASSERT_EQ(count, 2);
    ASSERT_EQ(graph.embeddingStore().count(), 2);
}

TEST(semantic_graph_preserve_traversal) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {}, "Alice");
    graph.addNodeWithEmbed("person", "2", {}, "Bob");
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));

    auto neighbors = graph.graph().neighbors(NodeRef("person", "1"), "KNOWS");
    ASSERT_EQ(neighbors.size(), 1);
}

TEST(semantic_graph_get_embedding) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    graph.addNodeWithEmbed("person", "1", {}, "Alice is an engineer");

    auto record = graph.getEmbedding(NodeRef("person", "1"));
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, "Alice is an engineer");
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(integration_full_workflow) {
    auto encoder = std::make_shared<MockEncoder>();
    SemanticGraph graph("test", encoder);

    // Add nodes
    graph.addNodeWithEmbed("user", "1", {{"name", "Alice"}}, "Alice is a software engineer");
    graph.addNodeWithEmbed("user", "2", {{"name", "Bob"}}, "Bob is a data scientist");
    graph.addNodeWithEmbed("user", "3", {{"name", "Carol"}}, "Carol is a product manager");
    graph.addNodeWithEmbed("project", "1", {{"name", "ML Platform"}}, "Machine learning infrastructure");
    graph.addNodeWithEmbed("project", "2", {{"name", "Web App"}}, "Frontend web application");

    // Add edges
    graph.graph().addEdge("WORKS_ON", NodeRef("user", "1"), NodeRef("project", "1"));
    graph.graph().addEdge("WORKS_ON", NodeRef("user", "2"), NodeRef("project", "1"));
    graph.graph().addEdge("WORKS_ON", NodeRef("user", "3"), NodeRef("project", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("user", "1"), NodeRef("user", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("user", "2"), NodeRef("user", "3"));

    // Similarity search - use exact text for reliable match with mock encoder
    auto similar = graph.similaritySearch("Machine learning infrastructure", 5);
    ASSERT_GT(similar.size(), 0);

    // Semantic multi-hop with lower threshold for mock encoder
    auto relatedPeople = graph.semanticMultiHop("Alice is a software engineer", "", 2, 5, 10, Direction::Out, 0.8f, 0.0f);
    ASSERT_GT(relatedPeople.size(), 0);

    // Verify graph structure
    ASSERT_EQ(graph.graph().nodeCount(), 5);
    ASSERT_EQ(graph.graph().edgeCount(), 5);
    ASSERT_EQ(graph.embeddingStore().count(), 5);
}

TEST(integration_cosine_similarity) {
    auto encoder = std::make_shared<MockEncoder>();
    EmbeddingStore store(encoder);

    // Add identical text - should have similarity = 1.0
    store.add(NodeRef("test", "1"), "hello world");

    auto results = store.similaritySearch("hello world", 1);
    ASSERT_EQ(results.size(), 1);
    // Cosine similarity with itself should be ~1.0
    ASSERT_GT(results[0].score, 0.99f);
}


// =============================================================================
// Regression tests for fixed defects
// =============================================================================

#define ASSERT_NEAR(a, b, tol) \
    if (std::fabs((a) - (b)) > (tol)) { \
        throw std::runtime_error("Assertion failed: " #a " != " #b " within " #tol); \
    }

#define ASSERT_THROWS(expr, exception_type) \
    { \
        bool threw = false; \
        try { expr; } catch (const exception_type&) { threw = true; } \
        if (!threw) throw std::runtime_error("Assertion failed: " #expr " did not throw " #exception_type); \
    }

namespace {

/// Asserted identically in the Python, TypeScript, JavaScript and Rust suites -
/// keeping the ports bit-identical is what lets an embedding store written by
/// one be searched by another.
const std::vector<float> GOLDEN_HELLO = {0.8381705284f, -0.8255031109f, 0.5899617672f, 0.4615051746f};
const std::vector<float> GOLDEN_ISON  = {-0.3658730984f, -0.2765417099f, 0.2944871187f, -0.6502785683f};
const std::vector<float> GOLDEN_EMPTY = {-0.4520676136f, -0.3797093630f, 0.3339765072f, 0.9023951292f};

/// Encoder returning a hand-picked vector so scores are exact.
class FixedEncoder : public EmbeddingEncoder {
public:
    size_t dimension() const override { return 4; }
    std::vector<float> encode(const std::string&) const override {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
};

SemanticGraph seededGraph() {
    SemanticGraph graph("seeds", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("person", "1");
    graph.addNode("person", "2");
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    // person:1 matches the query perfectly, person:2 slightly less well.
    graph.embeddingStore().add(NodeRef("person", "1"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "2"), "b", {0.9f, std::sqrt(0.19f), 0.0f, 0.0f});
    return graph;
}

const SemanticSearchResult& findById(const std::vector<SemanticSearchResult>& results,
                                     const std::string& id) {
    for (const auto& result : results) {
        if (result.node_ref.id == id) return result;
    }
    throw std::runtime_error("no result for id " + id);
}

}  // namespace

TEST(seed_keeps_its_own_score) {
    // Seeds were inserted only when absent from the result map, so an entry
    // written by an earlier traversal won even when it scored lower: person:2
    // is itself a 0.9 match at hop 0, but came back at seed(1.0) * decay(0.8)
    // = 0.8, at hop 1, down a path through person:1 that it never needed.
    SemanticGraph graph = seededGraph();

    MultiHopOptions options;
    options.relType = "KNOWS";
    options.maxHops = 2;
    options.decay = 0.8f;
    options.threshold = 0.3f;

    auto results = graph.semanticMultiHop("query", options);
    const auto& second = findById(results, "2");

    ASSERT_NEAR(second.score, 0.9f, 1e-5f);
    ASSERT_EQ(second.hop_count, 0u);
    ASSERT_EQ(second.path.size(), 1u);
    ASSERT_EQ(second.path[0].id, "2");
}

TEST(traversal_still_reaches_non_seeds) {
    SemanticGraph graph = seededGraph();
    graph.addNode("person", "3");
    graph.graph().addEdge("KNOWS", NodeRef("person", "2"), NodeRef("person", "3"));
    graph.embeddingStore().add(NodeRef("person", "3"), "c", {0.0f, 0.0f, 1.0f, 0.0f});

    MultiHopOptions options;
    options.relType = "KNOWS";
    options.maxHops = 2;
    options.threshold = 0.3f;

    auto results = graph.semanticMultiHop("query", options);
    const auto& third = findById(results, "3");
    ASSERT_GT(third.hop_count, 0u);
    ASSERT_GT(third.score, 0.0f);
}

TEST(negative_seed_does_not_grow_with_distance) {
    SemanticGraph graph("neg", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("person", "1");
    graph.addNode("person", "2");
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.embeddingStore().add(NodeRef("person", "1"), "a", {-1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "2"), "b", {0.0f, 0.0f, 1.0f, 0.0f});

    MultiHopOptions options;
    options.relType = "KNOWS";
    options.maxHops = 1;
    options.threshold = -1.0f;

    auto results = graph.semanticMultiHop("query", options);
    ASSERT_NEAR(findById(results, "1").score, -1.0f, 1e-5f);
    // 0.0, not -1.0 * 0.8 = -0.8, which would have outranked the seed.
    ASSERT_NEAR(findById(results, "2").score, 0.0f, 1e-5f);
}

TEST(blend_lifts_a_relevant_distant_node) {
    SemanticGraph graph("blend", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("person", "1");
    graph.addNode("person", "2");
    graph.addNode("person", "3");
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "3"));
    graph.embeddingStore().add(NodeRef("person", "1"), "seed", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "2"), "off", {0.0f, 1.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "3"), "on", {0.8f, 0.6f, 0.0f, 0.0f});

    MultiHopOptions options;
    options.relType = "KNOWS";
    options.maxHops = 1;
    options.topKSeeds = 1;
    options.threshold = 0.5f;

    auto plain = graph.semanticMultiHop("query", options);
    ASSERT_NEAR(findById(plain, "2").score, findById(plain, "3").score, 1e-5f);

    options.blend = 0.5f;
    auto blended = graph.semanticMultiHop("query", options);
    ASSERT_GT(findById(blended, "3").score, findById(blended, "2").score);
}

TEST(blend_out_of_range_is_rejected) {
    SemanticGraph graph = seededGraph();
    MultiHopOptions options;
    options.blend = 1.5f;
    ASSERT_THROWS(graph.semanticMultiHop("query", options), EmbeddingError);
}

TEST(dimension_mismatch_is_rejected) {
    auto encoder = std::make_shared<MockEncoder>(8);
    EmbeddingStore store(encoder);
    store.add(NodeRef("a", "1"), "full width");

    ASSERT_THROWS(store.add(NodeRef("a", "2"), "too short", {1.0f, 0.0f}), DimensionMismatchError);
    ASSERT_THROWS(store.similaritySearchVector({1.0f, 0.0f}, 1, "", -1.0f), DimensionMismatchError);
}

TEST(dimension_pinned_by_first_vector) {
    EmbeddingStore store;
    ASSERT_FALSE(store.dimension().has_value());
    store.add(NodeRef("a", "1"), "x", {1.0f, 0.0f, 0.0f});
    ASSERT_TRUE(store.dimension().has_value());
    ASSERT_EQ(*store.dimension(), 3u);
}

TEST(text_without_encoder_is_rejected) {
    EmbeddingStore store;
    ASSERT_THROWS(store.add(NodeRef("a", "1"), "no encoder here"), NoEncoderError);
}

TEST(mock_encoder_matches_other_ports) {
    MockEncoder encoder(4);
    const std::vector<std::pair<std::string, std::vector<float>>> cases = {
        {"hello", GOLDEN_HELLO}, {"ison", GOLDEN_ISON}, {"", GOLDEN_EMPTY},
    };
    for (const auto& [text, want] : cases) {
        auto got = encoder.encode(text);
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_NEAR(got[i], want[i], 1e-6f);
        }
    }
}

TEST(mock_encoder_has_no_collisions) {
    // The previous LCG scheme emitted the low 16 bits of the state, so every
    // vector was determined by seed mod 65536: 280 of these 5,000 texts
    // collided outright.
    MockEncoder encoder(8);
    std::set<std::vector<float>> seen;
    for (int i = 0; i < 5000; ++i) {
        seen.insert(encoder.encode("text" + std::to_string(i)));
    }
    ASSERT_EQ(seen.size(), 5000u);
}

TEST(mock_encoder_values_in_range) {
    for (float value : MockEncoder(128).encode("range check")) {
        ASSERT_GE(value, -1.0f);
        ASSERT_TRUE(value < 1.0f);
    }
}

TEST(mock_encoder_rejects_zero_dimension) {
    ASSERT_THROWS(MockEncoder(0), EmbeddingError);
}

TEST(add_node_auto_embeds_from_properties) {
    // autoEmbed used to apply only inside addNodeWithEmbed, so a node added
    // through the plain graph API came back unembedded and unsearchable.
    SemanticGraph graph("auto", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}, {"description", "engineer"}});

    ASSERT_EQ(graph.embeddingStore().count(), 1u);
    const auto* record = graph.getEmbedding(NodeRef("person", "1"));
    ASSERT_TRUE(record != nullptr);
    ASSERT_EQ(record->text, "Alice engineer");
}

TEST(add_node_leaves_nodes_unembedded_when_auto_embed_is_off) {
    SemanticGraph graph("manual", std::make_shared<MockEncoder>(16), false);
    graph.addNode("person", "1", {{"name", "Alice"}});
    ASSERT_EQ(graph.embeddingStore().count(), 0u);
}

TEST(semantic_path_finds_a_route) {
    SemanticGraph graph("paths", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("person", "1");
    graph.addNode("person", "2");
    graph.addNode("person", "3");
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("person", "2"), NodeRef("person", "3"));
    graph.embeddingStore().add(NodeRef("person", "1"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "2"), "b", {0.0f, 1.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "3"), "c", {0.0f, 0.0f, 1.0f, 0.0f});

    auto result = graph.semanticPath("query", NodeRef("person", "3"), "KNOWS", 5, 1,
                                     Direction::Out, 0.9f, 0.5f);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->node_ref.id, "3");
    ASSERT_EQ(result->hop_count, 2u);
    ASSERT_NEAR(result->score, 0.81f, 1e-5f);
}

TEST(semantic_path_returns_nothing_without_a_route) {
    SemanticGraph graph("paths", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("person", "1");
    graph.addNode("person", "2");
    // Only person:1 clears the threshold, so it is the sole seed - and it has
    // no edge to the target.
    graph.embeddingStore().add(NodeRef("person", "1"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("person", "2"), "b", {0.0f, 1.0f, 0.0f, 0.0f});

    auto result = graph.semanticPath("query", NodeRef("person", "2"), "KNOWS", 5, 1,
                                     Direction::Out, 0.9f, 0.5f);
    ASSERT_FALSE(result.has_value());
}

TEST(similar_to_node_excludes_itself) {
    SemanticGraph graph("similar", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.addNode("person", "2", {{"name", "Bob"}});
    graph.addNode("person", "3", {{"name", "Carol"}});

    auto results = graph.similarToNode(NodeRef("person", "1"), 2, "", -1.0f);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        ASSERT_NE(result.node_ref.id, "1");
    }
}

TEST(similar_to_node_without_embedding) {
    SemanticGraph graph("similar", std::make_shared<MockEncoder>(16), false);
    graph.addNode("person", "9");
    ASSERT_EQ(graph.similarToNode(NodeRef("person", "9"), 5, "", -1.0f).size(), 0u);
}

TEST(semantic_subgraph_is_a_working_graph) {
    SemanticGraph graph("full", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.addNode("person", "2", {{"name", "Bob"}});
    graph.addNode("person", "3", {{"name", "Carol"}});
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.graph().addEdge("KNOWS", NodeRef("person", "2"), NodeRef("person", "3"));

    MultiHopOptions options;
    options.relType = "KNOWS";
    options.maxHops = 1;
    options.topKSeeds = 1;
    options.topKResults = 2;
    options.threshold = -1.0f;

    SemanticGraph sub = graph.semanticSubgraph("Alice", options, "slice");
    ASSERT_EQ(sub.graph().name, "slice");
    ASSERT_GT(sub.graph().nodeCount(), 0u);
    ASSERT_TRUE(sub.graph().nodeCount() <= 2u);
    // every kept node brought its embedding along
    ASSERT_EQ(sub.embeddingStore().count(), sub.graph().nodeCount());
}

TEST(embed_all_nodes_batches) {
    SemanticGraph graph("batch", std::make_shared<MockEncoder>(16), false);
    for (int i = 0; i < 10; ++i) {
        graph.addNode("doc", std::to_string(i), {{"name", "Document"}});
    }
    ASSERT_EQ(graph.embedAllNodes(), 10u);
    ASSERT_EQ(graph.embeddingStore().count(), 10u);
}

TEST(embed_all_nodes_can_skip_existing) {
    SemanticGraph graph("skip", std::make_shared<MockEncoder>(16), false);
    graph.addNode("doc", "1", {{"name", "One"}});
    graph.addNode("doc", "2", {{"name", "Two"}});
    graph.embedNode(NodeRef("doc", "1"), "already embedded");

    ASSERT_EQ(graph.embedAllNodes(true), 1u);
    ASSERT_EQ(graph.getEmbedding(NodeRef("doc", "1"))->text, "already embedded");
}

TEST(embeddings_round_trip_through_json) {
    auto encoder = std::make_shared<MockEncoder>(8);
    EmbeddingStore store(encoder);
    store.add(NodeRef("person", "1"), "Alice");
    store.add(NodeRef("account", "007"), "Bond");

    std::string json = store.toJson();
    ASSERT_TRUE(json.find(EMBEDDING_FORMAT) != std::string::npos);

    EmbeddingStore other(encoder);
    ASSERT_EQ(other.fromJson(json, false), 2u);
    ASSERT_EQ(other.get(NodeRef("person", "1"))->text, "Alice");
    // a numeric-looking string id keeps its exact form
    ASSERT_EQ(other.get(NodeRef("account", "007"))->text, "Bond");
}

TEST(import_rejects_a_foreign_payload) {
    EmbeddingStore store(std::make_shared<MockEncoder>(8));
    ASSERT_THROWS(store.fromJson("{\"format\":\"something-else\",\"version\":1,\"embeddings\":[]}"),
                  EmbeddingError);
}

TEST(imports_a_payload_written_by_the_python_port) {
    // Captured verbatim from the Python port. Its node ids are typed, so an
    // int id arrives unquoted ("id":1) where this port always writes a string
    // - both have to land on the same NodeRef here.
    const std::string pythonPayload =
        R"PAYLOAD({"format":"ison-embeddings","version":1,"dimension":4,"count":2,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","model":"mock-4","vector":[-0.8610728979110718,-0.33433854579925537,0.8408418893814087,-0.2771846055984497]},{"type":"account","id":"007","id_type":"str","text":"Bond","model":"mock-4","vector":[-0.9531009197235107,0.9167747497558594,0.5073114633560181,-0.0420149564743042]}]})PAYLOAD";

    auto encoder = std::make_shared<MockEncoder>(4);
    EmbeddingStore store(encoder);
    ASSERT_EQ(store.fromJson(pythonPayload, false), 2u);
    ASSERT_EQ(store.get(NodeRef("person", "1"))->text, "Alice");
    ASSERT_EQ(store.get(NodeRef("account", "007"))->text, "Bond");

    // and the vectors match what this port encodes for the same text
    auto expected = encoder->encode("Alice");
    const auto& stored = store.get(NodeRef("person", "1"))->vector;
    ASSERT_EQ(stored.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(stored[i], expected[i], 1e-6f);
    }
}

TEST(embed_text_is_identical_across_ports) {
    // The same golden string is asserted in all six suites. ISONGraph 1.4.0
    // made property values typed, which is where the ports could drift:
    // Python would otherwise render None as "None" and True as "True", while
    // JavaScript renders "null" and "true". Different text means a different
    // vector, which would break the cross-port guarantee.
    SemanticGraph graph("props", std::make_shared<MockEncoder>(8));
    graph.addNode("doc", "1", {
        {"name", ison::Value("Report")},
        {"description", ison::Value()},
        {"content", ison::Value(1.5)},
        {"text", ison::Value(true)},
    });

    const auto* record = graph.getEmbedding(NodeRef("doc", "1"));
    ASSERT_TRUE(record != nullptr);
    ASSERT_EQ(record->text, "Report 1.5 true");
}

TEST(vendored_headers_are_the_versions_we_documented) {
    // The vendored copies have no version resolution behind them, so this is
    // the only thing that notices when they drift from vendor/VENDORED.md.
    // Update both together when re-copying from upstream.
    ASSERT_EQ(std::string(ison_graph::VERSION), "1.1.0");
    ASSERT_EQ(std::string(ison::VERSION), "1.2.0");
}

TEST(floats_render_the_same_way_in_every_port) {
    // The same values are asserted in all six suites. This port used the
    // default six significant figures and rendered 1/3 as "0.333333", which
    // meant a different embedding text - and so a different vector - from
    // every other port for the same graph.
    ASSERT_EQ(SemanticGraph::propertyText(ison::Value(1.0)), "1");
    ASSERT_EQ(SemanticGraph::propertyText(ison::Value(100.0)), "100");
    ASSERT_EQ(SemanticGraph::propertyText(ison::Value(1.5)), "1.5");
    ASSERT_EQ(SemanticGraph::propertyText(ison::Value(0.1)), "0.1");
    ASSERT_EQ(SemanticGraph::propertyText(ison::Value(1.0 / 3.0)), "0.3333333333333333");
}

TEST(null_contributes_nothing_to_embed_text) {
    SemanticGraph graph("props", std::make_shared<MockEncoder>(8));
    Properties props{{"name", ison::Value()}, {"description", ison::Value("kept")}};
    ASSERT_EQ(graph.embedTextFor(props), "kept");
}

// =============================================================================
// semanticPath: best seed wins
// =============================================================================

namespace {

/// Seeds A (similarity 1.0) and B (~0.9) both reach the target, but A needs
/// three hops and B one.
SemanticGraph twoSeedPathGraph() {
    SemanticGraph graph("paths2", std::make_shared<FixedEncoder>(), false, {});
    for (const char* id : {"A", "x", "y", "B", "T"}) graph.addNode("n", id);
    graph.graph().addEdge("R", NodeRef("n", "A"), NodeRef("n", "x"));
    graph.graph().addEdge("R", NodeRef("n", "x"), NodeRef("n", "y"));
    graph.graph().addEdge("R", NodeRef("n", "y"), NodeRef("n", "T"));
    graph.graph().addEdge("R", NodeRef("n", "B"), NodeRef("n", "T"));
    graph.embeddingStore().add(NodeRef("n", "A"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("n", "B"), "b", {0.9f, std::sqrt(0.19f), 0.0f, 0.0f});
    return graph;
}

} // namespace

TEST(semantic_path_returns_the_best_scoring_seed_not_the_first) {
    auto graph = twoSeedPathGraph();
    // A: 1.0 * 0.9^3 = 0.729 (first in seed order), B: 0.9 * 0.9 = 0.81.
    auto result = graph.semanticPath("query", NodeRef("n", "T"), "R", 5, 2,
                                     Direction::Out, 0.9f, 0.0f);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->hop_count, 1u);
    ASSERT_EQ(result->path.front().id, "B");
    ASSERT_NEAR(result->score, 0.81f, 1e-5f);
}

TEST(semantic_path_clamps_a_negative_seed_to_zero) {
    SemanticGraph graph("neg", std::make_shared<FixedEncoder>(), false, {});
    graph.addNode("n", "S");
    graph.addNode("n", "T");
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "T"));
    // cosine(query, S) = -1: decaying it would give -0.9, which outranks -1.
    graph.embeddingStore().add(NodeRef("n", "S"), "s", {-1.0f, 0.0f, 0.0f, 0.0f});

    auto result = graph.semanticPath("query", NodeRef("n", "T"), "R", 5, 1,
                                     Direction::Out, 0.9f, -1.0f);
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result->score, 0.0f, 1e-7f);
}

TEST(semantic_path_breaks_ties_toward_the_more_similar_seed) {
    SemanticGraph graph("ties", std::make_shared<FixedEncoder>(), false, {});
    for (const char* id : {"S1", "S2", "T"}) graph.addNode("n", id);
    graph.graph().addEdge("R", NodeRef("n", "S1"), NodeRef("n", "T"));
    graph.graph().addEdge("R", NodeRef("n", "S2"), NodeRef("n", "T"));
    // Both clamp to 0, so the scores tie; S2 (-0.5) is the more similar seed.
    graph.embeddingStore().add(NodeRef("n", "S1"), "s1", {-1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("n", "S2"), "s2", {-0.5f, 0.5f, 0.0f, 0.0f});

    auto result = graph.semanticPath("query", NodeRef("n", "T"), "R", 5, 2,
                                     Direction::Out, 0.9f, -1.0f);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->path.front().id, "S2");
}

// =============================================================================
// updateNode keeps embeddings in step
// =============================================================================

TEST(update_node_re_embeds_when_an_embed_field_changes) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}});
    auto before = graph.getEmbedding(NodeRef("person", "1"))->vector;

    Node& node = graph.updateNode("person", "1", {{"name", "Bob"}});
    ASSERT_EQ(node.properties.at("name").as_string(), "Bob");

    const auto* record = graph.getEmbedding(NodeRef("person", "1"));
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, "Bob");
    ASSERT_TRUE(record->vector != before);
    ASSERT_TRUE(record->vector == MockEncoder(16).encode("Bob"));
}

TEST(update_node_embeds_a_node_that_had_nothing_to_embed) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"age", ison::Value(int64_t(3))}});
    ASSERT_NULL(graph.getEmbedding(NodeRef("person", "1")));
    graph.updateNode("person", "1", {{"description", "engineer"}});
    ASSERT_EQ(graph.getEmbedding(NodeRef("person", "1"))->text, "engineer");
}

TEST(update_node_removes_the_vector_when_no_text_is_left) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}});
    ASSERT_NOT_NULL(graph.getEmbedding(NodeRef("person", "1")));

    graph.updateNode("person", "1", {{"name", ison::Value()}});
    ASSERT_NULL(graph.getEmbedding(NodeRef("person", "1")));
    ASSERT_EQ(graph.embeddingStore().count(), 0u);
}

TEST(update_node_leaves_the_vector_when_no_embed_field_changed) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.updateNode("person", "1", {{"age", ison::Value(int64_t(30))}});

    const auto* record = graph.getEmbedding(NodeRef("person", "1"));
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, "Alice");
    ASSERT_EQ(graph.graph().getNode("person", "1").properties.at("age").as_int(), int64_t(30));
}

TEST(update_node_does_nothing_to_embeddings_when_auto_embed_is_off) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16), false);
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.embedNode(NodeRef("person", "1"), "managed by the caller");

    graph.updateNode("person", "1", {{"name", "Bob"}});
    ASSERT_EQ(graph.getEmbedding(NodeRef("person", "1"))->text, "managed by the caller");

    graph.updateNode("person", "1", {{"name", ison::Value()}});
    ASSERT_NOT_NULL(graph.getEmbedding(NodeRef("person", "1")));
}

TEST(update_node_on_a_missing_node_throws) {
    SemanticGraph graph("upd", std::make_shared<MockEncoder>(16));
    ASSERT_THROWS(graph.updateNode("person", "nope", {{"name", "x"}}), GraphError);
}

// =============================================================================
// Persistence: graph file plus embeddings sidecar
// =============================================================================

namespace {

/// A path under the system temp directory that removes itself and its sidecar.
class TempGraphPath {
    std::string path_;
public:
    explicit TempGraphPath(const std::string& name) {
        path_ = (std::filesystem::temp_directory_path() / name).string();
        cleanup();
    }
    ~TempGraphPath() { cleanup(); }
    const std::string& path() const { return path_; }
    std::string sidecar() const { return SemanticGraph::embeddingsPathFor(path_); }
private:
    void cleanup() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        std::filesystem::remove(path_ + ".embeddings.json", ec);
    }
};

} // namespace

TEST(embeddings_path_for_appends_the_sidecar_suffix) {
    ASSERT_EQ(SemanticGraph::embeddingsPathFor("g.ison"), "g.ison.embeddings.json");
}

TEST(save_writes_the_graph_and_an_embeddings_sidecar) {
    TempGraphPath file("isongraph_vector_cpp_save.ison");
    SemanticGraph graph("persist", std::make_shared<MockEncoder>(8));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.addNode("person", "2", {{"name", "Bob"}});
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    graph.save(file.path());

    ASSERT_TRUE(std::filesystem::exists(file.path()));
    ASSERT_TRUE(std::filesystem::exists(file.sidecar()));

    std::ifstream in(file.sidecar(), std::ios::binary);
    std::string payload((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(payload.find("\"ison-embeddings\"") != std::string::npos);
}

TEST(save_then_load_round_trips_graph_and_embeddings) {
    TempGraphPath file("isongraph_vector_cpp_roundtrip.ison");
    auto encoder = std::make_shared<MockEncoder>(8);
    {
        SemanticGraph graph("persist", encoder);
        graph.addNode("person", "1", {{"name", "Alice"}, {"description", "engineer"}});
        graph.addNode("person", "2", {{"name", "Bob"}});
        graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
        graph.save(file.path());
    }

    auto loaded = SemanticGraph::load(file.path(), encoder);
    ASSERT_EQ(loaded.graph().nodeCount(), 2u);
    ASSERT_TRUE(loaded.graph().hasNode(NodeRef("person", "1")));
    ASSERT_EQ(loaded.graph().getEdges("KNOWS").size(), 1u);
    ASSERT_EQ(loaded.embeddingStore().count(), 2u);
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "1"))->text, "Alice engineer");
    ASSERT_TRUE(loaded.getEmbedding(NodeRef("person", "2"))->vector == encoder->encode("Bob"));

    // The restored vectors are searchable, not just present.
    auto hits = loaded.similaritySearch("Alice engineer", 1, "", -1.0f);
    ASSERT_EQ(hits.size(), 1u);
    ASSERT_EQ(hits[0].node_ref.id, "1");
    ASSERT_NEAR(hits[0].score, 1.0f, 1e-5f);
}

TEST(load_does_not_re_embed_or_overwrite_saved_vectors) {
    TempGraphPath file("isongraph_vector_cpp_noreembed.ison");
    {
        SemanticGraph graph("persist", std::make_shared<MockEncoder>(8));
        graph.addNode("person", "1", {{"name", "Alice"}});
        // A caller-supplied vector the encoder would never produce.
        graph.embeddingStore().add(NodeRef("person", "1"), "Alice",
                                   {1, 0, 0, 0, 0, 0, 0, 0});
        graph.save(file.path());
    }
    auto loaded = SemanticGraph::load(file.path(), std::make_shared<MockEncoder>(8));
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "1"))->vector[0], 1.0f);
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "1"))->vector[1], 0.0f);
}

TEST(save_can_skip_the_sidecar_and_load_tolerates_its_absence) {
    TempGraphPath file("isongraph_vector_cpp_nosidecar.ison");
    SemanticGraph graph("persist", std::make_shared<MockEncoder>(8));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.save(file.path(), "auto", false);
    ASSERT_FALSE(std::filesystem::exists(file.sidecar()));

    auto loaded = SemanticGraph::load(file.path());
    ASSERT_EQ(loaded.graph().nodeCount(), 1u);
    ASSERT_EQ(loaded.embeddingStore().count(), 0u);
}

TEST(load_can_ignore_an_existing_sidecar) {
    TempGraphPath file("isongraph_vector_cpp_ignore.ison");
    SemanticGraph graph("persist", std::make_shared<MockEncoder>(8));
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.save(file.path());

    auto loaded = SemanticGraph::load(file.path(), nullptr, "auto", false);
    ASSERT_EQ(loaded.embeddingStore().count(), 0u);
}

TEST(load_a_missing_graph_file_throws) {
    ASSERT_THROWS(SemanticGraph::load("definitely/not/here.ison"), GraphError);
}

TEST(sidecar_json_survives_quotes_braces_and_non_ascii_text) {
    auto encoder = std::make_shared<MockEncoder>(4);
    EmbeddingStore store(encoder);
    const std::string text =
        "He said \"hi\" {braces} [brackets]\nline two \xc3\xa9 \xe2\x82\xac \t tab \\ slash \x01";
    store.add(NodeRef("doc", "r\xc3\xa9sum\xc3\xa9"), text);

    EmbeddingStore other(encoder);
    ASSERT_EQ(other.fromJson(store.toJson()), 1u);
    const auto* record = other.get(NodeRef("doc", "r\xc3\xa9sum\xc3\xa9"));
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, text);
}

TEST(reads_a_python_sidecar_with_ascii_escaped_unicode) {
    // json.dumps escapes non-ASCII by default: "résumé", and an
    // astral character as a surrogate pair. Both must decode to UTF-8, and a
    // "}" inside the text must not end the object early.
    const std::string payload =
        R"PAYLOAD({"format": "ison-embeddings", "version": 1, "dimension": 2, "count": 1, "embeddings": [{"type": "doc", "id": "r\u00e9sum\u00e9", "id_type": "str", "text": "caf\u00e9 } \ud83d\ude00 end", "model": "mock-2", "vector": [0.25, -0.5]}]})PAYLOAD";
    EmbeddingStore store(std::make_shared<MockEncoder>(2));
    ASSERT_EQ(store.fromJson(payload), 1u);
    const auto* record = store.get(NodeRef("doc", "r\xc3\xa9sum\xc3\xa9"));
    ASSERT_NOT_NULL(record);
    ASSERT_EQ(record->text, "caf\xc3\xa9 } \xf0\x9f\x98\x80 end");
    ASSERT_EQ(record->vector.size(), 2u);
    ASSERT_EQ(record->vector[0], 0.25f);
    ASSERT_EQ(record->vector[1], -0.5f);
}

TEST(loads_a_sidecar_written_by_the_python_port_next_to_a_graph) {
    // The sidecar is json.dumps output: an int id, an escaped non-ASCII
    // character and the extra "model" key this port ignores.
    TempGraphPath file("isongraph_vector_cpp_frompy.ison");
    {
        SemanticGraph seed("persist", nullptr, false);
        seed.addNode("person", "1", {{"name", "Alice"}});
        seed.addNode("person", "2", {{"name", "Bob"}});
        seed.graph().save(file.path());
    }
    {
        std::ofstream sidecar(file.sidecar(), std::ios::binary);
        sidecar << R"PAYLOAD({"format": "ison-embeddings", "version": 1, "dimension": 4, "count": 2, "embeddings": [{"type": "person", "id": 1, "id_type": "int", "text": "Alice \u2603", "model": "mock-4", "vector": [-0.8610728979110718, -0.33433854579925537, 0.8408418893814087, -0.2771846055984497]}, {"type": "person", "id": 2, "id_type": "int", "text": "Bob", "model": "mock-4", "vector": [0.5, 0.5, 0.5, 0.5]}]})PAYLOAD";
    }

    auto loaded = SemanticGraph::load(file.path(), std::make_shared<MockEncoder>(4));
    ASSERT_EQ(loaded.embeddingStore().count(), 2u);
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "1"))->text, "Alice \xe2\x98\x83");
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "2"))->vector[0], 0.5f);
}

// =============================================================================
// Node filtering, edge weighting, connectivity, model lifecycle, consistency
// (the features the Python port gained; each mirrors a test class there)
// =============================================================================

namespace {

/// Collects the library's warnings for the lifetime of the object.
struct WarningCapture {
    std::vector<std::string> messages;
    WarningCapture() {
        setWarningHandler([this](const std::string& message) { messages.push_back(message); });
    }
    ~WarningCapture() { setWarningHandler(nullptr); }
    bool mentions(const std::string& needle) const {
        for (const auto& message : messages) {
            if (message.find(needle) != std::string::npos) return true;
        }
        return false;
    }
};

/// A scratch directory removed (with everything in it) on destruction.
class TempDir {
public:
    TempDir() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("isongraph_vector_cpp_" + std::to_string(std::random_device{}()) + "_" +
                 std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    std::string file(const std::string& name) const { return (path_ / name).string(); }
    std::vector<std::string> listing() const {
        std::vector<std::string> names;
        for (const auto& entry : std::filesystem::directory_iterator(path_)) {
            names.push_back(entry.path().filename().string());
        }
        std::sort(names.begin(), names.end());
        return names;
    }
private:
    std::filesystem::path path_;
};

std::string readWhole(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeWhole(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

std::string replaceAll(std::string text, const std::string& from, const std::string& to) {
    size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
    return text;
}

std::set<std::string> idsOf(const std::vector<SemanticSearchResult>& results) {
    std::set<std::string> ids;
    for (const auto& r : results) ids.insert(r.node_ref.type + ":" + r.node_ref.id);
    return ids;
}

const SemanticSearchResult& resultFor(const std::vector<SemanticSearchResult>& results,
                                      const std::string& key) {
    for (const auto& r : results) {
        if (r.node_ref.type + ":" + r.node_ref.id == key) return r;
    }
    throw std::runtime_error("no result for " + key);
}

std::set<std::string> nodeKeys(const SemanticGraph& graph) {
    std::set<std::string> keys;
    graph.graph().forEachNode([&](const Node& node) { keys.insert(node.type + ":" + node.id); });
    return keys;
}

bool isActive(const Node& node) {
    auto it = node.properties.find("active");
    return it != node.properties.end() && it->second.is_bool() && it->second.as_bool();
}

/// Auto-embedding graph whose vectors derive from node names (FNV mock encoder).
SemanticGraph namedGraph(std::shared_ptr<EmbeddingEncoder> encoder,
                         std::vector<std::string> names = {"Alice", "Bob", "Carol"}) {
    SemanticGraph graph("g", encoder, true);
    for (size_t i = 0; i < names.size(); ++i) {
        graph.addNode("person", std::to_string(i + 1), {{"name", names[i]}});
    }
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("person", "2"));
    return graph;
}

std::shared_ptr<MockEncoder> retagged(size_t dimension, const std::string& name) {
    auto encoder = std::make_shared<MockEncoder>(dimension);
    encoder->setModelName(name);
    return encoder;
}

/// Hand-vectored graph for traversal tests (every text query is [1,0,0,0]).
///
///     person 1  --KNOWS--> company 100 --KNOWS--> person 3
///     person 2  (isolated)                        person 3 has no vector
///
/// company 100 is the single best match, person 1 next.
SemanticGraph vectorGraph() {
    SemanticGraph graph("v", std::make_shared<FixedEncoder>(), false);
    graph.addNode("person", "1", {{"name", "P1"}, {"active", true}});
    graph.addNode("person", "2", {{"name", "P2"}, {"active", false}});
    graph.addNode("person", "3", {{"name", "P3"}, {"active", true}});
    graph.addNode("company", "100", {{"name", "C"}});
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("company", "100"));
    graph.graph().addEdge("KNOWS", NodeRef("company", "100"), NodeRef("person", "3"));
    auto& store = graph.embeddingStore();
    store.add(NodeRef("company", "100"), "c", {1.0f, 0.0f, 0.0f, 0.0f});
    store.add(NodeRef("person", "1"), "p1", {0.9f, 0.1f, 0.0f, 0.0f});
    store.add(NodeRef("person", "2"), "p2", {0.8f, 0.2f, 0.0f, 0.0f});
    return graph;
}

NodeFilterOptions onlyTypes(std::vector<std::string> types) {
    NodeFilterOptions filter;
    filter.nodeTypes = std::move(types);
    return filter;
}

MultiHopOptions wideOptions(size_t maxHops, size_t seeds) {
    MultiHopOptions options;
    options.maxHops = maxHops;
    options.topKSeeds = seeds;
    options.threshold = -1.0f;
    return options;
}

} // namespace

// -- node filtering ----------------------------------------------------------

TEST(filter_similarity_search_filters_before_taking_top_k) {
    SemanticGraph graph = vectorGraph();
    // the single best match is the company; filtering must look past it
    ASSERT_EQ(graph.similaritySearch("q", 1)[0].node_ref, NodeRef("company", "100"));
    auto top = graph.similaritySearch("q", 1, "", 0.0f, onlyTypes({"person"}));
    ASSERT_EQ(top.size(), 1u);
    ASSERT_EQ(top[0].node_ref, NodeRef("person", "1"));
}

TEST(filter_node_filter_and_types_combine_with_and) {
    SemanticGraph graph = vectorGraph();
    NodeFilterOptions active;
    active.nodeFilter = isActive;
    auto hits = graph.similaritySearch("q", 100, "", -1.0f, active);
    ASSERT_EQ(hits.size(), 1u);
    ASSERT_EQ(hits[0].node_ref, NodeRef("person", "1"));

    NodeFilterOptions both = onlyTypes({"company"});
    both.nodeFilter = isActive;
    ASSERT_EQ(graph.similaritySearch("q", 100, "", -1.0f, both).size(), 0u);
}

TEST(filter_empty_node_types_matches_nothing_and_a_single_type_matches_it) {
    SemanticGraph graph = vectorGraph();
    ASSERT_EQ(graph.similaritySearch("q", 100, "", 0.0f, onlyTypes({})).size(), 0u);
    auto only = graph.similaritySearch("q", 100, "", 0.0f, onlyTypes({"company"}));
    ASSERT_EQ(only.size(), 1u);
    ASSERT_EQ(only[0].node_ref, NodeRef("company", "100"));
}

TEST(filter_embeddings_without_a_node_fail_node_filter_only) {
    SemanticGraph graph = vectorGraph();
    graph.embeddingStore().add(NodeRef("ghost", "1"), "g", {1.0f, 0.0f, 0.0f, 0.0f});
    auto byType = graph.similaritySearch("q", 100, "", -1.0f, onlyTypes({"ghost"}));
    ASSERT_EQ(byType.size(), 1u);
    NodeFilterOptions anything;
    anything.nodeFilter = [](const Node&) { return true; };
    for (const auto& hit : graph.similaritySearch("q", 100, "", -1.0f, anything)) {
        ASSERT_NE(hit.node_ref.type, "ghost");
    }
}

TEST(filter_multi_hop_seeds_and_results_are_filtered) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions options = wideOptions(2, 1);
    options.nodeTypes = std::vector<std::string>{"person"};
    auto found = graph.semanticMultiHop("q", options);
    // the best match (company) did not take the seed slot, and is not returned
    ASSERT_EQ(found[0].node_ref, NodeRef("person", "1"));
    ASSERT_EQ(idsOf(found).count("company:100"), 0u);
}

TEST(filter_failing_nodes_are_traversed_but_not_returned_by_default) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions options = wideOptions(2, 1);
    options.nodeTypes = std::vector<std::string>{"person"};
    auto found = graph.semanticMultiHop("q", options);
    const auto& reach = resultFor(found, "person:3");
    ASSERT_EQ(reach.path.size(), 3u);
    ASSERT_EQ(reach.path[0], NodeRef("person", "1"));
    ASSERT_EQ(reach.path[1], NodeRef("company", "100"));
    ASSERT_EQ(reach.path[2], NodeRef("person", "3"));
    ASSERT_EQ(reach.hop_count, 2u);  // the company still cost a hop
    ASSERT_NEAR(reach.score, resultFor(found, "person:1").score * 0.64f, 1e-5f);
}

TEST(filter_strict_filter_walls_off_the_traversal) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions options = wideOptions(2, 1);
    options.nodeTypes = std::vector<std::string>{"person"};
    options.strictFilter = true;
    auto ids = idsOf(graph.semanticMultiHop("q", options));
    ASSERT_EQ(ids.count("person:3"), 0u);
    ASSERT_EQ(ids.count("company:100"), 0u);
}

TEST(filter_blend_path_respects_the_filter_too) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions options = wideOptions(1, 2);
    options.blend = 0.5f;
    options.nodeTypes = std::vector<std::string>{"person"};
    auto found = graph.semanticMultiHop("q", options);
    ASSERT_TRUE(!found.empty());
    for (const auto& r : found) ASSERT_EQ(r.node_ref.type, "person");
}

TEST(filter_semantic_path_requires_a_passing_target) {
    SemanticGraph graph = vectorGraph();
    ASSERT_FALSE(graph.semanticPath("q", NodeRef("company", "100"), "", 5, 3, Direction::Out,
                                    0.9f, -1.0f, onlyTypes({"person"})).has_value());
    ASSERT_TRUE(graph.semanticPath("q", NodeRef("company", "100"), "", 5, 3, Direction::Out,
                                   0.9f, -1.0f).has_value());
}

TEST(filter_semantic_path_may_cross_failing_nodes_unless_strict) {
    SemanticGraph graph = vectorGraph();
    NodeFilterOptions filter = onlyTypes({"person"});
    auto lax = graph.semanticPath("q", NodeRef("person", "3"), "", 5, 1, Direction::Out,
                                  0.9f, -1.0f, filter);
    ASSERT_TRUE(lax.has_value());
    ASSERT_EQ(lax->path.size(), 3u);
    ASSERT_EQ(lax->path[1], NodeRef("company", "100"));

    filter.strictFilter = true;
    ASSERT_FALSE(graph.semanticPath("q", NodeRef("person", "3"), "", 5, 1, Direction::Out,
                                    0.9f, -1.0f, filter).has_value());
}

TEST(filter_strict_semantic_path_finds_an_all_passing_route) {
    SemanticGraph graph("s", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"A", "B", "C"}) graph.addNode("n", id, {{"ok", true}});
    graph.addNode("n", "X", {{"ok", false}});
    // short route A -> X -> C (through the failing X), long route A -> B -> C
    graph.graph().addEdge("R", NodeRef("n", "A"), NodeRef("n", "X"));
    graph.graph().addEdge("R", NodeRef("n", "X"), NodeRef("n", "C"));
    graph.graph().addEdge("R", NodeRef("n", "A"), NodeRef("n", "B"));
    graph.graph().addEdge("R", NodeRef("n", "B"), NodeRef("n", "C"));
    graph.embeddingStore().add(NodeRef("n", "A"), "a", {1.0f, 0.0f, 0.0f, 0.0f});

    NodeFilterOptions filter;
    filter.nodeFilter = [](const Node& n) { return n.properties.at("ok").as_bool(); };
    auto lax = graph.semanticPath("q", NodeRef("n", "C"), "", 5, 1, Direction::Out, 0.9f, 0.0f, filter);
    ASSERT_EQ(lax->hop_count, 2u);
    ASSERT_EQ(lax->path[1], NodeRef("n", "X"));

    filter.strictFilter = true;
    auto strict = graph.semanticPath("q", NodeRef("n", "C"), "", 5, 1, Direction::Out, 0.9f, 0.0f, filter);
    ASSERT_TRUE(strict.has_value());
    ASSERT_EQ(strict->hop_count, 2u);
    ASSERT_EQ(strict->path[1], NodeRef("n", "B"));
}

TEST(filter_semantic_path_seeds_come_from_passing_nodes) {
    SemanticGraph graph = vectorGraph();
    NodeFilterOptions filter;
    filter.nodeFilter = [](const Node& n) { return n.type == "person"; };
    auto result = graph.semanticPath("q", NodeRef("person", "3"), "", 5, 1, Direction::Out,
                                     0.9f, -1.0f, filter);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->path[0], NodeRef("person", "1"));
}

TEST(filter_subgraph_holds_only_passing_nodes) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions options = wideOptions(2, 2);
    options.nodeTypes = std::vector<std::string>{"person"};
    SemanticGraph sub = graph.semanticSubgraph("q", options);
    auto keys = nodeKeys(sub);
    for (const auto& key : keys) ASSERT_EQ(key.rfind("person:", 0), 0u);
    ASSERT_EQ(keys.count("person:3"), 1u);
}

TEST(filter_defaults_are_unchanged) {
    SemanticGraph graph = vectorGraph();
    auto plain = graph.semanticMultiHop("q", wideOptions(2, 1));
    ASSERT_EQ(plain[0].node_ref, NodeRef("company", "100"));
    ASSERT_EQ(idsOf(plain).count("person:3"), 1u);
}

// -- edge weighting ----------------------------------------------------------

namespace {

/// S --R--> P (weight pWeight, if given), S --R--> Q (weight qWeight, if given).
SemanticGraph fan(const std::optional<ison::Value>& pWeight,
                  const std::optional<ison::Value>& qWeight = std::nullopt) {
    SemanticGraph graph("f", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"S", "P", "Q"}) graph.addNode("n", id);
    Properties p, q;
    if (pWeight) p["w"] = *pWeight;
    if (qWeight) q["w"] = *qWeight;
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "P"), p);
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "Q"), q);
    graph.embeddingStore().add(NodeRef("n", "S"), "s", {1.0f, 0.0f, 0.0f, 0.0f});
    return graph;
}

float scoreOf(const SemanticGraph& graph, const std::string& id, const std::string& weightField) {
    MultiHopOptions options = wideOptions(1, 1);
    options.weightField = weightField;
    return resultFor(graph.semanticMultiHop("q", options), "n:" + id).score;
}

} // namespace

TEST(weights_default_ignores_edge_properties) {
    SemanticGraph graph = fan(ison::Value(0.1));
    ASSERT_NEAR(scoreOf(graph, "P", ""), 0.8f, 1e-6f);
    ASSERT_NEAR(scoreOf(graph, "Q", ""), 0.8f, 1e-6f);
}

TEST(weights_multiply_the_decay) {
    SemanticGraph graph = fan(ison::Value(0.5));
    ASSERT_NEAR(scoreOf(graph, "S", "w"), 1.0f, 1e-6f);
    ASSERT_NEAR(scoreOf(graph, "P", "w"), 0.4f, 1e-6f);
    ASSERT_NEAR(scoreOf(graph, "Q", "w"), 0.8f, 1e-6f);  // missing -> 1.0
}

TEST(weights_are_clamped_and_junk_counts_as_one) {
    ASSERT_NEAR(scoreOf(fan(ison::Value(5.0)), "P", "w"), 0.8f, 1e-6f);
    ASSERT_NEAR(scoreOf(fan(ison::Value(-3.0)), "P", "w"), 0.0f, 1e-6f);
    ASSERT_NEAR(scoreOf(fan(ison::Value("heavy")), "P", "w"), 0.8f, 1e-6f);
    ASSERT_NEAR(scoreOf(fan(ison::Value(true)), "P", "w"), 0.8f, 1e-6f);
    ASSERT_NEAR(scoreOf(fan(ison::Value(std::nan(""))), "P", "w"), 0.8f, 1e-6f);
    ASSERT_NEAR(scoreOf(fan(ison::Value(2)), "P", "w"), 0.8f, 1e-6f);  // integer weight, clamped
}

TEST(weights_a_zero_weight_scores_zero_but_stays_reachable) {
    SemanticGraph graph = fan(ison::Value(0.0));
    MultiHopOptions options = wideOptions(1, 1);
    options.weightField = "w";
    auto found = graph.semanticMultiHop("q", options);
    ASSERT_EQ(idsOf(found).count("n:P"), 1u);
    ASSERT_NEAR(resultFor(found, "n:P").score, 0.0f, 1e-9f);
}

TEST(weights_best_route_can_be_longer_than_the_shortest) {
    SemanticGraph graph("w", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"S", "M", "T"}) graph.addNode("n", id);
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "T"), {{"w", 0.1}});  // direct but weak
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "M"), {{"w", 1.0}});
    graph.graph().addEdge("R", NodeRef("n", "M"), NodeRef("n", "T"), {{"w", 1.0}});  // 2 hops, strong
    graph.embeddingStore().add(NodeRef("n", "S"), "s", {1.0f, 0.0f, 0.0f, 0.0f});

    auto plain = graph.semanticMultiHop("q", wideOptions(2, 1));
    ASSERT_EQ(resultFor(plain, "n:T").hop_count, 1u);
    ASSERT_NEAR(resultFor(plain, "n:T").score, 0.8f, 1e-6f);

    MultiHopOptions options = wideOptions(2, 1);
    options.weightField = "w";
    auto weighted = graph.semanticMultiHop("q", options);
    const auto& t = resultFor(weighted, "n:T");
    ASSERT_EQ(t.hop_count, 2u);
    ASSERT_EQ(t.path.size(), 3u);
    ASSERT_EQ(t.path[1], NodeRef("n", "M"));
    ASSERT_NEAR(t.score, 0.64f, 1e-6f);
}

TEST(weights_hop_budget_still_applies_to_the_better_route) {
    SemanticGraph graph("w", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"S", "M", "T"}) graph.addNode("n", id);
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "T"), {{"w", 0.1}});
    graph.graph().addEdge("R", NodeRef("n", "S"), NodeRef("n", "M"), {{"w", 1.0}});
    graph.graph().addEdge("R", NodeRef("n", "M"), NodeRef("n", "T"), {{"w", 1.0}});
    graph.embeddingStore().add(NodeRef("n", "S"), "s", {1.0f, 0.0f, 0.0f, 0.0f});
    MultiHopOptions options = wideOptions(1, 1);
    options.weightField = "w";
    ASSERT_NEAR(resultFor(graph.semanticMultiHop("q", options), "n:T").score, 0.08f, 1e-6f);
}

TEST(weights_unweighted_scores_are_bit_identical_with_and_without_the_option_set_to_empty) {
    SemanticGraph graph = vectorGraph();
    MultiHopOptions a = wideOptions(2, 2);
    MultiHopOptions b = a;
    b.weightField = "";
    auto ra = graph.semanticMultiHop("q", a);
    auto rb = graph.semanticMultiHop("q", b);
    ASSERT_EQ(ra.size(), rb.size());
    for (size_t i = 0; i < ra.size(); ++i) {
        ASSERT_TRUE(ra[i].score == rb[i].score);  // exact, not approximate
        ASSERT_TRUE(ra[i].node_ref == rb[i].node_ref);
    }
}

TEST(weights_subgraph_selection_follows_the_weights) {
    SemanticGraph graph = fan(ison::Value(0.1));  // P weak, Q unweighted
    MultiHopOptions options = wideOptions(1, 1);
    options.topKResults = 2;
    options.weightField = "w";
    SemanticGraph sub = graph.semanticSubgraph("q", options);
    auto keys = nodeKeys(sub);
    ASSERT_EQ(keys.size(), 2u);
    ASSERT_EQ(keys.count("n:S"), 1u);
    ASSERT_EQ(keys.count("n:Q"), 1u);
}

// -- subgraph connectivity ---------------------------------------------------

namespace {

/// A and B are the two best matches; X bridges them (A -> X -> B by default).
SemanticGraph bridge(const std::string& relAX = "R", const std::string& relXB = "R",
                     bool aToX = true, bool bToX = false) {
    SemanticGraph graph("b", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"A", "B", "X"}) {
        graph.addNode("n", id, {{"kind", std::string(id) == "X" ? "hub" : "leaf"}});
    }
    auto& g = graph.graph();
    if (aToX) g.addEdge(relAX, NodeRef("n", "A"), NodeRef("n", "X"));
    else      g.addEdge(relAX, NodeRef("n", "X"), NodeRef("n", "A"));
    if (bToX) g.addEdge(relXB, NodeRef("n", "B"), NodeRef("n", "X"));
    else      g.addEdge(relXB, NodeRef("n", "X"), NodeRef("n", "B"));
    graph.embeddingStore().add(NodeRef("n", "A"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("n", "B"), "b", {0.9f, 0.1f, 0.0f, 0.0f});
    return graph;
}

MultiHopOptions pickTwo() {
    MultiHopOptions options = wideOptions(0, 2);
    options.topKResults = 2;
    return options;
}

} // namespace

TEST(connect_default_leaves_the_selection_disconnected) {
    SemanticGraph sub = bridge().semanticSubgraph("q", pickTwo());
    ASSERT_EQ(nodeKeys(sub), (std::set<std::string>{"n:A", "n:B"}));
    ASSERT_EQ(sub.graph().edgeCount(), 0u);
}

TEST(connect_adds_the_bridging_node_and_its_edges) {
    MultiHopOptions options = pickTwo();
    options.connect = true;
    SemanticGraph sub = bridge().semanticSubgraph("q", options);
    ASSERT_EQ(nodeKeys(sub), (std::set<std::string>{"n:A", "n:B", "n:X"}));
    ASSERT_EQ(sub.graph().edgeCount(), 2u);
    // the connector carries its own properties
    ASSERT_EQ(sub.graph().getNode("n", "X").properties.at("kind").as_string(), "hub");
}

TEST(connect_connector_nodes_may_exceed_top_k_results) {
    MultiHopOptions options = pickTwo();
    options.connect = true;
    options.topKResults = 2;
    ASSERT_EQ(bridge().semanticSubgraph("q", options).graph().nodeCount(), 3u);
}

TEST(connect_hop_budget_limits_the_search) {
    SemanticGraph graph = bridge();
    MultiHopOptions options = pickTwo();
    options.connect = true;
    options.connectMaxHops = 1;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)), (std::set<std::string>{"n:A", "n:B"}));
    options.connectMaxHops = 2;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)).count("n:X"), 1u);
    options.connectMaxHops = 0;
    ASSERT_THROWS(graph.semanticSubgraph("q", options), EmbeddingError);
}

TEST(connect_longer_budget_adds_a_chain_of_connectors) {
    SemanticGraph graph("c", std::make_shared<FixedEncoder>(), false);
    for (const char* id : {"A", "M1", "M2", "B"}) graph.addNode("n", id);
    graph.graph().addEdge("R", NodeRef("n", "A"), NodeRef("n", "M1"));
    graph.graph().addEdge("R", NodeRef("n", "M1"), NodeRef("n", "M2"));
    graph.graph().addEdge("R", NodeRef("n", "M2"), NodeRef("n", "B"));
    graph.embeddingStore().add(NodeRef("n", "A"), "a", {1.0f, 0.0f, 0.0f, 0.0f});
    graph.embeddingStore().add(NodeRef("n", "B"), "b", {0.9f, 0.1f, 0.0f, 0.0f});
    MultiHopOptions options = pickTwo();
    options.connect = true;
    options.connectMaxHops = 2;
    ASSERT_EQ(graph.semanticSubgraph("q", options).graph().nodeCount(), 2u);
    options.connectMaxHops = 3;
    ASSERT_EQ(graph.semanticSubgraph("q", options).graph().nodeCount(), 4u);
}

TEST(connect_direction_is_respected) {
    SemanticGraph graph = bridge("R", "R", true, true);  // A -> X <- B
    MultiHopOptions options = pickTwo();
    options.connect = true;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)), (std::set<std::string>{"n:A", "n:B"}));
    options.direction = Direction::Both;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)).count("n:X"), 1u);
}

TEST(connect_rel_type_is_respected) {
    SemanticGraph graph = bridge("KNOWS", "LIKES");
    MultiHopOptions options = pickTwo();
    options.connect = true;
    options.relType = "KNOWS";
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)), (std::set<std::string>{"n:A", "n:B"}));
    options.relType = "";
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)).count("n:X"), 1u);
}

TEST(connect_one_hub_joins_three_components) {
    SemanticGraph graph("h", std::make_shared<FixedEncoder>(), false);
    graph.addNode("n", "HUB");
    float i = 0;
    for (const char* id : {"A", "B", "C"}) {
        graph.addNode("n", id);
        graph.graph().addEdge("R", NodeRef("n", id), NodeRef("n", "HUB"));
        graph.embeddingStore().add(NodeRef("n", id), id, {1.0f - i * 0.05f, 0.1f * i, 0.0f, 0.0f});
        i += 1;
    }
    MultiHopOptions options = wideOptions(0, 3);
    options.topKResults = 3;
    options.connect = true;
    options.direction = Direction::Both;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)),
              (std::set<std::string>{"n:A", "n:B", "n:C", "n:HUB"}));
}

TEST(connect_strict_filter_bars_failing_nodes_as_connectors) {
    SemanticGraph graph = bridge();
    MultiHopOptions options = pickTwo();
    options.connect = true;
    options.nodeFilter = [](const Node& n) { return n.properties.at("kind").as_string() == "leaf"; };
    // lax: X fails the filter but may still bridge
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)).count("n:X"), 1u);
    options.strictFilter = true;
    ASSERT_EQ(nodeKeys(graph.semanticSubgraph("q", options)), (std::set<std::string>{"n:A", "n:B"}));
}

// -- model lifecycle ---------------------------------------------------------

TEST(lifecycle_models_counts_vectors_per_model) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    auto models = graph.embeddingStore().models();
    ASSERT_EQ(models.size(), 1u);
    ASSERT_EQ(models.at("mock-16"), 3u);
    ASSERT_EQ(graph.getEmbedding(NodeRef("person", "1"))->model, "mock-16");
}

TEST(lifecycle_stale_by_model_name_then_reembed) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    graph.setEncoder(retagged(16, "next-gen-16"));

    auto stale = graph.staleEmbeddings();
    ASSERT_EQ(stale.size(), 3u);
    ASSERT_EQ(stale[0], NodeRef("person", "1"));
    ASSERT_EQ(graph.embeddingStore().models().at("mock-16"), 3u);

    ASSERT_EQ(graph.reembedStale(), 3u);
    ASSERT_EQ(graph.staleEmbeddings().size(), 0u);
    auto models = graph.embeddingStore().models();
    ASSERT_EQ(models.size(), 1u);
    ASSERT_EQ(models.at("next-gen-16"), 3u);
}

TEST(lifecycle_nothing_is_stale_under_the_same_encoder) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    ASSERT_EQ(graph.staleEmbeddings().size(), 0u);
    ASSERT_EQ(graph.reembedStale(), 0u);
}

TEST(lifecycle_stale_by_dimension_resets_the_store) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(8));
    graph.setEncoder(std::make_shared<MockEncoder>(16));
    ASSERT_EQ(graph.staleEmbeddings().size(), 3u);

    ASSERT_EQ(graph.reembedStale(), 3u);
    ASSERT_EQ(*graph.embeddingStore().dimension(), 16u);
    ASSERT_EQ(graph.getEmbedding(NodeRef("person", "1"))->vector.size(), 16u);
    ASSERT_EQ(graph.staleEmbeddings().size(), 0u);
}

TEST(lifecycle_reembed_in_small_batches_rewrites_everything) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    graph.setEncoder(retagged(16, "v2"));
    ASSERT_EQ(graph.reembedStale(false, 2), 3u);
    ASSERT_EQ(graph.embeddingStore().models().at("v2"), 3u);
}

TEST(lifecycle_reembed_keeps_custom_embed_text) {
    SemanticGraph graph("g", std::make_shared<MockEncoder>(8), false);
    graph.addNodeWithEmbed("person", "1", {{"name", "A"}}, "a very specific blurb");
    graph.setEncoder(retagged(8, "v2"));
    graph.reembedStale();
    const auto* record = graph.getEmbedding(NodeRef("person", "1"));
    ASSERT_EQ(record->text, "a very specific blurb");
    ASSERT_EQ(record->model, "v2");
}

TEST(lifecycle_unknown_model_is_not_stale_unless_asked) {
    auto encoder = std::make_shared<MockEncoder>(8);
    SemanticGraph graph("g", encoder, false);
    graph.addNode("person", "1");
    graph.embeddingStore().setEncoder(nullptr);  // a vector handed in -> "provided"
    graph.embeddingStore().add(NodeRef("person", "1"), "x", std::vector<float>(8, 0.1f));
    graph.embeddingStore().setEncoder(encoder);
    auto models = graph.embeddingStore().models();
    ASSERT_EQ(models.size(), 1u);
    ASSERT_EQ(models.at("provided"), 1u);
    ASSERT_EQ(graph.staleEmbeddings().size(), 0u);
    auto withUnknown = graph.staleEmbeddings(true);
    ASSERT_EQ(withUnknown.size(), 1u);
    ASSERT_EQ(withUnknown[0], NodeRef("person", "1"));
}

TEST(lifecycle_stale_without_an_encoder_throws_but_the_store_can_still_judge) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    graph.setEncoder(nullptr);
    ASSERT_THROWS(graph.staleEmbeddings(), NoEncoderError);
    // the store can be asked explicitly
    ASSERT_EQ(graph.embeddingStore().stale().size(), 0u);
    ASSERT_EQ(graph.embeddingStore().stale(std::string("other")).size(), 3u);
}

TEST(lifecycle_text_search_with_a_different_model_warns_once) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    graph.setEncoder(retagged(16, "other-16"));
    WarningCapture warnings;
    graph.similaritySearch("hello", 1, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 1u);
    ASSERT_TRUE(warnings.mentions("other-16"));
    ASSERT_TRUE(warnings.mentions("mock-16"));
    graph.similaritySearch("hello again", 1, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 1u);
}

TEST(lifecycle_text_search_with_a_different_model_can_raise) {
    SemanticGraph graph("g", std::make_shared<MockEncoder>(16), true,
                        {"name", "description", "content", "text"}, ModelMismatchMode::Error);
    graph.addNode("person", "1", {{"name", "Alice"}});
    graph.setEncoder(retagged(16, "other-16"));

    bool named = false;
    try {
        graph.similaritySearch("hello", 1);
    } catch (const ModelMismatchError& error) {
        named = std::string(error.what()).find("mock-16") != std::string::npos;
    }
    ASSERT_TRUE(named);
    ASSERT_THROWS(graph.semanticMultiHop("hello"), ModelMismatchError);
    ASSERT_THROWS(graph.semanticPath("hello", NodeRef("person", "1")), ModelMismatchError);

    // re-embedding is the cure
    graph.reembedStale();
    ASSERT_EQ(graph.similaritySearch("hello", 1, "", -1.0f).size(), 1u);
}

TEST(lifecycle_vector_queries_and_ignore_mode_are_never_checked) {
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(16));
    graph.setEncoder(retagged(16, "other-16"));
    WarningCapture warnings;
    graph.embeddingStore().similaritySearchVector(MockEncoder(16).encode("x"), 1, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 0u);

    SemanticGraph quiet("g", std::make_shared<MockEncoder>(16), true,
                        {"name", "description", "content", "text"}, ModelMismatchMode::Ignore);
    quiet.addNode("person", "1", {{"name", "Alice"}});
    quiet.setEncoder(retagged(16, "other-16"));
    quiet.similaritySearch("x", 1, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 0u);
}

TEST(lifecycle_unknown_and_provided_vectors_never_trigger_a_mismatch) {
    auto encoder = retagged(8, "named");
    EmbeddingStore store(encoder);
    store.setEncoder(nullptr);
    store.add(NodeRef("a", "1"), "x", std::vector<float>(8, 0.5f));  // "provided"
    store.setEncoder(encoder);
    WarningCapture warnings;
    store.similaritySearch("q", 1, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 0u);
}

TEST(lifecycle_clear_keeps_the_pinned_dimension_unless_told_to_reset) {
    EmbeddingStore store(std::make_shared<MockEncoder>(8));
    store.add(NodeRef("a", "1"), "x");
    store.clear();
    ASSERT_EQ(store.count(), 0u);
    ASSERT_EQ(*store.dimension(), 8u);
    ASSERT_THROWS(store.add(NodeRef("a", "1"), "x", std::vector<float>(4, 1.0f)), DimensionMismatchError);
    store.clear(true);
    ASSERT_FALSE(store.dimension().has_value());
    store.add(NodeRef("a", "1"), "x", std::vector<float>(4, 1.0f));
    ASSERT_EQ(*store.dimension(), 4u);
}

TEST(lifecycle_version_is_recorded_in_the_sidecar) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8, "2026.1");
    SemanticGraph graph = namedGraph(encoder);
    std::string path = dir.file("g.ison");
    graph.save(path);
    ASSERT_TRUE(readWhole(SemanticGraph::embeddingsPathFor(path))
                    .find("\"encoder_version\":\"2026.1\"") != std::string::npos);

    // an explicit version wins over the encoder's
    EmbeddingStore store(encoder);
    store.setVersion(std::string("explicit"));
    store.add(NodeRef("a", "1"), "x");
    ASSERT_EQ(*store.exportEmbeddings().meta.encoderVersion, "explicit");

    // and nothing is written when no version is known
    EmbeddingStore plain(std::make_shared<MockEncoder>(8));
    plain.add(NodeRef("a", "1"), "x");
    ASSERT_FALSE(plain.exportEmbeddings().meta.encoderVersion.has_value());
    ASSERT_TRUE(plain.toJson().find("encoder_version") == std::string::npos);
}

TEST(lifecycle_loading_with_a_different_version_warns) {
    TempDir dir;
    std::string path = dir.file("g.ison");
    namedGraph(std::make_shared<MockEncoder>(8, "1")).save(path);

    {
        WarningCapture warnings;
        SemanticGraph::load(path, std::make_shared<MockEncoder>(8, "2"));
        ASSERT_EQ(warnings.messages.size(), 1u);
        ASSERT_TRUE(warnings.mentions("version '1'"));
    }
    WarningCapture quiet;
    SemanticGraph::load(path, std::make_shared<MockEncoder>(8, "1"));
    SemanticGraph::load(path, std::make_shared<MockEncoder>(8));  // unversioned: no claim
    ASSERT_EQ(quiet.messages.size(), 0u);
}

// -- save/load consistency ---------------------------------------------------

TEST(consistency_sidecar_records_digest_model_and_stays_version_1) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    std::string raw = readWhole(SemanticGraph::embeddingsPathFor(path));

    // optional metadata trails the array so naive readers meet
    // format/version/dimension/count/embeddings first
    size_t last = 0;
    for (const char* key : {"\"format\"", "\"version\"", "\"dimension\"", "\"count\"",
                            "\"embeddings\"", "\"encoder_model\"", "\"graph_digest\"",
                            "\"graph_digest_basis\""}) {
        size_t at = raw.find(key, last);
        ASSERT_TRUE(at != std::string::npos);
        ASSERT_GE(at, last);
        last = at;
    }
    auto payload = EmbeddingStore::parseJson(raw);
    ASSERT_EQ(payload.version, 1);
    ASSERT_EQ(*payload.meta.encoderModel, "mock-8");
    ASSERT_EQ(*payload.meta.graphDigestBasis, std::string(GRAPH_DIGEST_BASIS));
    ASSERT_EQ(payload.meta.graphDigest->rfind("sha256:", 0), 0u);
    ASSERT_EQ(payload.embeddings[0].model, "mock-8");
}

TEST(consistency_saves_leave_no_temp_files) {
    TempDir dir;
    namedGraph(std::make_shared<MockEncoder>(8)).save(dir.file("g.ison"));
    auto names = dir.listing();
    ASSERT_EQ(names.size(), 2u);
    ASSERT_EQ(names[0], "g.ison");
    ASSERT_EQ(names[1], "g.ison.embeddings.json");
}

TEST(consistency_saving_over_an_existing_file_replaces_it) {
    TempDir dir;
    std::string path = dir.file("g.ison");
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(8));
    graph.save(path);
    graph.addNode("person", "9", {{"name", "Zed"}});
    graph.save(path);
    auto loaded = SemanticGraph::load(path, std::make_shared<MockEncoder>(8));
    ASSERT_EQ(loaded.graph().nodeCount(), 4u);
    ASSERT_EQ(loaded.embeddingStore().count(), 4u);
    ASSERT_EQ(dir.listing().size(), 2u);
}

TEST(consistency_a_failed_write_leaves_no_temp_file_and_throws) {
    TempDir dir;
    // a directory where the graph file should go: the rename cannot succeed
    std::string path = dir.file("g.ison");
    std::filesystem::create_directory(path);
    std::filesystem::create_directory(path + "/keep");
    SemanticGraph graph = namedGraph(std::make_shared<MockEncoder>(8));
    ASSERT_THROWS(graph.save(path), GraphError);
    auto names = dir.listing();
    ASSERT_EQ(names.size(), 1u);
    ASSERT_EQ(names[0], "g.ison");

    // and a failing sidecar write is an EmbeddingError with the temp file gone too
    TempDir other;
    std::string good = other.file("h.ison");
    std::filesystem::create_directory(good + ".embeddings.json");
    std::filesystem::create_directory(good + ".embeddings.json/keep");
    ASSERT_THROWS(graph.save(good), EmbeddingError);
    for (const auto& name : other.listing()) {
        ASSERT_TRUE(name.size() < 4 || name.compare(name.size() - 4, 4, ".tmp") != 0);
    }
}

TEST(consistency_matching_pair_loads_clean_with_typed_values) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    SemanticGraph graph("t", encoder, false);
    graph.addNode("person", "1", {{"name", "A b"}, {"age", 30}, {"ratio", 1.0}, {"ok", true},
                                  {"note", ison::Value(nullptr)}, {"tiny", 1e-9}, {"big", 1e22}});
    graph.addNode("account", "007", {{"name", "Bond"}});
    graph.addNode("account", "42", {{"name", "Answer"}});
    graph.graph().addEdge("KNOWS", NodeRef("person", "1"), NodeRef("account", "007"),
                          {{"w", 0.5}, {"n", 2}});
    graph.embedNode(NodeRef("person", "1"), "a");
    std::string path = dir.file("t.ison");
    graph.save(path);
    WarningCapture warnings;
    auto loaded = SemanticGraph::load(path, encoder);
    ASSERT_EQ(loaded.embeddingStore().count(), 1u);
    ASSERT_EQ(warnings.messages.size(), 0u);
}

TEST(consistency_isonl_round_trip_verifies) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.isonl");
    namedGraph(encoder).save(path);
    ASSERT_EQ(SemanticGraph::load(path, encoder).embeddingStore().count(), 3u);
}

TEST(consistency_sidecar_from_another_save_is_rejected) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string pa = dir.file("a.ison"), pb = dir.file("b.ison");
    namedGraph(encoder, {"Alice", "Bob", "Carol"}).save(pa);
    namedGraph(encoder, {"Xavier", "Yolanda", "Zed"}).save(pb);
    // b's vectors dropped next to a's graph
    writeWhole(SemanticGraph::embeddingsPathFor(pa), readWhole(SemanticGraph::embeddingsPathFor(pb)));
    bool mentionsDigest = false;
    try {
        SemanticGraph::load(pa, encoder);
    } catch (const GraphEmbeddingMismatchError& error) {
        mentionsDigest = std::string(error.what()).find("digest") != std::string::npos;
    }
    ASSERT_TRUE(mentionsDigest);
}

TEST(consistency_edited_graph_is_rejected_by_default) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    writeWhole(path, replaceAll(readWhole(path), "Bob", "Robert"));
    ASSERT_THROWS(SemanticGraph::load(path, encoder), GraphEmbeddingMismatchError);
}

TEST(consistency_warn_mode_loads_the_old_vectors_anyway) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    writeWhole(path, replaceAll(readWhole(path), "Bob", "Robert"));
    WarningCapture warnings;
    auto loaded = SemanticGraph::load(path, encoder, "auto", true, true,
                                      {"name", "description", "content", "text"},
                                      GraphMismatchMode::Warn);
    ASSERT_TRUE(warnings.mentions("digest"));
    ASSERT_EQ(loaded.getEmbedding(NodeRef("person", "2"))->text, "Bob");  // the stale text
}

TEST(consistency_reembed_mode_rebuilds_from_the_edited_graph) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    writeWhole(path, replaceAll(readWhole(path), "Bob", "Robert"));
    auto loaded = SemanticGraph::load(path, encoder, "auto", true, true,
                                      {"name", "description", "content", "text"},
                                      GraphMismatchMode::Reembed);
    const auto* record = loaded.getEmbedding(NodeRef("person", "2"));
    ASSERT_EQ(record->text, "Robert");
    auto expected = encoder->encode("Robert");
    for (size_t i = 0; i < expected.size(); ++i) ASSERT_NEAR(record->vector[i], expected[i], 1e-6f);
    ASSERT_EQ(loaded.embeddingStore().count(), 3u);
}

TEST(consistency_sidecar_without_a_digest_loads_as_before) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    std::string sidecar = SemanticGraph::embeddingsPathFor(path);
    std::string raw = readWhole(sidecar);
    size_t cut = raw.find(",\"encoder_model\"");
    ASSERT_TRUE(cut != std::string::npos);
    writeWhole(sidecar, raw.substr(0, cut) + "}");  // no encoder_model, no digest
    // even an edited graph is accepted: there is nothing to compare with
    writeWhole(path, replaceAll(readWhole(path), "Bob", "Robert"));
    WarningCapture warnings;
    ASSERT_EQ(SemanticGraph::load(path, encoder).embeddingStore().count(), 3u);
    ASSERT_EQ(warnings.messages.size(), 0u);
}

TEST(consistency_digest_of_unknown_basis_is_not_compared) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    std::string sidecar = SemanticGraph::embeddingsPathFor(path);
    std::string raw = readWhole(sidecar);
    size_t cut = raw.find(",\"graph_digest\"");
    ASSERT_TRUE(cut != std::string::npos);
    writeWhole(sidecar, raw.substr(0, cut) + ",\"graph_digest\":\"sha256:" + std::string(64, '0') +
                            "\",\"graph_digest_basis\":\"some-other-port-v9\"}");
    ASSERT_EQ(SemanticGraph::load(path, encoder).embeddingStore().count(), 3u);
}

TEST(consistency_payload_with_unknown_fields_imports_and_exposes_the_meta) {
    EmbeddingStore source(std::make_shared<MockEncoder>(8));
    source.add(NodeRef("person", "1"), "Alice");
    std::string json = source.toJson(std::string("sha256:abc"));
    json.insert(json.size() - 1, ",\"some_future_field\":{\"x\":[1,2,{\"y\":null}]}");
    EmbeddingStore target(std::make_shared<MockEncoder>(8));
    ASSERT_EQ(target.fromJson(json), 1u);
    ASSERT_EQ(*target.lastImportMeta().graphDigest, "sha256:abc");
    ASSERT_EQ(*target.lastImportMeta().graphDigestBasis, std::string(GRAPH_DIGEST_BASIS));
    ASSERT_EQ(*target.lastImportMeta().encoderModel, "mock-8");
    ASSERT_EQ(target.get(NodeRef("person", "1"))->model, "mock-8");
}

TEST(consistency_a_sidecar_that_records_no_model_imports_as_unknown) {
    const std::string payload =
        R"PAYLOAD({"format": "ison-embeddings", "version": 1, "dimension": 2, "count": 1, "embeddings": [{"type": "doc", "id": 7, "id_type": "int", "text": "t", "vector": [0.25, -0.5]}]})PAYLOAD";
    EmbeddingStore store(std::make_shared<MockEncoder>(2));
    ASSERT_EQ(store.fromJson(payload), 1u);
    ASSERT_EQ(store.get(NodeRef("doc", "7"))->model, "unknown");
    ASSERT_EQ(store.stale().size(), 0u);  // unknown provenance is not stale ...
    ASSERT_EQ(store.stale(std::nullopt, std::nullopt, true).size(), 1u);  // ... unless asked
}

TEST(consistency_a_corrupt_sidecar_is_an_error_that_names_the_file) {
    TempDir dir;
    auto encoder = std::make_shared<MockEncoder>(8);
    std::string path = dir.file("g.ison");
    namedGraph(encoder).save(path);
    writeWhole(SemanticGraph::embeddingsPathFor(path), "{ not json");
    bool named = false;
    try {
        SemanticGraph::load(path, encoder);
    } catch (const EmbeddingError& error) {
        named = std::string(error.what()).find("embeddings.json") != std::string::npos;
    }
    ASSERT_TRUE(named);
}

TEST(consistency_replace_import_adopts_the_payload_dimension) {
    EmbeddingStore source(std::make_shared<MockEncoder>(4));
    source.add(NodeRef("a", "1"), "x");
    EmbeddingStore target(std::make_shared<MockEncoder>(8));
    target.add(NodeRef("a", "9"), "y");
    ASSERT_EQ(target.fromJson(source.toJson(), true), 1u);
    ASSERT_EQ(*target.dimension(), 4u);
    ASSERT_EQ(target.count(), 1u);
}

TEST(consistency_sha256_matches_the_standard_vectors) {
    using detail::sha256Hex;
    ASSERT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ASSERT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    ASSERT_EQ(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    ASSERT_EQ(sha256Hex(std::string(1000, 'a')),
              "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST(consistency_float_repr_matches_python) {
    using detail::pyFloatRepr;
    ASSERT_EQ(pyFloatRepr(0.1), "0.1");
    ASSERT_EQ(pyFloatRepr(1.0), "1.0");
    ASSERT_EQ(pyFloatRepr(100.0), "100.0");
    ASSERT_EQ(pyFloatRepr(-2.5), "-2.5");
    ASSERT_EQ(pyFloatRepr(-0.0), "-0.0");
    ASSERT_EQ(pyFloatRepr(1e15), "1000000000000000.0");
    ASSERT_EQ(pyFloatRepr(1e16), "1e+16");
    ASSERT_EQ(pyFloatRepr(1.5e16), "1.5e+16");
    ASSERT_EQ(pyFloatRepr(1e22), "1e+22");
    ASSERT_EQ(pyFloatRepr(123456789.123), "123456789.123");
    ASSERT_EQ(pyFloatRepr(123456789012345.6), "123456789012345.6");
    ASSERT_EQ(pyFloatRepr(12345678901234567890.0), "1.2345678901234567e+19");
    ASSERT_EQ(pyFloatRepr(0.0001), "0.0001");
    ASSERT_EQ(pyFloatRepr(0.00001), "1e-05");
    ASSERT_EQ(pyFloatRepr(2.5e-7), "2.5e-07");
    ASSERT_EQ(pyFloatRepr(0.00030000000000000003), "0.00030000000000000003");
    ASSERT_EQ(pyFloatRepr(0.30000000000000004), "0.30000000000000004");
    ASSERT_EQ(pyFloatRepr(1.0 / 3.0), "0.3333333333333333");
    ASSERT_EQ(pyFloatRepr(5e-324), "5e-324");
    ASSERT_EQ(pyFloatRepr(1.7976931348623157e308), "1.7976931348623157e+308");
    ASSERT_EQ(pyFloatRepr(9007199254740993.0), "9007199254740992.0");
}

TEST(consistency_digest_strings_are_ascii_escaped_like_python) {
    std::string out;
    detail::appendPyString(out, std::string("a\x7f") + "\xc3\xa9" + "\xf0\x9f\x98\x80" + "\n\x01\"\\");
    ASSERT_EQ(out, "\"a\\u007f\\u00e9\\ud83d\\ude00\\n\\u0001\\\"\\\\\"");
}

TEST(consistency_digest_ignores_insertion_order_but_sees_content) {
    ISONGraph a("a"), b("b");
    a.addNode("t", "1", {{"x", 1}});
    a.addNode("t", "2", {{"x", 2}});
    a.addEdge("R", NodeRef("t", "1"), NodeRef("t", "2"));
    b.addNode("t", "2", {{"x", 2}});
    b.addNode("t", "1", {{"x", 1}});
    b.addEdge("R", NodeRef("t", "1"), NodeRef("t", "2"));
    ASSERT_EQ(graphDigest(a), graphDigest(b));
    ASSERT_EQ(graphDigest(a).rfind("sha256:", 0), 0u);
    b.updateNode("t", "1", {{"x", 99}});
    ASSERT_NE(graphDigest(a), graphDigest(b));

    // an integer id and a string id that print alike are different nodes
    ISONGraph c("c"), d("d");
    c.addNode("t", "7");
    d.addNode("t", "7");
    auto isInt = [](const NodeRef&) { return true; };
    ASSERT_NE(graphDigest(c, isInt), graphDigest(d));
    ASSERT_EQ(graphDigest(c), graphDigest(d));
}

TEST(consistency_digest_matches_the_golden_fixture_shared_with_the_python_port) {
    namespace fs = std::filesystem;
    fs::path fixture = fs::path(__FILE__).parent_path().parent_path().parent_path() /
                       "tests" / "fixtures" / "graph_digest_golden.json";
    if (!fs::exists(fixture)) {
        // The fixture lives at the repository root; a checkout of this package alone has no copy.
        std::cout << "(fixture not found, skipped) ";
        return;
    }
    std::string raw = readWhole(fixture.string());
    detail::JsonValue root = detail::JsonParser(raw).parse();
    ASSERT_EQ(root.find("basis")->text, std::string(GRAPH_DIGEST_BASIS));
    const detail::JsonValue* cases = root.find("cases");
    ASSERT_TRUE(cases != nullptr);
    ASSERT_GE(cases->items.size(), 5u);

    auto toValue = [](const detail::JsonValue& v) -> ison::Value {
        using K = detail::JsonValue::Kind;
        switch (v.kind) {
            case K::Bool: return ison::Value(v.boolean);
            case K::String: return ison::Value(v.text);
            case K::Number:
                if (v.text.find_first_of(".eE") != std::string::npos) return ison::Value(v.number);
                return ison::Value(static_cast<long long>(std::stoll(v.text)));
            default: return ison::Value(nullptr);
        }
    };
    auto toProps = [&](const detail::JsonValue* obj) {
        Properties props;
        for (size_t i = 0; i < obj->keys.size(); ++i) props[obj->keys[i]] = toValue(obj->values[i]);
        return props;
    };

    for (const detail::JsonValue& c : cases->items) {
        std::string name = c.find("name")->text;
        ISONGraph graph(name, c.find("directed")->boolean);
        std::set<NodeRef> integerIds;
        for (const detail::JsonValue& n : c.find("nodes")->items) {
            const detail::JsonValue* id = n.find("id");
            NodeRef ref(n.find("type")->text, id->text);
            if (id->kind == detail::JsonValue::Kind::Number) integerIds.insert(ref);
            graph.addNode(ref.type, ref.id, toProps(n.find("properties")));
        }
        for (const detail::JsonValue& e : c.find("edges")->items) {
            auto refOf = [&](const detail::JsonValue& pair) {
                return NodeRef(pair.items[0].text, pair.items[1].text);
            };
            graph.addEdge(e.find("rel_type")->text, refOf(*e.find("source")),
                          refOf(*e.find("target")), toProps(e.find("properties")));
        }
        std::string actual = graphDigest(graph, [&](const NodeRef& r) { return integerIds.count(r) > 0; });
        std::string expected = c.find("digest")->text;
        if (actual != expected) {
            throw std::runtime_error("golden digest mismatch in case '" + name + "': got " +
                                     actual + ", expected " + expected);
        }
    }
}

TEST(consistency_digest_from_ison_text_tags_ids_by_how_the_text_wrote_them) {
    // `7` is an integer id and `"7"` a string one, exactly as the Python port reads them
    ison::Document asInt = ison::loads("nodes.t\nid x\n7 1\n");
    ison::Document asStr = ison::loads("nodes.t\nid x\n\"7\" 1\n");
    ASSERT_NE(detail::digestOfDocument(asInt), detail::digestOfDocument(asStr));

    ISONGraph g("g");
    g.addNode("t", "7", {{"x", 1}});
    ASSERT_EQ(detail::digestOfDocument(asStr), graphDigest(g));
    ASSERT_EQ(detail::digestOfDocument(asInt),
              graphDigest(g, [](const NodeRef&) { return true; }));
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== ISONGraph Vector C++ Tests ===" << std::endl << std::endl;

    // Tests are automatically run by static initialization

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
