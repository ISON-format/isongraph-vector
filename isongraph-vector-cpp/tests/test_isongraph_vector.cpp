/**
 * ISONGraph Vector C++ Tests
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
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
