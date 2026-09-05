/**
 * SQLite-backed store tests. Built only with -DISONGRAPH_VECTOR_SQLITE=ON.
 */

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../include/isongraph_vector_sqlite.hpp"

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
    if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " != " #b);
#define ASSERT_NE(a, b) \
    if ((a) == (b)) throw std::runtime_error("Assertion failed: " #a " == " #b);
#define ASSERT_TRUE(expr) \
    if (!(expr)) throw std::runtime_error("Assertion failed: " #expr);
#define ASSERT_FALSE(expr) \
    if (expr) throw std::runtime_error("Assertion failed: " #expr " should be false");
#define ASSERT_NEAR(a, b, tol) \
    if (std::fabs((a) - (b)) > (tol)) \
        throw std::runtime_error("Assertion failed: " #a " != " #b " within " #tol);
#define ASSERT_THROWS(expr, exception_type) \
    { \
        bool threw = false; \
        try { expr; } catch (const exception_type&) { threw = true; } \
        if (!threw) throw std::runtime_error("Assertion failed: " #expr " did not throw"); \
    }

namespace {

std::shared_ptr<MockEncoder> encoder(size_t dim = 32) {
    return std::make_shared<MockEncoder>(dim);
}

/// A temp database path that removes itself.
class TempDb {
public:
    explicit TempDb(const std::string& name) {
        path_ = std::string(std::tmpnam(nullptr)) + name;
    }
    ~TempDb() { std::remove(path_.c_str()); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

}  // namespace

TEST(md5_matches_the_other_ports) {
    // The `id` column is an md5 prefix; a row written here has to be
    // byte-identical to one written by Python, C#, Rust or Node.
    ASSERT_EQ(detail::Md5::hex(""), "d41d8cd98f00b204e9800998ecf8427e");
    ASSERT_EQ(detail::Md5::hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    ASSERT_EQ(detail::Md5::hex("person:1"), "427ee4ba613aeabc5c5098e5fd99aa9c");
}

TEST(adds_and_reads_back) {
    SqliteEmbeddingStore store(":memory:", encoder());
    store.add(NodeRef("person", "1"), "Alice is an engineer");

    auto record = store.get(NodeRef("person", "1"));
    ASSERT_TRUE(record.has_value());
    ASSERT_EQ(record->text, "Alice is an engineer");
    ASSERT_EQ(record->vector.size(), 32u);
    ASSERT_EQ(store.count(), 1u);
}

TEST(removes_and_clears) {
    SqliteEmbeddingStore store(":memory:", encoder());
    store.add(NodeRef("a", "1"), "one");
    store.add(NodeRef("a", "2"), "two");

    ASSERT_TRUE(store.remove(NodeRef("a", "1")));
    ASSERT_FALSE(store.remove(NodeRef("a", "1")));
    ASSERT_EQ(store.count(), 1u);

    store.clear();
    ASSERT_EQ(store.count(), 0u);
}

TEST(overwrites_rather_than_duplicating) {
    SqliteEmbeddingStore store(":memory:", encoder());
    store.add(NodeRef("a", "1"), "first");
    store.add(NodeRef("a", "1"), "second");
    ASSERT_EQ(store.count(), 1u);
    ASSERT_EQ(store.get(NodeRef("a", "1"))->text, "second");
}

TEST(validates_dimensions) {
    SqliteEmbeddingStore store(":memory:", encoder());
    store.add(NodeRef("a", "1"), "full width");

    ASSERT_THROWS(store.add(NodeRef("a", "2"), "short", {1.0f, 0.0f}), DimensionMismatchError);
    ASSERT_THROWS(store.similaritySearchVector({1.0f, 0.0f}, 1, "", -1.0f),
                  DimensionMismatchError);
}

TEST(rejects_text_without_an_encoder) {
    SqliteEmbeddingStore store(":memory:");
    ASSERT_THROWS(store.add(NodeRef("a", "1"), "no encoder here"), NoEncoderError);
}

TEST(batches_in_one_transaction) {
    SqliteEmbeddingStore store(":memory:", encoder());
    std::vector<std::pair<NodeRef, std::string>> entries;
    for (int i = 0; i < 10; ++i) {
        entries.emplace_back(NodeRef("doc", std::to_string(i)), "a document");
    }
    ASSERT_EQ(store.addBatch(entries), 10u);
    ASSERT_EQ(store.count(), 10u);
}

TEST(survives_close_and_reopen) {
    TempDb db("reopen.db");
    std::vector<float> probe;
    {
        SqliteEmbeddingStore store(db.path(), encoder());
        store.add(NodeRef("person", "1"), "Alice");
        probe = store.get(NodeRef("person", "1"))->vector;
    }

    SqliteEmbeddingStore reopened(db.path(), encoder());
    ASSERT_EQ(reopened.count(), 1u);
    ASSERT_EQ(reopened.get(NodeRef("person", "1"))->vector, probe);
    // dimension is recovered from the stored rows
    ASSERT_TRUE(reopened.dimension().has_value());
    ASSERT_EQ(*reopened.dimension(), 32u);
}

// =============================================================================
// sqlite-vec index
// =============================================================================

TEST(reports_the_backend) {
    SqliteEmbeddingStore plain(":memory:", encoder());
    SqliteEmbeddingStore indexed(":memory:", encoder(), true);
    ASSERT_FALSE(plain.sqliteVec());
    ASSERT_TRUE(indexed.sqliteVec());
}

TEST(index_matches_the_scan_exactly) {
    auto enc = encoder();
    std::vector<std::pair<NodeRef, std::string>> entries;
    for (int i = 0; i < 200; ++i) {
        entries.emplace_back(NodeRef("n", std::to_string(i)), "text " + std::to_string(i));
    }

    SqliteEmbeddingStore scan(":memory:", enc);
    SqliteEmbeddingStore indexed(":memory:", enc, true);
    scan.addBatch(entries);
    indexed.addBatch(entries);

    for (const std::string& probe : {"text 0", "text 57", "text 199"}) {
        auto query = enc->encode(probe);
        auto want = scan.similaritySearchVector(query, 10, "", -1.0f);
        auto got = indexed.similaritySearchVector(query, 10, "", -1.0f);

        ASSERT_EQ(want.size(), got.size());
        for (size_t i = 0; i < want.size(); ++i) {
            ASSERT_EQ(want[i].node_ref.id, got[i].node_ref.id);
            ASSERT_NEAR(want[i].score, got[i].score, 1e-5f);
        }
    }
}

TEST(type_filter_stays_exact) {
    // Filtering after a top-k fetch would drop the rare node entirely; the
    // type is a vec0 metadata column so the search itself is narrowed.
    SqliteEmbeddingStore store(":memory:", encoder(), true);
    for (int i = 0; i < 50; ++i) {
        store.add(NodeRef("crowd", std::to_string(i)), "crowd " + std::to_string(i));
    }
    store.add(NodeRef("rare", "1"), "the only rare node");

    auto results = store.similaritySearch("anything", 5, "rare", -1.0f);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].node_ref.type, "rare");
}

TEST(updates_and_deletes_reach_the_index) {
    SqliteEmbeddingStore store(":memory:", encoder(), true);
    store.add(NodeRef("a", "1"), "first");
    store.add(NodeRef("a", "2"), "second");

    auto second = store.get(NodeRef("a", "2"))->vector;
    store.add(NodeRef("a", "1"), "first", second);

    auto top = store.similaritySearchVector(second, 2, "", -1.0f);
    ASSERT_EQ(top.size(), 2u);
    for (const auto& r : top) ASSERT_NEAR(r.score, 1.0f, 1e-5f);

    store.remove(NodeRef("a", "1"));
    auto after = store.similaritySearchVector(second, 10, "", -1.0f);
    ASSERT_EQ(after.size(), 1u);
    ASSERT_EQ(after[0].node_ref.id, "2");
}

TEST(indexes_a_database_written_without_it) {
    TempDb db("backfill.db");
    std::vector<float> probe;
    {
        SqliteEmbeddingStore plain(db.path(), encoder());
        for (int i = 0; i < 20; ++i) {
            plain.add(NodeRef("n", std::to_string(i)), "text " + std::to_string(i));
        }
        probe = plain.get(NodeRef("n", "7"))->vector;
    }

    SqliteEmbeddingStore indexed(db.path(), encoder(), true);
    ASSERT_EQ(indexed.count(), 20u);

    auto results = indexed.similaritySearchVector(probe, 3, "", -1.0f);
    ASSERT_EQ(results[0].node_ref.id, "7");
    ASSERT_NEAR(results[0].score, 1.0f, 1e-5f);
}

TEST(score_map_still_covers_everything) {
    SqliteEmbeddingStore store(":memory:", encoder(), true);
    for (int i = 0; i < 30; ++i) {
        store.add(NodeRef("n", std::to_string(i)), "text " + std::to_string(i));
    }
    ASSERT_EQ(store.scoreMap(encoder()->encode("anything")).size(), 30u);
}

TEST(zero_query_vector_falls_back_to_the_scan) {
    SqliteEmbeddingStore store(":memory:", std::make_shared<MockEncoder>(4), true);
    store.add(NodeRef("a", "1"), "x", {1.0f, 0.0f, 0.0f, 0.0f});

    auto results = store.similaritySearchVector({0.0f, 0.0f, 0.0f, 0.0f}, 1, "", -1.0f);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].score, 0.0f);
}

TEST(round_trips_the_portable_payload) {
    SqliteEmbeddingStore source(":memory:", encoder());
    source.add(NodeRef("person", "1"), "Alice");
    source.add(NodeRef("account", "007"), "Bond");

    SqliteEmbeddingStore target(":memory:", encoder(), true);
    ASSERT_EQ(target.importEmbeddings(source.exportEmbeddings()), 2u);
    ASSERT_EQ(target.get(NodeRef("account", "007"))->text, "Bond");
}

int main() {
    std::cout << "\n=== isongraph-vector SQLite store tests ===" << std::endl << std::endl;
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
