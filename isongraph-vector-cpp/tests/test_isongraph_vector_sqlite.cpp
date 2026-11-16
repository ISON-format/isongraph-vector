/**
 * SQLite-backed store tests. Built only with -DISONGRAPH_VECTOR_SQLITE=ON.
 */

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
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

// =============================================================================
// Model lifecycle
// =============================================================================

namespace {

/// Collects the library's warnings for the lifetime of the object.
struct WarningCapture {
    std::vector<std::string> messages;
    WarningCapture() {
        setWarningHandler([this](const std::string& message) { messages.push_back(message); });
    }
    ~WarningCapture() { setWarningHandler(nullptr); }
};

/// A path in the system temp directory that removes itself.
class TempFile {
public:
    explicit TempFile(const std::string& suffix) {
        static int counter = 0;
        path_ = (std::filesystem::temp_directory_path() /
                 ("isongraph_vector_sqlite_" + std::to_string(std::random_device{}()) + "_" +
                  std::to_string(counter++) + suffix)).string();
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::shared_ptr<MockEncoder> named(size_t dim, const std::string& model) {
    auto enc = std::make_shared<MockEncoder>(dim);
    enc->setModelName(model);
    return enc;
}

/// Run raw SQL against a database file, as another port (or an older release) would.
void rawSql(const std::string& path, const std::string& sql) {
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) throw std::runtime_error("open failed");
    char* err = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    std::string message = err ? err : "";
    sqlite3_free(err);
    sqlite3_close(db);
    if (rc != SQLITE_OK) throw std::runtime_error("raw sql failed: " + message);
}

/// First column of the first row of a query, as text ("" for none/NULL).
std::string rawScalar(const std::string& path, const std::string& sql) {
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) throw std::runtime_error("open failed");
    sqlite3_stmt* stmt = nullptr;
    std::string out;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_ROW) {
        auto* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (text) out = text;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return out;
}

/// An encoder whose batches come back with a stray vector width.
struct RaggedEncoder : EmbeddingEncoder {
    size_t dimension() const override { return 4; }
    std::vector<float> encode(const std::string&) const override { return {1, 2, 3, 4}; }
    std::vector<std::vector<float>> encodeBatch(const std::vector<std::string>& texts) const override {
        std::vector<std::vector<float>> out;
        for (size_t i = 0; i < texts.size(); ++i) {
            out.push_back(std::vector<float>(i == 0 ? 4 : 3, 1.0f));
        }
        return out;
    }
};

}  // namespace

TEST(stamps_the_encoders_model_on_writes) {
    SqliteEmbeddingStore store(":memory:", named(8, "model-a"));
    store.add(NodeRef("a", "1"), "text one");
    store.addBatch({{NodeRef("a", "2"), "text two"}, {NodeRef("a", "3"), "text three"}});
    store.add(NodeRef("a", "4"), "handed in", std::vector<float>(8, 0.5f));

    ASSERT_EQ(store.get(NodeRef("a", "1"))->model, "model-a");
    ASSERT_EQ(store.get(NodeRef("a", "3"))->model, "model-a");
    // the vector was supplied, but the store has an encoder: stamped like the in-memory store
    ASSERT_EQ(store.get(NodeRef("a", "4"))->model, "model-a");
}

TEST(vectors_added_without_an_encoder_are_marked_provided) {
    SqliteEmbeddingStore store(":memory:");
    store.add(NodeRef("a", "1"), "text", {1.0f, 0.0f});
    ASSERT_EQ(store.get(NodeRef("a", "1"))->model, "provided");
    ASSERT_EQ(store.models().at("provided"), 1u);
}

TEST(models_counts_follow_every_write) {
    SqliteEmbeddingStore store(":memory:", named(8, "model-a"));
    ASSERT_TRUE(store.models().empty());
    store.add(NodeRef("a", "1"), "one");
    store.add(NodeRef("a", "2"), "two");
    ASSERT_EQ(store.models().at("model-a"), 2u);

    store.setEncoder(named(8, "model-b"));
    store.add(NodeRef("a", "2"), "two again");  // overwrite moves a row between buckets
    auto counts = store.models();
    ASSERT_EQ(counts.at("model-a"), 1u);
    ASSERT_EQ(counts.at("model-b"), 1u);

    store.remove(NodeRef("a", "1"));
    ASSERT_TRUE(store.models().count("model-a") == 0);
    store.clear();
    ASSERT_TRUE(store.models().empty());
}

TEST(stale_finds_vectors_from_another_model) {
    SqliteEmbeddingStore store(":memory:", named(8, "old-model"));
    store.add(NodeRef("a", "2"), "two");
    store.add(NodeRef("a", "1"), "one");
    ASSERT_TRUE(store.stale().empty());

    store.setEncoder(named(8, "new-model"));
    auto stale = store.stale();
    ASSERT_EQ(stale.size(), 2u);
    ASSERT_EQ(stale[0].id, "1");  // sorted
    ASSERT_EQ(stale[1].id, "2");

    store.add(NodeRef("a", "1"), "one");  // re-embedded with the new model
    ASSERT_EQ(store.stale().size(), 1u);
    // an explicit model overrides the encoder's
    ASSERT_EQ(store.stale(std::string("old-model")).size(), 1u);
}

TEST(stale_treats_unknown_models_separately) {
    SqliteEmbeddingStore store(":memory:");
    store.add(NodeRef("a", "1"), "provided", std::vector<float>(8, 1.0f));
    ASSERT_EQ(store.stale(std::string("anything")).size(), 0u);  // cannot judge
    ASSERT_EQ(store.stale(std::string("anything"), std::nullopt, true).size(), 1u);
    // a dimension mismatch always counts, known model or not
    ASSERT_EQ(store.stale(std::nullopt, size_t(4)).size(), 1u);
    // no encoder, no arguments: nothing can be judged
    ASSERT_TRUE(store.stale().empty());
}

TEST(a_text_query_from_another_model_warns_once) {
    WarningCapture warnings;
    SqliteEmbeddingStore store(":memory:", named(8, "old-model"));
    store.add(NodeRef("a", "1"), "one");

    store.setEncoder(named(8, "new-model"));
    store.similaritySearch("query", 5, "", -1.0f);
    store.similaritySearch("query again", 5, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 1u);
    ASSERT_TRUE(warnings.messages[0].find("new-model") != std::string::npos);
    ASSERT_TRUE(warnings.messages[0].find("old-model") != std::string::npos);

    // a vector query is never checked
    store.similaritySearchVector(std::vector<float>(8, 0.1f), 5, "", -1.0f);
    ASSERT_EQ(warnings.messages.size(), 1u);
}

TEST(model_mismatch_can_be_an_error_or_ignored) {
    WarningCapture warnings;
    SqliteEmbeddingStore strict(":memory:", named(8, "old-model"), false, ModelMismatchMode::Error);
    strict.add(NodeRef("a", "1"), "one");
    strict.setEncoder(named(8, "new-model"));
    ASSERT_THROWS(strict.similaritySearch("q", 5, "", -1.0f), ModelMismatchError);
    strict.similaritySearchVector(std::vector<float>(8, 0.1f), 5, "", -1.0f);  // not a text query

    strict.setOnModelMismatch(ModelMismatchMode::Ignore);
    strict.similaritySearch("q", 5, "", -1.0f);
    ASSERT_TRUE(warnings.messages.empty());
}

TEST(unknown_and_provided_vectors_never_trigger_a_mismatch) {
    WarningCapture warnings;
    SqliteEmbeddingStore store(":memory:", nullptr, false, ModelMismatchMode::Error);
    store.add(NodeRef("a", "1"), "handed in", std::vector<float>(8, 0.5f));
    store.setEncoder(named(8, "some-model"));
    store.similaritySearch("q", 5, "", -1.0f);  // would throw if "provided" counted
    ASSERT_TRUE(warnings.messages.empty());
}

TEST(clear_keeps_the_pinned_dimension_unless_told_to_reset) {
    SqliteEmbeddingStore store(":memory:", named(8, "m"));
    store.add(NodeRef("a", "1"), "one");
    store.clear();
    ASSERT_EQ(store.count(), 0u);
    ASSERT_TRUE(store.dimension().has_value());
    ASSERT_THROWS(store.add(NodeRef("a", "2"), "wide", std::vector<float>(16, 1.0f)),
                  DimensionMismatchError);

    store.clear(true);
    ASSERT_FALSE(store.dimension().has_value());
    store.add(NodeRef("a", "2"), "wide", std::vector<float>(16, 1.0f));
    ASSERT_EQ(*store.dimension(), 16u);
}

TEST(clear_with_reset_lets_the_vec_index_change_width) {
    SqliteEmbeddingStore store(":memory:", named(8, "m"), true);
    store.add(NodeRef("a", "1"), "one");
    store.clear(true);
    store.setEncoder(named(16, "m16"));
    store.add(NodeRef("a", "2"), "two");
    store.add(NodeRef("a", "3"), "three");

    auto hits = store.similaritySearch("two", 2, "", -1.0f);
    ASSERT_EQ(hits.size(), 2u);
    ASSERT_EQ(hits[0].node_ref.id, "2");
}

TEST(export_carries_models_and_encoder_metadata) {
    auto enc = named(8, "model-a");
    enc->setVersion(std::string("rev-7"));
    SqliteEmbeddingStore store(":memory:", enc);
    store.add(NodeRef("person", "1"), "Alice");

    auto payload = store.exportEmbeddings(std::string("digest-1"));
    ASSERT_EQ(payload.embeddings.size(), 1u);
    ASSERT_EQ(payload.embeddings[0].model, "model-a");
    ASSERT_EQ(*payload.meta.encoderModel, "model-a");
    ASSERT_EQ(*payload.meta.encoderVersion, "rev-7");
    ASSERT_EQ(*payload.meta.graphDigest, "digest-1");
    ASSERT_EQ(*payload.meta.graphDigestBasis, std::string(GRAPH_DIGEST_BASIS));

    // an explicit version wins over the encoder's
    store.setVersion(std::string("override"));
    ASSERT_EQ(*store.exportEmbeddings().meta.encoderVersion, "override");
}

TEST(import_keeps_each_entrys_model) {
    SqliteEmbeddingStore source(":memory:", named(8, "model-a"));
    source.add(NodeRef("a", "1"), "one");
    auto payload = source.exportEmbeddings();
    payload.embeddings.push_back(ExportedEmbedding{"a", "2", "str", "two",
                                                   std::vector<float>(8, 0.25f), "model-b"});
    payload.embeddings.push_back(ExportedEmbedding{"a", "3", "str", "three",
                                                   std::vector<float>(8, 0.5f)});  // default model

    SqliteEmbeddingStore target(":memory:");
    ASSERT_EQ(target.importEmbeddings(payload), 3u);
    ASSERT_EQ(target.get(NodeRef("a", "1"))->model, "model-a");
    ASSERT_EQ(target.get(NodeRef("a", "2"))->model, "model-b");
    ASSERT_EQ(target.get(NodeRef("a", "3"))->model, "unknown");
}

TEST(json_round_trip_and_metadata) {
    auto enc = named(8, "model-a");
    enc->setVersion(std::string("rev-1"));
    SqliteEmbeddingStore source(":memory:", enc);
    source.add(NodeRef("person", "1"), "Alice \"the\" engineer\n\xc3\xa9");
    source.add(NodeRef("doc", "x"), "a document");

    SqliteEmbeddingStore target(":memory:");
    ASSERT_EQ(target.fromJson(source.toJson(std::string("abc123"))), 2u);
    ASSERT_EQ(target.get(NodeRef("person", "1"))->text, "Alice \"the\" engineer\n\xc3\xa9");
    ASSERT_EQ(target.get(NodeRef("doc", "x"))->model, "model-a");
    ASSERT_EQ(*target.lastImportMeta().encoderModel, "model-a");
    ASSERT_EQ(*target.lastImportMeta().encoderVersion, "rev-1");
    ASSERT_EQ(*target.lastImportMeta().graphDigest, "abc123");
}

TEST(json_matches_the_in_memory_store_byte_for_byte) {
    auto enc = named(8, "model-a");
    enc->setVersion(std::string("rev-1"));
    SqliteEmbeddingStore sql(":memory:", enc);
    EmbeddingStore memory(enc);
    for (const char* id : {"b", "a", "c"}) {
        sql.add(NodeRef("t", id), std::string("text ") + id);
        memory.add(NodeRef("t", id), std::string("text ") + id);
    }
    ASSERT_EQ(sql.toJson(std::string("d")), memory.toJson(std::string("d")));
    ASSERT_EQ(sql.toJson(), memory.toJson());
}

TEST(save_and_load_a_sidecar_atomically) {
    TempFile sidecar(".embeddings.json");
    SqliteEmbeddingStore source(":memory:", named(8, "model-a"));
    source.add(NodeRef("a", "1"), "one");
    source.add(NodeRef("a", "2"), "two");
    ASSERT_EQ(source.saveEmbeddings(sidecar.path(), std::string("dg")), 2u);

    // a second save replaces the first, leaving no temp file behind
    source.add(NodeRef("a", "3"), "three");
    ASSERT_EQ(source.saveEmbeddings(sidecar.path()), 3u);
    std::filesystem::path file(sidecar.path());
    size_t siblings = 0;
    for (const auto& entry : std::filesystem::directory_iterator(file.parent_path())) {
        if (entry.path().filename().string().find(file.filename().string()) != std::string::npos) {
            ++siblings;
        }
    }
    ASSERT_EQ(siblings, 1u);

    SqliteEmbeddingStore target(":memory:");
    ASSERT_EQ(target.loadEmbeddings(sidecar.path()), 3u);
    ASSERT_EQ(target.get(NodeRef("a", "3"))->model, "model-a");
    ASSERT_THROWS(target.loadEmbeddings(sidecar.path() + ".missing"), EmbeddingError);
}

TEST(a_failed_import_changes_nothing) {
    SqliteEmbeddingStore store(":memory:", named(8, "model-a"), true);
    store.add(NodeRef("a", "1"), "one");
    store.add(NodeRef("a", "2"), "two");

    ExportedEmbeddings bad;
    bad.dimension = 4;
    bad.embeddings.push_back(ExportedEmbedding{"b", "1", "str", "ok", std::vector<float>(4, 1.0f)});
    bad.embeddings.push_back(ExportedEmbedding{"b", "2", "str", "wrong width",
                                               std::vector<float>(5, 1.0f)});

    ASSERT_THROWS(store.importEmbeddings(bad, true), DimensionMismatchError);
    ASSERT_EQ(store.count(), 2u);
    ASSERT_EQ(*store.dimension(), 8u);
    ASSERT_EQ(store.get(NodeRef("a", "1"))->model, "model-a");
    ASSERT_EQ(store.similaritySearch("one", 2, "", -1.0f).size(), 2u);  // the index still answers
}

TEST(import_with_replace_swaps_the_contents_and_the_width) {
    SqliteEmbeddingStore store(":memory:", named(8, "model-a"), true);
    store.add(NodeRef("a", "1"), "one");

    ExportedEmbeddings wider;
    wider.dimension = 16;
    wider.embeddings.push_back(
        ExportedEmbedding{"b", "1", "str", "x", std::vector<float>(16, 1.0f), "m16"});
    wider.embeddings.push_back(
        ExportedEmbedding{"b", "2", "str", "y", std::vector<float>(16, -1.0f), "m16"});
    ASSERT_EQ(store.importEmbeddings(wider, true), 2u);

    ASSERT_EQ(store.count(), 2u);
    ASSERT_EQ(*store.dimension(), 16u);
    ASSERT_FALSE(store.get(NodeRef("a", "1")).has_value());
    auto hits = store.similaritySearchVector(std::vector<float>(16, 1.0f), 1, "", -1.0f);
    ASSERT_EQ(hits[0].node_ref.id, "1");
    ASSERT_EQ(store.models().at("m16"), 2u);
}

TEST(a_failed_batch_writes_nothing_and_pins_nothing) {
    SqliteEmbeddingStore store(":memory:", std::make_shared<RaggedEncoder>());
    ASSERT_THROWS(store.addBatch({{NodeRef("a", "1"), "x"}, {NodeRef("a", "2"), "y"}}),
                  DimensionMismatchError);
    ASSERT_EQ(store.count(), 0u);
    ASSERT_FALSE(store.dimension().has_value());
}

// -- shared schema -------------------------------------------------------------

TEST(opens_a_database_from_a_release_without_the_model_column) {
    TempFile db(".db");
    rawSql(db.path(), R"(
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            node_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node_type, node_id)
        );
        INSERT INTO embeddings (id, node_type, node_id, vector, text)
        VALUES ('x', 'person', '1', X'0000803F000000000000000000000000', 'legacy row');
    )");

    {
        SqliteEmbeddingStore store(db.path(), named(4, "model-a"));
        ASSERT_EQ(store.count(), 1u);
        ASSERT_EQ(store.get(NodeRef("person", "1"))->model, "unknown");
        ASSERT_EQ(store.models().at("unknown"), 1u);
        ASSERT_EQ(*store.dimension(), 4u);
        store.add(NodeRef("person", "2"), "new row");
    }
    // the column was added with the documented default, and the new row carries the model
    ASSERT_EQ(rawScalar(db.path(), "SELECT model FROM embeddings WHERE node_id = '1'"), "unknown");
    ASSERT_EQ(rawScalar(db.path(), "SELECT model FROM embeddings WHERE node_id = '2'"), "model-a");
    ASSERT_EQ(rawScalar(db.path(), "SELECT node_id_type FROM embeddings WHERE node_id = '1'"),
              "str");
}

TEST(reads_rows_written_by_another_port_and_leaves_the_schema_as_it_is) {
    // The schema every port writes: a model column defaulting to the old model name.
    TempFile db(".db");
    rawSql(db.path(), R"(
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            node_id TEXT NOT NULL,
            node_id_type TEXT NOT NULL DEFAULT 'str',
            vector BLOB NOT NULL,
            text TEXT NOT NULL,
            model TEXT DEFAULT 'all-MiniLM-L6-v2',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node_type, node_id)
        );
        CREATE INDEX idx_embeddings_node ON embeddings(node_type, node_id);
        INSERT INTO embeddings (id, node_type, node_id, node_id_type, vector, text, model)
        VALUES ('p', 'person', '2', 'int', X'0000803F000000000000000000000000', 'from python', 'mock-4');
    )");

    {
        SqliteEmbeddingStore store(db.path(), named(4, "mock-4"));
        ASSERT_EQ(store.get(NodeRef("person", "2"))->model, "mock-4");
        ASSERT_TRUE(store.stale().empty());
        auto payload = store.exportEmbeddings();
        ASSERT_EQ(payload.embeddings[0].id_type, "int");
        ASSERT_EQ(payload.embeddings[0].model, "mock-4");
    }
    ASSERT_EQ(rawScalar(db.path(), "SELECT COUNT(*) FROM pragma_table_info('embeddings')"), "8");
}

TEST(rows_written_here_are_what_the_other_ports_expect) {
    TempFile db(".db");
    {
        SqliteEmbeddingStore store(db.path(), named(4, "model-a"));
        store.add(NodeRef("person", "1"), "Alice");
        store.addBatch({{NodeRef("doc", "r\xc3\xa9sum\xc3\xa9"), "caf\xc3\xa9"}});
    }
    // Python, Rust, C# and TypeScript select exactly these columns; each must hold what they read.
    ASSERT_EQ(rawScalar(db.path(),
                        "SELECT model FROM embeddings WHERE node_type = 'person' AND node_id = '1'"),
              "model-a");
    ASSERT_EQ(rawScalar(db.path(), "SELECT node_id_type FROM embeddings WHERE node_type = 'person'"),
              "str");
    ASSERT_EQ(rawScalar(db.path(), "SELECT LENGTH(vector) FROM embeddings WHERE node_type = 'person'"),
              "16");
    ASSERT_EQ(rawScalar(db.path(), "SELECT id FROM embeddings WHERE node_type = 'person'"),
              detail::Md5::hex("person:1").substr(0, 16));
    ASSERT_EQ(rawScalar(db.path(), "SELECT text FROM embeddings WHERE node_type = 'doc'"),
              "caf\xc3\xa9");
    ASSERT_EQ(rawScalar(db.path(), "SELECT COUNT(*) FROM pragma_table_info('embeddings')"), "8");
}

int main() {
    std::cout << "\n=== isongraph-vector SQLite store tests ===" << std::endl << std::endl;
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
