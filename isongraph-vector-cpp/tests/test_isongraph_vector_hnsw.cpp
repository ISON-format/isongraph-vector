/**
 * ISONGraph Vector C++ Tests: the opt-in HNSW index (AnnMode::Hnsw)
 *
 * Mirrors the Rust crate's HNSW tests and DESIGN.md section 12: recall against
 * the exact scan, exactness below minSize, bit-identical scores, filters and
 * their fallbacks, tombstones / updates / compaction, thread-count independence
 * of the graph, batch == single, concurrent const searches, and "never
 * persisted". The existing suite (test_isongraph_vector.cpp) is untouched.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
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

#define ASSERT_EQ(a, b) \
    if (!((a) == (b))) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #a " != " #b); \
    }

#define ASSERT_NE(a, b) \
    if ((a) == (b)) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #a " == " #b); \
    }

#define ASSERT_TRUE(expr) \
    if (!(expr)) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #expr); \
    }

#define ASSERT_FALSE(expr) \
    if (expr) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #expr " should be false"); \
    }

#define ASSERT_GE(a, b) \
    if ((a) < (b)) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #a " < " #b); \
    }

#define ASSERT_GT(a, b) \
    if ((a) <= (b)) { \
        throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #a " <= " #b); \
    }

#define ASSERT_THROWS(expr, type) \
    { \
        bool threw_ = false; \
        try { (void)(expr); } catch (const type&) { threw_ = true; } \
        if (!threw_) throw std::runtime_error("Assertion failed (line " + std::to_string(__LINE__) + "): " #expr " did not throw " #type); \
    }

namespace {

// =============================================================================
// Data and helpers
// =============================================================================

using Vec = std::vector<float>;

/** Clustered vectors (the shape of real embeddings) and queries drawn from the same clusters. */
struct Clustered {
    size_t dim;
    double noise;
    std::vector<Vec> centers;
    std::mt19937_64 rng;

    Clustered(size_t clusters, size_t dimension, uint64_t seed, double noiseNorm = 0.6)
        : dim(dimension), noise(noiseNorm), rng(seed) {
        for (size_t c = 0; c < clusters; ++c) {
            Vec center(dim);
            double sq = 0;
            for (auto& x : center) { x = static_cast<float>(gauss()); sq += double(x) * double(x); }
            for (auto& x : center) x = static_cast<float>(double(x) / std::sqrt(sq));
            centers.push_back(std::move(center));
        }
    }

    double gauss() {
        double u1 = (static_cast<double>(rng() >> 11) + 1.0) / 9007199254740993.0;
        double u2 = static_cast<double>(rng() >> 11) / 9007199254740992.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    Vec draw() {
        const auto& center = centers[rng() % centers.size()];
        Vec v(dim);
        for (size_t d = 0; d < dim; ++d) v[d] = center[d] + static_cast<float>(noise * gauss() / std::sqrt(double(dim)));
        return v;
    }

    std::vector<Vec> many(size_t count) {
        std::vector<Vec> out;
        out.reserve(count);
        for (size_t i = 0; i < count; ++i) out.push_back(draw());
        return out;
    }
};

/** Plain gaussian vectors: no cluster structure to lean on. */
std::vector<Vec> randomVectors(size_t count, size_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<Vec> out(count, Vec(dim));
    for (auto& v : out) for (auto& x : v) x = normal(rng);
    return out;
}

uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

std::string keyOf(const SimilarityResult& r) { return r.node_ref.type + ":" + r.node_ref.id; }

bool sameResults(const std::vector<SimilarityResult>& a, const std::vector<SimilarityResult>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (keyOf(a[i]) != keyOf(b[i]) || floatBits(a[i].score) != floatBits(b[i].score)) return false;
    }
    return true;
}

AnnOptions hnswOptions(size_t minSize = 0) {
    AnnOptions o = AnnOptions::makeHnsw();
    o.minSize = minSize;
    return o;
}

void fill(EmbeddingStore& store, const std::vector<Vec>& data,
          const std::function<std::string(size_t)>& typeOf = nullptr) {
    for (size_t i = 0; i < data.size(); ++i) {
        store.add(NodeRef(typeOf ? typeOf(i) : "doc", std::to_string(i)), "", data[i]);
    }
}

/** Mean overlap of the approximate and exact top-k sets. */
double recallOf(const EmbeddingStore& store, const std::vector<Vec>& queries, size_t k,
                const std::string& type = "", float threshold = -1.0f) {
    double total = 0;
    size_t counted = 0;
    for (const auto& q : queries) {
        auto truth = store.similaritySearchVector(q, k, type, threshold, true);
        if (truth.empty()) continue;
        std::set<std::string> want;
        for (const auto& r : truth) want.insert(keyOf(r));
        auto got = store.similaritySearchVector(q, k, type, threshold);
        size_t hit = 0;
        for (const auto& r : got) hit += want.count(keyOf(r));
        total += static_cast<double>(hit) / static_cast<double>(want.size());
        ++counted;
    }
    return counted ? total / static_cast<double>(counted) : 1.0;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct Dir {
    std::filesystem::path root;
    explicit Dir(const std::string& tag) {
        root = std::filesystem::temp_directory_path() / ("isongraph_hnsw_" + tag + "_" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(root);
    }
    ~Dir() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
    std::string file(const std::string& name) const { return (root / name).string(); }
};

/** A raw table the index can be built over without a store (for graph-level tests). */
struct RawTable {
    detail::ChunkedMatrix matrix;
    std::vector<uint8_t> alive;
    std::vector<uint32_t> codes;
    size_t dim;

    RawTable(const std::vector<Vec>& data, size_t dimension) : dim(dimension) {
        matrix.reset(dim);
        for (const auto& v : data) {
            matrix.append(v.data());
            alive.push_back(1);
            codes.push_back(0);
        }
    }

    detail::HnswRows rows() const {
        detail::HnswRows r;
        r.matrix = &matrix;
        r.alive = alive.data();
        r.codes = codes.data();
        r.dim = dim;
        return r;
    }

    detail::HnswIndex build(size_t threads, uint64_t seed = 0, size_t m = 16, size_t efc = 200) const {
        return detail::HnswIndex::build(rows(), alive.size(), m, efc, seed, threads);
    }
};

// =============================================================================
// Pieces
// =============================================================================

}  // namespace

TEST(hnsw_options_default_to_ivf_and_validate) {
    AnnOptions defaults;
    ASSERT_TRUE(defaults.mode == AnnMode::Ivf);
    ASSERT_EQ(defaults.hnsw.m, 16u);
    ASSERT_EQ(defaults.hnsw.efConstruction, 200u);
    ASSERT_EQ(defaults.hnsw.efSearch, 0u);
    ASSERT_EQ(defaults.minSize, 50000u);
    AnnOptions h = AnnOptions::makeHnsw();
    ASSERT_TRUE(h.mode == AnnMode::Hnsw);
    ASSERT_EQ(h.minSize, 50000u);

    EmbeddingStore store;
    AnnOptions bad = hnswOptions();
    bad.hnsw.m = 1;
    ASSERT_THROWS(store.enableAnn(bad), EmbeddingError);
    bad.hnsw.m = 65;
    ASSERT_THROWS(store.enableAnn(bad), EmbeddingError);
    bad.hnsw.m = 2;
    store.enableAnn(bad);
    bad.hnsw.m = 64;
    store.enableAnn(bad);
    bad.hnsw.efConstruction = 0;
    ASSERT_THROWS(store.enableAnn(bad), EmbeddingError);

    EmbeddingStore none;
    ASSERT_THROWS(none.setAnnEfSearch(10), EmbeddingError);
    ASSERT_THROWS(none.similaritySearchVectorEf(Vec(4, 1.0f), 3, "", 0.0f, 0), EmbeddingError);
    AnnStats s = none.annStats();
    ASSERT_FALSE(s.enabled);
    ASSERT_TRUE(s.mode == AnnMode::Ivf);
}

TEST(hnsw_splitmix64_and_level_law_are_portable_integers) {
    // the published SplitMix64 outputs for seed 0
    detail::SplitMix64 rng(0);
    ASSERT_EQ(rng.next(), 0xE220A8397B1DCDAFull);
    ASSERT_EQ(rng.next(), 0x6E789E6AA1B965F4ull);
    ASSERT_EQ(rng.next(), 0x06C45D188009454Full);

#if defined(__SIZEOF_INT128__)
    // below(n) is the high 64 bits of next() * n
    detail::SplitMix64 a(77), b(77);
    for (int i = 0; i < 2000; ++i) {
        const uint64_t n = (i % 3 == 0) ? 16u : (i % 3 == 1 ? 64u : static_cast<uint64_t>(0xFFFFFFFFFFFF) + static_cast<uint64_t>(i));
        __extension__ typedef unsigned __int128 Wide;
        const Wide wide = static_cast<Wide>(a.next()) * n;
        ASSERT_TRUE(b.below(n) == static_cast<uint64_t>(wide >> 64));
    }
#endif

    // level = 0; while level < 24 and below(m) == 0: level += 1  =>  P(level >= l) = m^-l
    detail::SplitMix64 gen(0 ^ 0x4857535748575357ull);
    const size_t draws = 160000;
    size_t atLeast1 = 0, atLeast2 = 0;
    for (size_t i = 0; i < draws; ++i) {
        size_t level = 0;
        while (level < 24 && gen.below(16) == 0) ++level;
        atLeast1 += level >= 1;
        atLeast2 += level >= 2;
    }
    ASSERT_TRUE(std::fabs(double(atLeast1) / double(draws) - 1.0 / 16.0) < 0.004);
    ASSERT_TRUE(std::fabs(double(atLeast2) / double(draws) - 1.0 / 256.0) < 0.001);
}

TEST(hnsw_four_row_scorer_equals_the_canonical_scorer_bit_for_bit) {
    std::mt19937_64 rng(5);
    std::normal_distribution<float> normal;
    for (size_t dim : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 33u, 100u, 384u}) {
        std::vector<Vec> rows(4, Vec(dim));
        Vec q(dim);
        for (int trial = 0; trial < 40; ++trial) {
            for (auto& r : rows) for (auto& x : r) x = normal(rng) * (trial % 5 == 0 ? 1e-3f : 1.0f);
            for (auto& x : q) x = normal(rng);
            if (trial % 7 == 3) std::fill(rows[static_cast<size_t>(trial % 4)].begin(), rows[static_cast<size_t>(trial % 4)].end(), 0.0f);
            if (trial % 11 == 5) rows[1][0] = -0.0f;
            float quad[4];
            detail::cosineRows4(q.data(), rows[0].data(), rows[1].data(), rows[2].data(), rows[3].data(), dim, quad);
            for (size_t j = 0; j < 4; ++j) {
                ASSERT_EQ(floatBits(quad[j]), floatBits(detail::cosineRows(q.data(), rows[j].data(), dim)));
            }
        }
    }
}

/** `slots` scored through the graph's scorer against what cosineRows says for each, bit for bit. */
static void expectScorerMatchesCanonical(const detail::HnswRows& rows, const Vec& q, const std::vector<uint32_t>& slots) {
    std::vector<float> got(slots.size());
    detail::hnswScore(rows, q.data(), detail::hnswRowNorm(q.data(), q.size()), slots.data(), slots.size(), got.data());
    for (size_t i = 0; i < slots.size(); ++i) {
        float want = detail::cosineRows(q.data(), rows.row(slots[i]), rows.dim);
        if (std::isnan(want)) want = -std::numeric_limits<float>::infinity();   // the scorer ranks NaN last
        ASSERT_EQ(floatBits(got[i]), floatBits(want));
    }
}

TEST(hnsw_graph_scorer_equals_the_canonical_scorer_bit_for_bit) {
    // Every batch size (1 row, a partial group, whole groups, a repeated slot), every row
    // width around the SIMD step, and the rows that make scores awkward: zero, -0.0,
    // tiny, huge, non-finite.
    std::mt19937_64 rng(11);
    std::normal_distribution<float> normal;
    for (size_t dim : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 33u, 100u, 384u}) {
        const size_t count = 40;
        std::vector<Vec> data(count, Vec(dim));
        for (size_t r = 0; r < count; ++r) {
            for (auto& x : data[r]) x = normal(rng) * (r % 5 == 0 ? 1e-3f : (r % 7 == 0 ? 1e3f : 1.0f));
        }
        std::fill(data[3].begin(), data[3].end(), 0.0f);
        data[4][0] = -0.0f;
        data[9][dim - 1] = std::numeric_limits<float>::quiet_NaN();
        data[10][0] = std::numeric_limits<float>::infinity();
        RawTable table(data, dim);
        std::vector<float> norms(count);
        for (size_t r = 0; r < count; ++r) norms[r] = detail::hnswRowNorm(table.matrix.row(r), dim);
        detail::HnswRows rows = table.rows();
        rows.norms = norms.data();
        for (int trial = 0; trial < 6; ++trial) {
            Vec q(dim);
            for (auto& x : q) x = normal(rng);
            if (trial == 4) std::fill(q.begin(), q.end(), 0.0f);
            for (size_t n = 1; n <= 20; ++n) {
                std::vector<uint32_t> slots(n);
                for (auto& s : slots) s = static_cast<uint32_t>(rng() % count);
                expectScorerMatchesCanonical(rows, q, slots);
            }
        }
    }
}

TEST(hnsw_cached_norms_follow_updates_and_compaction) {
    // The graph keeps each row's norm; the scores it navigates by must stay exactly what
    // cosineRows gives for the row as it is NOW (an overwritten or moved row, in
    // particular, must not keep its old norm).
    Clustered gen(8, 24, 77);
    auto data = gen.many(3000);
    RawTable table(data, 24);
    detail::HnswIndex graph = table.build(0);
    auto checkHits = [&](const detail::HnswIndex& g, const detail::HnswRows& rows) {
        for (const auto& q : gen.many(20)) {
            detail::HnswFound found;
            g.search(rows, q.data(), detail::IvfIndex::kAnyCode, -std::numeric_limits<float>::infinity(), 64,
                     std::numeric_limits<size_t>::max(), found);
            ASSERT_FALSE(found.hits.empty());
            for (const auto& hit : found.hits) {
                ASSERT_EQ(floatBits(hit.score), floatBits(detail::cosineRows(q.data(), rows.row(hit.slot), 24)));
            }
        }
    };
    checkHits(graph, table.rows());

    // overwrite 200 rows with different vectors and link them again
    const auto moved = gen.many(200);
    for (uint32_t i = 0; i < 200; ++i) {
        const uint32_t slot = i * 7;
        table.matrix.set(slot, moved[i].data());
        graph.notePut(table.rows(), slot);
    }
    checkHits(graph, table.rows());

    // tombstone every fourth row, squeeze the table as the store does, renumber the graph
    std::vector<uint32_t> remap(3000, detail::kHnswNone);
    size_t kept = 0;
    for (uint32_t i = 0; i < 3000; ++i) {
        if (i % 4 == 1 && i != graph.entry()) table.alive[i] = 0;
        if (table.alive[i]) remap[i] = static_cast<uint32_t>(kept++);
    }
    for (uint32_t i = 0; i < 3000; ++i) {
        if (remap[i] != detail::kHnswNone && remap[i] != i) table.matrix.moveRow(i, remap[i]);
    }
    table.matrix.truncate(kept);
    table.alive.assign(kept, 1);
    table.codes.assign(kept, 0);
    graph.compact(remap, kept);
    ASSERT_EQ(graph.check(table.rows(), kept), std::string());
    checkHits(graph, table.rows());
}

TEST(hnsw_recall_on_clustered_and_random_data) {
    // clustered: the shape of real embeddings
    {
        Clustered gen(40, 32, 11);
        auto data = gen.many(6000);
        EmbeddingStore store;
        store.enableAnn(hnswOptions());
        fill(store, data);
        auto queries = gen.many(100);
        const double recall = recallOf(store, queries, 10);
        ASSERT_GE(recall, 0.97);
        AnnStats s = store.annStats();
        ASSERT_TRUE(s.active);
        ASSERT_TRUE(s.mode == AnnMode::Hnsw);
        ASSERT_EQ(s.graphNodes, 6000u);
        ASSERT_EQ(s.graphDead, 0u);
        ASSERT_EQ(s.builds, 1u);
        ASSERT_EQ(s.nlist, 0u);
        ASSERT_EQ(s.efSearch, 64u);                      // auto, under 10,000 rows
        ASSERT_GT(s.indexBytes, 0u);
        ASSERT_GE(s.approximateSearches, 100u);          // the graph answered, not the fallback
        // 146 bytes per node at m = 16, plus chunk padding
        ASSERT_TRUE(s.indexBytes < 6000u * 200u);
        store.verifyInvariants();

        // the same through recallCheck()
        RecallReport report = store.recallCheck(queries, 10);
        ASSERT_TRUE(report.approximate);
        ASSERT_GE(report.recall, 0.97);
    }
    // plain gaussian noise, no clusters (16-d: in 24-d the walk scores so many rows that the
    // visit budget sends most queries to the scan, which is the design but would test nothing here)
    {
        auto data = randomVectors(12000, 16, 12);
        EmbeddingStore store;
        store.enableAnn(hnswOptions());
        fill(store, data);
        auto queries = randomVectors(100, 16, 13);
        ASSERT_GE(recallOf(store, queries, 10), 0.93);
        ASSERT_GE(store.annStats().approximateSearches, 90u);
        store.verifyInvariants();
        // a larger ef is better, and ef >= live is the exact scan
        store.setAnnEfSearch(400);
        ASSERT_GE(recallOf(store, queries, 10), 0.99);
        ASSERT_EQ(store.annStats().efSearch, 400u);
    }
}

TEST(hnsw_is_exact_below_min_size_and_when_top_k_covers_every_row) {
    Clustered gen(20, 16, 21);
    auto data = gen.many(2500);
    EmbeddingStore plain;
    fill(plain, data);

    EmbeddingStore store;
    store.enableAnn(hnswOptions(/*minSize=*/5000));
    fill(store, data);
    auto queries = gen.many(10);
    for (const auto& q : queries) {
        ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 10, "", -1.0f), plain.similaritySearchVector(q, 10, "", -1.0f)));
    }
    AnnStats s = store.annStats();
    ASSERT_FALSE(s.active);
    ASSERT_EQ(s.builds, 0u);
    ASSERT_EQ(s.approximateSearches, 0u);

    // an eligible store, but top_k asks for every row: the scan answers
    store.enableAnn(hnswOptions(0));
    for (const auto& q : queries) {
        ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 2500, "", -1.0f), plain.similaritySearchVector(q, 2500, "", -1.0f)));
        ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 100000, "", -1.0f), plain.similaritySearchVector(q, 100000, "", -1.0f)));
    }
    ASSERT_EQ(store.annStats().approximateSearches, 0u);
    // exact = true skips the graph even when it would answer
    for (const auto& q : queries) {
        ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 10, "", -1.0f, true), plain.similaritySearchVector(q, 10, "", -1.0f)));
    }
    ASSERT_EQ(store.annStats().approximateSearches, 0u);
    // a zero query has no direction to walk by
    Vec zero(16, 0.0f);
    ASSERT_TRUE(sameResults(store.similaritySearchVector(zero, 10, "", -1.0f), plain.similaritySearchVector(zero, 10, "", -1.0f)));
    // ef >= live is the scan too
    ASSERT_TRUE(sameResults(store.similaritySearchVectorEf(queries[0], 10, "", -1.0f, 2500),
                            plain.similaritySearchVector(queries[0], 10, "", -1.0f)));
    store.verifyInvariants();
}

TEST(hnsw_scores_are_bit_identical_to_the_exact_scan) {
    Clustered gen(30, 48, 31);
    auto data = gen.many(5000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    store.buildAnnIndex();
    size_t checked = 0;
    for (const auto& q : gen.many(60)) {
        auto exactAll = store.similaritySearchVector(q, 5000, "", -2.0f, true);
        std::map<std::string, float> truth;
        for (const auto& r : exactAll) truth[keyOf(r)] = r.score;
        auto got = store.similaritySearchVector(q, 10, "", -2.0f);
        ASSERT_EQ(got.size(), 10u);
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(floatBits(got[i].score), floatBits(truth.at(keyOf(got[i]))));
            ASSERT_EQ(floatBits(got[i].score),
                      floatBits(detail::cosineRows(q.data(), store.vectorData(got[i].node_ref), q.size())));
            if (i > 0) ASSERT_TRUE(got[i - 1].score >= got[i].score);   // best first
            ++checked;
        }
    }
    ASSERT_EQ(checked, 600u);
    ASSERT_GE(store.annStats().approximateSearches, 60u);
}

TEST(hnsw_type_filter_threshold_and_selective_filters) {
    Clustered gen(30, 32, 41);
    auto data = gen.many(6000);
    // three types in a 1 : 1 : 1 mix, plus "rare" for 1 row in 300
    auto typeOf = [](size_t i) -> std::string {
        if (i % 300 == 7) return "rare";
        return i % 2 == 0 ? "alpha" : "beta";
    };
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data, typeOf);
    store.verifyInvariants();
    auto queries = gen.many(60);

    // a common type: the graph answers, every hit has the type, recall is high
    for (const std::string type : {"alpha", "beta"}) {
        ASSERT_GE(recallOf(store, queries, 10, type), 0.95);
        for (const auto& r : store.similaritySearchVector(queries[0], 10, type, -1.0f)) ASSERT_EQ(r.node_ref.type, type);
    }
    // a type that does not exist: nothing
    ASSERT_EQ(store.similaritySearchVector(queries[0], 10, "nope", -1.0f).size(), 0u);

    // threshold: nothing below it is returned, and the hits equal the exact scan's
    const float threshold = 0.6f;
    size_t shortLists = 0;
    for (const auto& q : queries) {
        auto got = store.similaritySearchVector(q, 10, "", threshold);
        auto truth = store.similaritySearchVector(q, 10, "", threshold, true);
        for (const auto& r : got) ASSERT_GE(r.score, threshold);
        ASSERT_EQ(got.size(), truth.size());       // a threshold that few rows pass widens, then scans
        if (truth.size() < 10) ++shortLists;
    }
    ASSERT_GE(recallOf(store, queries, 10, "", threshold), 0.95);
    // a threshold nobody passes is an empty answer, as the scan says
    ASSERT_EQ(store.similaritySearchVector(queries[0], 10, "", 1.5f).size(), 0u);
    (void)shortLists;

    // a selective filter (20 rows of 6000): ef widens, the visit budget runs out
    // and the scan answers, so the answer is exactly the scan's
    size_t exactBefore = store.annStats().exactSearches;
    for (const auto& q : queries) {
        ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 10, "rare", -1.0f),
                                store.similaritySearchVector(q, 10, "rare", -1.0f, true)));
    }
    ASSERT_GT(store.annStats().exactSearches, exactBefore + queries.size());   // the fallbacks plus the exact = true calls
    ASSERT_EQ(store.similaritySearchVector(queries[0], 10, "rare", -1.0f).size(), 10u);
    // asking for more than the type holds: every such row, as the scan returns them
    ASSERT_TRUE(sameResults(store.similaritySearchVector(queries[1], 25, "rare", -1.0f),
                            store.similaritySearchVector(queries[1], 25, "rare", -1.0f, true)));
}

TEST(hnsw_graph_node_filter_widens_and_ends_in_the_scan) {
    Clustered gen(30, 32, 51);
    auto data = gen.many(5000);
    AnnOptions o = hnswOptions();
    SemanticGraph graph("g", nullptr, false, {"name"}, ModelMismatchMode::Warn, std::nullopt, std::optional<AnnOptions>(o));
    for (size_t i = 0; i < data.size(); ++i) {
        graph.addNode("item", std::to_string(i), {{"group", static_cast<int>(i % 50)}});
        graph.embeddingStore().add(NodeRef("item", std::to_string(i)), "", data[i]);
    }
    auto q = gen.draw();

    auto plain = graph.similaritySearchVector(q, 5);
    ASSERT_EQ(plain.size(), 5u);
    ASSERT_GE(graph.embeddingStore().annStats().approximateSearches, 1u);
    ASSERT_TRUE(graph.embeddingStore().annStats().mode == AnnMode::Hnsw);

    // one node in 50 passes: the first ef holds ~2, so the search widens (and may scan), and still hands back k
    NodeFilterOptions filter;
    filter.nodeFilter = [](const Node& n) { return n.properties.at("group").as_int() == 7; };
    auto filtered = graph.similaritySearchVector(q, 10, "", -1.0f, filter);
    auto truth = graph.similaritySearchVector(q, 10, "", -1.0f, filter, /*exact=*/true);
    ASSERT_EQ(filtered.size(), 10u);
    ASSERT_EQ(truth.size(), 10u);
    for (const auto& r : filtered) {
        ASSERT_EQ(graph.graph().getNode(r.node_ref.type, r.node_ref.id).properties.at("group").as_int(), 7);
    }
    size_t same = 0;
    std::set<std::string> want;
    for (const auto& r : truth) want.insert(keyOf(r));
    for (const auto& r : filtered) same += want.count(keyOf(r));
    ASSERT_GE(same, 8u);
    ASSERT_TRUE(graph.embeddingStore().annStats().widenedSearches >= 1u);

    // a gate that passes nothing: an empty answer, as the scan gives
    NodeFilterOptions none;
    none.nodeFilter = [](const Node&) { return false; };
    ASSERT_EQ(graph.similaritySearchVector(q, 10, "", -1.0f, none).size(), 0u);
    graph.embeddingStore().verifyInvariants();
}

TEST(hnsw_writes_keep_the_graph_current_and_every_invariant) {
    Clustered gen(25, 24, 61);
    auto data = gen.many(4000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    store.buildAnnIndex();
    ASSERT_EQ(store.annStats().builds, 1u);
    ASSERT_EQ(store.annStats().inserts, 0u);

    // inserts after the build link at once (no pending list, no rebuild)
    auto fresh = gen.many(300);
    for (size_t i = 0; i < fresh.size(); ++i) store.add(NodeRef("new", std::to_string(i)), "", fresh[i]);
    AnnStats s = store.annStats();
    ASSERT_EQ(s.builds, 1u);
    ASSERT_EQ(s.inserts, 300u);
    ASSERT_EQ(s.graphNodes, 4300u);
    ASSERT_EQ(s.pending, 0u);
    store.verifyInvariants();
    // every fresh row finds itself first, through the graph
    size_t selfFirst = 0;
    for (size_t i = 0; i < fresh.size(); ++i) {
        auto got = store.similaritySearchVector(fresh[i], 3, "", -1.0f);
        if (!got.empty() && got[0].node_ref == NodeRef("new", std::to_string(i))) ++selfFirst;
    }
    ASSERT_GE(selfFirst, 295u);
    ASSERT_EQ(store.annStats().builds, 1u);

    // updates: the node keeps its slot and level, its links are recomputed
    auto moved = gen.many(200);
    for (size_t i = 0; i < moved.size(); ++i) store.add(NodeRef("doc", std::to_string(i * 3)), "", moved[i]);
    s = store.annStats();
    ASSERT_EQ(s.graphNodes, 4300u);
    ASSERT_EQ(s.builds, 1u);
    ASSERT_EQ(s.churn, 200u);
    store.verifyInvariants();
    size_t found = 0;
    for (size_t i = 0; i < moved.size(); ++i) {
        auto got = store.similaritySearchVector(moved[i], 3, "", -1.0f);
        if (!got.empty() && got[0].node_ref == NodeRef("doc", std::to_string(i * 3))) ++found;
    }
    ASSERT_GE(found, 195u);
    ASSERT_GE(recallOf(store, gen.many(60), 10), 0.96);

    // deletes tombstone only: no graph work, nodes stay (counted as dead)
    for (size_t i = 1; i < 4000; i += 7) store.remove(NodeRef("doc", std::to_string(i)));   // ~570 of 4300: under a quarter
    s = store.annStats();
    ASSERT_GT(s.graphDead, 0u);
    ASSERT_EQ(s.graphNodes, 4300u);
    store.verifyInvariants();
    for (const auto& q : gen.many(30)) {
        for (const auto& r : store.similaritySearchVector(q, 10, "", -1.0f)) {
            ASSERT_TRUE(store.contains(r.node_ref));        // a tombstone is never returned
        }
    }
    ASSERT_GE(recallOf(store, gen.many(60), 10), 0.95);
    ASSERT_EQ(store.annStats().builds, 1u);
}

TEST(hnsw_compaction_renumbers_bridges_and_keeps_recall) {
    Clustered gen(25, 24, 71);
    auto data = gen.many(4000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    store.buildAnnIndex();

    // 40% deleted: the table compacts itself once a quarter is dead (and again, as it grows)
    std::set<size_t> gone;
    for (size_t i = 0; i < 4000; ++i) {
        if (i % 5 < 2) { store.remove(NodeRef("doc", std::to_string(i))); gone.insert(i); }
        if (i % 500 == 0) store.verifyInvariants();
    }
    store.verifyInvariants();
    AnnStats s = store.annStats();
    ASSERT_EQ(store.count(), 2400u);
    ASSERT_EQ(s.builds, 1u);                           // compaction follows the table, no rebuild
    ASSERT_TRUE(s.graphNodes < 4000u);                 // tombstones left the graph at compaction

    for (const auto& q : gen.many(40)) {
        for (const auto& r : store.similaritySearchVector(q, 10, "", -1.0f)) {
            ASSERT_TRUE(gone.count(static_cast<size_t>(std::stoul(r.node_ref.id))) == 0);   // no dead id comes back
        }
    }
    ASSERT_GE(recallOf(store, gen.many(80), 10), 0.95);
    ASSERT_EQ(store.annStats().builds, 1u);
    // and the compacted graph keeps taking writes
    for (size_t i = 0; i < 200; ++i) store.add(NodeRef("late", std::to_string(i)), "", gen.draw());
    store.verifyInvariants();
    ASSERT_EQ(store.annStats().builds, 1u);
}

TEST(hnsw_graph_level_compaction_bridges_over_tombstones) {
    Clustered gen(8, 12, 81);
    auto data = gen.many(4000);
    RawTable table(data, 12);
    detail::HnswIndex graph = table.build(0);
    ASSERT_EQ(graph.check(table.rows(), 4000), std::string());

    // BFS over layer 0 from the entry point: how many live nodes can it reach
    auto reachable = [](const detail::HnswIndex& g, const RawTable& t) {
        std::vector<char> seen(t.alive.size(), 0);
        std::vector<uint32_t> stack{g.entry()};
        seen[g.entry()] = 1;
        while (!stack.empty()) {
            uint32_t s = stack.back();
            stack.pop_back();
            for (uint32_t nb : g.layer0(s)) {
                if (!seen[nb]) { seen[nb] = 1; stack.push_back(nb); }
            }
        }
        size_t live = 0;
        for (size_t s = 0; s < seen.size(); ++s) live += (seen[s] && t.alive[s]) ? 1u : 0u;
        return live;
    };
    ASSERT_EQ(reachable(graph, table), 4000u);

    // kill every third row except the entry point row, then squeeze the table as the store does
    std::vector<uint32_t> remap(4000, detail::kHnswNone);
    size_t kept = 0;
    for (uint32_t i = 0; i < 4000; ++i) {
        if (i % 3 == 0 && i != graph.entry()) table.alive[i] = 0;
        if (table.alive[i]) remap[i] = static_cast<uint32_t>(kept++);
    }
    for (uint32_t i = 0; i < 4000; ++i) {
        if (remap[i] != detail::kHnswNone && remap[i] != i) table.matrix.moveRow(i, remap[i]);
    }
    table.matrix.truncate(kept);
    table.alive.assign(kept, 1);
    table.codes.assign(kept, 0);
    for (int i = 0; i < 3; ++i) graph.noteDrop();
    graph.compact(remap, kept);
    ASSERT_EQ(graph.check(table.rows(), kept), std::string());
    ASSERT_EQ(graph.members(), kept);
    ASSERT_EQ(graph.dead(table.rows()), 0u);
    ASSERT_GE(reachable(graph, table) * 100, kept * 99);

    // lists were refilled from the links of the tombstones: the mean degree stays near a fresh build
    auto degree = [&](const detail::HnswIndex& g) {
        size_t total = 0;
        for (uint32_t s = 0; s < kept; ++s) total += g.layer0(s).size();
        return double(total) / double(kept);
    };
    detail::HnswIndex rebuilt = table.build(0);
    ASSERT_TRUE(degree(graph) >= 0.85 * degree(rebuilt));
}

TEST(hnsw_rebuilds_once_updates_and_deletes_exceed_the_node_count) {
    Clustered gen(20, 16, 91);
    auto data = gen.many(2000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    auto q = gen.draw();
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_EQ(store.annStats().builds, 1u);

    // 1,500 updates (< 2,000 nodes): no rebuild
    for (size_t i = 0; i < 1500; ++i) store.add(NodeRef("doc", std::to_string(i)), "", gen.draw());
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_EQ(store.annStats().builds, 1u);
    ASSERT_EQ(store.annStats().churn, 1500u);
    // inserts do not count
    for (size_t i = 0; i < 1000; ++i) store.add(NodeRef("extra", std::to_string(i)), "", gen.draw());
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_EQ(store.annStats().builds, 1u);
    // more updates take the count past the 3,000 nodes: the next search rebuilds
    for (size_t i = 1500; i < 2000; ++i) store.add(NodeRef("doc", std::to_string(i)), "", gen.draw());
    for (size_t i = 0; i < 1100; ++i) store.add(NodeRef("extra", std::to_string(i % 1000)), "", gen.draw());
    ASSERT_GT(store.annStats().churn, 3000u);
    store.similaritySearchVector(q, 10, "", -1.0f);
    AnnStats s = store.annStats();
    ASSERT_EQ(s.builds, 2u);
    ASSERT_EQ(s.churn, 0u);
    ASSERT_EQ(s.graphNodes, 3000u);
    store.verifyInvariants();
    ASSERT_GE(recallOf(store, gen.many(50), 10), 0.97);
}

TEST(hnsw_is_dropped_below_min_size_and_rebuilt_when_the_store_grows_back) {
    Clustered gen(10, 16, 95);
    auto data = gen.many(1200);
    EmbeddingStore store;
    store.enableAnn(hnswOptions(/*minSize=*/1000));
    fill(store, data);
    auto q = gen.draw();
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_TRUE(store.annStats().active);
    for (size_t i = 0; i < 300; ++i) store.remove(NodeRef("doc", std::to_string(i)));   // 900 left
    ASSERT_FALSE(store.annStats().active);                                              // dropped by the delete
    // under minSize the answer is the scan
    EmbeddingStore plain;
    for (size_t i = 300; i < 1200; ++i) plain.add(NodeRef("doc", std::to_string(i)), "", data[i]);
    ASSERT_TRUE(sameResults(store.similaritySearchVector(q, 10, "", -1.0f), plain.similaritySearchVector(q, 10, "", -1.0f)));
    for (size_t i = 0; i < 300; ++i) store.add(NodeRef("back", std::to_string(i)), "", gen.draw());   // 1200 again
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_TRUE(store.annStats().active);
    ASSERT_EQ(store.annStats().builds, 2u);
    store.verifyInvariants();
}

TEST(hnsw_graph_does_not_depend_on_the_thread_count) {
    Clustered gen(12, 16, 101);
    auto data = gen.many(3000);
    RawTable table(data, 16);
    detail::HnswIndex one = table.build(1);
    detail::HnswIndex three = table.build(3);
    detail::HnswIndex many = table.build(0);
    detail::HnswIndex again = table.build(0);
    ASSERT_EQ(one.check(table.rows(), 3000), std::string());
    ASSERT_TRUE(one.sameGraphAs(three));
    ASSERT_TRUE(one.sameGraphAs(many));
    ASSERT_TRUE(many.sameGraphAs(again));
    ASSERT_TRUE(one.dumpLinks() == many.dumpLinks());
    ASSERT_FALSE(one.sameGraphAs(table.build(1, /*seed=*/1)));    // the seed matters
    ASSERT_FALSE(one.sameGraphAs(table.build(1, 0, /*m=*/8)));    // and so does m

    // through the store: the same graph means the same answers
    auto build = [&](size_t threads) {
        AnnOptions o = hnswOptions();
        o.threads = threads;
        auto store = std::make_unique<EmbeddingStore>();
        store->enableAnn(o);
        fill(*store, data);
        store->buildAnnIndex();
        return store;
    };
    auto a = build(1), b = build(8), c = build(0);
    for (const auto& q : gen.many(40)) {
        auto ra = a->similaritySearchVector(q, 10, "", -1.0f);
        ASSERT_TRUE(sameResults(ra, b->similaritySearchVector(q, 10, "", -1.0f)));
        ASSERT_TRUE(sameResults(ra, c->similaritySearchVector(q, 10, "", -1.0f)));
    }
}

TEST(hnsw_batch_equals_single_query_bit_for_bit) {
    Clustered gen(30, 32, 111);
    auto data = gen.many(5000);
    auto typeOf = [](size_t i) -> std::string { return i % 3 == 0 ? "a" : "b"; };
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data, typeOf);
    auto queries = gen.many(97);
    for (const std::string type : {"", "a"}) {
        for (float threshold : {-1.0f, 0.5f}) {
            std::vector<std::vector<SimilarityResult>> single;
            for (const auto& q : queries) single.push_back(store.similaritySearchVector(q, 10, type, threshold));
            for (size_t threads : {size_t(1), size_t(3), size_t(0)}) {
                auto batch = store.similaritySearchBatch(queries, 10, type, threshold, false, threads);
                ASSERT_EQ(batch.size(), queries.size());
                for (size_t i = 0; i < queries.size(); ++i) ASSERT_TRUE(sameResults(batch[i], single[i]));
            }
        }
    }
    ASSERT_GE(store.annStats().approximateSearches, 97u);
    // exact = true is the exact batch, not the graph
    auto exactBatch = store.similaritySearchBatch(queries, 10, "", -1.0f, true, 2);
    for (size_t i = 0; i < queries.size(); ++i) {
        ASSERT_TRUE(sameResults(exactBatch[i], store.similaritySearchVector(queries[i], 10, "", -1.0f, true)));
    }
}

TEST(hnsw_concurrent_const_searches_agree_with_serial_ones) {
    Clustered gen(30, 32, 121);
    auto data = gen.many(5000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);          // the graph is not built yet: the threads below race to build it lazily
    auto queries = gen.many(64);

    std::vector<std::vector<SimilarityResult>> got(queries.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= queries.size()) break;
                got[i] = store.similaritySearchVector(queries[i], 10, "", -1.0f);
            }
        });
    }
    for (auto& t : threads) t.join();
    ASSERT_EQ(store.annStats().builds, 1u);
    for (size_t i = 0; i < queries.size(); ++i) {
        ASSERT_TRUE(sameResults(got[i], store.similaritySearchVector(queries[i], 10, "", -1.0f)));
    }

    // and many threads at once, searches and small batches mixed
    std::atomic<bool> bad{false};
    std::vector<std::thread> mixed;
    for (int t = 0; t < 6; ++t) {
        mixed.emplace_back([&, t]() {
            for (size_t i = 0; i < queries.size(); ++i) {
                if (t % 2 == 0) {
                    if (!sameResults(store.similaritySearchVector(queries[i], 10, "", -1.0f), got[i])) bad = true;
                } else {
                    auto batch = store.similaritySearchBatch({queries[i], queries[(i + 1) % queries.size()]}, 10, "", -1.0f, false, 2);
                    if (!sameResults(batch[0], got[i])) bad = true;
                }
            }
        });
    }
    for (auto& t : mixed) t.join();
    ASSERT_FALSE(bad.load());
}

TEST(hnsw_is_never_persisted_files_and_digest_are_byte_identical) {
    Clustered gen(20, 16, 131);
    auto data = gen.many(1500);
    auto build = [&](bool withHnsw) {
        SemanticGraph graph("g", nullptr, false, {"name"}, ModelMismatchMode::Warn, std::nullopt,
                            withHnsw ? std::optional<AnnOptions>(hnswOptions()) : std::nullopt);
        for (size_t i = 0; i < data.size(); ++i) {
            graph.addNode("item", std::to_string(i), {{"name", "n" + std::to_string(i)}});
            graph.embeddingStore().add(NodeRef("item", std::to_string(i)), "n" + std::to_string(i), data[i]);
        }
        return graph;
    };
    SemanticGraph plain = build(false);
    SemanticGraph indexed = build(true);
    indexed.embeddingStore().buildAnnIndex();
    ASSERT_TRUE(indexed.embeddingStore().annStats().active);
    indexed.similaritySearchVector(data[3], 5);
    indexed.embeddingStore().add(NodeRef("item", "7"), "n7", data[100]);            // an update and a delete after the build
    plain.embeddingStore().add(NodeRef("item", "7"), "n7", data[100]);
    indexed.embeddingStore().remove(NodeRef("item", "9"));
    plain.embeddingStore().remove(NodeRef("item", "9"));

    ASSERT_EQ(graphDigest(plain.graph()), graphDigest(indexed.graph()));
    ASSERT_EQ(plain.embeddingStore().toJson(), indexed.embeddingStore().toJson());

    Dir dir("persist");
    const std::string a = dir.file("plain.ison"), b = dir.file("indexed.ison");
    plain.save(a);
    indexed.save(b);
    ASSERT_EQ(slurp(a), slurp(b));
    const std::string sideA = slurp(SemanticGraph::embeddingsPathFor(a));
    const std::string sideB = slurp(SemanticGraph::embeddingsPathFor(b));
    ASSERT_EQ(sideA, sideB);
    ASSERT_TRUE(sideB.find("hnsw") == std::string::npos && sideB.find("level") == std::string::npos);

    // the binary snapshot as well
    plain.embeddingStore().saveSnapshot(dir.file("plain.snap"));
    indexed.embeddingStore().saveSnapshot(dir.file("indexed.snap"));
    const std::string snapA = slurp(dir.file("plain.snap")), snapB = slurp(dir.file("indexed.snap"));
    ASSERT_GT(snapA.size(), 0u);
    ASSERT_EQ(snapA, snapB);

    // a loaded store has no index (never persisted); enabling it builds one on the first search
    SemanticGraph loaded = SemanticGraph::load(b, nullptr, "auto", true, false, {"name"});
    ASSERT_EQ(loaded.embeddingStore().count(), 1499u);
    ASSERT_FALSE(loaded.embeddingStore().annStats().enabled);
    loaded.embeddingStore().enableAnn(hnswOptions());
    ASSERT_FALSE(loaded.embeddingStore().annStats().active);
    auto hits = loaded.similaritySearchVector(data[50], 8, "", -1.0f);
    ASSERT_EQ(hits.size(), 8u);
    ASSERT_TRUE(loaded.embeddingStore().annStats().active);
    ASSERT_EQ(loaded.embeddingStore().annStats().builds, 1u);
    ASSERT_TRUE(sameResults(hits, loaded.similaritySearchVector(data[50], 8, "", -1.0f)));
    ASSERT_GE(recallOf(loaded.embeddingStore(), gen.many(40), 10), 0.95);

    // a snapshot load drops a graph built on the old rows
    EmbeddingStore snapLoaded;
    snapLoaded.enableAnn(hnswOptions());
    snapLoaded.loadSnapshot(dir.file("indexed.snap"));
    snapLoaded.similaritySearchVector(data[1], 5, "", -1.0f);
    ASSERT_TRUE(snapLoaded.annStats().active);
    snapLoaded.loadSnapshot(dir.file("indexed.snap"));
    ASSERT_FALSE(snapLoaded.annStats().active);
    snapLoaded.similaritySearchVector(data[1], 5, "", -1.0f);
    ASSERT_TRUE(snapLoaded.annStats().active);
    snapLoaded.verifyInvariants();
}

TEST(hnsw_ef_per_call_and_configured) {
    auto data = randomVectors(12000, 16, 141);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    auto queries = randomVectors(60, 16, 142);
    EmbeddingStore plain;
    fill(plain, data);

    auto recallWith = [&](size_t ef) {
        double total = 0;
        for (const auto& q : queries) {
            auto truth = plain.similaritySearchVector(q, 10, "", -2.0f);
            std::set<std::string> want;
            for (const auto& r : truth) want.insert(keyOf(r));
            size_t hit = 0;
            for (const auto& r : store.similaritySearchVectorEf(q, 10, "", -2.0f, ef)) hit += want.count(keyOf(r));
            total += double(hit) / 10.0;
        }
        return total / double(queries.size());
    };
    const double low = recallWith(16), mid = recallWith(64), high = recallWith(512);
    ASSERT_TRUE(low <= mid + 0.02 && mid <= high + 0.01);
    ASSERT_GE(high, 0.99);
    // an ef below k is raised to k
    ASSERT_EQ(store.similaritySearchVectorEf(queries[0], 10, "", -2.0f, 1).size(), 10u);
    // the per-call value does not stick; the configured one does, without a rebuild
    ASSERT_EQ(store.annStats().efSearch, 96u);        // automatic: 96 from 10,000 to 200,000 rows
    const size_t builds = store.annStats().builds;
    store.setAnnEfSearch(300);
    ASSERT_EQ(store.annStats().efSearch, 300u);
    ASSERT_GE(recallOf(store, queries, 10), 0.98);
    store.setAnnEfSearch(0);
    ASSERT_EQ(store.annStats().efSearch, 96u);
    ASSERT_EQ(store.annStats().builds, builds);
    // the per-call ef is ignored by IVF
    EmbeddingStore ivf;
    AnnOptions io;
    io.minSize = 0;
    io.nlist = 32;
    io.nprobe = 32;
    ivf.enableAnn(io);
    fill(ivf, std::vector<Vec>(data.begin(), data.begin() + 2000));
    ASSERT_EQ(ivf.similaritySearchVectorEf(queries[0], 10, "", -2.0f, 5).size(), 10u);
}

TEST(hnsw_mode_can_be_switched_and_cleared) {
    Clustered gen(15, 16, 151);
    auto data = gen.many(2500);
    EmbeddingStore store;
    fill(store, data);
    AnnOptions ivf;
    ivf.minSize = 0;
    ivf.nlist = 32;
    ivf.nprobe = 8;
    store.enableAnn(ivf);
    auto q = gen.draw();
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_TRUE(store.annStats().mode == AnnMode::Ivf);
    ASSERT_EQ(store.annStats().nlist, 32u);

    store.enableAnn(hnswOptions());                      // replaces the options and drops the IVF index
    ASSERT_FALSE(store.annStats().active);
    store.buildAnnIndex();
    AnnStats s = store.annStats();
    ASSERT_TRUE(s.active && s.mode == AnnMode::Hnsw);
    ASSERT_EQ(s.nlist, 0u);
    ASSERT_EQ(s.graphNodes, 2500u);
    store.verifyInvariants();

    store.buildAnnIndex(/*force=*/true);                 // a forced rebuild
    ASSERT_EQ(store.annStats().builds, s.builds + 1u);

    store.clear();
    ASSERT_FALSE(store.annStats().active);
    ASSERT_EQ(store.count(), 0u);
    fill(store, data);                                    // still configured, builds again on demand
    store.similaritySearchVector(q, 10, "", -1.0f);
    ASSERT_TRUE(store.annStats().active);
    store.verifyInvariants();

    store.disableAnn();
    ASSERT_FALSE(store.annStats().enabled);
    ASSERT_EQ(store.similaritySearchVector(q, 10, "", -1.0f).size(), 10u);

    // copying drops the graph; the copy rebuilds from its own rows
    store.enableAnn(hnswOptions());
    store.similaritySearchVector(q, 10, "", -1.0f);
    EmbeddingStore copy = store;
    ASSERT_FALSE(copy.annStats().active);
    ASSERT_TRUE(copy.annStats().enabled && copy.annStats().mode == AnnMode::Hnsw);
    ASSERT_TRUE(sameResults(copy.similaritySearchVector(q, 10, "", -1.0f), store.similaritySearchVector(q, 10, "", -1.0f)));
    EmbeddingStore moved = std::move(store);
    ASSERT_TRUE(moved.annStats().active);
    moved.verifyInvariants();
}

TEST(hnsw_handles_zero_rows_duplicates_and_non_finite_rows) {
    Clustered gen(10, 16, 161);
    auto data = gen.many(2000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    // a zero row scores 0 against everything and must be linkable; a NaN / inf row never is
    store.add(NodeRef("odd", "zero"), "", Vec(16, 0.0f));
    Vec nan(16, 1.0f);
    nan[3] = std::numeric_limits<float>::quiet_NaN();
    store.add(NodeRef("odd", "nan"), "", nan);
    Vec inf(16, 1.0f);
    inf[5] = std::numeric_limits<float>::infinity();
    store.add(NodeRef("odd", "inf"), "", inf);
    for (int i = 0; i < 5; ++i) store.add(NodeRef("dup", std::to_string(i)), "", data[10]);   // identical vectors
    store.buildAnnIndex();
    store.verifyInvariants();
    ASSERT_EQ(store.annStats().graphNodes, 2000u + 1u + 5u);                                    // NaN and inf rows are not in the graph
    auto got = store.similaritySearchVector(data[10], 7, "", -1.0f);
    ASSERT_EQ(got.size(), 7u);
    // ties order by slot: the original row, then the five copies
    ASSERT_EQ(keyOf(got[0]), "doc:10");
    for (size_t i = 1; i <= 5; ++i) ASSERT_EQ(got[i].node_ref.type, "dup");
    for (const auto& r : got) ASSERT_TRUE(r.node_ref.id != "nan" && r.node_ref.id != "inf");
    // writes keep working with such rows around
    store.add(NodeRef("odd", "nan"), "", Vec(16, 0.5f));                                        // the NaN row becomes finite: it joins the graph
    store.verifyInvariants();
    ASSERT_EQ(store.annStats().graphNodes, 2000u + 1u + 5u + 1u);
    store.add(NodeRef("odd", "zero"), "", data[11]);
    store.remove(NodeRef("odd", "inf"));
    store.verifyInvariants();
}

TEST(hnsw_tombstones_are_routed_through_but_never_returned) {
    Clustered gen(20, 24, 171);
    auto data = gen.many(5000);
    EmbeddingStore store;
    store.enableAnn(hnswOptions());
    fill(store, data);
    store.buildAnnIndex();
    // delete the nearest neighbours of a query, so the walk reaches tombstones first
    auto q = gen.draw();
    auto top = store.similaritySearchVector(q, 300, "", -2.0f, true);
    std::set<std::string> dead;
    for (size_t i = 0; i < 200; ++i) {
        store.remove(top[i].node_ref);
        dead.insert(keyOf(top[i]));
    }
    ASSERT_EQ(store.annStats().graphDead, 200u);
    auto got = store.similaritySearchVector(q, 10, "", -2.0f);
    ASSERT_EQ(got.size(), 10u);
    for (const auto& r : got) ASSERT_EQ(dead.count(keyOf(r)), 0u);
    auto truth = store.similaritySearchVector(q, 10, "", -2.0f, true);
    size_t hit = 0;
    std::set<std::string> want;
    for (const auto& r : truth) want.insert(keyOf(r));
    for (const auto& r : got) hit += want.count(keyOf(r));
    ASSERT_GE(hit, 9u);
    store.verifyInvariants();
}

TEST(hnsw_works_through_the_text_search_api_with_an_encoder) {
    auto encoder = std::make_shared<MockEncoder>(32);
    EmbeddingStore store(encoder);
    store.enableAnn(hnswOptions());
    for (size_t i = 0; i < 3000; ++i) store.add(NodeRef("doc", std::to_string(i)), "document number " + std::to_string(i));
    auto got = store.similaritySearch("document number 1234", 5);
    ASSERT_EQ(got.size(), 5u);
    ASSERT_EQ(got[0].node_ref.id, "1234");
    ASSERT_TRUE(store.annStats().active);
    auto many = store.similaritySearchTextBatch({"document number 5", "document number 99"}, 3);
    ASSERT_EQ(many[0][0].node_ref.id, "5");
    ASSERT_EQ(many[1][0].node_ref.id, "99");
    RecallReport report = store.recallCheckTexts({"document number 7", "something else"}, 10);
    ASSERT_TRUE(report.approximate);
    ASSERT_TRUE(report.recall >= 0.8);
}

int main() {
    std::cout << "\n=== ISONGraph Vector C++ HNSW Tests ===" << std::endl << std::endl;
    // Tests are automatically run by static initialization
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Tests passed: " << tests_passed << "/" << tests_run << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
