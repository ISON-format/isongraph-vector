// HNSW against IVF and the exact scan: build time, recall@10, single-query and
// batch throughput, write latency and memory. The C++ counterpart of
// isongraph-vector-rs/examples/hnsw_bench.rs (same measurements, synthetic data).
//
//   hnsw_bench <phase> [n=100000] [dim=384] [queries=300] [data=random|clustered] [noise=1.0]
//
// phases (run each in its own process so the resident-set numbers are clean):
//   mem     resident memory after importing n unit vectors, and after the HNSW
//           graph is built, against the raw vector bytes
//   search  HNSW build, recall and QPS at the automatic ef and a sweep of ef,
//           batch QPS, insert / update / delete latency; then IVF (default
//           nprobe and a sweep) and the exact scan on the same data and queries
//
// data=random is uniform noise on the unit sphere (the hardest case: in 384-d
// every pair is nearly orthogonal, so no approximate index has structure to
// exploit); data=clustered is n/100 centres with Gaussian spread, the family
// the Rust benchmark uses (closer to real embeddings). Prints one "key=value" line per measurement so a
// script can collect the medians of several runs.
//
//   g++ -O2 -I include -I vendor/ison-graph-cpp bench/hnsw_bench.cpp -pthread

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#include "isongraph_vector.hpp"

using namespace isongraph_vector;
using Clock = std::chrono::steady_clock;

namespace {

size_t residentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) return counters.WorkingSetSize;
    return 0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return static_cast<size_t>(std::stoull(line.substr(6))) * 1024;
    }
    return 0;
#endif
}

using Vec = std::vector<float>;

Vec unitNoise(detail::SplitMix64& rng, size_t dim) {
    Vec v(dim);
    double sq = 0;
    for (auto& x : v) {
        x = static_cast<float>(rng.next() >> 40) / 8388608.0f - 1.0f;
        sq += double(x) * double(x);
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(sq));
    for (auto& x : v) x *= inv;
    return v;
}

struct Source {
    bool clustered;
    size_t dim;
    float noise;
    std::vector<Vec> centers;

    /**
     * clustered: n / 100 unit centres, a row is a centre plus Gaussian noise of
     * norm about 1, renormalised (the same family as the Rust benchmark's
     * synthetic data, so the two ports can be compared on one machine).
     */
    Source(bool isClustered, size_t dimension, size_t rows, float noiseNorm)
        : clustered(isClustered), dim(dimension), noise(noiseNorm) {
        detail::SplitMix64 rng(99);
        if (clustered) {
            const size_t count = std::max<size_t>(8, rows / 100);
            for (size_t c = 0; c < count; ++c) {
                Vec v(dim);
                double sq = 0;
                for (auto& x : v) { x = gauss(rng); sq += double(x) * double(x); }
                const float inv = static_cast<float>(1.0 / std::sqrt(sq));
                for (auto& x : v) x *= inv;
                centers.push_back(std::move(v));
            }
        }
    }

    static float gauss(detail::SplitMix64& rng) {
        const double u1 = (static_cast<double>(rng.next() >> 40) + 0.5) / 16777216.0;
        const double u2 = (static_cast<double>(rng.next() >> 40) + 0.5) / 16777216.0;
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }

    Vec draw(detail::SplitMix64& rng) const {
        if (!clustered) return unitNoise(rng, dim);
        const Vec& c = centers[static_cast<size_t>(rng.below(centers.size()))];
        Vec v(dim);
        const float scale = noise / std::sqrt(static_cast<float>(dim));
        double sq = 0;
        for (size_t d = 0; d < dim; ++d) {
            v[d] = c[d] + gauss(rng) * scale;
            sq += double(v[d]) * double(v[d]);
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(sq));
        for (auto& x : v) x *= inv;
        return v;
    }
};

double secondsSince(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
double mib(double bytes) { return bytes / 1048576.0; }
void report(const std::string& key, double value) { std::printf("%s=%.4f\n", key.c_str(), value); }

void fill(EmbeddingStore& store, const Source& src, size_t n) {
    detail::SplitMix64 rng(12345);
    for (size_t i = 0; i < n; ++i) store.add(NodeRef("doc", std::to_string(i)), "", src.draw(rng));
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(std::llround(static_cast<double>(v.size() - 1) * p))];
}

/** Mean overlap of `found` with the exact top-10 of each query. */
double recallAgainst(const std::vector<std::vector<SimilarityResult>>& found,
                     const std::vector<std::vector<SimilarityResult>>& truth) {
    double total = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        std::set<std::string> want;
        for (const auto& r : truth[i]) want.insert(r.node_ref.id);
        size_t hit = 0;
        for (const auto& r : found[i]) hit += want.count(r.node_ref.id);
        total += static_cast<double>(hit) / static_cast<double>(std::max<size_t>(1, want.size()));
    }
    return total / static_cast<double>(found.size());
}

struct Measured {
    double recall = 0, qps1 = 0, qpsBatch = 0, approxFraction = 0;
};

/** Recall, single-thread QPS and batch QPS of the store as configured. */
Measured measure(EmbeddingStore& store, const std::vector<Vec>& queries,
                 const std::vector<std::vector<SimilarityResult>>& truth, size_t threads) {
    Measured m;
    store.similaritySearchVector(queries[0], 10, "", 0.0f);   // warm up
    const size_t before = store.annStats().approximateSearches;
    auto t0 = Clock::now();
    std::vector<std::vector<SimilarityResult>> found;
    found.reserve(queries.size());
    for (const auto& q : queries) found.push_back(store.similaritySearchVector(q, 10, "", 0.0f));
    m.qps1 = static_cast<double>(queries.size()) / secondsSince(t0);
    m.recall = recallAgainst(found, truth);
    m.approxFraction = static_cast<double>(store.annStats().approximateSearches - before) / static_cast<double>(queries.size());
    store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);   // warm up
    t0 = Clock::now();
    store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);
    m.qpsBatch = static_cast<double>(queries.size()) / secondsSince(t0);
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string phase = argc > 1 ? argv[1] : "search";
    const size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100000;
    const size_t dim = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 384;
    const size_t nq = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 300;
    const std::string kind = argc > 5 ? argv[5] : "random";
    const float noise = argc > 6 ? std::strtof(argv[6], nullptr) : 1.0f;
    const Source src(kind == "clustered", dim, n, noise);
    const double raw = static_cast<double>(n) * static_cast<double>(dim) * 4.0;
    const size_t threads = std::max<unsigned>(1, std::thread::hardware_concurrency());

    if (phase == "mem") {
        const size_t before = residentBytes();
        EmbeddingStore store;
        fill(store, src, n);
        const size_t imported = residentBytes();
        AnnOptions o = AnnOptions::makeHnsw();
        o.minSize = 0;
        store.enableAnn(o);
        store.buildAnnIndex();
        const size_t indexed = residentBytes();
        report("raw_mib", mib(raw));
        report("rss_import_over_raw", static_cast<double>(imported - before) / raw);
        report("rss_hnsw_over_raw", static_cast<double>(indexed - before) / raw);
        report("graph_mib", mib(static_cast<double>(store.annStats().indexBytes)));
        report("graph_bytes_per_node", static_cast<double>(store.annStats().indexBytes) / static_cast<double>(n));
        report("graph_over_raw", static_cast<double>(store.annStats().indexBytes) / raw);
        return 0;
    }

    if (phase != "search") {
        std::fprintf(stderr, "unknown phase %s\n", phase.c_str());
        return 2;
    }

    EmbeddingStore store;
    fill(store, src, n);
    detail::SplitMix64 qrng(777);
    std::vector<Vec> queries;
    for (size_t i = 0; i < nq; ++i) queries.push_back(src.draw(qrng));

    // ground truth and the exact scan
    std::vector<std::vector<SimilarityResult>> truth;
    truth.reserve(nq);
    auto t0 = Clock::now();
    for (const auto& q : queries) truth.push_back(store.similaritySearchVector(q, 10, "", 0.0f, true));
    report("exact_qps_1t", static_cast<double>(nq) / secondsSince(t0));
    store.similaritySearchBatch(queries, 10, "", 0.0f, true, threads);
    t0 = Clock::now();
    store.similaritySearchBatch(queries, 10, "", 0.0f, true, threads);
    report("exact_batch_qps", static_cast<double>(nq) / secondsSince(t0));

    // ---- HNSW ----
    AnnOptions h = AnnOptions::makeHnsw();
    h.minSize = 0;
    store.enableAnn(h);
    t0 = Clock::now();
    store.buildAnnIndex();
    report("hnsw_build_s", secondsSince(t0));
    report("hnsw_graph_mib", mib(static_cast<double>(store.annStats().indexBytes)));
    report("hnsw_auto_ef", static_cast<double>(store.annStats().efSearch));
    {
        Measured m = measure(store, queries, truth, threads);
        report("hnsw_auto_recall10", m.recall);
        report("hnsw_auto_qps_1t", m.qps1);
        report("hnsw_auto_batch_qps", m.qpsBatch);
        report("hnsw_auto_graph_fraction", m.approxFraction);
    }
    for (size_t ef : {size_t(32), size_t(64), size_t(128), size_t(192), size_t(256), size_t(512)}) {
        store.setAnnEfSearch(ef);
        Measured m = measure(store, queries, truth, threads);
        const std::string tag = "hnsw_ef" + std::to_string(ef) + "_";
        report(tag + "recall10", m.recall);
        report(tag + "qps_1t", m.qps1);
        report(tag + "graph_fraction", m.approxFraction);
    }
    store.setAnnEfSearch(0);

    // write latency with the graph on: inserts, updates, deletes (microseconds)
    {
        detail::SplitMix64 wrng(4242);
        std::vector<double> insert, update, remove;
        const size_t writes = 1000;
        for (size_t i = 0; i < writes; ++i) {
            Vec v = src.draw(wrng);
            auto w = Clock::now();
            store.add(NodeRef("late", std::to_string(i)), "", v);
            insert.push_back(secondsSince(w) * 1e6);
        }
        for (size_t i = 0; i < writes; ++i) {
            Vec v = src.draw(wrng);
            auto w = Clock::now();
            store.add(NodeRef("doc", std::to_string(i * 7)), "", v);
            update.push_back(secondsSince(w) * 1e6);
        }
        for (size_t i = 0; i < writes; ++i) {
            auto w = Clock::now();
            store.remove(NodeRef("late", std::to_string(i)));
            remove.push_back(secondsSince(w) * 1e6);
        }
        report("hnsw_insert_us_median", percentile(insert, 0.5));
        report("hnsw_insert_us_p99", percentile(insert, 0.99));
        report("hnsw_update_us_median", percentile(update, 0.5));
        report("hnsw_delete_us_median", percentile(remove, 0.5));
    }

    // the updates above changed 1,000 rows: the ground truth must follow
    truth.clear();
    for (const auto& q : queries) truth.push_back(store.similaritySearchVector(q, 10, "", 0.0f, true));

    // ---- IVF on the same rows ----
    AnnOptions ivf;
    ivf.minSize = 0;
    store.enableAnn(ivf);
    t0 = Clock::now();
    store.buildAnnIndex();
    report("ivf_build_s", secondsSince(t0));
    report("ivf_nlist", static_cast<double>(store.annStats().nlist));
    {
        Measured m = measure(store, queries, truth, threads);
        report("ivf_auto_nprobe", static_cast<double>(store.annStats().nprobe));
        report("ivf_auto_recall10", m.recall);
        report("ivf_auto_qps_1t", m.qps1);
        report("ivf_auto_batch_qps", m.qpsBatch);
    }
    const size_t nlist = store.annStats().nlist;
    for (double frac : {0.10, 0.20, 0.35, 0.50}) {
        const size_t nprobe = std::max<size_t>(1, static_cast<size_t>(std::llround(frac * static_cast<double>(nlist))));
        store.setAnnNprobe(nprobe);
        Measured m = measure(store, queries, truth, threads);
        const std::string tag = "ivf_np" + std::to_string(static_cast<int>(frac * 100)) + "pct_";
        report(tag + "recall10", m.recall);
        report(tag + "qps_1t", m.qps1);
        report(tag + "batch_qps", m.qpsBatch);
    }
    return 0;
}
