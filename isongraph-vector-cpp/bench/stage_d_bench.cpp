// Memory, search and persistence benchmark for the vector store.
//
//   stage_d_bench <phase> [n=100000] [dim=384] [queries=200]
//
// phases (run each in its own process so the resident-set numbers are clean):
//   mem     resident memory after importing n random unit vectors, and after the
//           approximate index is built, against the raw vector bytes
//   search  single-query exact and approximate QPS, parallel batch QPS
//   io      JSON sidecar save/load, and the binary snapshot save/load (when this
//           build has it), with file sizes
//
// Builds against either header set; compile with -DISONGRAPH_BENCH_NEW when the
// store has similaritySearchBatch() and the snapshot API.
//
//   g++ -O2 -I include -I vendor/ison-graph-cpp -DISONGRAPH_BENCH_NEW bench/stage_d_bench.cpp -pthread
//
// Prints one "key=value" line per measurement so a script can collect the
// medians of several runs.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

/// Resident set size of this process in bytes (0 where unknown).
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

struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float unit() { return static_cast<float>(next() >> 40) / 8388608.0f - 1.0f; }
};

std::vector<float> unitVector(Rng& rng, size_t dim) {
    std::vector<float> v(dim);
    double sq = 0;
    for (auto& x : v) { x = rng.unit(); sq += double(x) * x; }
    const float inv = static_cast<float>(1.0 / std::sqrt(sq));
    for (auto& x : v) x *= inv;
    return v;
}

double secondsSince(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

void fill(EmbeddingStore& store, size_t n, size_t dim) {
    Rng rng(12345);
    for (size_t i = 0; i < n; ++i) store.add(NodeRef("doc", std::to_string(i)), "", unitVector(rng, dim));
}

double mib(double bytes) { return bytes / 1048576.0; }

void report(const char* key, double value) { std::printf("%s=%.4f\n", key, value); }

}  // namespace

int main(int argc, char** argv) {
    const std::string phase = argc > 1 ? argv[1] : "mem";
    const size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100000;
    const size_t dim = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 384;
    const size_t nq = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 200;
    const double raw = static_cast<double>(n) * static_cast<double>(dim) * 4.0;

    if (phase == "mem") {
        const size_t before = residentBytes();
        EmbeddingStore store;
        fill(store, n, dim);
        const size_t imported = residentBytes();
        report("raw_mib", mib(raw));
        report("rss_import_mib", mib(static_cast<double>(imported - before)));
        report("rss_import_over_raw", static_cast<double>(imported - before) / raw);
        AnnOptions o;
        o.minSize = 0;
        store.enableAnn(o);
        store.buildAnnIndex();
        const size_t indexed = residentBytes();
        report("rss_indexed_mib", mib(static_cast<double>(indexed - before)));
        report("rss_indexed_over_raw", static_cast<double>(indexed - before) / raw);
        return 0;
    }

    if (phase == "search") {
        EmbeddingStore store;
        fill(store, n, dim);
        Rng qrng(777);
        std::vector<std::vector<float>> queries;
        for (size_t i = 0; i < nq; ++i) queries.push_back(unitVector(qrng, dim));

        store.similaritySearchVector(queries[0], 10, "", 0.0f);  // warm up
        auto t0 = Clock::now();
        for (const auto& q : queries) store.similaritySearchVector(q, 10, "", 0.0f);
        report("exact_single_qps", static_cast<double>(nq) / secondsSince(t0));

        // parallel batch of the exact scan
        const size_t threads = std::max<unsigned>(1, std::thread::hardware_concurrency());
#if defined(ISONGRAPH_BENCH_NEW)
        store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);  // warm up
        t0 = Clock::now();
        auto batch = store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);
        report("exact_batch_qps", static_cast<double>(nq) / secondsSince(t0));
#else
        // no batch API: the same work spread over threads by hand, one query per task
        {
            std::vector<std::vector<SimilarityResult>> out(nq);
            auto runAll = [&]() {
                std::atomic<size_t> next{0};
                std::vector<std::thread> pool;
                for (size_t t = 0; t < threads; ++t) {
                    pool.emplace_back([&]() {
                        for (;;) {
                            size_t i = next.fetch_add(1);
                            if (i >= nq) break;
                            out[i] = store.similaritySearchVector(queries[i], 10, "", 0.0f);
                        }
                    });
                }
                for (auto& t : pool) t.join();
            };
            runAll();
            t0 = Clock::now();
            runAll();
            report("exact_batch_qps", static_cast<double>(nq) / secondsSince(t0));
        }
#endif

        AnnOptions o;
        o.minSize = 0;
        store.enableAnn(o);
        auto b0 = Clock::now();
        store.buildAnnIndex();
        report("ann_build_s", secondsSince(b0));
        store.similaritySearchVector(queries[0], 10, "", 0.0f);
        t0 = Clock::now();
        for (const auto& q : queries) store.similaritySearchVector(q, 10, "", 0.0f);
        report("ann_single_qps", static_cast<double>(nq) / secondsSince(t0));
#if defined(ISONGRAPH_BENCH_NEW)
        t0 = Clock::now();
        auto annBatch = store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);
        report("ann_batch_qps", static_cast<double>(nq) / secondsSince(t0));
#endif
        RecallReport recall = store.recallCheck(std::vector<std::vector<float>>(queries.begin(), queries.begin() + std::min<size_t>(nq, 50)), 10);
        report("ann_recall10", recall.recall);

        // write cost: single adds with the index on (new rows, then updates)
        Rng wrng(4242);
        auto w0 = Clock::now();
        const size_t writes = 2000;
        for (size_t i = 0; i < writes; ++i) store.add(NodeRef("late", std::to_string(i)), "", unitVector(wrng, dim));
        report("ann_write_us", secondsSince(w0) / static_cast<double>(writes) * 1e6);
        return 0;
    }

    if (phase == "io") {
        EmbeddingStore store;
        fill(store, n, dim);
        const auto dir = std::filesystem::temp_directory_path() / "isongraph_stage_d_bench";
        std::filesystem::create_directories(dir);
        const std::string json = (dir / "bench.embeddings.json").string();
        auto t0 = Clock::now();
        store.saveEmbeddings(json);
        report("json_save_s", secondsSince(t0));
        report("json_mib", mib(static_cast<double>(std::filesystem::file_size(json))));
        {
            EmbeddingStore back;
            t0 = Clock::now();
            back.loadEmbeddings(json);
            report("json_load_s", secondsSince(t0));
        }
#if defined(ISONGRAPH_BENCH_NEW)
        const std::string snap = (dir / "bench.embeddings.snap").string();
        t0 = Clock::now();
        store.saveSnapshot(snap);
        report("snap_save_s", secondsSince(t0));
        report("snap_mib", mib(static_cast<double>(std::filesystem::file_size(snap))));
        {
            EmbeddingStore back;
            t0 = Clock::now();
            back.loadSnapshot(snap);
            report("snap_load_s", secondsSince(t0));
        }
#endif
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        return 0;
    }

    std::cerr << "unknown phase " << phase << "\n";
    return 2;
}
