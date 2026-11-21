// HNSW against IVF and the exact scan: build time, recall@10, single-query and
// batch throughput, write latency and memory. The C++ counterpart of
// isongraph-vector-rs/examples/hnsw_bench.rs (same measurements, synthetic data).
//
//   hnsw_bench <phase> [n=100000] [dim=384] [queries=300] [data=random|clustered] [noise=1.0]
//   hnsw_bench real dir=<prep/text384> [n=100000] [queries=1000] [ef=64,96,128,192] [nprobe=a,b]
//                   [budget=4] [pin=1] [batch=0] [hnsw=1] [ivf=1]
//
// phases (run each in its own process so the resident-set numbers are clean):
//   mem     resident memory after importing n unit vectors, and after the HNSW
//           graph is built, against the raw vector bytes
//   search  HNSW build, recall and QPS at the automatic ef and a sweep of ef,
//           batch QPS, insert / update / delete latency; then IVF (default
//           nprobe and a sweep) and the exact scan on the same data and queries
//   real    the benchmarks/competitive/ real vectors (prep/text384: base.f32,
//           queries.f32, gt_<n>.i32, meta.json), read as isongraph-vector-rs/
//           examples/hnsw_bench.rs reads them with --ds-dir: HNSW build, recall
//           and single-thread / batch QPS over ef, insert latency, then IVF over
//           nprobe (pin=1 pins the single-thread loops to one CPU on Windows,
//           like the Rust --pin)
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
#include <iterator>
#include <map>
#include <set>
#include <sstream>
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

// ---- real data (the Rust benchmark's --ds-dir mode) ---------------------------------------

std::vector<float> readF32(const std::string& path, size_t startRow, size_t rows, size_t d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    f.seekg(static_cast<std::streamoff>(startRow * d * 4));
    std::vector<float> out(rows * d);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
    if (!f) { std::fprintf(stderr, "short read of %s\n", path.c_str()); std::exit(2); }
    return out;
}

std::vector<int32_t> readI32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<int32_t> out(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
    return out;
}

/** Run `f`, on one pinned CPU at high priority when `pin` (Windows; elsewhere just runs it). Like the Rust bench's --pin. */
template <class F>
void pinned(bool pin, F&& f) {
#if defined(_WIN32)
    if (pin) {
        char* env = nullptr;
        size_t envLen = 0;
        _dupenv_s(&env, &envLen, "BENCH_PIN_CPU");
        const unsigned cpu = env ? static_cast<unsigned>(std::atoi(env)) : 8u;
        std::free(env);
        DWORD_PTR processMask = 0, systemMask = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask);
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(1) << cpu);
        f();
        SetProcessAffinityMask(GetCurrentProcess(), processMask);
        return;
    }
#else
    (void)pin;
#endif
    f();
}

struct RealPoint {
    double recall = 0, qps1 = 0, medianMs = 0, p99Ms = 0, qpsBatch = 0;
};

RealPoint measureReal(EmbeddingStore& store, const std::vector<Vec>& queries,
                      const std::vector<std::vector<int>>& truth, size_t threads, double budget, bool pin,
                      bool withBatch) {
    RealPoint p;
    std::vector<std::vector<SimilarityResult>> found;
    found.reserve(queries.size());
    for (const auto& q : queries) found.push_back(store.similaritySearchVector(q, 10, "", 0.0f));
    size_t hit = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        std::set<std::string> want;
        for (size_t j = 0; j < 10 && j < truth[i].size(); ++j) want.insert(std::to_string(truth[i][j]));
        for (const auto& r : found[i]) hit += want.count(r.node_ref.id);
    }
    p.recall = static_cast<double>(hit) / static_cast<double>(10 * found.size());
    std::vector<double> ms;
    const auto loopStart = Clock::now();
    pinned(pin, [&] {
        for (int pass = 0; pass < 3; ++pass) {
            for (const auto& q : queries) {
                const auto t = Clock::now();
                auto r = store.similaritySearchVector(q, 10, "", 0.0f);
                ms.push_back(secondsSince(t) * 1000.0);
                if (r.empty()) std::fputc('!', stderr);
                if (secondsSince(loopStart) > budget) return;
            }
        }
    });
    p.qps1 = static_cast<double>(ms.size()) / secondsSince(loopStart);
    p.medianMs = percentile(ms, 0.5);
    p.p99Ms = percentile(ms, 0.99);
    if (!withBatch) return p;
    const auto t = Clock::now();
    size_t done = 0;
    while (secondsSince(t) < std::min(budget, 3.0)) {
        store.similaritySearchBatch(queries, 10, "", 0.0f, false, threads);
        done += queries.size();
    }
    p.qpsBatch = static_cast<double>(done) / secondsSince(t);
    return p;
}

void printHeader(const char* title) {
    std::printf("\n%s\n  %-14s %8s %10s %10s %10s %12s\n", title, "setting", "recall", "qps(1T)", "median ms", "p99 ms", "qps(batch)");
}

void printPoint(const std::string& label, const RealPoint& p) {
    std::printf("  %-14s %8.4f %10.0f %10.3f %10.3f %12.0f\n", label.c_str(), p.recall, p.qps1, p.medianMs, p.p99Ms, p.qpsBatch);
    std::fflush(stdout);
}

int runReal(int argc, char** argv) {
    std::map<std::string, std::string> opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const size_t eq = a.find('=');
        if (eq != std::string::npos) opt[a.substr(0, eq)] = a.substr(eq + 1);
    }
    auto get = [&](const char* key, const char* def) {
        auto it = opt.find(key);
        return it == opt.end() ? std::string(def) : it->second;
    };
    auto list = [&](const std::string& text) {
        std::vector<size_t> out;
        std::stringstream ss(text);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) out.push_back(static_cast<size_t>(std::stoull(item)));
        }
        return out;
    };
    const std::string dir = get("dir", "");
    if (dir.empty()) { std::fprintf(stderr, "real: dir=<prep/text384> is required\n"); return 2; }
    const size_t n = std::stoull(get("n", "100000"));
    const size_t nq = std::stoull(get("queries", "1000"));
    const double budget = std::stod(get("budget", "4"));
    const bool pin = get("pin", "0") == "1";
    const bool withBatch = get("batch", "1") == "1";
    const bool doHnsw = get("hnsw", "1") == "1", doIvf = get("ivf", "1") == "1";
    const std::vector<size_t> efs = list(get("ef", "64,96,128,192"));
    const size_t threads = std::max<unsigned>(1, std::thread::hardware_concurrency());

    // meta.json: only "d" is needed
    size_t d = 0;
    {
        std::ifstream f(dir + "/meta.json");
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const size_t at = text.find("\"d\"");
        if (at == std::string::npos) { std::fprintf(stderr, "no d in meta.json\n"); return 2; }
        d = static_cast<size_t>(std::strtoull(text.c_str() + text.find(':', at) + 1, nullptr, 10));
    }
    std::printf("hnsw_bench real: n=%zu d=%zu queries=%zu threads=%zu M=16 efC=200 data=%s pin=%d\n", n, d, nq,
                threads, dir.c_str(), pin ? 1 : 0);

    std::vector<Vec> queries;
    {
        const std::vector<float> q = readF32(dir + "/queries.f32", 0, nq, d);
        for (size_t i = 0; i < nq; ++i) {
            queries.emplace_back(q.begin() + static_cast<std::ptrdiff_t>(i * d), q.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
        }
    }
    std::vector<std::vector<int>> truth;
    {
        const std::vector<int32_t> gt = readI32(dir + "/gt_" + std::to_string(n) + ".i32");
        for (size_t i = 0; i < nq; ++i) {
            truth.emplace_back(gt.begin() + static_cast<std::ptrdiff_t>(i * 100), gt.begin() + static_cast<std::ptrdiff_t>((i + 1) * 100));
        }
    }
    const std::vector<float> base = readF32(dir + "/base.f32", 0, n + 2000, d);
    auto rowVec = [&](size_t i) {
        return Vec(base.begin() + static_cast<std::ptrdiff_t>(i * d), base.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
    };
    auto ingest = [&](EmbeddingStore& store) {
        const auto t = Clock::now();
        for (size_t i = 0; i < n; ++i) store.add(NodeRef("doc", std::to_string(i)), "", rowVec(i));
        return secondsSince(t);
    };

    if (doHnsw) {
        EmbeddingStore store;
        const double ingestS = ingest(store);
        std::printf("ingest %.2fs (%.0f vec/s)\n", ingestS, static_cast<double>(n) / ingestS);
        AnnOptions h = AnnOptions::makeHnsw();
        h.minSize = 0;
        store.enableAnn(h);
        const auto t = Clock::now();
        store.buildAnnIndex();
        const double buildS = secondsSince(t);
        std::printf("\nHNSW build %.1fs (%.0f nodes/s), max level %zu, graph %.1f MB (%.0f B/node)\n", buildS,
                    static_cast<double>(n) / buildS, store.annStats().maxLevel,
                    mib(static_cast<double>(store.annStats().indexBytes)),
                    static_cast<double>(store.annStats().indexBytes) / static_cast<double>(n));
        printHeader("HNSW: recall@10 vs speed over ef_search");
        for (size_t ef : efs) {
            store.setAnnEfSearch(ef);
            printPoint("ef=" + std::to_string(ef), measureReal(store, queries, truth, threads, budget, pin, withBatch));
        }
        store.setAnnEfSearch(0);
        // writes into the built graph
        const size_t extra = std::min<size_t>(2000, std::max<size_t>(nq, 500));
        std::vector<double> ins;
        for (size_t i = 0; i < extra; ++i) {
            Vec v = rowVec(n + i);
            const auto w = Clock::now();
            store.add(NodeRef("doc", std::to_string(n + i)), "", v);
            ins.push_back(secondsSince(w) * 1000.0);
        }
        std::printf("\ninsert into the built graph (ms): median %.3f p99 %.3f max %.3f\n", percentile(ins, 0.5),
                    percentile(ins, 0.99), percentile(ins, 1.0));
    }

    if (doIvf) {
        EmbeddingStore store;
        ingest(store);
        AnnOptions o;
        o.minSize = 0;
        store.enableAnn(o);
        const auto t = Clock::now();
        store.buildAnnIndex();
        const size_t nlist = store.annStats().nlist;
        std::printf("\nIVF build %.1fs nlist=%zu\n", secondsSince(t), nlist);
        printHeader("IVF-flat: recall@10 vs speed over nprobe");
        std::vector<size_t> probes = list(get("nprobe", ""));
        if (probes.empty()) {
            for (double pct : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0}) {
                probes.push_back(std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(nlist) * pct / 100.0))));
            }
        }
        for (size_t np : probes) {
            store.setAnnNprobe(np);
            printPoint("nprobe=" + std::to_string(np), measureReal(store, queries, truth, threads, budget, pin, withBatch));
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "real") return runReal(argc, argv);
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
