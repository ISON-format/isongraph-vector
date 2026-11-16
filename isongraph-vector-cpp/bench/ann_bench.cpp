/**
 * Approximate-index benchmark for the C++ port: latency and recall@10 of the
 * opt-in IVF index (EmbeddingStore::enableAnn) against the exact scan, on
 * synthetic clustered 384-d unit vectors.
 *
 * This is the C++ counterpart of benchmarks/ann.py and uses the same two data
 * regimes (the generators are not bit-identical to numpy's, so the datasets
 * differ; their structure does not):
 *
 *   clustered  N/100 Gaussian clusters, noise norm 1.0: well separated topics
 *   diffuse    N/40 clusters grouped under 20 super-topics, noise norm 2.0 and
 *              a super-topic spread of 1.0: hierarchical, overlapping, harder
 *
 * Both are synthetic. An IVF index's recall depends entirely on how the data
 * clusters; use EmbeddingStore::recallCheck on your own queries before trusting
 * a setting.
 *
 * Build (not part of the default build):
 *   cmake -S . -B build -DISONGRAPH_VECTOR_BENCH=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release --target ann_bench
 * Add -DISONGRAPH_VECTOR_BENCH_NATIVE=ON to compile it with AVX2/FMA (or
 * -march=native); the default is the portable SSE2 baseline.
 *
 * Run:
 *   ann_bench                                  # 10k and 100k, both regimes
 *   ann_bench --sizes 10000,100000,1000000     # adds 1M (needs ~4 GB RAM)
 *   ann_bench --sizes 1000000 --regime clustered --queries 100 --json out.jsonl
 *   ann_bench --threads 1                      # single-threaded index build
 *
 * Output is a markdown table per (regime, size) on stdout; --json also appends
 * one JSON object per row to the given file for the benchmark wave to collect.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

#include "../include/isongraph_vector.hpp"

using namespace isongraph_vector;
using Clock = std::chrono::steady_clock;

namespace {

constexpr size_t DIM = 384;
constexpr size_t K = 10;
constexpr uint64_t SEED = 4242;

struct Regime {
    const char* name;
    size_t perCluster;
    double noise;
    size_t supers;  // 0 = none
    double spread;
};

const Regime REGIMES[] = {
    {"clustered", 100, 1.0, 0, 0.0},
    {"diffuse", 40, 2.0, 20, 1.0},
};

double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double pos = p * (v.size() - 1);
    size_t lo = static_cast<size_t>(pos), hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

struct Gaussian {
    std::mt19937_64 rng;
    explicit Gaussian(uint64_t seed) : rng(seed) {}
    double next() {
        double u1 = (static_cast<double>(rng() >> 11) + 1.0) / 9007199254740993.0;
        double u2 = static_cast<double>(rng() >> 11) / 9007199254740992.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

void normalise(std::vector<float>& v) {
    double sq = 0;
    for (float x : v) sq += double(x) * x;
    double inv = sq > 0 ? 1.0 / std::sqrt(sq) : 0.0;
    for (float& x : v) x = static_cast<float>(x * inv);
}

class Generator {
public:
    Generator(const Regime& regime, size_t n) : regime_(regime) {
        Gaussian g(SEED ^ (n * 1000003u) ^ regime.perCluster);
        size_t clusters = std::max<size_t>(8, n / regime.perCluster);
        std::vector<std::vector<float>> supers;
        for (size_t s = 0; s < regime.supers; ++s) supers.push_back(randomUnit(g));
        for (size_t c = 0; c < clusters; ++c) {
            std::vector<float> center = randomUnit(g);
            if (regime.supers) {
                const auto& parent = supers[g.rng() % supers.size()];
                for (size_t d = 0; d < DIM; ++d) center[d] = parent[d] + static_cast<float>(regime.spread) * center[d];
                normalise(center);
            }
            centers_.push_back(std::move(center));
        }
    }

    /** `count` vectors of stream `stream` (a different stream gives different vectors). */
    std::vector<std::vector<float>> sample(size_t count, uint64_t stream) const {
        Gaussian g(SEED * 31 + stream);
        std::vector<std::vector<float>> out;
        out.reserve(count);
        const double scale = regime_.noise / std::sqrt(static_cast<double>(DIM));
        for (size_t i = 0; i < count; ++i) {
            const auto& center = centers_[g.rng() % centers_.size()];
            std::vector<float> v(DIM);
            for (size_t d = 0; d < DIM; ++d) v[d] = center[d] + static_cast<float>(scale * g.next());
            normalise(v);
            out.push_back(std::move(v));
        }
        return out;
    }

private:
    static std::vector<float> randomUnit(Gaussian& g) {
        std::vector<float> v(DIM);
        for (float& x : v) x = static_cast<float>(g.next());
        normalise(v);
        return v;
    }
    Regime regime_;
    std::vector<std::vector<float>> centers_;
};

struct Options {
    std::vector<size_t> sizes = {10000, 100000};
    std::vector<std::string> regimes = {"clustered", "diffuse"};
    size_t queries = 100;
    size_t threads = 0;
    std::vector<size_t> nprobes = {1, 2, 4, 8, 16, 32, 64};
    std::string json;
};

std::vector<size_t> parseList(const std::string& text) {
    std::vector<size_t> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(static_cast<size_t>(std::stoull(item)));
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--sizes") o.sizes = parseList(next());
        else if (a == "--regime") {
            std::string r = next();
            o.regimes = r == "both" ? std::vector<std::string>{"clustered", "diffuse"} : std::vector<std::string>{r};
        }
        else if (a == "--queries") o.queries = static_cast<size_t>(std::stoull(next()));
        else if (a == "--threads") o.threads = static_cast<size_t>(std::stoull(next()));
        else if (a == "--nprobe") o.nprobes = parseList(next());
        else if (a == "--json") o.json = next();
        else if (a == "--help" || a == "-h") {
            std::cout << "ann_bench [--sizes 10000,100000] [--regime clustered|diffuse|both] [--queries 100]\n"
                         "          [--threads N] [--nprobe 1,2,4,...] [--json file]\n";
            std::exit(0);
        }
        else { std::cerr << "unknown argument " << a << "\n"; std::exit(2); }
    }
    return o;
}

const Regime& regimeNamed(const std::string& name) {
    for (const auto& r : REGIMES) if (name == r.name) return r;
    std::cerr << "unknown regime " << name << "\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    std::ofstream json;
    if (!opt.json.empty()) json.open(opt.json, std::ios::app);

    std::cout << "# ann_bench: IVF-flat vs exact scan, dim " << DIM << ", top_k " << K << ", "
              << opt.queries << " queries\n";
    std::cout << "SIMD backend lanes: " << detail::simd::W << " floats; hardware threads: "
              << std::thread::hardware_concurrency() << "; index build threads: "
              << (opt.threads ? std::to_string(opt.threads) : std::string("auto")) << "\n\n";

    for (const std::string& regimeName : opt.regimes) {
        const Regime& regime = regimeNamed(regimeName);
        for (size_t n : opt.sizes) {
            std::cout << "## " << regime.name << ", N = " << n << "\n\n";
            Generator gen(regime, n);

            EmbeddingStore store;
            AnnOptions ann;
            ann.minSize = 0;
            ann.threads = opt.threads;
            store.enableAnn(ann);

            // The same vectors, kept flat for the contiguous brute-force baseline.
            std::vector<float> flat;
            flat.reserve(n * DIM);
            double added = 0;
            const size_t chunk = 5000;
            for (size_t start = 0; start < n; start += chunk) {
                size_t count = std::min(chunk, n - start);
                auto vectors = gen.sample(count, 1000 + start / chunk);
                auto a = Clock::now();
                for (size_t i = 0; i < count; ++i) {
                    store.add(NodeRef("doc", std::to_string(start + i)), "", vectors[i]);
                }
                added += millis(a, Clock::now());
                for (const auto& v : vectors) flat.insert(flat.end(), v.begin(), v.end());
            }
            std::cout << "* ingest: " << static_cast<size_t>(n / (added / 1000.0)) << " vectors/s (index not built yet)\n";

            auto queries = gen.sample(opt.queries, 7);

            // exact scan: warm latency, and the ground truth for recall
            std::vector<std::set<std::string>> truth(queries.size());
            std::vector<double> exactMs;
            store.similaritySearchVector(queries[0], K, "", 0.0f, true);  // warm up
            for (size_t q = 0; q < queries.size(); ++q) {
                auto a = Clock::now();
                auto res = store.similaritySearchVector(queries[q], K, "", 0.0f, true);
                auto b = Clock::now();
                exactMs.push_back(millis(a, b));
                for (const auto& r : res) truth[q].insert(r.node_ref.id);
            }
            double exactMedian = percentile(exactMs, 0.5);
            std::cout << "* exact scan: median " << exactMedian << " ms, p95 " << percentile(exactMs, 0.95) << " ms\n";

            // Contiguous single-threaded SIMD scan of the same vectors (they are
            // unit length, so the dot product is the cosine): what an exact
            // search costs when it is written for speed rather than as a loop
            // over a hash map. This is the fair yardstick for the index.
            std::vector<double> flatMs;
            size_t flatAgree = 0;
            for (size_t q = 0; q < queries.size(); ++q) {
                auto a = Clock::now();
                std::vector<detail::IvfHit> heap;
                for (size_t row = 0; row < n; ++row) {
                    float s = detail::simd::dot(flat.data() + row * DIM, queries[q].data(), DIM);
                    detail::IvfHit hit{s, static_cast<uint32_t>(row)};
                    if (heap.size() < K) { heap.push_back(hit); std::push_heap(heap.begin(), heap.end(), detail::hitBefore); }
                    else if (detail::hitBefore(hit, heap.front())) {
                        std::pop_heap(heap.begin(), heap.end(), detail::hitBefore);
                        heap.back() = hit;
                        std::push_heap(heap.begin(), heap.end(), detail::hitBefore);
                    }
                }
                flatMs.push_back(millis(a, Clock::now()));
                for (const auto& h : heap) flatAgree += truth[q].count(std::to_string(h.slot));
            }
            double flatMedian = percentile(flatMs, 0.5);
            std::cout << "* flat SIMD scan (contiguous, one thread, no index): median " << flatMedian
                      << " ms, p95 " << percentile(flatMs, 0.95) << " ms (agrees with exact on "
                      << flatAgree << "/" << queries.size() * K << " top-10 entries)\n";

            AnnStats built = store.buildAnnIndex();
            std::cout << "* index build: " << built.buildSeconds << " s, nlist " << built.nlist
                      << " (automatic nprobe " << built.nprobe << "), index overhead "
                      << built.indexBytes / 1048576.0 << " MB plus a " << built.vectorBytes / 1048576.0
                      << " MB contiguous copy of the vectors\n\n";

            std::cout << "| nprobe | % of cells | recall@10 mean | worst | median ms | p95 ms | vs exact scan | vs flat SIMD scan |\n"
                         "|---|---|---|---|---|---|---|---|\n";
            std::vector<size_t> probes = opt.nprobes;
            probes.push_back(0);  // the automatic default
            for (size_t probe : probes) {
                if (probe != 0 && probe >= built.nlist) continue;
                store.setAnnNprobe(probe);
                size_t effective = store.annStats().nprobe;
                // one untimed pass so the first measured query is not a cold one
                store.similaritySearchVector(queries[0], K, "", 0.0f);
                std::vector<double> ms, recalls;
                for (size_t q = 0; q < queries.size(); ++q) {
                    auto a = Clock::now();
                    auto res = store.similaritySearchVector(queries[q], K, "", 0.0f);
                    auto b = Clock::now();
                    ms.push_back(millis(a, b));
                    size_t hit = 0;
                    for (const auto& r : res) hit += truth[q].count(r.node_ref.id);
                    recalls.push_back(truth[q].empty() ? 1.0 : double(hit) / truth[q].size());
                }
                double mean = 0, worst = 1;
                for (double r : recalls) { mean += r; worst = std::min(worst, r); }
                mean /= recalls.size();
                double median = percentile(ms, 0.5), p95 = percentile(ms, 0.95);
                char line[256];
                std::snprintf(line, sizeof line, "| %zu%s | %.1f | %.3f | %.2f | %.2f | %.2f | %.1fx | %.1fx |",
                              effective, probe == 0 ? " (auto)" : "",
                              100.0 * effective / built.nlist, mean, worst, median, p95, exactMedian / median,
                              flatMedian / median);
                std::cout << line << "\n";
                if (json.is_open()) {
                    json << "{\"port\":\"cpp\",\"regime\":\"" << regime.name << "\",\"n\":" << n
                         << ",\"dim\":" << DIM << ",\"k\":" << K << ",\"queries\":" << queries.size()
                         << ",\"nlist\":" << built.nlist << ",\"nprobe\":" << effective
                         << ",\"recall_mean\":" << mean << ",\"recall_worst\":" << worst
                         << ",\"ann_median_ms\":" << median << ",\"ann_p95_ms\":" << p95
                         << ",\"exact_median_ms\":" << exactMedian << ",\"flat_simd_median_ms\":" << flatMedian
                         << ",\"build_seconds\":" << built.buildSeconds
                         << ",\"simd_lanes\":" << detail::simd::W << "}\n";
                }
            }
            std::cout << "\n";
        }
    }
    return 0;
}
