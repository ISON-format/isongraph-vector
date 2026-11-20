// The C++ side of the HNSW cross-port check: the same deterministic data, the
// same phases and the same output format as isongraph-vector-rs/examples/
// hnsw_xcheck.rs, so the two outputs can be compared line by line.
//
//   hnsw_xcheck [n=5000] [dim=32] [queries=100] [k=10] [m=16] [efc=200] [seed=1] [ef=0] [noise=500]
//
// Every vector component is `integer / 1024` (exact in float), drawn from the
// SplitMix64 generator of the level draw, so no floating-point library sits
// between the two ports' data. Output per query: `<phase> q<i> id:score_bits ...`
// (HNSW) and `<phase> x<i> ...` (exact scan); then a `stats` line per phase.
//
//   g++ -O2 -I include -I vendor/ison-graph-cpp bench/hnsw_xcheck.cpp -pthread

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "isongraph_vector.hpp"

using namespace isongraph_vector;

namespace {

constexpr size_t kClusters = 30;

std::vector<std::vector<int>> makeCenters(detail::SplitMix64& rng, size_t dim) {
    std::vector<std::vector<int>> centers(kClusters, std::vector<int>(dim));
    for (auto& c : centers) {
        for (auto& x : c) x = static_cast<int>(rng.below(2001)) - 1000;
    }
    return centers;
}

std::vector<float> draw(detail::SplitMix64& rng, const std::vector<std::vector<int>>& centers, int noiseRange) {
    const auto& c = centers[static_cast<size_t>(rng.below(centers.size()))];
    std::vector<float> v(c.size());
    for (size_t d = 0; d < c.size(); ++d) {
        const int noise = static_cast<int>(rng.below(2 * static_cast<uint64_t>(noiseRange) + 1)) - noiseRange;
        v[d] = static_cast<float>(c[d] + noise) / 1024.0f;
    }
    return v;
}

void line(const char* phase, char tag, size_t i, const std::vector<SimilarityResult>& hits) {
    std::printf("%s %c%zu", phase, tag, i);
    for (const auto& h : hits) {
        uint32_t bits;
        std::memcpy(&bits, &h.score, sizeof bits);
        std::printf(" %s:%08x", h.node_ref.id.c_str(), bits);
    }
    std::printf("\n");
}

void dump(const char* phase, const EmbeddingStore& store, const std::vector<std::vector<float>>& queries, size_t k) {
    for (size_t i = 0; i < queries.size(); ++i) {
        line(phase, 'q', i, store.similaritySearchVector(queries[i], k, "", -2.0f));
    }
    for (size_t i = 0; i < queries.size(); ++i) {
        line(phase, 'x', i, store.similaritySearchVector(queries[i], k, "", -2.0f, true));
    }
    const AnnStats s = store.annStats();
    std::printf("%s stats nodes=%zu dead=%zu level=%zu approximate=%zu\n", phase, s.graphNodes, s.graphDead,
                s.maxLevel, s.approximateSearches);
}

}  // namespace

int main(int argc, char** argv) {
    auto arg = [&](int i, size_t d) { return argc > i ? static_cast<size_t>(std::strtoull(argv[i], nullptr, 10)) : d; };
    const size_t n = arg(1, 5000), dim = arg(2, 32), nq = arg(3, 100), k = arg(4, 10);
    const size_t m = arg(5, 16), efc = arg(6, 200), ef = arg(8, 0);
    const int noise = static_cast<int>(arg(9, 500));
    const uint64_t seed = arg(7, 1);

    detail::SplitMix64 centerRng(seed ^ 0x51), data(seed ^ 0x52), qrng(seed ^ 0x53), ops(seed ^ 0x54);
    const auto centers = makeCenters(centerRng, dim);
    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < nq; ++i) queries.push_back(draw(qrng, centers, noise));

    EmbeddingStore store;
    AnnOptions o = AnnOptions::makeHnsw();
    o.minSize = 0;
    o.hnsw.m = m;
    o.hnsw.efConstruction = efc;
    o.hnsw.efSearch = ef;
    store.enableAnn(o);
    for (size_t i = 0; i < n; ++i) store.add(NodeRef("doc", std::to_string(i)), "", draw(data, centers, noise));
    store.buildAnnIndex();
    std::printf("params n=%zu dim=%zu queries=%zu k=%zu m=%zu efc=%zu seed=%llu\n", n, dim, nq, k, m, efc,
                static_cast<unsigned long long>(seed));
    dump("built", store, queries, k);

    for (size_t i = 0; i < n / 10; ++i) store.add(NodeRef("doc", std::to_string(n + i)), "", draw(ops, centers, noise));
    for (size_t i = 0; i < n / 20; ++i) store.add(NodeRef("doc", std::to_string((i * 13) % n)), "", draw(ops, centers, noise));
    for (size_t i = 0; i < n / 25; ++i) store.remove(NodeRef("doc", std::to_string((i * 17 + 5) % n)));
    dump("writes", store, queries, k);

    for (size_t i = 0; i < n; ++i) {
        if (i % 3 == 1) store.remove(NodeRef("doc", std::to_string(i)));
    }
    store.verifyInvariants();
    dump("compacted", store, queries, k);
    return 0;
}
