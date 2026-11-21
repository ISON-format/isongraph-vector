/**
 * ISONGraph Vector - HNSW approximate index (numerics only)
 *
 * The opt-in graph index behind AnnOptions::mode == AnnMode::Hnsw, beside the
 * IVF-flat one in isongraph_vector_ivf.hpp. A port of the Rust crate's
 * src/hnsw.rs; DESIGN.md section 12 is the specification, and where it says
 * "portable" this file follows it exactly (the level draw, the (score, slot)
 * order, the layout, the plan/apply linking, the heuristics, compaction).
 *
 * Like IvfIndex it knows nothing about node references or the store. It holds
 * slot numbers and links ONLY: every score is computed from the store's one
 * ChunkedMatrix by hnswScore, which returns exactly the float
 * detail::cosineRows would (the same ascending single-precision sums as the
 * exact scan: only the schedule differs, several rows per SIMD pass and the
 * row norms cached instead of re-summed), so a hit's score is bit-identical to
 * the exact scan's, and ties order by slot as everywhere else.
 *
 * Layout per table slot (DESIGN.md 12.2):
 *   levels_[slot]   u8, 255 = not in the graph
 *   l0_             layer 0: one block of 2m+1 u32 (link count, then links) per
 *                   node, in chunks of kChunkRows nodes so growth never copies
 *   upperAt_[slot]  u32 offset into upper_, which holds, for a node of level L,
 *                   L consecutive blocks of m+1 u32 for layers 1..L
 *   norms_[slot]    the row's Euclidean norm, summed exactly as cosineRows sums it
 *                   (so a cached norm and a recomputed one are the same float)
 *   entry_, top_    the entry slot and its level
 *
 * Determinism: a build inserts in slot order in batches of size/16 (1 at the
 * start, at most 512). Every member of a batch is planned against the same
 * frozen graph (in parallel) and the plans are applied in member order, so the
 * graph depends on the seed, the data and the insertion order, never on the
 * number of threads.
 *
 * Thread safety: search() is const and uses only per-thread scratch, so any
 * number may run at once; every other method needs exclusive access, which
 * EmbeddingStore provides with its reader/writer lock.
 */

#ifndef ISONGRAPH_VECTOR_HNSW_HPP
#define ISONGRAPH_VECTOR_HNSW_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "isongraph_vector_ivf.hpp"

namespace isongraph_vector {
namespace detail {

constexpr size_t kHnswMaxCap = 128;             ///< most links a layer holds: 2 * m with m <= 64
constexpr size_t kHnswMaxLevel = 24;
constexpr size_t kHnswBuildBatch = 512;         ///< largest batch planned against one frozen graph
constexpr size_t kHnswBuildRatio = 16;          ///< a batch is at most size / 16 nodes
constexpr uint32_t kHnswNone = std::numeric_limits<uint32_t>::max();
constexpr uint8_t kHnswAbsent = 255;

/** SplitMix64 (DESIGN.md 12.3): integers only, the same on every platform. */
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed = 0) : state_(seed) {}

    uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /** Uniform in [0, n): the high 64 bits of next() * n (a portable 64x64 -> 128 multiply). */
    uint64_t below(uint64_t n) {
        const uint64_t a = next();
        const uint64_t aLo = a & 0xFFFFFFFFull, aHi = a >> 32;
        const uint64_t bLo = n & 0xFFFFFFFFull, bHi = n >> 32;
        const uint64_t ll = aLo * bLo;
        const uint64_t lh = aLo * bHi;
        const uint64_t hl = aHi * bLo;
        const uint64_t hh = aHi * bHi;
        const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
        return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    }

    uint64_t state() const { return state_; }

private:
    uint64_t state_;
};

/** What a layer search may look at: the matrix, which slots are live, their type codes. */
struct HnswRows {
    const ChunkedMatrix* matrix = nullptr;
    const uint8_t* alive = nullptr;
    const uint32_t* codes = nullptr;
    size_t dim = 0;
    /** Per-slot row norms; the index fills this in itself (callers leave it null). */
    const float* norms = nullptr;

    const float* row(uint32_t slot) const { return matrix->row(slot); }
};

/**
 * Euclidean norm of a row (the finiteness test of DESIGN.md 12.4, and the
 * cached norm of the scorers below). The same strict single-precision
 * ascending sum as the norm inside cosineRows, so the two are the same float;
 * out of line for the same reason cosineRows is (see there).
 */
ISONGRAPH_VECTOR_NOINLINE inline float hnswRowNorm(const float* v, size_t n) {
    ISONGRAPH_VECTOR_STRICT_FP
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

inline bool hnswHitWorse(const IvfHit& a, const IvfHit& b) { return hitBefore(b, a); }

// The lane kernel below needs SSE2 (every x86-64 compiler enables it) and the
// <emmintrin.h> that isongraph_vector_ivf.hpp already included for it.
#if defined(ISONGRAPH_VECTOR_SIMD_AVX2) || defined(ISONGRAPH_VECTOR_SIMD_SSE2)
#define ISONGRAPH_VECTOR_HNSW_LANES 1
#endif

/** cos = dot / (|q| * |r|), or 0 when either norm is 0: the closing step of cosineRows. */
inline float hnswCosine(float dot, float qNorm, float rowNorm) {
    return (qNorm == 0.0f || rowNorm == 0.0f) ? 0.0f : dot / (qNorm * rowNorm);
}

#if defined(ISONGRAPH_VECTOR_HNSW_LANES)

/**
 * The dot products of q with 4 * G rows, one row per SIMD lane. Every lane runs
 * the very sequence cosineRows runs for its row - acc = acc + q[i] * r[i], i
 * ascending, a separate multiply and add, no fused multiply-add - so each
 * lane's sum is bit for bit cosineRows's dot; the lanes just advance together.
 * (The alternative, several elements of one row per lane, would change the
 * order of the sum; this does not.) The rows are loaded four elements at a
 * time and transposed in registers, so the matrix keeps its row-major layout.
 * Out of line and contraction-free for the reason cosineRows is.
 */
template <int G>
ISONGRAPH_VECTOR_NOINLINE inline void hnswDotsLanes(const float* q, const float* const* rows, size_t n,
                                                    float* dots /* [4 * G] */) {
    ISONGRAPH_VECTOR_STRICT_FP
    __m128 acc[G];
    for (int g = 0; g < G; ++g) acc[g] = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 b0 = _mm_set1_ps(q[i]);
        const __m128 b1 = _mm_set1_ps(q[i + 1]);
        const __m128 b2 = _mm_set1_ps(q[i + 2]);
        const __m128 b3 = _mm_set1_ps(q[i + 3]);
        for (int g = 0; g < G; ++g) {
            const __m128 a = _mm_loadu_ps(rows[4 * g] + i);
            const __m128 b = _mm_loadu_ps(rows[4 * g + 1] + i);
            const __m128 c = _mm_loadu_ps(rows[4 * g + 2] + i);
            const __m128 d = _mm_loadu_ps(rows[4 * g + 3] + i);
            const __m128 t0 = _mm_unpacklo_ps(a, b);
            const __m128 t1 = _mm_unpacklo_ps(c, d);
            const __m128 t2 = _mm_unpackhi_ps(a, b);
            const __m128 t3 = _mm_unpackhi_ps(c, d);
            __m128 s = acc[g];
            s = _mm_add_ps(s, _mm_mul_ps(b0, _mm_movelh_ps(t0, t1)));   // element i of the four rows
            s = _mm_add_ps(s, _mm_mul_ps(b1, _mm_movehl_ps(t1, t0)));   // i + 1
            s = _mm_add_ps(s, _mm_mul_ps(b2, _mm_movelh_ps(t2, t3)));   // i + 2
            s = _mm_add_ps(s, _mm_mul_ps(b3, _mm_movehl_ps(t3, t2)));   // i + 3
            acc[g] = s;
        }
    }
    for (; i < n; ++i) {
        const __m128 b = _mm_set1_ps(q[i]);
        for (int g = 0; g < G; ++g) {
            const __m128 c = _mm_setr_ps(rows[4 * g][i], rows[4 * g + 1][i], rows[4 * g + 2][i], rows[4 * g + 3][i]);
            acc[g] = _mm_add_ps(acc[g], _mm_mul_ps(b, c));
        }
    }
    for (int g = 0; g < G; ++g) _mm_storeu_ps(dots + 4 * g, acc[g]);
}

#else

/** The dot products of q with four rows: the scalar form of the above (cosineRows4 without the norm sums). */
ISONGRAPH_VECTOR_NOINLINE inline void hnswDots4(const float* q, const float* r0, const float* r1,
                                                const float* r2, const float* r3, size_t n, float* dots) {
    ISONGRAPH_VECTOR_STRICT_FP
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        d0 += q[i] * r0[i];
        d1 += q[i] * r1[i];
        d2 += q[i] * r2[i];
        d3 += q[i] * r3[i];
    }
    dots[0] = d0;
    dots[1] = d1;
    dots[2] = d2;
    dots[3] = d3;
}

ISONGRAPH_VECTOR_NOINLINE inline float hnswDot1(const float* q, const float* r, size_t n) {
    ISONGRAPH_VECTOR_STRICT_FP
    float d = 0.0f;
    for (size_t i = 0; i < n; ++i) d += q[i] * r[i];
    return d;
}

#endif

/**
 * Cosine of the query (norm qNorm) with each of slots[0..n), equal bit for bit
 * to cosineRows(q, row, dim) for every one: the dot is the same ascending
 * single-precision sum (see hnswDotsLanes) and the two norms are the floats
 * cosineRows would sum (qNorm from hnswRowNorm, the row's from the index's
 * cache of the same sum), so only the work differs, not the answer. A NaN
 * score (a row that cannot be compared) ranks last as -infinity.
 */
inline void hnswScore(const HnswRows& rows, const float* q, float qNorm, const uint32_t* slots, size_t n,
                      float* out) {
    size_t i = 0;
#if defined(ISONGRAPH_VECTOR_HNSW_LANES)
    while (i < n) {
        // Eight rows per pass (two lane groups) while there are that many left, else
        // four; a short last pass repeats its final row in the spare lanes.
        const size_t left = n - i;
        const size_t take = std::min<size_t>(left, left > 4 ? 8 : 4);
        const float* ptr[8];
        float dots[8];
        const size_t lanes = take > 4 ? 8 : 4;
        for (size_t j = 0; j < lanes; ++j) ptr[j] = rows.row(slots[i + std::min(j, take - 1)]);
        if (lanes == 8) hnswDotsLanes<2>(q, ptr, rows.dim, dots); else hnswDotsLanes<1>(q, ptr, rows.dim, dots);
        for (size_t j = 0; j < take; ++j) out[i + j] = hnswCosine(dots[j], qNorm, rows.norms[slots[i + j]]);
        i += take;
    }
#else
    for (; i + 4 <= n; i += 4) {
        float dots[4];
        hnswDots4(q, rows.row(slots[i]), rows.row(slots[i + 1]), rows.row(slots[i + 2]), rows.row(slots[i + 3]),
                  rows.dim, dots);
        for (size_t j = 0; j < 4; ++j) out[i + j] = hnswCosine(dots[j], qNorm, rows.norms[slots[i + j]]);
    }
    for (; i < n; ++i) out[i] = hnswCosine(hnswDot1(q, rows.row(slots[i]), rows.dim), qNorm, rows.norms[slots[i]]);
#endif
    for (i = 0; i < n; ++i) {
        if (std::isnan(out[i])) out[i] = -std::numeric_limits<float>::infinity();
    }
}

inline IvfHit hnswScoreOne(const HnswRows& rows, const float* q, float qNorm, uint32_t slot) {
    float s;
    hnswScore(rows, q, qNorm, &slot, 1, &s);
    return IvfHit{s, slot};
}

/** Hint the first two cache lines of a row (a hint only; never reads past the row). */
inline void hnswPrefetch(const float* row, size_t dim) {
    simd::prefetch(row);
    if (dim > 16) simd::prefetch(row + 16);
}

/** Which nodes a layer-0 search may return (the others are only routed through). */
struct HnswAdmit {
    uint32_t code = IvfIndex::kAnyCode;
    float threshold = -std::numeric_limits<float>::infinity();
    uint32_t exclude = kHnswNone;

    bool ok(const HnswRows& rows, uint32_t slot, float score) const {
        return slot != exclude && rows.alive[slot] && score >= threshold &&
               score > -std::numeric_limits<float>::infinity() &&
               (code == IvfIndex::kAnyCode || rows.codes[slot] == code);
    }
};

/**
 * The marks of the nodes a search has touched. One per thread, reused: a query
 * clears only the words it set, so it costs what it visited, and the memory is
 * one bit per slot per searching thread.
 */
struct HnswVisited {
    std::vector<uint64_t> bits;
    std::vector<uint32_t> words;

    void begin(size_t slots) {
        clear();
        const size_t need = (slots + 63) / 64;
        if (bits.size() < need) bits.resize(need, 0);
    }

    void clear() {
        for (uint32_t w : words) bits[w] = 0;
        words.clear();
    }

    /** Mark `slot`; true if it already was. */
    bool testSet(uint32_t slot) {
        const size_t w = slot >> 6;
        const uint64_t mask = uint64_t(1) << (slot & 63u);
        const uint64_t cur = bits[w];
        if (cur & mask) return true;
        if (cur == 0) words.push_back(static_cast<uint32_t>(w));
        bits[w] = cur | mask;
        return false;
    }
};

inline HnswVisited& hnswThreadVisited() {
    thread_local HnswVisited visited;
    return visited;
}

/** Borrow this thread's visited set for one operation (cleared on the way out, even by an exception). */
class HnswVisitedScope {
public:
    explicit HnswVisitedScope(size_t slots) : v_(hnswThreadVisited()) { v_.begin(slots); }
    ~HnswVisitedScope() { v_.clear(); }
    HnswVisitedScope(const HnswVisitedScope&) = delete;
    HnswVisitedScope& operator=(const HnswVisitedScope&) = delete;
    HnswVisited& get() { return v_; }

private:
    HnswVisited& v_;
};

/** What a layer search returns. */
struct HnswFound {
    std::vector<IvfHit> hits;   ///< admitted nodes, best first
    bool budgetHit = false;     ///< gave up after scoring too many nodes (a filter so selective that scanning is cheaper)
};

/** `ef_search` by store size when none is configured (DESIGN.md 12.1). */
inline size_t hnswAutoEf(size_t live) {
    if (live < 10000) return 64;
    if (live < 200000) return 96;
    if (live < 600000) return 128;
    return 192;
}

class HnswIndex {
public:
    size_t churn = 0;               ///< updates + deletes since the build
    double buildSeconds = 0.0;
    uint64_t inserts = 0;
    double insertSeconds = 0.0;

    HnswIndex(size_t m, size_t efConstruction, uint64_t seed)
        : m_(m), m0_(2 * m), efC_(std::max(efConstruction, m)),
          rng_(seed ^ 0x4857535748575357ull) {}

    size_t members() const { return members_; }
    size_t topLevel() const { return top_; }
    size_t slots() const { return levels_.size(); }

    /** Updates plus deletes since the build beyond the node count call for a rebuild. */
    bool stale() const {
        return static_cast<double>(churn) > 1.0 * static_cast<double>(std::max<size_t>(members_, 1));
    }

    /** Nodes whose rows are tombstones. */
    size_t dead(const HnswRows& rows) const {
        size_t d = 0;
        for (size_t s = 0; s < levels_.size(); ++s) {
            if (levels_[s] != kHnswAbsent && !rows.alive[s]) ++d;
        }
        return d;
    }

    /** Bytes held by the graph. It holds no copy of any vector. */
    size_t bytes() const {
        size_t total = levels_.capacity() + upperAt_.capacity() * sizeof(uint32_t) +
                       upper_.capacity() * sizeof(uint32_t) + norms_.capacity() * sizeof(float);
        for (const auto& chunk : l0_) total += chunk.capacity() * sizeof(uint32_t);
        return total;
    }

    /** A graph over every live row with a finite norm, inserted in slot order. */
    static HnswIndex build(const HnswRows& rows, size_t slotCount, size_t m, size_t efConstruction,
                           uint64_t seed, size_t threads) {
        const auto started = std::chrono::steady_clock::now();
        HnswIndex h(m, efConstruction, seed);
        h.levels_.reserve(slotCount);
        h.upperAt_.reserve(slotCount);
        h.norms_.reserve(slotCount);
        if (slotCount > 0) h.ensureSlot(slotCount - 1);
        std::vector<uint32_t> order;
        order.reserve(slotCount);
        for (size_t s = 0; s < slotCount; ++s) {
            if (!rows.alive[s]) continue;
            const float norm = hnswRowNorm(rows.row(static_cast<uint32_t>(s)), rows.dim);
            h.norms_[s] = norm;
            if (std::isfinite(norm)) order.push_back(static_cast<uint32_t>(s));
        }
        size_t at = 0;
        std::vector<Job> jobs;
        while (at < order.size()) {
            const size_t size = std::min(std::max<size_t>(at / kHnswBuildRatio, 1), kHnswBuildBatch);
            const size_t end = std::min(at + size, order.size());
            jobs.clear();
            for (size_t i = at; i < end; ++i) {
                const uint8_t level = h.drawLevel();
                jobs.push_back(Job{order[i], level, false});
            }
            h.link(rows, jobs, threads);
            at = end;
        }
        h.buildSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return h;
    }

    /** A row was written (new, or its vector replaced); link it now. */
    void notePut(const HnswRows& rows, uint32_t slot) {
        const auto started = std::chrono::steady_clock::now();
        ensureSlot(slot);
        const float norm = hnswRowNorm(rows.row(slot), rows.dim);
        norms_[slot] = norm;
        if (!std::isfinite(norm)) {
            if (inGraph(slot)) ++churn;
            return;
        }
        Job job{slot, 0, false};
        if (inGraph(slot)) {
            ++churn;
            job.level = levels_[slot];
            job.relink = true;
        } else {
            job.level = drawLevel();
        }
        std::vector<Job> one(1, job);
        link(rows, one, 1);
        ++inserts;
        insertSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    /** A row was deleted: the node stays (routes through it keep working) and counts as churn. */
    void noteDrop() { ++churn; }

    /**
     * Follow the table's compaction: `remap[old]` is the new slot (kHnswNone for
     * a tombstone), `kept` the new slot count. Tombstones leave the graph, every
     * link is renumbered, and a node that lost links takes up the links of the
     * tombstones it pointed to ("bridging"), without computing a distance.
     */
    void compact(const std::vector<uint32_t>& remap, size_t kept) {
        const size_t oldN = levels_.size();
        std::vector<uint32_t> newOf(oldN, kHnswNone);
        for (size_t old = 0; old < oldN && old < remap.size(); ++old) newOf[old] = remap[old];

        HnswIndex fresh(*this, kEmptyCopy);
        if (kept > 0) fresh.ensureSlot(kept - 1);
        std::vector<uint32_t> lost, out;
        for (size_t old = 0; old < remap.size(); ++old) {
            const uint32_t nw = remap[old];
            if (nw == kHnswNone) continue;
            const uint8_t level = old < oldN ? levels_[old] : kHnswAbsent;
            // Only live rows survive; a tombstone has no new slot, so every
            // surviving node is either a live graph member or absent.
            if (level == kHnswAbsent) continue;
            fresh.addNode(nw, level);
            fresh.norms_[nw] = norms_[old];
            for (size_t l = 0; l <= level; ++l) {
                const size_t cap = capacity(l);
                out.clear();
                lost.clear();
                const Links own = list(old, l);
                for (uint32_t i = 0; i < own.count; ++i) {
                    const uint32_t nb = own.items[i];
                    const uint32_t mapped = nb < oldN ? newOf[nb] : kHnswNone;
                    if (mapped != kHnswNone) out.push_back(mapped); else lost.push_back(nb);
                }
                for (uint32_t d : lost) {
                    if (out.size() >= cap) break;
                    if (d >= oldN || levels_[d] == kHnswAbsent || levels_[d] < l) continue;
                    const Links theirs = list(d, l);
                    bool full = false;
                    for (uint32_t i = 0; i < theirs.count && !full; ++i) {
                        const uint32_t w = theirs.items[i];
                        const uint32_t nwLink = w < oldN ? newOf[w] : kHnswNone;
                        if (nwLink != kHnswNone && nwLink != nw &&
                            std::find(out.begin(), out.end(), nwLink) == out.end()) {
                            out.push_back(nwLink);
                            if (out.size() >= cap) full = true;
                        }
                    }
                    if (full) break;
                }
                fresh.setList(nw, l, out.data(), out.size());
            }
            if (fresh.entry_ == kHnswNone || level > fresh.top_) {
                fresh.entry_ = nw;
                fresh.top_ = level;
            }
        }
        *this = std::move(fresh);
    }

    /**
     * Top `ef` admitted nodes for a query, best first (ties by slot). Greedy
     * descent through the upper layers, then a best-first search of layer 0.
     * Gives up (budgetHit) after scoring more than `budget` nodes.
     */
    void search(const HnswRows& given, const float* q, uint32_t code, float threshold, size_t ef,
                size_t budget, HnswFound& found) const {
        found.hits.clear();
        found.budgetHit = false;
        if (entry_ == kHnswNone) return;
        const HnswRows rows = withNorms(given);
        const float qNorm = hnswRowNorm(q, rows.dim);
        HnswVisitedScope scope(levels_.size());
        IvfHit cur = hnswScoreOne(rows, q, qNorm, entry_);
        for (size_t level = top_; level >= 1; --level) cur = greedy(rows, q, qNorm, cur, level);
        HnswAdmit admit;
        admit.code = code;
        admit.threshold = threshold;
        const std::vector<IvfHit> eps(1, cur);
        searchLayer(rows, q, qNorm, eps, ef, 0, admit, budget, scope.get(), found);
    }

    /**
     * Every invariant of DESIGN.md 12.6 against the table it indexes; an empty
     * string when they all hold, else a description of the first that fails.
     */
    std::string check(const HnswRows& rows, size_t tableSlots) const {
        const size_t n = tableSlots;
        if (levels_.size() != n || upperAt_.size() != n) {
            return "graph covers " + std::to_string(levels_.size()) + " rows, table has " + std::to_string(n);
        }
        if (l0_.size() != (n + kChunkRows - 1) / kChunkRows) {
            return std::to_string(l0_.size()) + " link chunks for " + std::to_string(n) + " rows";
        }
        size_t members = 0;
        size_t maxLevel = 0;
        std::vector<uint32_t> seen;
        for (size_t s = 0; s < n; ++s) {
            const uint8_t level = levels_[s];
            if (level == kHnswAbsent) {
                if (rows.alive[s] && std::isfinite(hnswRowNorm(rows.row(static_cast<uint32_t>(s)), rows.dim))) {
                    return "live slot " + std::to_string(s) + " is not in the graph";
                }
                continue;
            }
            ++members;
            maxLevel = std::max<size_t>(maxLevel, level);
            for (size_t l = 0; l <= level; ++l) {
                const Links links = list(s, l);
                if (links.count > capacity(l)) {
                    return "slot " + std::to_string(s) + " layer " + std::to_string(l) + " holds " +
                           std::to_string(links.count) + " links";
                }
                seen.assign(links.items, links.items + links.count);
                std::sort(seen.begin(), seen.end());
                if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
                    return "slot " + std::to_string(s) + " layer " + std::to_string(l) + " links a node twice";
                }
                for (uint32_t i = 0; i < links.count; ++i) {
                    const uint32_t nb = links.items[i];
                    if (nb >= n || nb == s) {
                        return "slot " + std::to_string(s) + " layer " + std::to_string(l) + " links " +
                               std::to_string(nb);
                    }
                    if (levels_[nb] == kHnswAbsent || levels_[nb] < l) {
                        return "slot " + std::to_string(s) + " layer " + std::to_string(l) + " links " +
                               std::to_string(nb) + ", which is not on that layer";
                    }
                }
            }
        }
        if (members != members_) {
            return "member counter " + std::to_string(members_) + " but " + std::to_string(members) + " nodes";
        }
        if (members == 0) {
            if (entry_ != kHnswNone) return "an empty graph has an entry point";
        } else if (entry_ == kHnswNone || levels_[entry_] != top_ || maxLevel != top_) {
            return "entry " + std::to_string(entry_) + " does not sit on top level " +
                   std::to_string(top_) + " (highest node " + std::to_string(maxLevel) + ")";
        }
        return std::string();
    }

    /** Layer-0 links of `slot` (for tests of reachability). */
    std::vector<uint32_t> layer0(uint32_t slot) const {
        const Links l = list(slot, 0);
        return std::vector<uint32_t>(l.items, l.items + l.count);
    }

    uint32_t entry() const { return entry_; }

    /** Does `other` hold exactly the same graph (levels, links, entry)? */
    bool sameGraphAs(const HnswIndex& other) const {
        return levels_ == other.levels_ && l0_ == other.l0_ && upperAt_ == other.upperAt_ &&
               upper_ == other.upper_ && entry_ == other.entry_ && top_ == other.top_ &&
               members_ == other.members_;
    }

    /** The graph's links as one flat vector (for comparing two ports bit for bit). */
    std::vector<uint32_t> dumpLinks() const {
        std::vector<uint32_t> out;
        for (size_t s = 0; s < levels_.size(); ++s) {
            if (levels_[s] == kHnswAbsent) continue;
            for (size_t l = 0; l <= levels_[s]; ++l) {
                const Links links = list(s, l);
                out.push_back(static_cast<uint32_t>(s));
                out.push_back(static_cast<uint32_t>(l));
                out.push_back(links.count);
                for (uint32_t i = 0; i < links.count; ++i) out.push_back(links.items[i]);
            }
        }
        return out;
    }

private:
    struct Job {
        uint32_t slot;
        uint8_t level;
        bool relink;
    };

    struct Links {
        const uint32_t* items;
        uint32_t count;
    };

    struct Add {
        uint8_t layer;
        uint32_t target;
        uint32_t source;
    };

    using Plan = std::vector<std::vector<uint32_t>>;

    struct EmptyCopy {};
    static constexpr EmptyCopy kEmptyCopy{};

    /** A graph with `src`'s settings and counters but no nodes (compaction builds into one). */
    HnswIndex(const HnswIndex& src, EmptyCopy)
        : churn(src.churn), buildSeconds(src.buildSeconds), inserts(src.inserts),
          insertSeconds(src.insertSeconds), m_(src.m_), m0_(src.m0_), efC_(src.efC_),
          rng_(src.rng_) {}

public:
    HnswIndex(HnswIndex&&) = default;
    HnswIndex& operator=(HnswIndex&&) = default;

private:
    size_t m_, m0_, efC_;
    SplitMix64 rng_;
    std::vector<uint8_t> levels_;
    std::vector<std::vector<uint32_t>> l0_;     ///< kChunkRows nodes per chunk, m0 + 1 words each
    std::vector<uint32_t> upperAt_;
    std::vector<uint32_t> upper_;
    std::vector<float> norms_;                  ///< per slot; NaN until the row is written (see the layout above)
    uint32_t entry_ = kHnswNone;
    uint8_t top_ = 0;
    size_t members_ = 0;

    size_t capacity(size_t level) const { return level == 0 ? m0_ : m_; }

    /** `rows` with this graph's cached norms attached (callers hand in a view without them). */
    HnswRows withNorms(HnswRows rows) const {
        rows.norms = norms_.data();
        return rows;
    }

    /** Cover slot `s` (new slots start absent, with empty link blocks). */
    void ensureSlot(size_t s) {
        while (levels_.size() <= s) {
            levels_.push_back(kHnswAbsent);
            upperAt_.push_back(kHnswNone);
            norms_.push_back(std::numeric_limits<float>::quiet_NaN());
        }
        while (l0_.size() <= s / kChunkRows) l0_.emplace_back(kChunkRows * (m0_ + 1), 0u);
    }

    uint32_t* block(size_t s, size_t level) {
        if (level == 0) return l0_[s / kChunkRows].data() + (s % kChunkRows) * (m0_ + 1);
        return upper_.data() + upperAt_[s] + (level - 1) * (m_ + 1);
    }
    const uint32_t* block(size_t s, size_t level) const {
        if (level == 0) return l0_[s / kChunkRows].data() + (s % kChunkRows) * (m0_ + 1);
        return upper_.data() + upperAt_[s] + (level - 1) * (m_ + 1);
    }

    /** The links of `s` on `level` (`s` must have that layer). */
    Links list(size_t s, size_t level) const {
        const uint32_t* b = block(s, level);
        return Links{b + 1, b[0]};
    }

    void setList(size_t s, size_t level, const uint32_t* items, size_t count) {
        uint32_t* b = block(s, level);
        b[0] = static_cast<uint32_t>(count);
        std::copy(items, items + count, b + 1);
    }

    void addNode(size_t s, uint8_t level) {
        ensureSlot(s);
        levels_[s] = level;
        if (level >= 1) {
            upperAt_[s] = static_cast<uint32_t>(upper_.size());
            upper_.resize(upper_.size() + static_cast<size_t>(level) * (m_ + 1), 0u);
        }
        ++members_;
    }

    /** P(level >= l) = m^-l, from integer draws only (portable, no logarithm). */
    uint8_t drawLevel() {
        size_t level = 0;
        while (level < kHnswMaxLevel && rng_.below(m_) == 0) ++level;
        return static_cast<uint8_t>(level);
    }

    bool inGraph(size_t s) const { return s < levels_.size() && levels_[s] != kHnswAbsent; }

    // -- search -----------------------------------------------------------------

    /** Walk to the best node reachable by single steps on `level`. */
    IvfHit greedy(const HnswRows& rows, const float* q, float qNorm, IvfHit cur, size_t level) const {
        float scores[kHnswMaxCap];
        for (;;) {
            const Links links = list(cur.slot, level);
            if (links.count == 0) return cur;
            for (uint32_t i = 0; i < links.count; ++i) hnswPrefetch(rows.row(links.items[i]), rows.dim);
            hnswScore(rows, q, qNorm, links.items, links.count, scores);
            IvfHit best = cur;
            for (uint32_t i = 0; i < links.count; ++i) {
                const IvfHit c{scores[i], links.items[i]};
                if (hitBefore(c, best)) best = c;
            }
            if (best.slot == cur.slot) return cur;
            cur = best;
        }
    }

    /**
     * Best-first search of one layer from `eps`, keeping the `ef` best admitted
     * nodes. Tombstones and non-admitted nodes are routed through but never
     * kept. Stops when the best unexpanded node is worse than the worst kept one
     * (once `ef` are kept), or after `budget` nodes were scored.
     */
    void searchLayer(const HnswRows& rows, const float* q, float qNorm, const std::vector<IvfHit>& eps, size_t ef,
                     size_t level, const HnswAdmit& admit, size_t budget, HnswVisited& visited,
                     HnswFound& found) const {
        visited.clear();
        std::vector<IvfHit> cand;     // max-heap: the best candidate on top
        std::vector<IvfHit> kept;     // min-heap: the worst kept node on top
        cand.reserve(ef * 2 + 16);
        kept.reserve(ef + 1);
        auto keep = [&](const IvfHit& node) {
            kept.push_back(node);
            std::push_heap(kept.begin(), kept.end(), hitBefore);
            if (kept.size() > ef) {
                std::pop_heap(kept.begin(), kept.end(), hitBefore);
                kept.pop_back();
            }
        };
        for (const IvfHit& e : eps) {
            if (!visited.testSet(e.slot)) {
                cand.push_back(e);
                std::push_heap(cand.begin(), cand.end(), hnswHitWorse);
                if (admit.ok(rows, e.slot, e.score)) keep(e);
            }
        }
        uint32_t fresh[kHnswMaxCap];
        float scores[kHnswMaxCap];
        size_t scored = 0;
        found.budgetHit = false;
        while (!cand.empty()) {
            std::pop_heap(cand.begin(), cand.end(), hnswHitWorse);
            const IvfHit c = cand.back();
            cand.pop_back();
            if (kept.size() >= ef && hitBefore(kept.front(), c)) break;   // c is worse than the worst kept
            size_t nFresh = 0;
            const Links links = list(c.slot, level);
            for (uint32_t i = 0; i < links.count; ++i) {
                const uint32_t x = links.items[i];
                if (!visited.testSet(x)) {
                    fresh[nFresh++] = x;
                    hnswPrefetch(rows.row(x), rows.dim);
                }
            }
            if (nFresh == 0) continue;
            hnswScore(rows, q, qNorm, fresh, nFresh, scores);
            scored += nFresh;
            for (size_t i = 0; i < nFresh; ++i) {
                const IvfHit node{scores[i], fresh[i]};
                const bool promising = kept.size() < ef || hitBefore(node, kept.front());
                if (promising) {
                    cand.push_back(node);
                    std::push_heap(cand.begin(), cand.end(), hnswHitWorse);
                    if (admit.ok(rows, node.slot, node.score)) keep(node);
                }
            }
            if (scored > budget) {
                found.budgetHit = true;
                break;
            }
        }
        std::sort(kept.begin(), kept.end(), hitBefore);
        found.hits.assign(kept.begin(), kept.end());
    }

    // -- linking ------------------------------------------------------------------

    /**
     * The diversity heuristic: walk `cands` (best first) and keep one only if it
     * is closer to the base than to every node already kept, until `cap` are
     * kept. Fewer than `cap` candidates are all kept.
     */
    std::vector<uint32_t> select(const HnswRows& rows, const std::vector<IvfHit>& cands,
                                 size_t cap) const {
        std::vector<uint32_t> chosen;
        if (cands.size() <= cap) {
            chosen.reserve(cands.size());
            for (const IvfHit& c : cands) chosen.push_back(c.slot);
            return chosen;
        }
        chosen.reserve(cap);
        float scores[kHnswMaxCap];
        for (const IvfHit& c : cands) {
            if (chosen.size() >= cap) break;
            hnswScore(rows, rows.row(c.slot), rows.norms[c.slot], chosen.data(), chosen.size(), scores);
            bool rejected = false;
            for (size_t i = 0; i < chosen.size(); ++i) {
                if (scores[i] > c.score) { rejected = true; break; }
            }
            if (!rejected) chosen.push_back(c.slot);
        }
        return chosen;
    }

    /**
     * Neighbours chosen for `job` on every layer, against the graph as it is now
     * plus the other new members of its batch (`intra`).
     */
    Plan plan(const HnswRows& rows, const Job& job, const std::vector<std::pair<uint32_t, uint8_t>>& intra,
              HnswVisited& visited) const {
        const uint32_t x = job.slot;
        const size_t level = job.level;
        const float* q = rows.row(x);
        const float qNorm = rows.norms[x];
        HnswAdmit admit;
        admit.exclude = x;
        const size_t top = top_;
        Plan lists(level + 1);
        std::vector<IvfHit> eps;
        if (entry_ != kHnswNone) {
            IvfHit cur = hnswScoreOne(rows, q, qNorm, entry_);
            for (size_t l = top; l > level; --l) cur = greedy(rows, q, qNorm, cur, l);
            eps.push_back(cur);
        }
        std::vector<uint32_t> members;
        std::vector<float> memberScores;
        HnswFound found;
        for (size_t li = level + 1; li-- > 0;) {
            std::vector<IvfHit> cands;
            if (li <= top && !eps.empty()) {
                searchLayer(rows, q, qNorm, eps, efC_, li, admit, std::numeric_limits<size_t>::max(), visited, found);
                cands = found.hits;
                if (!cands.empty()) eps = cands;
            }
            members.clear();
            for (const auto& entry : intra) {
                if (entry.first != x && entry.second >= li) members.push_back(entry.first);
            }
            if (!members.empty()) {
                memberScores.assign(members.size(), 0.0f);
                hnswScore(rows, q, qNorm, members.data(), members.size(), memberScores.data());
                for (size_t i = 0; i < members.size(); ++i) cands.push_back(IvfHit{memberScores[i], members[i]});
                std::sort(cands.begin(), cands.end(), hitBefore);
            }
            lists[li] = select(rows, cands, m_);
        }
        return lists;
    }

    /**
     * `t`'s list on `layer` with `incoming` added, pruned to the layer's
     * capacity when it overflows (tombstones are dropped then). False if
     * nothing changes.
     */
    bool grownList(const HnswRows& rows, size_t layer, uint32_t t, const std::vector<uint32_t>& incoming,
                   std::vector<uint32_t>& all) const {
        const size_t cap = capacity(layer);
        const Links cur = list(t, layer);
        all.assign(cur.items, cur.items + cur.count);
        for (uint32_t n : incoming) {
            if (n != t && std::find(all.begin(), all.end(), n) == all.end()) all.push_back(n);
        }
        if (all.size() == cur.count) return false;
        if (all.size() <= cap) return true;
        all.erase(std::remove_if(all.begin(), all.end(), [&](uint32_t s) { return !rows.alive[s]; }),
                  all.end());
        if (all.size() <= cap) return true;
        std::vector<float> scores(all.size());
        hnswScore(rows, rows.row(t), rows.norms[t], all.data(), all.size(), scores.data());
        std::vector<IvfHit> cands;
        cands.reserve(all.size());
        for (size_t i = 0; i < all.size(); ++i) cands.push_back(IvfHit{scores[i], all[i]});
        std::sort(cands.begin(), cands.end(), hitBefore);
        all = select(rows, cands, cap);
        return true;
    }

    /** Tasks per worker below which another thread is not worth starting. */
    static size_t workersFor(size_t tasks, size_t threads) {
        if (threads == 0) threads = autoThreads();
        return std::max<size_t>(1, std::min(threads, tasks / 2));
    }

    /**
     * Link `jobs` into the graph: plan every job against the current graph (in
     * parallel, read-only), then apply the plans in job order. The result does
     * not depend on `threads`.
     */
    void link(const HnswRows& given, const std::vector<Job>& jobs, size_t threads) {
        const HnswRows rows = withNorms(given);
        const size_t slots = levels_.size();
        std::vector<std::pair<uint32_t, uint8_t>> intra;
        for (const Job& j : jobs) {
            if (!j.relink) intra.emplace_back(j.slot, j.level);
        }

        std::vector<Plan> plans(jobs.size());
        {
            const HnswIndex& self = *this;
            const size_t workers = workersFor(jobs.size(), threads);
            std::atomic<size_t> next{0};
            runWorkers(workers, [&](size_t) {
                HnswVisitedScope scope(slots);
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size()) break;
                    plans[i] = self.plan(rows, jobs[i], intra, scope.get());
                }
            });
        }

        // forward links
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (!jobs[i].relink) addNode(jobs[i].slot, jobs[i].level);
            for (size_t l = 0; l < plans[i].size(); ++l) {
                setList(jobs[i].slot, l, plans[i][l].data(), plans[i][l].size());
            }
        }

        // backward links, grouped by (layer, target) in a fixed order
        std::vector<Add> adds;
        for (size_t i = 0; i < jobs.size(); ++i) {
            for (size_t l = 0; l < plans[i].size(); ++l) {
                for (uint32_t nb : plans[i][l]) {
                    if (rows.alive[nb]) adds.push_back(Add{static_cast<uint8_t>(l), nb, jobs[i].slot});
                }
            }
        }
        std::stable_sort(adds.begin(), adds.end(), [](const Add& a, const Add& b) {
            return a.layer != b.layer ? a.layer < b.layer : a.target < b.target;
        });
        std::vector<std::pair<size_t, size_t>> groups;   // [start, end) into adds
        for (size_t at = 0; at < adds.size();) {
            size_t end = at + 1;
            while (end < adds.size() && adds[end].layer == adds[at].layer && adds[end].target == adds[at].target) ++end;
            groups.emplace_back(at, end);
            at = end;
        }
        std::vector<std::vector<uint32_t>> grown(groups.size());
        std::vector<uint8_t> changed(groups.size(), 0);
        {
            const HnswIndex& self = *this;
            const size_t workers = workersFor(groups.size(), threads);
            std::atomic<size_t> next{0};
            runWorkers(workers, [&](size_t) {
                std::vector<uint32_t> incoming;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= groups.size()) break;
                    incoming.clear();
                    for (size_t a = groups[i].first; a < groups[i].second; ++a) incoming.push_back(adds[a].source);
                    changed[i] = self.grownList(rows, adds[groups[i].first].layer,
                                                adds[groups[i].first].target, incoming, grown[i]) ? 1 : 0;
                }
            });
        }
        for (size_t i = 0; i < groups.size(); ++i) {
            if (changed[i]) {
                setList(adds[groups[i].first].target, adds[groups[i].first].layer, grown[i].data(), grown[i].size());
            }
        }

        // the entry point is the first new node of the highest level
        for (const Job& j : jobs) {
            if (j.relink) continue;
            if (entry_ == kHnswNone || j.level > top_) {
                entry_ = j.slot;
                top_ = j.level;
            }
        }
    }
};

}  // namespace detail
}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_HNSW_HPP
