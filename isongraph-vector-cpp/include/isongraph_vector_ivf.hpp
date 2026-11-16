/**
 * ISONGraph Vector - IVF-flat approximate index (numerics only)
 *
 * The inverted-file index behind EmbeddingStore::enableAnn(). It knows nothing
 * about node references or the store: it holds rows of floats, addressed by
 * slot number, and answers "which slots score highest against this query".
 * isongraph_vector.hpp owns the slot <-> record mapping and the policy
 * (when to build, fold, retrain); this file is the arithmetic.
 *
 * Layout (all contiguous, nothing allocated per query):
 *   data       slots x dim floats, the vectors as given (not normalised)
 *   invNorm    1 / |row| per slot (0 for a zero row, which then scores 0)
 *   assign     cell of each slot, -1 = in no cell yet (new/updated) or dead
 *   order      the slots of every cell, grouped by cell
 *   offsets    order[offsets[c] .. offsets[c+1]) are the slots of cell c
 *   centroids  nlist x dim unit vectors
 *
 * Training is spherical k-means (centroids are unit vectors and rows are
 * compared by direction, which is what cosine similarity measures) on a
 * sample of at most 40 x nlist rows, 8 iterations, with a seeded Mersenne
 * Twister. For a given build the result does not depend on the number of
 * threads used: every row's cell is independent of the others, and a
 * centroid sums its rows in ascending row order.
 *
 * Not thread-safe by itself; EmbeddingStore serialises access.
 */

#ifndef ISONGRAPH_VECTOR_IVF_HPP
#define ISONGRAPH_VECTOR_IVF_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <system_error>
#include <thread>
#include <vector>

// The kernels use SIMD where the compiler enables it: AVX2+FMA (compile with
// -mavx2 -mfma, /arch:AVX2 or -march=native), else SSE2 (every x86-64), else
// NEON (AArch64), else plain scalar code. Define ISONGRAPH_VECTOR_NO_SIMD to
// force the scalar path.
#if defined(ISONGRAPH_VECTOR_NO_SIMD)
// scalar
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))  // MSVC has no __FMA__; /arch:AVX2 allows FMA
#include <immintrin.h>
#define ISONGRAPH_VECTOR_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define ISONGRAPH_VECTOR_SIMD_SSE2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ISONGRAPH_VECTOR_SIMD_NEON 1
#endif

namespace isongraph_vector {
namespace detail {

// =============================================================================
// SIMD lane abstraction. One implementation of each kernel serves every
// backend; the scalar fallback is the same code with one-float lanes.
// =============================================================================

namespace simd {

#if defined(ISONGRAPH_VECTOR_SIMD_AVX2)
using V = __m256;
constexpr size_t W = 8;
inline V zero() { return _mm256_setzero_ps(); }
inline V load(const float* p) { return _mm256_loadu_ps(p); }
inline V fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
inline V add(V a, V b) { return _mm256_add_ps(a, b); }
inline float hsum(V v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
inline void prefetch(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0); }
#elif defined(ISONGRAPH_VECTOR_SIMD_SSE2)
using V = __m128;
constexpr size_t W = 4;
inline V zero() { return _mm_setzero_ps(); }
inline V load(const float* p) { return _mm_loadu_ps(p); }
inline V fma(V a, V b, V c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
inline V add(V a, V b) { return _mm_add_ps(a, b); }
inline float hsum(V v) {
    __m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
inline void prefetch(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0); }
#elif defined(ISONGRAPH_VECTOR_SIMD_NEON)
using V = float32x4_t;
constexpr size_t W = 4;
inline V zero() { return vdupq_n_f32(0.0f); }
inline V load(const float* p) { return vld1q_f32(p); }
inline V fma(V a, V b, V c) { return vmlaq_f32(c, a, b); }
inline V add(V a, V b) { return vaddq_f32(a, b); }
inline float hsum(V v) { return vaddvq_f32(v); }
inline void prefetch(const float*) {}
#else
using V = float;
constexpr size_t W = 1;
inline V zero() { return 0.0f; }
inline V load(const float* p) { return *p; }
inline V fma(V a, V b, V c) { return a * b + c; }
inline V add(V a, V b) { return a + b; }
inline float hsum(V v) { return v; }
inline void prefetch(const float*) {}
#endif

/** Dot product of two n-float rows. Fixed summation order for a given backend. */
inline float dot(const float* a, const float* b, size_t n) {
    V s0 = zero(), s1 = zero(), s2 = zero(), s3 = zero();
    size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        s0 = fma(load(a + i), load(b + i), s0);
        s1 = fma(load(a + i + W), load(b + i + W), s1);
        s2 = fma(load(a + i + 2 * W), load(b + i + 2 * W), s2);
        s3 = fma(load(a + i + 3 * W), load(b + i + 3 * W), s3);
    }
    for (; i + W <= n; i += W) s0 = fma(load(a + i), load(b + i), s0);
    float total = hsum(add(add(s0, s1), add(s2, s3)));
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

/**
 * Four rows against two centroids at once (eight dot products): each centroid
 * chunk is loaded once for four rows, which is what makes assigning a million
 * rows to thousands of cells compute-bound rather than memory-bound.
 * `rows` points at four consecutive rows of length `n`.
 */
inline void tile4x2(const float* rows, size_t n, const float* c0, const float* c1,
                    float* out /* [4][2] */) {
    const float* r0 = rows;
    const float* r1 = rows + n;
    const float* r2 = rows + 2 * n;
    const float* r3 = rows + 3 * n;
    V a00 = zero(), a01 = zero(), a10 = zero(), a11 = zero();
    V a20 = zero(), a21 = zero(), a30 = zero(), a31 = zero();
    size_t i = 0;
    for (; i + W <= n; i += W) {
        V b0 = load(c0 + i);
        V b1 = load(c1 + i);
        V x0 = load(r0 + i);
        V x1 = load(r1 + i);
        V x2 = load(r2 + i);
        V x3 = load(r3 + i);
        a00 = fma(x0, b0, a00); a01 = fma(x0, b1, a01);
        a10 = fma(x1, b0, a10); a11 = fma(x1, b1, a11);
        a20 = fma(x2, b0, a20); a21 = fma(x2, b1, a21);
        a30 = fma(x3, b0, a30); a31 = fma(x3, b1, a31);
    }
    out[0] = hsum(a00); out[1] = hsum(a01);
    out[2] = hsum(a10); out[3] = hsum(a11);
    out[4] = hsum(a20); out[5] = hsum(a21);
    out[6] = hsum(a30); out[7] = hsum(a31);
    for (; i < n; ++i) {
        out[0] += r0[i] * c0[i]; out[1] += r0[i] * c1[i];
        out[2] += r1[i] * c0[i]; out[3] += r1[i] * c1[i];
        out[4] += r2[i] * c0[i]; out[5] += r2[i] * c1[i];
        out[6] += r3[i] * c0[i]; out[7] += r3[i] * c1[i];
    }
}

}  // namespace simd

// =============================================================================
// Small helpers
// =============================================================================

/** Run fn(0..tasks) on up to `threads` threads; serial when that is 1 or the work is tiny. */
template <class F>
void parallelFor(size_t tasks, size_t threads, F&& fn) {
    if (tasks == 0) return;
    if (threads > tasks) threads = tasks;
    if (threads <= 1) {
        for (size_t i = 0; i < tasks; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::atomic<bool> failed{false};
    auto worker = [&]() {
        try {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= tasks || failed.load()) break;
                fn(i);
            }
        } catch (...) {
            if (!failed.exchange(true)) failure = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    try {
        for (size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    } catch (const std::system_error&) {
        // No threads available (for example a static build without -pthread):
        // the caller's thread does the rest, same result.
    }
    worker();
    for (auto& t : pool) t.join();
    if (failure) std::rethrow_exception(failure);
}

inline size_t autoThreads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

/** One search hit: higher score first, lower slot first on a tie. */
struct IvfHit {
    float score;
    uint32_t slot;
};

inline bool hitBefore(const IvfHit& a, const IvfHit& b) {
    return a.score > b.score || (a.score == b.score && a.slot < b.slot);
}

// =============================================================================
// The index
// =============================================================================

class IvfIndex {
public:
    static constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();

    size_t dim = 0;
    size_t nlist = 0;
    size_t trainedN = 0;       ///< live rows when the centroids were trained
    size_t churn = 0;          ///< writes + deletes since then
    size_t pendingLive = 0;    ///< live rows not in any cell yet
    double buildSeconds = 0.0;

    size_t slots() const { return alive_.size(); }
    size_t live() const { return alive_.size() - dead_; }
    size_t dead() const { return dead_; }

    /**
     * Train on `n` rows laid out contiguously in `data` (slot i = row i, all
     * alive), then put every row in its nearest cell. `codes` gives each row
     * a small integer type code for filtered searches.
     */
    static IvfIndex build(size_t dim, std::vector<float>&& data, std::vector<uint32_t>&& codes,
                          size_t nlistWanted, uint64_t seed, size_t threads, int iterations = 8) {
        auto started = std::chrono::steady_clock::now();
        IvfIndex index;
        index.dim = dim;
        index.data_ = std::move(data);
        index.codes_ = std::move(codes);
        const size_t n = index.codes_.size();
        index.alive_.assign(n, 1);
        index.pend_.assign(n, 0);
        index.assign_.assign(n, -1);
        index.invNorm_.resize(n);
        if (threads == 0) threads = autoThreads();
        parallelFor((n + 8191) / 8192, threads, [&](size_t block) {
            size_t lo = block * 8192, hi = std::min(n, lo + 8192);
            for (size_t i = lo; i < hi; ++i) index.invNorm_[i] = inverseNorm(index.row(i), dim);
        });

        index.nlist = std::max<size_t>(1, std::min(nlistWanted, n));
        index.trainedN = n;
        index.train(seed, threads, iterations);
        index.assignAll(threads);
        index.relist();
        index.buildSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return index;
    }

    // -- writes -----------------------------------------------------------------

    /** A new row; it is scored exactly by every query until folded into a cell. */
    uint32_t append(const float* v, uint32_t code) {
        uint32_t slot = static_cast<uint32_t>(alive_.size());
        data_.insert(data_.end(), v, v + dim);
        invNorm_.push_back(inverseNorm(v, dim));
        assign_.push_back(-1);
        alive_.push_back(1);
        pend_.push_back(1);
        codes_.push_back(code);
        pending_.push_back(slot);
        ++pendingLive;
        ++churn;
        return slot;
    }

    /** Replace a row in place: it leaves its cell and is pending again. */
    void overwrite(uint32_t slot, const float* v, uint32_t code) {
        std::copy(v, v + dim, data_.begin() + static_cast<std::ptrdiff_t>(slot) * dim);
        invNorm_[slot] = inverseNorm(v, dim);
        codes_[slot] = code;
        assign_[slot] = -1;
        if (!pend_[slot]) {
            pend_[slot] = 1;
            pending_.push_back(slot);
            ++pendingLive;
        }
        ++churn;
    }

    /** Tombstone a row. Its stale entry in a cell list is skipped at query time. */
    void drop(uint32_t slot) {
        if (!alive_[slot]) return;
        alive_[slot] = 0;
        assign_[slot] = -1;
        if (pend_[slot]) {
            pend_[slot] = 0;
            --pendingLive;
        }
        ++dead_;
        ++churn;
    }

    bool needsCompaction() const { return dead_ >= 64 && dead_ * 4 > alive_.size(); }

    /**
     * Squeeze out the tombstones, keeping live rows in slot order and their
     * cells. Returns, for every old slot, its new slot (kNoSlot when dead);
     * the owner uses it to renumber its own slot tables.
     */
    std::vector<uint32_t> compact() {
        const size_t n = alive_.size();
        std::vector<uint32_t> remap(n, kNoSlot);
        size_t kept = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!alive_[i]) continue;
            remap[i] = static_cast<uint32_t>(kept);
            if (kept != i) {
                std::copy(row(i), row(i) + dim, data_.begin() + static_cast<std::ptrdiff_t>(kept) * dim);
                invNorm_[kept] = invNorm_[i];
                assign_[kept] = assign_[i];
                pend_[kept] = pend_[i];
                codes_[kept] = codes_[i];
            }
            ++kept;
        }
        data_.resize(kept * dim);
        invNorm_.resize(kept);
        assign_.resize(kept);
        pend_.resize(kept);
        codes_.resize(kept);
        alive_.assign(kept, 1);
        dead_ = 0;
        std::vector<uint32_t> pending;
        pending.reserve(pendingLive);
        for (uint32_t old : pending_) {
            if (old < n && remap[old] != kNoSlot && pend_[remap[old]]) {
                pending.push_back(remap[old]);
            }
        }
        // a slot can be listed twice only if it was dropped and re-listed, which
        // overwrite() prevents; sort+unique keeps the list a set regardless
        std::sort(pending.begin(), pending.end());
        pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
        pending_ = std::move(pending);
        relist();
        return remap;
    }

    /** Put every pending row in its nearest cell and regroup. */
    void fold(size_t threads) {
        std::vector<uint32_t> ids;
        ids.reserve(pendingLive);
        for (uint32_t slot : pending_) {
            if (alive_[slot] && pend_[slot]) ids.push_back(slot);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (!ids.empty()) {
            std::vector<float> gathered(ids.size() * dim);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::copy(row(ids[i]), row(ids[i]) + dim, gathered.begin() + static_cast<std::ptrdiff_t>(i) * dim);
            }
            std::vector<int32_t> cells(ids.size());
            if (threads == 0) threads = autoThreads();
            nearest(gathered.data(), ids.size(), cells.data(), threads);
            for (size_t i = 0; i < ids.size(); ++i) {
                assign_[ids[i]] = cells[i];
                pend_[ids[i]] = 0;
            }
        }
        pending_.clear();
        pendingLive = 0;
        relist();
    }

    // -- queries ----------------------------------------------------------------

    /**
     * Score the rows of the `lists` cells whose centroids are nearest `q`, plus
     * every pending row, and keep the best `k` that score at least `threshold`
     * (and have type `code`, unless it is kAnyCode). `out` is filled best
     * first. Scores are cosine similarities computed from the cached inverse
     * norms.
     */
    static constexpr uint32_t kAnyCode = std::numeric_limits<uint32_t>::max();

    void search(const float* q, float invQ, uint32_t code, float threshold, size_t lists,
                size_t k, std::vector<IvfHit>& out) {
        out.clear();
        if (k == 0 || live() == 0) return;
        chooseCells(q, lists);
        heap_.clear();
        heap_.reserve(k);
        scan(q, invQ, code, threshold, k, /*collectAll=*/false);
        out.assign(heap_.begin(), heap_.end());
        std::sort(out.begin(), out.end(), hitBefore);
    }

    /** Like search() but returns every candidate that passes the filters, best first. */
    void candidates(const float* q, float invQ, uint32_t code, float threshold, size_t lists,
                    std::vector<IvfHit>& out) {
        out.clear();
        if (live() == 0) return;
        chooseCells(q, lists);
        heap_.clear();
        scan(q, invQ, code, threshold, 0, /*collectAll=*/true);
        out.assign(heap_.begin(), heap_.end());
        std::sort(out.begin(), out.end(), hitBefore);
    }

    /** Bytes held by the index apart from the vectors themselves. */
    size_t overheadBytes() const {
        return centroids_.capacity() * sizeof(float) + assign_.capacity() * sizeof(int32_t) +
               order_.capacity() * sizeof(uint32_t) + offsets_.capacity() * sizeof(uint32_t) +
               invNorm_.capacity() * sizeof(float) + codes_.capacity() * sizeof(uint32_t) +
               alive_.capacity() + pend_.capacity() + pending_.capacity() * sizeof(uint32_t);
    }
    /** Bytes of the contiguous vector copy the index scans. */
    size_t vectorBytes() const { return data_.capacity() * sizeof(float); }

    static float inverseNorm(const float* v, size_t n) {
        float s = simd::dot(v, v, n);
        return s > 0.0f ? 1.0f / std::sqrt(s) : 0.0f;
    }

private:
    std::vector<float> data_;
    std::vector<float> invNorm_;
    std::vector<int32_t> assign_;
    std::vector<uint8_t> alive_;
    std::vector<uint8_t> pend_;
    std::vector<uint32_t> codes_;
    std::vector<uint32_t> pending_;
    std::vector<uint32_t> order_;
    std::vector<uint32_t> offsets_;
    std::vector<float> centroids_;
    size_t dead_ = 0;

    // per-query scratch, sized once
    std::vector<float> cellScore_;
    std::vector<uint32_t> cellIdx_;
    std::vector<uint32_t> chosen_;
    std::vector<IvfHit> heap_;

    const float* row(size_t slot) const { return data_.data() + slot * dim; }

    // -- training -----------------------------------------------------------------

    static uint64_t below(std::mt19937_64& rng, uint64_t bound) { return rng() % bound; }

    /** `count` distinct values of [0, bound) by a partial Fisher-Yates shuffle. */
    static std::vector<uint32_t> pick(std::mt19937_64& rng, size_t bound, size_t count) {
        std::vector<uint32_t> all(bound);
        std::iota(all.begin(), all.end(), 0u);
        for (size_t i = 0; i < count; ++i) {
            size_t j = i + static_cast<size_t>(below(rng, bound - i));
            std::swap(all[i], all[j]);
        }
        all.resize(count);
        return all;
    }

    /** Nearest centroid (by dot product) of each of `count` rows at `rows`. */
    void nearest(const float* rows, size_t count, int32_t* out, size_t threads) const {
        constexpr size_t kChunk = 256;
        parallelFor((count + kChunk - 1) / kChunk, threads, [&](size_t chunk) {
            size_t lo = chunk * kChunk;
            nearestRange(rows + lo * dim, std::min(kChunk, count - lo), out + lo);
        });
    }

    /** Blocked nearest-centroid kernel: 4 rows x 2 centroids per step, centroids in L2-sized panels. */
    void nearestRange(const float* rows, size_t count, int32_t* out) const {
        constexpr size_t kRowBlock = 32;
        constexpr size_t kCentroidPanel = 64;
        float best[kRowBlock];
        for (size_t rb = 0; rb < count; rb += kRowBlock) {
            const size_t rows_here = std::min(kRowBlock, count - rb);
            for (size_t j = 0; j < rows_here; ++j) {
                best[j] = -std::numeric_limits<float>::infinity();
                out[rb + j] = 0;
            }
            for (size_t cp = 0; cp < nlist; cp += kCentroidPanel) {
                const size_t cEnd = std::min(nlist, cp + kCentroidPanel);
                size_t j = 0;
                for (; j + 4 <= rows_here; j += 4) {
                    const float* block = rows + (rb + j) * dim;
                    size_t c = cp;
                    for (; c + 2 <= cEnd; c += 2) {
                        float s[8];
                        simd::tile4x2(block, dim, centroid(c), centroid(c + 1), s);
                        for (size_t r = 0; r < 4; ++r) {
                            if (s[2 * r] > best[j + r]) { best[j + r] = s[2 * r]; out[rb + j + r] = static_cast<int32_t>(c); }
                            if (s[2 * r + 1] > best[j + r]) { best[j + r] = s[2 * r + 1]; out[rb + j + r] = static_cast<int32_t>(c + 1); }
                        }
                    }
                    for (; c < cEnd; ++c) {
                        for (size_t r = 0; r < 4; ++r) {
                            float s = simd::dot(block + r * dim, centroid(c), dim);
                            if (s > best[j + r]) { best[j + r] = s; out[rb + j + r] = static_cast<int32_t>(c); }
                        }
                    }
                }
                for (; j < rows_here; ++j) {
                    const float* r = rows + (rb + j) * dim;
                    for (size_t c = cp; c < cEnd; ++c) {
                        float s = simd::dot(r, centroid(c), dim);
                        if (s > best[j]) { best[j] = s; out[rb + j] = static_cast<int32_t>(c); }
                    }
                }
            }
        }
    }

    const float* centroid(size_t c) const { return centroids_.data() + c * dim; }
    float* centroid(size_t c) { return centroids_.data() + c * dim; }

    void train(uint64_t seed, size_t threads, int iterations) {
        const size_t total = alive_.size();
        std::mt19937_64 rng(seed);
        const size_t sampleN = std::min(total, 40 * nlist);

        std::vector<uint32_t> sample;
        if (sampleN == total) {
            sample.resize(total);
            std::iota(sample.begin(), sample.end(), 0u);
        } else {
            sample = pick(rng, total, sampleN);
            std::sort(sample.begin(), sample.end());
        }

        // unit-length copy of the sample (a zero row stays zero)
        std::vector<float> x(sampleN * dim);
        parallelFor((sampleN + 4095) / 4096, threads, [&](size_t block) {
            size_t lo = block * 4096, hi = std::min(sampleN, lo + 4096);
            for (size_t i = lo; i < hi; ++i) {
                const float* src = row(sample[i]);
                float inv = invNorm_[sample[i]];
                float* dst = x.data() + i * dim;
                for (size_t d = 0; d < dim; ++d) dst[d] = src[d] * inv;
            }
        });

        centroids_.assign(nlist * dim, 0.0f);
        {
            std::vector<uint32_t> init = pick(rng, sampleN, nlist);
            for (size_t c = 0; c < nlist; ++c) {
                std::copy(x.data() + static_cast<size_t>(init[c]) * dim,
                          x.data() + (static_cast<size_t>(init[c]) + 1) * dim, centroid(c));
            }
        }

        std::vector<int32_t> cell(sampleN);
        std::vector<uint32_t> start(nlist + 1), byCell(sampleN), fill(nlist);
        std::vector<char> reseed(nlist);
        for (int it = 0; it < iterations; ++it) {
            nearest(x.data(), sampleN, cell.data(), threads);

            // rows of each cell in ascending row order (counting sort)
            std::fill(start.begin(), start.end(), 0u);
            for (size_t i = 0; i < sampleN; ++i) ++start[static_cast<size_t>(cell[i]) + 1];
            for (size_t c = 0; c < nlist; ++c) start[c + 1] += start[c];
            std::copy(start.begin(), start.end() - 1, fill.begin());
            for (size_t i = 0; i < sampleN; ++i) byCell[fill[static_cast<size_t>(cell[i])]++] = static_cast<uint32_t>(i);

            constexpr size_t kCellsPerTask = 16;
            parallelFor((nlist + kCellsPerTask - 1) / kCellsPerTask, threads, [&](size_t task) {
                std::vector<double> sum(dim);
                size_t cLo = task * kCellsPerTask, cHi = std::min(nlist, cLo + kCellsPerTask);
                for (size_t c = cLo; c < cHi; ++c) {
                    std::fill(sum.begin(), sum.end(), 0.0);
                    for (uint32_t p = start[c]; p < start[c + 1]; ++p) {
                        const float* r = x.data() + static_cast<size_t>(byCell[p]) * dim;
                        for (size_t d = 0; d < dim; ++d) sum[d] += r[d];
                    }
                    float* out = centroid(c);
                    double sq = 0.0;
                    for (size_t d = 0; d < dim; ++d) {
                        out[d] = static_cast<float>(sum[d]);
                        sq += static_cast<double>(out[d]) * out[d];
                    }
                    double length = std::sqrt(sq);
                    reseed[c] = (start[c + 1] == start[c] || length == 0.0) ? 1 : 0;
                    if (!reseed[c]) {
                        float inv = static_cast<float>(1.0 / length);
                        for (size_t d = 0; d < dim; ++d) out[d] *= inv;
                    }
                }
            });
            // a cell that lost all its rows restarts on a random sample point
            for (size_t c = 0; c < nlist; ++c) {
                if (!reseed[c]) continue;
                size_t pickRow = static_cast<size_t>(below(rng, sampleN));
                std::copy(x.data() + pickRow * dim, x.data() + (pickRow + 1) * dim, centroid(c));
            }
        }
    }

    void assignAll(size_t threads) {
        const size_t n = alive_.size();
        nearest(data_.data(), n, assign_.data(), threads);
        for (size_t i = 0; i < n; ++i) {
            if (!alive_[i]) assign_[i] = -1;
        }
        pending_.clear();
        pendingLive = 0;
        std::fill(pend_.begin(), pend_.end(), uint8_t(0));
    }

    /** Regroup the live rows by cell (a stable counting sort of assign_). */
    void relist() {
        offsets_.assign(nlist + 1, 0u);
        const size_t n = alive_.size();
        for (size_t i = 0; i < n; ++i) {
            if (assign_[i] >= 0) ++offsets_[static_cast<size_t>(assign_[i]) + 1];
        }
        for (size_t c = 0; c < nlist; ++c) offsets_[c + 1] += offsets_[c];
        order_.resize(offsets_[nlist]);
        std::vector<uint32_t> fill(offsets_.begin(), offsets_.end() - 1);
        for (size_t i = 0; i < n; ++i) {
            if (assign_[i] >= 0) order_[fill[static_cast<size_t>(assign_[i])]++] = static_cast<uint32_t>(i);
        }
    }

    // -- query internals ---------------------------------------------------------

    void chooseCells(const float* q, size_t lists) {
        if (cellScore_.size() != nlist) {
            cellScore_.assign(nlist, 0.0f);
            cellIdx_.resize(nlist);
        }
        chosen_.clear();
        if (lists >= nlist) {
            for (size_t c = 0; c < nlist; ++c) chosen_.push_back(static_cast<uint32_t>(c));
            return;
        }
        for (size_t c = 0; c < nlist; ++c) {
            cellScore_[c] = simd::dot(q, centroid(c), dim);
            cellIdx_[c] = static_cast<uint32_t>(c);
        }
        auto better = [this](uint32_t a, uint32_t b) {
            return cellScore_[a] > cellScore_[b] || (cellScore_[a] == cellScore_[b] && a < b);
        };
        std::nth_element(cellIdx_.begin(), cellIdx_.begin() + static_cast<std::ptrdiff_t>(lists - 1),
                         cellIdx_.end(), better);
        chosen_.assign(cellIdx_.begin(), cellIdx_.begin() + static_cast<std::ptrdiff_t>(lists));
        std::sort(chosen_.begin(), chosen_.end());
    }

    void scan(const float* q, float invQ, uint32_t code, float threshold, size_t k, bool collectAll) {
        const float* base = data_.data();
        auto take = [&](uint32_t slot) {
            float s = simd::dot(base + static_cast<size_t>(slot) * dim, q, dim) * invNorm_[slot] * invQ;
            if (!(s >= threshold)) return;
            IvfHit hit{s, slot};
            if (collectAll) {
                heap_.push_back(hit);
            } else if (heap_.size() < k) {
                heap_.push_back(hit);
                std::push_heap(heap_.begin(), heap_.end(), hitBefore);
            } else if (hitBefore(hit, heap_.front())) {
                std::pop_heap(heap_.begin(), heap_.end(), hitBefore);
                heap_.back() = hit;
                std::push_heap(heap_.begin(), heap_.end(), hitBefore);
            }
        };

        for (uint32_t c : chosen_) {
            const uint32_t lo = offsets_[c], hi = offsets_[c + 1];
            for (uint32_t p = lo; p < hi; ++p) {
                if (p + 2 < hi) simd::prefetch(base + static_cast<size_t>(order_[p + 2]) * dim);
                uint32_t slot = order_[p];
                if (assign_[slot] != static_cast<int32_t>(c)) continue;  // moved or deleted since
                if (code != kAnyCode && codes_[slot] != code) continue;
                take(slot);
            }
        }
        for (uint32_t slot : pending_) {
            if (!alive_[slot] || !pend_[slot]) continue;
            if (code != kAnyCode && codes_[slot] != code) continue;
            take(slot);
        }
    }
};

}  // namespace detail
}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_IVF_HPP
