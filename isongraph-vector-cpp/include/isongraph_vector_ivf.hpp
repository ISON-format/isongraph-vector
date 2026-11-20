/**
 * ISONGraph Vector - IVF-flat approximate index (numerics only)
 *
 * The inverted-file index behind EmbeddingStore::enableAnn(). It knows nothing
 * about node references or the store: it holds rows of floats, addressed by
 * slot number, and answers "which slots score highest against this query".
 * isongraph_vector.hpp owns the slot <-> record mapping and the policy
 * (when to build, fold, retrain); this file is the arithmetic.
 *
 * The index holds NO copy of the vectors. The rows live once, in the store's
 * ChunkedMatrix (below); every method that scores or trains takes that matrix
 * and the per-slot type codes by reference. Layout of what the index itself keeps:
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
 * Thread safety: search() and candidates() are const and use only per-call
 * scratch, so any number may run at once; every other method needs exclusive
 * access, which EmbeddingStore provides with a reader/writer lock.
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
#include <memory>
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

/**
 * Run fn(worker) once on each of `workers` threads (worker 0 is the caller's),
 * and return when all are done. The first exception any of them throws is
 * rethrown on the caller's thread after every worker has been joined, so a
 * failure neither leaks a thread nor terminates the process. If the platform
 * cannot start a thread the remaining workers simply do not exist: callers
 * must hand out work from a shared counter, so the workers that did start
 * finish all of it.
 */
template <class F>
void runWorkers(size_t workers, F&& fn) {
    if (workers == 0) workers = 1;
    std::exception_ptr failure;
    std::atomic<bool> failed{false};
    auto guarded = [&](size_t w) {
        try {
            fn(w);
        } catch (...) {
            if (!failed.exchange(true)) failure = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    try {
        for (size_t w = 1; w < workers; ++w) pool.emplace_back(guarded, w);
    } catch (const std::system_error&) {
        // fewer threads than asked for; see above
    }
    guarded(0);
    for (auto& t : pool) t.join();
    if (failure) std::rethrow_exception(failure);
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
// The scoring of one row (moved here from isongraph_vector.hpp so the HNSW
// index, which also lives below that header, can use it unchanged)
// =============================================================================

/** The scoring of one row. Strict IEEE float arithmetic, one fixed order; see cosineRows. */
#if defined(__clang__)
#define ISONGRAPH_VECTOR_NOINLINE __attribute__((noinline))
// Apple clang contracts a*b+c into fused multiply-adds (-ffp-contract=on) and may
// choose differently per loop shape, which made cosineRows4 differ from cosineRows
// in the last bit on Apple Silicon. The scorers switch contraction off themselves.
#define ISONGRAPH_VECTOR_STRICT_FP _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
// optimize("fp-contract=off"): GCC fuses by default on targets that have an FMA
// (aarch64 always, x86 with -mfma); the scorers must not depend on that.
#define ISONGRAPH_VECTOR_NOINLINE __attribute__((noinline, noclone, optimize("fp-contract=off")))
#elif defined(_MSC_VER)
#define ISONGRAPH_VECTOR_NOINLINE __declspec(noinline)
#else
#define ISONGRAPH_VECTOR_NOINLINE
#endif
#ifndef ISONGRAPH_VECTOR_STRICT_FP
#define ISONGRAPH_VECTOR_STRICT_FP
#endif

/**
 * The cosine of two n-float rows: dot / (|a| * |b|), 0 when either norm is 0.
 * The three sums run in ascending order in single-precision accumulators.
 *
 * THE one scoring routine. The exact scan, the batch search, the approximate
 * index's re-score, scoreMap and a store loaded from a snapshot all call this
 * function and nothing else, so one query/row pair has one score everywhere.
 * It is out of line on purpose: two inlined copies are free to be contracted
 * (fused multiply-add) or vectorised differently by an optimiser, which showed
 * up as last-bit differences on GCC 15. Never compile this with -ffast-math.
 */
ISONGRAPH_VECTOR_NOINLINE inline float cosineRows(const float* a, const float* b, size_t n) {
    ISONGRAPH_VECTOR_STRICT_FP
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm1 += a[i] * a[i];
        norm2 += b[i] * b[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    return dot / (norm1 * norm2);
}

/**
 * cosineRows(q, r0, n), cosineRows(q, r1, n), cosineRows(q, r2, n) and
 * cosineRows(q, r3, n) in one pass (the "1 query x 4 rows" kernel of DESIGN.md
 * section 2): each of the twelve sums is the same ascending single-precision
 * running sum, written with the same expression shapes, so every out[j] equals
 * cosineRows(q, rj, n) bit for bit; only the number of independent dependency
 * chains per pass differs (and the query's own norm is summed once). Out of
 * line for the same reason as cosineRows. tests check the equality.
 */
ISONGRAPH_VECTOR_NOINLINE inline void cosineRows4(const float* q, const float* r0, const float* r1,
                                                  const float* r2, const float* r3, size_t n,
                                                  float* out) {
    ISONGRAPH_VECTOR_STRICT_FP
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float norm1 = 0.0f;
    float n20 = 0.0f, n21 = 0.0f, n22 = 0.0f, n23 = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        dot0 += q[i] * r0[i];
        dot1 += q[i] * r1[i];
        dot2 += q[i] * r2[i];
        dot3 += q[i] * r3[i];
        norm1 += q[i] * q[i];
        n20 += r0[i] * r0[i];
        n21 += r1[i] * r1[i];
        n22 += r2[i] * r2[i];
        n23 += r3[i] * r3[i];
    }

    norm1 = std::sqrt(norm1);
    n20 = std::sqrt(n20);
    n21 = std::sqrt(n21);
    n22 = std::sqrt(n22);
    n23 = std::sqrt(n23);

    out[0] = (norm1 == 0.0f || n20 == 0.0f) ? 0.0f : dot0 / (norm1 * n20);
    out[1] = (norm1 == 0.0f || n21 == 0.0f) ? 0.0f : dot1 / (norm1 * n21);
    out[2] = (norm1 == 0.0f || n22 == 0.0f) ? 0.0f : dot2 / (norm1 * n22);
    out[3] = (norm1 == 0.0f || n23 == 0.0f) ? 0.0f : dot3 / (norm1 * n23);
}

// =============================================================================
// The vector matrix: one copy of every vector
// =============================================================================

/** Rows per chunk of the matrix (and per scan block). */
constexpr size_t kChunkRows = 1024;

/**
 * Row-major float matrix stored in chunks of kChunkRows rows. Growth never
 * copies stored rows (a new chunk is allocated; only the first chunk, which
 * starts small, is regrown while it is still below kChunkRows rows), a deletion
 * is a row move done by the owner during compaction, and the rows of one chunk
 * are contiguous, so a scan block never straddles chunks. Chunks are allocated
 * uninitialised, so memory is only touched when a row is written.
 *
 * Row pointers are stable until the next write to the matrix.
 */
class ChunkedMatrix {
public:
    ChunkedMatrix() = default;
    ChunkedMatrix(ChunkedMatrix&& other) noexcept { takeFrom(other); }
    ChunkedMatrix& operator=(ChunkedMatrix&& other) noexcept {
        if (this != &other) takeFrom(other);
        return *this;
    }
    ChunkedMatrix(const ChunkedMatrix& other) { copyFrom(other); }
    ChunkedMatrix& operator=(const ChunkedMatrix& other) {
        if (this != &other) {
            chunks_.clear();
            copyFrom(other);
        }
        return *this;
    }

    size_t dim() const { return dim_; }
    size_t rows() const { return rows_; }
    size_t chunks() const { return chunks_.size(); }
    /** Rows held by chunk `c` (kChunkRows except for the last). */
    size_t rowsInChunk(size_t c) const {
        return c + 1 < chunks_.size() ? kChunkRows : rows_ - c * kChunkRows;
    }
    /** Bytes allocated for rows (not counting the table of chunk pointers). */
    size_t bytes() const {
        size_t total = 0;
        for (size_t c = 0; c < chunks_.size(); ++c) {
            total += (c == 0 ? firstCap_ : kChunkRows) * dim_ * sizeof(float);
        }
        return total;
    }

    /** Forget everything and set the row width. */
    void reset(size_t dim) {
        chunks_.clear();
        rows_ = 0;
        firstCap_ = 0;
        dim_ = dim;
    }

    const float* row(size_t slot) const {
        return chunks_[slot / kChunkRows].get() + (slot % kChunkRows) * dim_;
    }
    float* row(size_t slot) {
        return chunks_[slot / kChunkRows].get() + (slot % kChunkRows) * dim_;
    }
    const float* chunkData(size_t c) const { return chunks_[c].get(); }

    /** Add a row at the end; returns its slot. */
    size_t append(const float* v) {
        const size_t slot = rows_;
        if (slot % kChunkRows == 0) {
            if (slot == 0) {
                growFirst(4);
            } else {
                chunks_.emplace_back(new float[kChunkRows * dim_]);
            }
        } else if (slot < kChunkRows && slot == firstCap_) {
            growFirst(std::min(kChunkRows, firstCap_ * 2));
        }
        ++rows_;
        std::copy(v, v + dim_, row(slot));
        return slot;
    }

    void set(size_t slot, const float* v) { std::copy(v, v + dim_, row(slot)); }

    /** Copy row `from` over row `to` (from > to). */
    void moveRow(size_t from, size_t to) {
        const float* src = row(from);
        std::copy(src, src + dim_, row(to));
    }

    /** Keep the first `keep` rows and free the chunks no longer needed. */
    void truncate(size_t keep) {
        rows_ = keep;
        const size_t needed = (keep + kChunkRows - 1) / kChunkRows;
        while (chunks_.size() > needed) chunks_.pop_back();
        if (keep == 0) firstCap_ = 0;
    }

private:
    std::vector<std::unique_ptr<float[]>> chunks_;
    size_t dim_ = 0;
    size_t rows_ = 0;
    size_t firstCap_ = 0;   // rows the first chunk has room for (<= kChunkRows)

    void takeFrom(ChunkedMatrix& other) noexcept {
        chunks_ = std::move(other.chunks_);
        dim_ = other.dim_;
        rows_ = other.rows_;
        firstCap_ = other.firstCap_;
        other.chunks_.clear();
        other.rows_ = 0;
        other.firstCap_ = 0;
    }

    void growFirst(size_t wantRows) {
        wantRows = std::min(wantRows, kChunkRows);
        std::unique_ptr<float[]> next(new float[wantRows * dim_]);
        if (!chunks_.empty()) {
            const size_t used = std::min(rows_, kChunkRows);
            std::copy(chunks_[0].get(), chunks_[0].get() + used * dim_, next.get());
            chunks_[0] = std::move(next);
        } else {
            chunks_.push_back(std::move(next));
        }
        firstCap_ = wantRows;
    }

    void copyFrom(const ChunkedMatrix& other) {
        dim_ = other.dim_;
        rows_ = other.rows_;
        firstCap_ = other.firstCap_;
        chunks_.clear();
        chunks_.reserve(other.chunks_.size());
        for (size_t c = 0; c < other.chunks_.size(); ++c) {
            const size_t cap = c == 0 ? other.firstCap_ : kChunkRows;
            std::unique_ptr<float[]> mine(new float[cap * dim_]);
            std::copy(other.chunks_[c].get(), other.chunks_[c].get() + other.rowsInChunk(c) * dim_, mine.get());
            chunks_.push_back(std::move(mine));
        }
    }
};

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
     * Train on the live rows of `m` and put every one in its nearest cell.
     * Slot i of the index is row i of the matrix; `alive` says which slots hold a
     * row. `trainOrder` lists the live slots in the order training should see
     * them (the store sorts them by key, so the same data gives the same index
     * however it was inserted).
     */
    static IvfIndex build(const ChunkedMatrix& m, const std::vector<uint8_t>& alive,
                          const std::vector<uint32_t>& trainOrder, size_t nlistWanted,
                          uint64_t seed, size_t threads, int iterations = 8) {
        auto started = std::chrono::steady_clock::now();
        IvfIndex index;
        index.dim = m.dim();
        const size_t n = m.rows();
        index.alive_ = alive;
        index.dead_ = 0;
        for (uint8_t a : index.alive_) index.dead_ += a ? 0 : 1;
        index.pend_.assign(n, 0);
        index.assign_.assign(n, -1);
        index.invNorm_.assign(n, 0.0f);
        if (threads == 0) threads = autoThreads();
        const size_t dim = index.dim;
        parallelFor((n + 8191) / 8192, threads, [&](size_t block) {
            size_t lo = block * 8192, hi = std::min(n, lo + 8192);
            for (size_t i = lo; i < hi; ++i) {
                if (index.alive_[i]) index.invNorm_[i] = inverseNorm(m.row(i), dim);
            }
        });

        const size_t liveRows = trainOrder.size();
        index.nlist = std::max<size_t>(1, std::min(nlistWanted, liveRows));
        index.trainedN = liveRows;
        index.train(m, trainOrder, seed, threads, iterations);
        index.assignAll(m, threads);
        index.relist();
        index.buildSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return index;
    }

    // -- writes (the owner has already written the row to the matrix) -------------

    /** A new row at the next slot; it is scored exactly by every query until folded into a cell. */
    uint32_t append(const float* v) {
        uint32_t slot = static_cast<uint32_t>(alive_.size());
        invNorm_.push_back(inverseNorm(v, dim));
        assign_.push_back(-1);
        alive_.push_back(1);
        pend_.push_back(1);
        pending_.push_back(slot);
        ++pendingLive;
        ++churn;
        return slot;
    }

    /** A row was replaced in place: it leaves its cell and is pending again. */
    void overwrite(uint32_t slot, const float* v) {
        invNorm_[slot] = inverseNorm(v, dim);
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

    /**
     * The owner squeezed the tombstones out of its matrix, keeping live rows in
     * slot order: `remapTable[old]` is the new slot (kNoSlot for a dead one) and
     * `kept` the new slot count. Renumber everything slot-indexed to match.
     */
    void remap(const std::vector<uint32_t>& remapTable, size_t kept) {
        const size_t n = alive_.size();
        for (size_t i = 0; i < n; ++i) {
            const uint32_t to = remapTable[i];
            if (to == kNoSlot || to == i) continue;
            invNorm_[to] = invNorm_[i];
            assign_[to] = assign_[i];
            pend_[to] = pend_[i];
        }
        invNorm_.resize(kept);
        assign_.resize(kept);
        pend_.resize(kept);
        alive_.assign(kept, 1);
        dead_ = 0;
        std::vector<uint32_t> pending;
        pending.reserve(pendingLive);
        for (uint32_t old : pending_) {
            if (old < n && remapTable[old] != kNoSlot && pend_[remapTable[old]]) {
                pending.push_back(remapTable[old]);
            }
        }
        // a slot can be listed twice only if it was dropped and re-listed, which
        // overwrite() prevents; sort+unique keeps the list a set regardless
        std::sort(pending.begin(), pending.end());
        pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
        pending_ = std::move(pending);
        relist();
    }

    /** Put every pending row in its nearest cell and regroup. */
    void fold(const ChunkedMatrix& m, size_t threads) {
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
                const float* src = m.row(ids[i]);
                std::copy(src, src + dim, gathered.begin() + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(dim));
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
     * (and have type `code` in `codes`, unless it is kAnyCode). `out` is filled
     * best first. Scores are cosine similarities computed from the cached
     * inverse norms. const, and all scratch is local to the call: any number of
     * searches may run at once.
     */
    static constexpr uint32_t kAnyCode = std::numeric_limits<uint32_t>::max();

    void search(const ChunkedMatrix& m, const uint32_t* codes, const float* q, float invQ,
                uint32_t code, float threshold, size_t lists, size_t k,
                std::vector<IvfHit>& out) const {
        out.clear();
        if (k == 0 || live() == 0) return;
        std::vector<uint32_t> chosen;
        chooseCells(q, lists, chosen);
        scan(m, codes, q, invQ, code, threshold, k, /*collectAll=*/false, chosen, out);
        std::sort(out.begin(), out.end(), hitBefore);
    }

    /** Like search() but returns every candidate that passes the filters, best first. */
    void candidates(const ChunkedMatrix& m, const uint32_t* codes, const float* q, float invQ,
                    uint32_t code, float threshold, size_t lists,
                    std::vector<IvfHit>& out) const {
        out.clear();
        if (live() == 0) return;
        std::vector<uint32_t> chosen;
        chooseCells(q, lists, chosen);
        scan(m, codes, q, invQ, code, threshold, 0, /*collectAll=*/true, chosen, out);
        std::sort(out.begin(), out.end(), hitBefore);
    }

    /** Bytes held by the index. It holds no copy of the vectors. */
    size_t overheadBytes() const {
        return centroids_.capacity() * sizeof(float) + assign_.capacity() * sizeof(int32_t) +
               order_.capacity() * sizeof(uint32_t) + offsets_.capacity() * sizeof(uint32_t) +
               invNorm_.capacity() * sizeof(float) +
               alive_.capacity() + pend_.capacity() + pending_.capacity() * sizeof(uint32_t);
    }

    static float inverseNorm(const float* v, size_t n) {
        float s = simd::dot(v, v, n);
        return s > 0.0f ? 1.0f / std::sqrt(s) : 0.0f;
    }

private:
    std::vector<float> invNorm_;
    std::vector<int32_t> assign_;
    std::vector<uint8_t> alive_;
    std::vector<uint8_t> pend_;
    std::vector<uint32_t> pending_;
    std::vector<uint32_t> order_;
    std::vector<uint32_t> offsets_;
    std::vector<float> centroids_;
    size_t dead_ = 0;

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

    /** nearest() over the first `count` rows of the matrix (256-row blocks never cross a chunk). */
    void nearestMatrix(const ChunkedMatrix& m, size_t count, int32_t* out, size_t threads) const {
        constexpr size_t kChunk = 256;
        static_assert(kChunkRows % kChunk == 0, "a 256-row block must lie inside one chunk");
        parallelFor((count + kChunk - 1) / kChunk, threads, [&](size_t chunk) {
            size_t lo = chunk * kChunk;
            nearestRange(m.row(lo), std::min(kChunk, count - lo), out + lo);
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

    void train(const ChunkedMatrix& m, const std::vector<uint32_t>& trainOrder, uint64_t seed,
               size_t threads, int iterations) {
        const size_t total = trainOrder.size();
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
                const uint32_t slot = trainOrder[sample[i]];
                const float* src = m.row(slot);
                float inv = invNorm_[slot];
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

    void assignAll(const ChunkedMatrix& m, size_t threads) {
        const size_t n = alive_.size();
        nearestMatrix(m, n, assign_.data(), threads);
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

    void chooseCells(const float* q, size_t lists, std::vector<uint32_t>& chosen) const {
        chosen.clear();
        if (lists >= nlist) {
            for (size_t c = 0; c < nlist; ++c) chosen.push_back(static_cast<uint32_t>(c));
            return;
        }
        std::vector<float> cellScore(nlist);
        std::vector<uint32_t> cellIdx(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            cellScore[c] = simd::dot(q, centroid(c), dim);
            cellIdx[c] = static_cast<uint32_t>(c);
        }
        auto better = [&cellScore](uint32_t a, uint32_t b) {
            return cellScore[a] > cellScore[b] || (cellScore[a] == cellScore[b] && a < b);
        };
        std::nth_element(cellIdx.begin(), cellIdx.begin() + static_cast<std::ptrdiff_t>(lists - 1),
                         cellIdx.end(), better);
        chosen.assign(cellIdx.begin(), cellIdx.begin() + static_cast<std::ptrdiff_t>(lists));
        std::sort(chosen.begin(), chosen.end());
    }

    void scan(const ChunkedMatrix& m, const uint32_t* codes, const float* q, float invQ,
              uint32_t code, float threshold, size_t k, bool collectAll,
              const std::vector<uint32_t>& chosen, std::vector<IvfHit>& heap) const {
        auto take = [&](uint32_t slot) {
            float s = simd::dot(m.row(slot), q, dim) * invNorm_[slot] * invQ;
            if (!(s >= threshold)) return;
            IvfHit hit{s, slot};
            if (collectAll) {
                heap.push_back(hit);
            } else if (heap.size() < k) {
                heap.push_back(hit);
                std::push_heap(heap.begin(), heap.end(), hitBefore);
            } else if (hitBefore(hit, heap.front())) {
                std::pop_heap(heap.begin(), heap.end(), hitBefore);
                heap.back() = hit;
                std::push_heap(heap.begin(), heap.end(), hitBefore);
            }
        };

        for (uint32_t c : chosen) {
            const uint32_t lo = offsets_[c], hi = offsets_[c + 1];
            for (uint32_t p = lo; p < hi; ++p) {
                if (p + 2 < hi) simd::prefetch(m.row(order_[p + 2]));
                uint32_t slot = order_[p];
                if (assign_[slot] != static_cast<int32_t>(c)) continue;  // moved or deleted since
                if (code != kAnyCode && codes[slot] != code) continue;
                take(slot);
            }
        }
        for (uint32_t slot : pending_) {
            if (!alive_[slot] || !pend_[slot]) continue;
            if (code != kAnyCode && codes[slot] != code) continue;
            take(slot);
        }
    }
};

}  // namespace detail
}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_IVF_HPP
