/**
 * SQLite-backed embedding storage for isongraph-vector, optionally indexed
 * with sqlite-vec.
 *
 * A separate header from isongraph_vector.hpp, which stays header-only and
 * dependency-free: this one needs SQLite. Build it with
 * `-DISONGRAPH_VECTOR_SQLITE=ON`, which fetches the SQLite and sqlite-vec
 * amalgamations, or link your own SQLite and include this directly.
 *
 * The schema is identical to the one the Python, C#, Rust, TypeScript and
 * JavaScript ports write, so a database created by any of them can be opened
 * by any other. Vectors are little-endian float32, four bytes per dimension.
 *
 * @example
 * ```cpp
 * #include "isongraph_vector_sqlite.hpp"
 *
 * using namespace isongraph_vector;
 *
 * auto encoder = std::make_shared<MockEncoder>(384);
 * SqliteEmbeddingStore store("embeddings.db", encoder, true);
 * store.add(NodeRef("person", "1"), "Alice is a software engineer");
 * auto hits = store.similaritySearch("engineering", 5, "", -1.0f);
 * ```
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */

#ifndef ISONGRAPH_VECTOR_SQLITE_HPP
#define ISONGRAPH_VECTOR_SQLITE_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "isongraph_vector.hpp"

extern "C" int sqlite3_vec_init(sqlite3*, char**, const sqlite3_api_routines*);

namespace isongraph_vector {

/// Virtual table holding the vec0 index, alongside the `embeddings` table.
constexpr const char* VEC_TABLE = "vec_embeddings";

namespace detail {

/**
 * MD5, used only to compute the `id` surrogate key so that a row written here
 * is byte-identical to one written by the other ports. Not used for anything
 * security-related.
 */
class Md5 {
public:
    static std::string hex(const std::string& input) {
        Md5 md5;
        md5.update(reinterpret_cast<const uint8_t*>(input.data()), input.size());
        auto digest = md5.finish();

        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(32);
        for (uint8_t byte : digest) {
            out.push_back(digits[byte >> 4]);
            out.push_back(digits[byte & 0x0f]);
        }
        return out;
    }

private:
    uint32_t state_[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    uint64_t length_ = 0;
    uint8_t buffer_[64] = {};
    size_t buffered_ = 0;

    static uint32_t rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

    void update(const uint8_t* data, size_t len) {
        length_ += len;
        while (len > 0) {
            size_t take = std::min(len, size_t(64) - buffered_);
            std::memcpy(buffer_ + buffered_, data, take);
            buffered_ += take;
            data += take;
            len -= take;
            if (buffered_ == 64) {
                transform(buffer_);
                buffered_ = 0;
            }
        }
    }

    std::array<uint8_t, 16> finish() {
        uint64_t bits = length_ * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = static_cast<uint8_t>((bits >> (8 * i)) & 0xff);
            // length_ must not count the length field itself
            length_ -= 1;
            update(&byte, 1);
        }

        std::array<uint8_t, 16> digest{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                digest[i * 4 + j] = static_cast<uint8_t>((state_[i] >> (8 * j)) & 0xff);
            }
        }
        return digest;
    }

    void transform(const uint8_t block[64]) {
        static const uint32_t K[64] = {
            0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au,
            0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
            0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u,
            0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
            0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
            0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
            0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
            0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
            0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u,
            0xffeff47du, 0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
            0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};
        static const int S[64] = {
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
            5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

        uint32_t M[16];
        for (int i = 0; i < 16; ++i) {
            M[i] = static_cast<uint32_t>(block[i * 4]) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 8) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 3]) << 24);
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        for (int i = 0; i < 64; ++i) {
            uint32_t f;
            int g;
            if (i < 16)      { f = (b & c) | (~b & d);        g = i; }
            else if (i < 32) { f = (d & b) | (~d & c);        g = (5 * i + 1) % 16; }
            else if (i < 48) { f = b ^ c ^ d;                 g = (3 * i + 5) % 16; }
            else             { f = c ^ (b | ~d);              g = (7 * i) % 16; }

            uint32_t tmp = d;
            d = c;
            c = b;
            b = b + rotl(a + f + K[i] + M[g], S[i]);
            a = tmp;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
    }
};

}  // namespace detail

/// SQLite-backed embedding storage.
class SqliteEmbeddingStore {
private:
    sqlite3* db_ = nullptr;
    std::shared_ptr<EmbeddingEncoder> encoder_;
    std::optional<size_t> dimension_;
    bool sqlite_vec_ = false;
    bool vec_ready_ = false;

    [[noreturn]] void fail(const std::string& what) const {
        throw EmbeddingError(what + ": " + (db_ ? sqlite3_errmsg(db_) : "no connection"));
    }

    void exec(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "unknown error";
            sqlite3_free(err);
            throw EmbeddingError("sql failed: " + message);
        }
    }

    /// Prepared-statement holder that always finalizes.
    class Stmt {
    public:
        Stmt(sqlite3* db, const std::string& sql) {
            if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt_, nullptr) != SQLITE_OK) {
                throw EmbeddingError(std::string("prepare failed: ") + sqlite3_errmsg(db));
            }
        }
        ~Stmt() { sqlite3_finalize(stmt_); }
        Stmt(const Stmt&) = delete;
        Stmt& operator=(const Stmt&) = delete;

        sqlite3_stmt* get() const { return stmt_; }

        void bindText(int i, const std::string& value) {
            sqlite3_bind_text(stmt_, i, value.c_str(), -1, SQLITE_TRANSIENT);
        }
        void bindBlob(int i, const std::vector<float>& vector) {
            sqlite3_bind_blob(stmt_, i, vector.data(),
                              static_cast<int>(vector.size() * sizeof(float)), SQLITE_TRANSIENT);
        }
        void bindInt(int i, sqlite3_int64 value) { sqlite3_bind_int64(stmt_, i, value); }

        bool step() {
            int rc = sqlite3_step(stmt_);
            if (rc == SQLITE_ROW) return true;
            if (rc == SQLITE_DONE) return false;
            throw EmbeddingError("step failed with code " + std::to_string(rc));
        }

        std::string columnText(int i) const {
            auto* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt_, i));
            return text ? text : "";
        }
        std::vector<float> columnVector(int i) const {
            const void* blob = sqlite3_column_blob(stmt_, i);
            int bytes = sqlite3_column_bytes(stmt_, i);
            std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
            if (bytes > 0) std::memcpy(out.data(), blob, static_cast<size_t>(bytes));
            return out;
        }
        double columnDouble(int i) const { return sqlite3_column_double(stmt_, i); }
        sqlite3_int64 columnInt(int i) const { return sqlite3_column_int64(stmt_, i); }

    private:
        sqlite3_stmt* stmt_ = nullptr;
    };

    void createSchema() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS embeddings (
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
            CREATE INDEX IF NOT EXISTS idx_embeddings_node
                ON embeddings(node_type, node_id);
        )");
    }

    void adoptStoredDimension() {
        Stmt stmt(db_, "SELECT LENGTH(vector) FROM embeddings LIMIT 1");
        if (stmt.step()) {
            auto bytes = stmt.columnInt(0);
            if (bytes > 0) dimension_ = static_cast<size_t>(bytes) / sizeof(float);
        }
    }

    /// Create the vec0 index once the vector width is known, and backfill it.
    void ensureVecTable(size_t dimension) {
        if (!sqlite_vec_ || vec_ready_) return;

        // node_type is a vec0 metadata column so a type-filtered search stays
        // exact - filtering after a top-k fetch would silently drop matches.
        std::ostringstream ddl;
        ddl << "CREATE VIRTUAL TABLE IF NOT EXISTS " << VEC_TABLE << " USING vec0("
            << "node_type text, embedding float[" << dimension << "] distance_metric=cosine)";
        exec(ddl.str());

        sqlite3_int64 indexed = 0, stored = 0;
        {
            Stmt count(db_, std::string("SELECT COUNT(*) FROM ") + VEC_TABLE);
            if (count.step()) indexed = count.columnInt(0);
        }
        {
            Stmt count(db_, "SELECT COUNT(*) FROM embeddings");
            if (count.step()) stored = count.columnInt(0);
        }

        if (indexed != stored) {
            // Opening a database written without the index, or with it off.
            exec(std::string("DELETE FROM ") + VEC_TABLE);
            std::vector<std::tuple<sqlite3_int64, std::string, std::vector<float>>> rows;
            {
                Stmt read(db_, "SELECT rowid, node_type, vector FROM embeddings");
                while (read.step()) {
                    rows.emplace_back(read.columnInt(0), read.columnText(1), read.columnVector(2));
                }
            }
            for (const auto& [rowid, nodeType, vector] : rows) {
                insertVecRow(rowid, nodeType, vector);
            }
        }

        vec_ready_ = true;
    }

    void insertVecRow(sqlite3_int64 rowid, const std::string& nodeType,
                      const std::vector<float>& vector) {
        Stmt stmt(db_, std::string("INSERT INTO ") + VEC_TABLE +
                           "(rowid, node_type, embedding) VALUES (?, ?, ?)");
        stmt.bindInt(1, rowid);
        stmt.bindText(2, nodeType);
        stmt.bindBlob(3, vector);
        stmt.step();
    }

    /// Surrogate primary key, matching what the other ports compute.
    static std::string embeddingId(const NodeRef& ref) {
        return detail::Md5::hex(ref.type + ":" + ref.id).substr(0, 16);
    }

    void checkDimension(const std::vector<float>& vector, const std::string& what) {
        if (vector.empty()) throw DimensionMismatchError(what + " is empty");
        if (!dimension_.has_value()) {
            dimension_ = vector.size();
        } else if (vector.size() != *dimension_) {
            throw DimensionMismatchError(
                what + " has " + std::to_string(vector.size()) + " dimensions, store holds " +
                std::to_string(*dimension_));
        }
    }

    void checkQuery(const std::vector<float>& vector) const {
        if (vector.empty()) throw DimensionMismatchError("query vector is empty");
        if (dimension_.has_value() && vector.size() != *dimension_) {
            throw DimensionMismatchError(
                "query vector has " + std::to_string(vector.size()) + " dimensions, store holds " +
                std::to_string(*dimension_));
        }
    }

    static float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        n1 = std::sqrt(n1);
        n2 = std::sqrt(n2);
        if (n1 == 0.0f || n2 == 0.0f) return 0.0f;
        return dot / (n1 * n2);
    }

    std::string modelName() const { return encoder_ ? "unknown" : "provided"; }

    /**
     * Insert or update one row, keeping the vec0 index in step.
     *
     * An UPSERT rather than INSERT OR REPLACE: REPLACE assigns a new rowid,
     * which would orphan the matching vec0 entry and reset created_at.
     */
    std::string writeRow(const NodeRef& ref, const std::vector<float>& vector,
                         const std::string& text, const std::string& model,
                         const std::string& nodeIdType = "str") {
        std::string embId = embeddingId(ref);
        {
            Stmt stmt(db_, R"(
                INSERT INTO embeddings
                (id, node_type, node_id, node_id_type, vector, text, model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_type, node_id) DO UPDATE SET
                    node_id_type = excluded.node_id_type,
                    vector = excluded.vector,
                    text = excluded.text,
                    model = excluded.model
            )");
            stmt.bindText(1, embId);
            stmt.bindText(2, ref.type);
            stmt.bindText(3, ref.id);
            stmt.bindText(4, nodeIdType);
            stmt.bindBlob(5, vector);
            stmt.bindText(6, text);
            stmt.bindText(7, model);
            stmt.step();
        }

        if (sqlite_vec_) {
            ensureVecTable(vector.size());
            sqlite3_int64 rowid = 0;
            {
                Stmt find(db_,
                          "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?");
                find.bindText(1, ref.type);
                find.bindText(2, ref.id);
                if (find.step()) rowid = find.columnInt(0);
            }
            {
                // vec0 has no upsert; replace the entry at this rowid.
                Stmt drop(db_, std::string("DELETE FROM ") + VEC_TABLE + " WHERE rowid = ?");
                drop.bindInt(1, rowid);
                drop.step();
            }
            insertVecRow(rowid, ref.type, vector);
        }

        return embId;
    }

    /// Score every stored vector against a query. Always an exact scan.
    std::vector<SimilarityResult> scan(const std::vector<float>& queryVector,
                                       const std::string& nodeType) const {
        std::vector<SimilarityResult> results;
        Stmt stmt(db_, "SELECT node_type, node_id, vector, text FROM embeddings");
        while (stmt.step()) {
            std::string type = stmt.columnText(0);
            if (!nodeType.empty() && type != nodeType) continue;
            results.push_back(SimilarityResult{
                NodeRef(type, stmt.columnText(1)),
                cosineSimilarity(queryVector, stmt.columnVector(2)),
                stmt.columnText(3)});
        }
        return results;
    }

    /// Top-k from the vec0 index.
    std::vector<SimilarityResult> vecSearch(const std::vector<float>& queryVector, size_t topK,
                                            const std::string& nodeType, float threshold) {
        ensureVecTable(queryVector.size());

        std::string sql = std::string(
            "SELECT e.node_type, e.node_id, e.text, v.distance FROM ") + VEC_TABLE +
            " v JOIN embeddings e ON e.rowid = v.rowid WHERE ";
        sql += nodeType.empty() ? "v.embedding MATCH ? AND k = ?"
                                : "v.node_type = ? AND v.embedding MATCH ? AND k = ?";

        Stmt stmt(db_, sql);
        int arg = 1;
        if (!nodeType.empty()) stmt.bindText(arg++, nodeType);
        stmt.bindBlob(arg++, queryVector);
        stmt.bindInt(arg, static_cast<sqlite3_int64>(topK));

        // Rows arrive ordered by distance ascending, i.e. score descending.
        std::vector<SimilarityResult> results;
        while (stmt.step()) {
            float score = 1.0f - static_cast<float>(stmt.columnDouble(3));
            if (score >= threshold) {
                results.push_back(SimilarityResult{
                    NodeRef(stmt.columnText(0), stmt.columnText(1)), score, stmt.columnText(2)});
            }
        }
        return results;
    }

public:
    /**
     * Open (or create) a store. Use ":memory:" for a scratch store.
     *
     * @param path       Database path, or ":memory:"
     * @param encoder    Encoder used when a text is added without a vector
     * @param sqliteVec  Answer top-k searches through a sqlite-vec vec0
     *                   virtual table instead of scanning every row. Results
     *                   are the same either way - vec0 runs an exact
     *                   brute-force KNN in C.
     */
    explicit SqliteEmbeddingStore(const std::string& path = ":memory:",
                                  std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
                                  bool sqliteVec = false)
        : encoder_(std::move(encoder)), sqlite_vec_(sqliteVec) {
        if (sqlite_vec_) {
            // Registering as an auto-extension applies to connections opened
            // afterwards, so this must run before sqlite3_open_v2.
            static bool registered = false;
            if (!registered) {
                int rc = sqlite3_auto_extension(
                    reinterpret_cast<void (*)()>(sqlite3_vec_init));
                if (rc != SQLITE_OK) {
                    throw EmbeddingError("could not register sqlite-vec, code " +
                                         std::to_string(rc));
                }
                registered = true;
            }
        }

        if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
            std::string message = db_ ? sqlite3_errmsg(db_) : "unknown error";
            sqlite3_close(db_);
            db_ = nullptr;
            throw EmbeddingError("could not open " + path + ": " + message);
        }

        createSchema();
        adoptStoredDimension();
        if (sqlite_vec_ && dimension_.has_value()) ensureVecTable(*dimension_);
    }

    ~SqliteEmbeddingStore() { sqlite3_close(db_); }

    SqliteEmbeddingStore(const SqliteEmbeddingStore&) = delete;
    SqliteEmbeddingStore& operator=(const SqliteEmbeddingStore&) = delete;

    /// Whether top-k search is answered from the sqlite-vec index.
    bool sqliteVec() const { return sqlite_vec_; }

    /// Vector length this store holds, or nullopt while it is still empty.
    std::optional<size_t> dimension() const { return dimension_; }

    void setEncoder(std::shared_ptr<EmbeddingEncoder> encoder) { encoder_ = std::move(encoder); }
    const std::shared_ptr<EmbeddingEncoder>& encoder() const { return encoder_; }

    std::string add(const NodeRef& ref, const std::string& text,
                    const std::vector<float>& vector = {}) {
        std::vector<float> vec = vector;
        if (vec.empty()) {
            if (!encoder_) throw NoEncoderError("No encoder provided and no vector given");
            vec = encoder_->encode(text);
        }
        checkDimension(vec, "vector");
        return writeRow(ref, vec, text, modelName());
    }

    /// Add many embeddings in one transaction, encoding their texts together.
    size_t addBatch(const std::vector<std::pair<NodeRef, std::string>>& entries) {
        if (entries.empty()) return 0;
        if (!encoder_) throw NoEncoderError("No encoder provided and no vectors given");

        std::vector<std::string> texts;
        texts.reserve(entries.size());
        for (const auto& entry : entries) texts.push_back(entry.second);

        auto vectors = encoder_->encodeBatch(texts);
        if (vectors.size() != entries.size()) {
            throw EmbeddingError("encoder returned " + std::to_string(vectors.size()) +
                                 " vectors for " + std::to_string(entries.size()) + " texts");
        }
        for (const auto& vector : vectors) checkDimension(vector, "vector");

        std::string model = modelName();
        exec("BEGIN");
        try {
            for (size_t i = 0; i < entries.size(); ++i) {
                writeRow(entries[i].first, vectors[i], entries[i].second, model);
            }
            exec("COMMIT");
        } catch (...) {
            exec("ROLLBACK");
            throw;
        }
        return entries.size();
    }

    std::optional<EmbeddingRecord> get(const NodeRef& ref) const {
        Stmt stmt(db_,
                  "SELECT id, node_type, node_id, vector, text FROM embeddings "
                  "WHERE node_type = ? AND node_id = ?");
        stmt.bindText(1, ref.type);
        stmt.bindText(2, ref.id);
        if (!stmt.step()) return std::nullopt;

        return EmbeddingRecord{stmt.columnText(0),
                               NodeRef(stmt.columnText(1), stmt.columnText(2)),
                               stmt.columnVector(3), stmt.columnText(4)};
    }

    bool remove(const NodeRef& ref) {
        if (sqlite_vec_ && vec_ready_) {
            sqlite3_int64 rowid = -1;
            {
                Stmt find(db_,
                          "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?");
                find.bindText(1, ref.type);
                find.bindText(2, ref.id);
                if (find.step()) rowid = find.columnInt(0);
            }
            if (rowid >= 0) {
                Stmt drop(db_, std::string("DELETE FROM ") + VEC_TABLE + " WHERE rowid = ?");
                drop.bindInt(1, rowid);
                drop.step();
            }
        }

        Stmt stmt(db_, "DELETE FROM embeddings WHERE node_type = ? AND node_id = ?");
        stmt.bindText(1, ref.type);
        stmt.bindText(2, ref.id);
        stmt.step();
        return sqlite3_changes(db_) > 0;
    }

    size_t count() const {
        Stmt stmt(db_, "SELECT COUNT(*) FROM embeddings");
        return stmt.step() ? static_cast<size_t>(stmt.columnInt(0)) : 0;
    }

    void clear() {
        exec("DELETE FROM embeddings");
        if (sqlite_vec_ && vec_ready_) exec(std::string("DELETE FROM ") + VEC_TABLE);
        dimension_.reset();
    }

    std::vector<SimilarityResult> similaritySearch(const std::string& query, size_t topK = 10,
                                                   const std::string& nodeType = "",
                                                   float threshold = 0.0f) {
        if (!encoder_) throw NoEncoderError("No encoder provided for text query");
        return similaritySearchVector(encoder_->encode(query), topK, nodeType, threshold);
    }

    std::vector<SimilarityResult> similaritySearchVector(const std::vector<float>& queryVector,
                                                         size_t topK = 10,
                                                         const std::string& nodeType = "",
                                                         float threshold = 0.0f) {
        checkQuery(queryVector);

        // A zero vector has no direction for cosine to measure; vec0 returns
        // NaN there, while the scan defines it as 0.0.
        bool nonZero = false;
        for (float v : queryVector) {
            if (v != 0.0f) { nonZero = true; break; }
        }
        if (sqlite_vec_ && topK > 0 && nonZero) {
            return vecSearch(queryVector, topK, nodeType, threshold);
        }

        auto results = scan(queryVector, nodeType);
        results.erase(std::remove_if(results.begin(), results.end(),
                                     [&](const SimilarityResult& r) { return r.score < threshold; }),
                      results.end());
        auto by_score = [](const SimilarityResult& a, const SimilarityResult& b) {
            return a.score > b.score;
        };
        // Top-k selection rather than a full ordering; same rows, same order.
        if (results.size() > topK) {
            std::partial_sort(results.begin(), results.begin() + topK, results.end(), by_score);
            results.resize(topK);
        } else {
            std::sort(results.begin(), results.end(), by_score);
        }
        return results;
    }

    /**
     * Cosine score for every embedded node, keyed by "type:id".
     *
     * Always scans, even with the index on: the index answers top-k queries,
     * and blending needs a score for every node.
     */
    std::map<std::string, float> scoreMap(const std::vector<float>& queryVector,
                                          const std::string& nodeType = "") const {
        checkQuery(queryVector);
        std::map<std::string, float> scores;
        for (const auto& result : scan(queryVector, nodeType)) {
            scores[result.node_ref.type + ":" + result.node_ref.id] = result.score;
        }
        return scores;
    }

    /// Serialize every embedding into the portable cross-language payload.
    ExportedEmbeddings exportEmbeddings() const {
        ExportedEmbeddings payload;
        payload.dimension = dimension_;

        Stmt stmt(db_,
                  "SELECT node_type, node_id, node_id_type, vector, text FROM embeddings "
                  "ORDER BY node_type, node_id");
        while (stmt.step()) {
            payload.embeddings.push_back(ExportedEmbedding{
                stmt.columnText(0), stmt.columnText(1), stmt.columnText(2),
                stmt.columnText(4), stmt.columnVector(3)});
        }
        payload.count = payload.embeddings.size();
        return payload;
    }

    /// Load a payload produced by exportEmbeddings in any language port.
    size_t importEmbeddings(const ExportedEmbeddings& payload, bool replace = false) {
        if (payload.format != std::string(EMBEDDING_FORMAT)) {
            throw EmbeddingError(std::string("not an ") + EMBEDDING_FORMAT +
                                 " payload: format=" + payload.format);
        }
        if (payload.version != EMBEDDING_FORMAT_VERSION) {
            throw EmbeddingError(std::string("unsupported ") + EMBEDDING_FORMAT + " version " +
                                 std::to_string(payload.version));
        }
        if (replace) clear();

        exec("BEGIN");
        size_t written = 0;
        try {
            for (const auto& item : payload.embeddings) {
                std::vector<float> vector = item.vector;
                checkDimension(vector, "vector");
                writeRow(NodeRef(item.type, item.id), vector, item.text, "unknown", item.id_type);
                ++written;
            }
            exec("COMMIT");
        } catch (...) {
            exec("ROLLBACK");
            throw;
        }
        return written;
    }
};

}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_SQLITE_HPP
