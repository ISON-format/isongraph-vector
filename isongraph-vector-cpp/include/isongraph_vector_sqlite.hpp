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
 * Like the in-memory EmbeddingStore it records, per row, which model made the
 * vector (the `model` column every port already has), warns when a text query
 * is encoded by a different model, reports stale vectors, and reads and writes
 * the portable payload / sidecar with its optional metadata. A database from
 * an older release that lacks the `model` column gets it added (default
 * 'unknown') on open; nothing else about the schema changes.
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
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
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
    std::optional<std::string> version_;
    ModelMismatchMode onModelMismatch_ = ModelMismatchMode::Warn;
    mutable std::set<std::string> warnedMismatch_;
    /// {model: count}, filled lazily and dropped by every write made here.
    /// A change made through another connection is not seen until the next write.
    mutable std::optional<std::map<std::string, size_t>> modelsCache_;
    EmbeddingMeta lastImportMeta_;

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

        // CREATE TABLE IF NOT EXISTS does nothing against a database written
        // by an older release, which may lack columns added since. Add them
        // rather than fail on every read; existing rows take the default.
        std::set<std::string> columns;
        {
            Stmt info(db_, "PRAGMA table_info(embeddings)");
            while (info.step()) columns.insert(info.columnText(1));
        }
        if (columns.count("node_id_type") == 0) {
            exec("ALTER TABLE embeddings ADD COLUMN node_id_type TEXT NOT NULL DEFAULT 'str'");
        }
        if (columns.count("model") == 0) {
            exec("ALTER TABLE embeddings ADD COLUMN model TEXT DEFAULT 'unknown'");
        }
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

    /// Model stamped on vectors written now: the encoder's, "provided" with none.
    std::string currentModelStamp() const {
        if (!encoder_) return "provided";
        std::string name = encoder_->modelName();
        return name.empty() ? "unknown" : name;
    }

    /// SQL for a row's model, reading a NULL or empty value as "unknown".
    static constexpr const char* MODEL_EXPR = "COALESCE(NULLIF(model, ''), 'unknown')";

    /// Complain when a text query's encoder is not the stored vectors' model.
    void checkModelCompat() const {
        if (onModelMismatch_ == ModelMismatchMode::Ignore) return;
        std::optional<std::string> current = currentModel();
        if (!current.has_value()) return;

        std::set<std::string> foreign;  // sorted, which also orders the message
        for (const auto& [model, count] : models()) {
            if (!isUnknownModel(model) && model != *current) foreign.insert(model);
        }
        if (foreign.empty()) return;

        std::string listed;
        for (const std::string& model : foreign) {
            if (!listed.empty()) listed += ", ";
            listed += "'" + model + "'";
        }
        std::string message =
            "encoder model '" + *current + "' differs from the model that produced the "
            "stored vectors (" + listed + "); scores compare vectors from different spaces. "
            "Re-embed with the current encoder or search with the original one.";

        if (onModelMismatch_ == ModelMismatchMode::Error) {
            throw ModelMismatchError(message);
        }
        std::string key = *current + "\n";
        for (const std::string& model : foreign) key += model + "\n";
        if (warnedMismatch_.insert(key).second) {
            emitWarning(message);
        }
    }

    void dropVecTable() {
        if (sqlite_vec_) exec(std::string("DROP TABLE IF EXISTS ") + VEC_TABLE);
        vec_ready_ = false;
    }

    /**
     * Run `body` in one transaction. On failure the transaction is rolled back
     * and the in-memory state it may have changed (the pinned dimension, the
     * index-ready flag, the cached model counts) is put back, so a failed
     * write leaves the store as it found it.
     */
    template <typename F>
    auto transact(F&& body) -> decltype(body()) {
        const auto savedDimension = dimension_;
        const bool savedVecReady = vec_ready_;
        exec("BEGIN");
        try {
            if constexpr (std::is_void_v<decltype(body())>) {
                body();
                exec("COMMIT");
            } else {
                auto result = body();
                exec("COMMIT");
                return result;
            }
        } catch (...) {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            dimension_ = savedDimension;
            vec_ready_ = savedVecReady;
            modelsCache_.reset();
            throw;
        }
    }

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
        modelsCache_.reset();
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
     * @param path             Database path, or ":memory:"
     * @param encoder          Encoder used when a text is added without a vector
     * @param sqliteVec        Answer top-k searches through a sqlite-vec vec0
     *                         virtual table instead of scanning every row. Results
     *                         are the same either way - vec0 runs an exact
     *                         brute-force KNN in C.
     * @param onModelMismatch  What a *text* query does when the encoder's model
     *                         differs from the model that produced the stored
     *                         vectors (vector queries are never checked;
     *                         "unknown"/"provided" vectors never trigger it).
     */
    explicit SqliteEmbeddingStore(const std::string& path = ":memory:",
                                  std::shared_ptr<EmbeddingEncoder> encoder = nullptr,
                                  bool sqliteVec = false,
                                  ModelMismatchMode onModelMismatch = ModelMismatchMode::Warn)
        : encoder_(std::move(encoder)), sqlite_vec_(sqliteVec),
          onModelMismatch_(onModelMismatch) {
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

        try {
            createSchema();
            adoptStoredDimension();
            if (sqlite_vec_ && dimension_.has_value()) ensureVecTable(*dimension_);
        } catch (...) {
            // a throwing constructor never runs the destructor
            sqlite3_close(db_);
            db_ = nullptr;
            throw;
        }
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

    ModelMismatchMode onModelMismatch() const { return onModelMismatch_; }
    void setOnModelMismatch(ModelMismatchMode mode) { onModelMismatch_ = mode; }

    /**
     * Encoder/model version recorded in the sidecar: the one set with
     * setVersion, else the encoder's own, else nullopt.
     */
    std::optional<std::string> version() const {
        if (version_.has_value()) return version_;
        return encoder_ ? encoder_->version() : std::nullopt;
    }
    void setVersion(std::optional<std::string> version) { version_ = std::move(version); }

    /** Metadata of the most recently imported payload (all unset before any). */
    const EmbeddingMeta& lastImportMeta() const { return lastImportMeta_; }

    // -- model lifecycle -------------------------------------------------

    /**
     * Which models produced the stored vectors: {modelName: count}.
     * "unknown" / "provided" appear as-is (rows written by a port that records
     * no model, or handed in with no encoder attached).
     */
    std::map<std::string, size_t> models() const {
        if (!modelsCache_.has_value()) {
            std::map<std::string, size_t> counts;
            Stmt stmt(db_, std::string("SELECT m, COUNT(*) FROM (SELECT ") + MODEL_EXPR +
                               " AS m FROM embeddings) GROUP BY m");
            while (stmt.step()) counts[stmt.columnText(0)] = static_cast<size_t>(stmt.columnInt(1));
            modelsCache_ = std::move(counts);
        }
        return *modelsCache_;
    }

    /** The encoder's model name, or nullopt with no encoder or an unnamed ("unknown") one. */
    std::optional<std::string> currentModel() const {
        if (!encoder_) return std::nullopt;
        std::string name = encoder_->modelName();
        if (name.empty() || isUnknownModel(name)) return std::nullopt;
        return name;
    }

    /**
     * Refs whose stored vector does not belong to the current encoder, sorted.
     *
     * A vector is stale when its recorded model differs from `model` (default:
     * the encoder's modelName) or its length differs from `dimension` (default:
     * the encoder's dimension). Vectors whose model is "unknown"/"provided"
     * cannot be judged by model and are skipped unless `includeUnknown` (a
     * dimension mismatch still counts). With no encoder and no arguments
     * nothing can be judged and the result is empty.
     */
    std::vector<NodeRef> stale(std::optional<std::string> model = std::nullopt,
                               std::optional<size_t> dimension = std::nullopt,
                               bool includeUnknown = false) const {
        if (!model.has_value()) model = currentModel();
        if (!dimension.has_value() && encoder_) dimension = encoder_->dimension();

        std::vector<NodeRef> out;
        Stmt stmt(db_, std::string("SELECT node_type, node_id, ") + MODEL_EXPR +
                           ", LENGTH(vector) FROM embeddings");
        while (stmt.step()) {
            std::string recorded = stmt.columnText(2);
            size_t width = static_cast<size_t>(stmt.columnInt(3)) / sizeof(float);
            bool byModel = model.has_value() && recorded != *model &&
                           (includeUnknown || !isUnknownModel(recorded));
            bool byDim = dimension.has_value() && width != *dimension;
            if (byModel || byDim) out.emplace_back(stmt.columnText(0), stmt.columnText(1));
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    /**
     * Encode a text query, first checking that the encoder's model is the one
     * that made the stored vectors (per the store's ModelMismatchMode).
     * Only text queries go through here; vector queries are never checked.
     *
     * @throws NoEncoderError with no encoder
     * @throws ModelMismatchError with ModelMismatchMode::Error on a mismatch
     */
    std::vector<float> encodeQuery(const std::string& query) const {
        if (!encoder_) throw NoEncoderError("No encoder provided for text query");
        checkModelCompat();
        return encoder_->encode(query);
    }

    // -- writes ------------------------------------------------------------

    std::string add(const NodeRef& ref, const std::string& text,
                    const std::vector<float>& vector = {}) {
        std::vector<float> vec = vector;
        if (vec.empty()) {
            if (!encoder_) throw NoEncoderError("No encoder provided and no vector given");
            vec = encoder_->encode(text);
        }
        std::string model = currentModelStamp();
        return transact([&]() {
            checkDimension(vec, "vector");
            return writeRow(ref, vec, text, model);
        });
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

        std::string model = currentModelStamp();
        transact([&]() {
            for (const auto& vector : vectors) checkDimension(vector, "vector");
            for (size_t i = 0; i < entries.size(); ++i) {
                writeRow(entries[i].first, vectors[i], entries[i].second, model);
            }
        });
        return entries.size();
    }

    std::optional<EmbeddingRecord> get(const NodeRef& ref) const {
        Stmt stmt(db_,
                  std::string("SELECT id, node_type, node_id, vector, text, ") + MODEL_EXPR +
                      " FROM embeddings WHERE node_type = ? AND node_id = ?");
        stmt.bindText(1, ref.type);
        stmt.bindText(2, ref.id);
        if (!stmt.step()) return std::nullopt;

        return EmbeddingRecord{stmt.columnText(0),
                               NodeRef(stmt.columnText(1), stmt.columnText(2)),
                               stmt.columnVector(3), stmt.columnText(4), stmt.columnText(5)};
    }

    bool remove(const NodeRef& ref) {
        return transact([&]() {
            modelsCache_.reset();
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
        });
    }

    size_t count() const {
        Stmt stmt(db_, "SELECT COUNT(*) FROM embeddings");
        return stmt.step() ? static_cast<size_t>(stmt.columnInt(0)) : 0;
    }

    /**
     * Clear all embeddings.
     *
     * @param resetDimension also forget the pinned vector length (and drop the
     *        sqlite-vec index, whose width is fixed), so the next vector may
     *        have a different one - switching to an encoder of a different
     *        dimension. Off by default: a store normally refuses vectors of the
     *        wrong length even when empty. (Before this parameter existed,
     *        clear() always reset it.)
     */
    void clear(bool resetDimension = false) {
        transact([&]() {
            modelsCache_.reset();
            exec("DELETE FROM embeddings");
            if (resetDimension) {
                dropVecTable();
                dimension_.reset();
            } else if (sqlite_vec_ && vec_ready_) {
                exec(std::string("DELETE FROM ") + VEC_TABLE);
            }
        });
    }

    // -- search --------------------------------------------------------------

    std::vector<SimilarityResult> similaritySearch(const std::string& query, size_t topK = 10,
                                                   const std::string& nodeType = "",
                                                   float threshold = 0.0f) {
        return similaritySearchVector(encodeQuery(query), topK, nodeType, threshold);
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

    // -- portability ---------------------------------------------------------

    /**
     * Serialize every embedding into the portable cross-language payload.
     *
     * Besides the embeddings (each with its recorded model) it carries the
     * optional metadata: the encoder's model and the store's version when
     * known, and `graphDigest` (with its basis) when given - see graphDigest().
     */
    ExportedEmbeddings exportEmbeddings(
        const std::optional<std::string>& graphDigest = std::nullopt) const {
        ExportedEmbeddings payload;
        payload.dimension = dimension_;

        Stmt stmt(db_,
                  std::string("SELECT node_type, node_id, node_id_type, vector, text, ") +
                      MODEL_EXPR + " FROM embeddings ORDER BY node_type, node_id");
        while (stmt.step()) {
            payload.embeddings.push_back(ExportedEmbedding{
                stmt.columnText(0), stmt.columnText(1), stmt.columnText(2),
                stmt.columnText(4), stmt.columnVector(3), stmt.columnText(5)});
        }
        payload.count = payload.embeddings.size();

        payload.meta.encoderModel = currentModel();
        payload.meta.encoderVersion = version();
        if (graphDigest.has_value()) {
            payload.meta.graphDigest = graphDigest;
            payload.meta.graphDigestBasis = GRAPH_DIGEST_BASIS;
        }
        return payload;
    }

    /**
     * Load a payload produced by exportEmbeddings in any language port. Each
     * entry keeps the model it was exported with ("unknown" when the source
     * records none). All-or-nothing: a payload that fails part way (a vector of
     * the wrong width) changes nothing, with `replace` as well.
     */
    size_t importEmbeddings(const ExportedEmbeddings& payload, bool replace = false) {
        if (payload.format != std::string(EMBEDDING_FORMAT)) {
            throw EmbeddingError(std::string("not an ") + EMBEDDING_FORMAT +
                                 " payload: format=" + payload.format);
        }
        if (payload.version != EMBEDDING_FORMAT_VERSION) {
            throw EmbeddingError(std::string("unsupported ") + EMBEDDING_FORMAT + " version " +
                                 std::to_string(payload.version) + " (this build reads version " +
                                 std::to_string(EMBEDDING_FORMAT_VERSION) + ")");
        }

        lastImportMeta_ = payload.meta;
        return transact([&]() {
            if (replace) {
                modelsCache_.reset();
                exec("DELETE FROM embeddings");
                dropVecTable();
                dimension_.reset();
                if (payload.dimension.has_value() && *payload.dimension > 0) {
                    dimension_ = payload.dimension;
                }
            }

            size_t written = 0;
            for (const auto& item : payload.embeddings) {
                std::vector<float> vector = item.vector;
                checkDimension(vector, "embedding for :" + item.type + ":" + item.id);
                writeRow(NodeRef(item.type, item.id), vector, item.text,
                         item.model.empty() ? "unknown" : item.model, item.id_type);
                ++written;
            }
            return written;
        });
    }

    /** Serialize the portable payload to a JSON string (see EmbeddingStore::toJson). */
    std::string toJson(const std::optional<std::string>& graphDigest = std::nullopt) const {
        return EmbeddingStore::serializePayload(exportEmbeddings(graphDigest));
    }

    /** Load a JSON payload written by any language port (see EmbeddingStore::fromJson). */
    size_t fromJson(const std::string& json, bool replace = false) {
        return importEmbeddings(EmbeddingStore::parseJson(json), replace);
    }

    /**
     * Write the portable payload to `path` (the file EmbeddingStore::saveEmbeddings
     * and Python's save_embeddings write), atomically: a temp file is written
     * and renamed into place, so a crash never leaves a half-written sidecar.
     * `graphDigest` ties the sidecar to one particular graph (see graphDigest()).
     * Returns the number of embeddings written.
     */
    size_t saveEmbeddings(const std::string& path,
                          const std::optional<std::string>& graphDigest = std::nullopt) const {
        ExportedEmbeddings payload = exportEmbeddings(graphDigest);
        detail::atomicWriteText<EmbeddingError>(path, EmbeddingStore::serializePayload(payload));
        return payload.count;
    }

    /** Read a payload file written by saveEmbeddings (or any port's equivalent). */
    size_t loadEmbeddings(const std::string& path, bool replace = false) {
        return importEmbeddings(EmbeddingStore::parseJson(EmbeddingStore::readFile(path)), replace);
    }
};

}  // namespace isongraph_vector

#endif  // ISONGRAPH_VECTOR_SQLITE_HPP
