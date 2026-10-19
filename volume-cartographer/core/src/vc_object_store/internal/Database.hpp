#pragma once
#include "vc/core/objects/ObjectStore.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <ctime>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "vc/core/util/Uuid.hpp"

namespace vc::objects {
namespace detail { class GeometryReadCache; class SpatialReadCache; }
// Connections are leased to one thread per transaction. Cache idle statements
// on the connection; the thread-local registry only identifies the active lease.
struct StatementPool {
    std::shared_ptr<detail::GeometryReadCache> geometryCache;
    std::shared_ptr<detail::SpatialReadCache> spatialCache;
    std::unordered_multimap<std::string, sqlite3_stmt*> idle;
    ~StatementPool() { for (auto& [sql, stmt] : idle) sqlite3_finalize(stmt); }
};
inline thread_local std::unordered_map<sqlite3*, StatementPool*> activeStatementPools;
namespace detail {
// Called only after a read connection has pinned its new SQLite snapshot.
void beginReadCaches(sqlite3*, StatementPool&, size_t geometryBudget, size_t spatialBudget);
void beginSpatialRead(StatementPool&, size_t budget, int version);
}
struct ObjectStore::Impl {
    std::filesystem::path path;
    OpenOptions options;
    std::string hostname;
    sqlite3* writer = nullptr;
    std::mutex writerMutex;
    std::vector<sqlite3*> readerPool;
    std::mutex poolMutex;
    std::condition_variable poolCv;
    bool closing = false;
    std::map<sqlite3*, std::unique_ptr<StatementPool>> statementPools;

    ~Impl()
    {
        statementPools.clear();
        for (sqlite3* c : readerPool)
            if (c) sqlite3_close(c);
        if (writer) sqlite3_close(writer);
    }

    sqlite3* checkoutReader()
    {
        std::unique_lock<std::mutex> lock(poolMutex);
        poolCv.wait(lock, [&] { return closing || !readerPool.empty(); });
        if (readerPool.empty())
            throw std::runtime_error("object store: handle is closing");
        sqlite3* conn = readerPool.back();
        readerPool.pop_back();
        return conn;
    }
    void checkinReader(sqlite3* conn)
    {
        {
            std::lock_guard<std::mutex> lock(poolMutex);
            readerPool.push_back(conn);
        }
        poolCv.notify_one();
    }
};

struct ObjectStoreImplAccess {
    static sqlite3* conn(const Txn& txn)
    {
        if (!txn.active())
            throw std::runtime_error("object store: use of finished transaction");
        return static_cast<sqlite3*>(txn.handle_);
    }
};

namespace {
[[noreturn]] void fail(sqlite3* db, const std::string& what)
{
    throw std::runtime_error(what + ": " + (db ? sqlite3_errmsg(db) : "no database"));
}

void check(int rc, sqlite3* db, const char* what = "sqlite")
{
    if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE)
        fail(db, what);
}

void exec(sqlite3* db, const char* sql)
{
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : sqlite3_errmsg(db);
        sqlite3_free(err);
        throw std::runtime_error("sqlite exec failed: " + msg);
    }
}

class Statement {
public:
    Statement(sqlite3* db, const char* sql) : db_(db), sql_(sql)
    {
        if (auto lease = activeStatementPools.find(db); lease != activeStatementPools.end()) {
            pool_ = lease->second;
            if (auto cached = pool_->idle.find(sql_); cached != pool_->idle.end()) {
                value = cached->second; pool_->idle.erase(cached); return;
            }
        }
        check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr), db, "prepare");
    }
    Statement(sqlite3* db, const std::string& sql) : Statement(db, sql.c_str()) {}
    ~Statement() {
        sqlite3_reset(value);
        sqlite3_clear_bindings(value); // release SQLITE_STATIC buffers too
        const auto lease = activeStatementPools.find(db_);
        if (pool_ && lease != activeStatementPools.end() && lease->second == pool_ && pool_->idle.size() < 128) {
            try { pool_->idle.emplace(sql_, value); return; } catch (...) {}
        }
        sqlite3_finalize(value);
    }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    void bind(int i, int64_t v) { check(sqlite3_bind_int64(value, i, v), db_, "bind"); }
    void bind(int i, double v) { check(sqlite3_bind_double(value, i, v), db_, "bind"); }
    void bind(int i, const std::string& v) { check(sqlite3_bind_text(value, i, v.c_str(), int(v.size()), SQLITE_TRANSIENT), db_, "bind"); }
    void bindNull(int i) { check(sqlite3_bind_null(value, i), db_, "bind"); }
    void bindBlob(int i, const std::array<unsigned char, 16>& v)
    {
        check(sqlite3_bind_blob(value, i, v.data(), 16, SQLITE_TRANSIENT), db_, "bind");
    }
    bool row()
    {
        const int rc = sqlite3_step(value);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        fail(db_, "step");
    }
    void stepDone()
    {
        if (row()) fail(db_, "unexpected row");
    }
    void reset()
    {
        check(sqlite3_reset(value), db_, "reset");
        check(sqlite3_clear_bindings(value), db_, "clear_bindings");
    }

    bool isNull(int c) const { return sqlite3_column_type(value, c) == SQLITE_NULL; }
    int64_t i64(int c) const { return sqlite3_column_int64(value, c); }
    double f64(int c) const { return sqlite3_column_double(value, c); }
    std::string text(int c) const
    {
        const auto* p = sqlite3_column_text(value, c);
        return p ? std::string(reinterpret_cast<const char*>(p), size_t(sqlite3_column_bytes(value, c))) : "";
    }
    std::optional<std::string> optText(int c) const
    {
        if (isNull(c)) return std::nullopt;
        return text(c);
    }
    std::array<unsigned char, 16> blob16(int c) const
    {
        if (sqlite3_column_bytes(value, c) != 16)
            throw std::runtime_error("object store: uuid column is not 16 bytes");
        std::array<unsigned char, 16> out{};
        std::memcpy(out.data(), sqlite3_column_blob(value, c), 16);
        return out;
    }

    sqlite3_stmt* value = nullptr;

private:
    sqlite3* db_;
    std::string sql_;
    StatementPool* pool_ = nullptr;
};

std::array<unsigned char, 16> parseUuid(const std::string& uuid)
{
    if (!util::isUuid(uuid))
        throw std::runtime_error("object store: not a UUID: " + uuid);
    std::array<unsigned char, 16> out{};
    int nibble = 0;
    for (char ch : uuid) {
        if (ch == '-') continue;
        int v;
        if (ch >= '0' && ch <= '9') v = ch - '0';
        else if (ch >= 'a' && ch <= 'f') v = ch - 'a' + 10;
        else v = ch - 'A' + 10;
        out[size_t(nibble / 2)] = static_cast<unsigned char>((out[size_t(nibble / 2)] << 4) | v);
        ++nibble;
    }
    return out;
}

std::string formatUuid(const std::array<unsigned char, 16>& bytes)
{
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out.push_back('-');
        out.push_back(hex[bytes[i] >> 4]);
        out.push_back(hex[bytes[i] & 0x0f]);
    }
    return out;
}

std::string nowIsoUtc()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string detectHostname()
{
#ifdef _WIN32
    const char* name = std::getenv("COMPUTERNAME");
    return name && *name ? name : "unknown";
#else
    char buf[256]{};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0])
        return buf;
    return "unknown";
#endif
}

void applyPragmas(sqlite3* db, const OpenOptions& options, bool queryOnly)
{
    sqlite3_busy_timeout(db, options.busyTimeoutMs);
    exec(db, "PRAGMA foreign_keys=ON;");
    exec(db, "PRAGMA temp_store=MEMORY;");
    exec(db, "PRAGMA cache_size=-8000;");
    if (queryOnly)
        exec(db, "PRAGMA query_only=ON;");
}

sqlite3* openConnection(const std::filesystem::path& path, int flags, const OpenOptions& options, bool queryOnly)
{
    sqlite3* db = nullptr;
    const auto name = path.u8string();
    const int rc = sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()), &db, flags | SQLITE_OPEN_FULLMUTEX, nullptr);
    if (rc != SQLITE_OK) {
        std::string msg = db ? sqlite3_errmsg(db) : "open failed";
        if (db) sqlite3_close(db);
        throw std::runtime_error("object store: cannot open " + path.string() + ": " + msg);
    }
    try {
        applyPragmas(db, options, queryOnly);
    } catch (...) {
        sqlite3_close(db);
        throw;
    }
    return db;
}

int pragmaInt(sqlite3* db, const char* pragma)
{
    Statement s(db, (std::string("PRAGMA ") + pragma).c_str());
    if (!s.row())
        throw std::runtime_error(std::string("object store: PRAGMA ") + pragma + " returned no row");
    return int(s.i64(0));
}

} // namespace
} // namespace vc::objects
