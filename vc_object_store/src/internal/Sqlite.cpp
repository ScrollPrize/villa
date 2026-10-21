#include "internal/Database.hpp"
#include "internal/ReadRequest.hpp"

namespace vc::objects::detail {
[[noreturn]] void fail(sqlite3* db, const std::string& what)
{
    throw std::runtime_error(what + ": " + (db ? sqlite3_errmsg(db) : "no database"));
}

void check(int rc, sqlite3* db, const char* what)
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

Statement::Statement(sqlite3* db, const char* sql) : db_(db), sql_(sql)
{
    if (auto lease = activeStatementPools.find(db); lease != activeStatementPools.end()) {
        pool_ = lease->second;
        if (auto cached = pool_->idle.find(sql_); cached != pool_->idle.end()) {
            value = cached->second; pool_->idle.erase(cached); return;
        }
    }
    check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr), db, "prepare");
}

Statement::Statement(sqlite3* db, const std::string& sql) : Statement(db, sql.c_str())
{}

Statement::~Statement()
{
    sqlite3_reset(value);
    sqlite3_clear_bindings(value); // release SQLITE_STATIC buffers too
    const auto lease = activeStatementPools.find(db_);
    if (pool_ && lease != activeStatementPools.end() && lease->second == pool_ && pool_->idle.size() < 128) {
        try { pool_->idle.emplace(sql_, value); return; } catch (...) {}
    }
    sqlite3_finalize(value);
}

void Statement::bind(int i, int64_t v)
{ check(sqlite3_bind_int64(value, i, v), db_, "bind"); }

void Statement::bind(int i, double v)
{ check(sqlite3_bind_double(value, i, v), db_, "bind"); }

void Statement::bind(int i, const std::string& v)
{ check(sqlite3_bind_text(value, i, v.c_str(), int(v.size()), SQLITE_TRANSIENT), db_, "bind"); }

void Statement::bindNull(int i)
{ check(sqlite3_bind_null(value, i), db_, "bind"); }

void Statement::bindBlob(int i, const std::array<unsigned char, 16>& v)
{
    check(sqlite3_bind_blob(value, i, v.data(), 16, SQLITE_TRANSIENT), db_, "bind");
}

bool Statement::row()
{
    requestPoll();
    const int rc = sqlite3_step(value);
    requestPoll();
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    fail(db_, "step");
}

void Statement::stepDone()
{
    if (row()) fail(db_, "unexpected row");
}

void Statement::reset()
{
    check(sqlite3_reset(value), db_, "reset");
    check(sqlite3_clear_bindings(value), db_, "clear_bindings");
}

bool Statement::isNull(int c) const
{ return sqlite3_column_type(value, c) == SQLITE_NULL; }

int64_t Statement::i64(int c) const
{ return sqlite3_column_int64(value, c); }

double Statement::f64(int c) const
{ return sqlite3_column_double(value, c); }

std::string Statement::text(int c) const
{
    if (readRequest) requestBytes(sqlite3_column_bytes(value, c), &FiberRegionStats::metadataBytes);
    const auto* p = sqlite3_column_text(value, c);
    return p ? std::string(reinterpret_cast<const char*>(p), size_t(sqlite3_column_bytes(value, c))) : "";
}

std::optional<std::string> Statement::optText(int c) const
{
    return isNull(c) ? std::nullopt : std::optional(text(c));
}

std::array<unsigned char, 16> Statement::blob16(int c) const
{
    if (sqlite3_column_bytes(value, c) != 16)
        throw std::runtime_error("object store: uuid column is not 16 bytes");
    std::array<unsigned char, 16> out{};
    std::memcpy(out.data(), sqlite3_column_blob(value, c), 16);
    return out;
}

Savepoint::Savepoint(sqlite3* db) : db_(db) { exec(db_, "SAVEPOINT object_operation"); }
Savepoint::~Savepoint()
{
    if (db_) sqlite3_exec(db_, "ROLLBACK TO object_operation; RELEASE object_operation", nullptr, nullptr, nullptr);
}
void Savepoint::release()
{
    exec(db_, "RELEASE object_operation");
    db_ = nullptr;
}
} // namespace vc::objects::detail
