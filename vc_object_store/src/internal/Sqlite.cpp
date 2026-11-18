#include "internal/Database.hpp"
#include "internal/ReadRequest.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include <limits>

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
        if(sqlite3_extended_errcode(db)==SQLITE_CONSTRAINT_FOREIGNKEY){
            // Failed commits otherwise hide which reference could not be
            // satisfied. This diagnostic does no work on successful writes.
            sqlite3_stmt* check=nullptr;
            if(sqlite3_prepare_v2(db,"PRAGMA foreign_key_check",-1,&check,nullptr)==SQLITE_OK&&sqlite3_step(check)==SQLITE_ROW){
                msg+=" ["+std::string(reinterpret_cast<const char*>(sqlite3_column_text(check,0)))+
                    " -> "+std::string(reinterpret_cast<const char*>(sqlite3_column_text(check,2)))+
                    ", row "+std::to_string(sqlite3_column_int64(check,1))+
                    ", key "+std::to_string(sqlite3_column_int(check,3))+"]";
            }
            sqlite3_finalize(check);
        }
        throw std::runtime_error("sqlite exec failed: " + msg + " (" + sql + ")");
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

HashKey parseHash(const std::string& hex)
{
    auto digit = [&](char ch) {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        throw std::runtime_error("object store: not a lowercase SHA-256 hex digest: " + hex);
    };
    if (hex.size() != 64) throw std::runtime_error("object store: not a lowercase SHA-256 hex digest: " + hex);
    HashKey out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<unsigned char>(digit(hex[2 * i]) << 4 | digit(hex[2 * i + 1]));
    return out;
}

std::string formatHash(const unsigned char* bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < 32; ++i) { out[2 * i] = digits[bytes[i] >> 4]; out[2 * i + 1] = digits[bytes[i] & 15]; }
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

static std::string isoUtc(std::chrono::system_clock::time_point now, bool fractional)
{
    const auto second = std::chrono::floor<std::chrono::seconds>(now);
    const std::time_t t = std::chrono::system_clock::to_time_t(second);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    if (!fractional) return std::string(buf) + "Z";
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now-second).count();
    return std::string(buf) + "." + std::to_string(1000000 + micros).substr(1) + "Z";
}

std::string nowIsoUtc(bool fractional)
{
    return isoUtc(std::chrono::system_clock::now(), fractional);
}

std::string refineCreationStamp(const std::string& created, std::optional<int64_t> publicationUs)
{
    if (created.size() != 20 || !publicationUs) return created;
    const auto publication = isoUtc(std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::microseconds(*publicationUs))), true);
    return publication.compare(0, 19, created, 0, 19) == 0 ? publication : created;
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
    const auto cacheKiB=options.sqliteCacheBytes/1024+(options.sqliteCacheBytes%1024!=0);
    if(cacheKiB>static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("object store: SQLite cache budget exceeds supported range");
    const auto cachePragma="PRAGMA cache_size=-"+std::to_string(cacheKiB)+";";
    exec(db,cachePragma.c_str());
    if (queryOnly)
        exec(db, "PRAGMA query_only=ON;");
}

sqlite3* openConnection(const std::filesystem::path& path, int flags, const OpenOptions& options, bool queryOnly)
{
    sqlite3* db = nullptr;
    const bool packedLocator = isPackedDatabasePath(path);
    const bool packedFile = !packedLocator && isPackedStore(path);
    const bool packed = packedLocator || packedFile;
    const auto name = packedFile ? packedDatabasePath(path).string() : path.string();
    const char* vfs = packed ? packedStoreVfs() : nullptr;
    if (vfs && !sqlite3_vfs_find(vfs))
        throw std::runtime_error("object store: this SQLite build lacks the unix-dotfile VFS needed for shared files");
    const int rc = sqlite3_open_v2(packed ? name.c_str() : reinterpret_cast<const char*>(path.u8string().c_str()), &db,
                                   flags & SQLITE_OPEN_NOMUTEX ? flags : flags | SQLITE_OPEN_FULLMUTEX, vfs);
    if (rc != SQLITE_OK) {
        std::string msg = db ? sqlite3_errmsg(db) : "open failed";
        if (packed && !packedStoreLastError().empty()) msg += " (" + packedStoreLastError() + ")";
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

void Statement::bindBorrowedText(int i, const std::string& v)
{ check(sqlite3_bind_text(value, i, v.data(), int(v.size()), SQLITE_STATIC), db_, "bind"); }

void Statement::bind(int i, std::span<const uint8_t> bytes)
{ check(sqlite3_bind_blob64(value, i, bytes.data(), bytes.size(), SQLITE_TRANSIENT), db_, "bind"); }

void Statement::bindNull(int i)
{ check(sqlite3_bind_null(value, i), db_, "bind"); }

void Statement::bind(int i, const std::array<unsigned char, 16>& v)
{
    check(sqlite3_bind_blob(value, i, v.data(), 16, SQLITE_TRANSIENT), db_, "bind");
}

void Statement::bind(int i, const HashKey& v)
{
    check(sqlite3_bind_blob(value, i, v.data(), int(v.size()), SQLITE_TRANSIENT), db_, "bind");
}

std::string Statement::hash(int c) const
{
    if (sqlite3_column_bytes(value, c) != 32)
        throw std::runtime_error("object store: hash column is not 32 bytes");
    return formatHash(static_cast<const unsigned char*>(sqlite3_column_blob(value, c)));
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
    resetKeepingBindings();
    check(sqlite3_clear_bindings(value), db_, "clear_bindings");
}

void Statement::resetKeepingBindings()
{ check(sqlite3_reset(value), db_, "reset"); }

bool Statement::isNull(int c) const
{ return sqlite3_column_type(value, c) == SQLITE_NULL; }

int64_t Statement::i64(int c) const
{ return sqlite3_column_int64(value, c); }

double Statement::f64(int c) const
{ return sqlite3_column_double(value, c); }

std::string Statement::text(int c) const
{
    if (readRequest) requestBytes(sqlite3_column_bytes(value, c), &RegionStats::metadataBytes);
    const auto* p = sqlite3_column_text(value, c);
    return p ? std::string(reinterpret_cast<const char*>(p), size_t(sqlite3_column_bytes(value, c))) : "";
}

std::optional<std::string> Statement::optText(int c) const
{
    return isNull(c) ? std::nullopt : std::optional(text(c));
}

std::string Statement::uuid(int c) const { return formatUuid(blob16(c)); }

std::optional<std::string> Statement::optUuid(int c) const
{
    return isNull(c) ? std::nullopt : std::optional(uuid(c));
}

std::optional<std::array<unsigned char, 16>> parseOptionalUuid(const std::optional<std::string>& uuid)
{
    return uuid ? std::optional(parseUuid(*uuid)) : std::nullopt;
}

std::array<unsigned char, 16> Statement::blob16(int c) const
{
    if (sqlite3_column_bytes(value, c) != 16)
        throw std::runtime_error("object store: uuid column is not 16 bytes");
    std::array<unsigned char, 16> out{};
    std::memcpy(out.data(), sqlite3_column_blob(value, c), 16);
    return out;
}

Savepoint::Savepoint(sqlite3* db) : db_(db) {
    if(const auto found=activeStatementPools.find(db);found!=activeStatementPools.end()) {
        pool_=found->second;currentWrite_=pool_->currentWrite;
    }
    exec(db_, "SAVEPOINT object_operation");
}
Savepoint::~Savepoint()
{
    if (db_) {
        sqlite3_exec(db_, "ROLLBACK TO object_operation; RELEASE object_operation", nullptr, nullptr, nullptr);
        if(pool_)pool_->currentWrite=std::move(currentWrite_);
    }
}
void Savepoint::release()
{
    exec(db_, "RELEASE object_operation");
    db_ = nullptr;
}
} // namespace vc::objects::detail
