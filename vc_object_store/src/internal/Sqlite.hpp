#pragma once
#include <sqlite3.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects { struct StatementPool; }
namespace vc::objects::detail {
[[noreturn]] void fail(sqlite3* db, const std::string& what);
void check(int rc, sqlite3* db, const char* what = "sqlite");
void exec(sqlite3* db, const char* sql);
std::array<unsigned char, 16> parseUuid(const std::string& uuid);
std::string formatUuid(const std::array<unsigned char, 16>& bytes);
std::string nowIsoUtc();
std::string detectHostname();
void applyPragmas(sqlite3* db, const OpenOptions& options, bool queryOnly);
sqlite3* openConnection(const std::filesystem::path& path, int flags, const OpenOptions& options, bool queryOnly);
int pragmaInt(sqlite3* db, const char* pragma);

class Statement {
public:
    Statement(sqlite3* db, const char* sql);
    Statement(sqlite3* db, const std::string& sql);
    template<class Sql, class... Args> requires (sizeof...(Args) > 0)
    Statement(sqlite3* db, const Sql& sql, const Args&... args) : Statement(db, sql) {
        bindAll(args...);
    }
    ~Statement();
    void bind(int i, int64_t v);
    void bind(int i, double v);
    void bind(int i, const std::string& v);
    void bind(int i, std::span<const uint8_t> bytes);
    template<class T> void bind(int i, const std::optional<T>& value) {
        if (value) bind(i, *value); else bindNull(i);
    }
    void bindNull(int i);
    void bind(int i, const std::array<unsigned char, 16>& v);
    template<class... Args> void bindAll(const Args&... args) {
        int index = 1;
        (bind(index++, args), ...);
    }
    template<class... Args> void run(const Args&... args) {
        reset();
        bindAll(args...);
        stepDone();
    }
    bool row();
    void stepDone();
    void reset();
    bool isNull(int c) const;
    int64_t i64(int c) const;
    double f64(int c) const;
    std::string text(int c) const;
    std::optional<std::string> optText(int c) const;
    std::array<unsigned char, 16> blob16(int c) const;
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    sqlite3_stmt* value = nullptr;
private:
    sqlite3* db_;
    std::string sql_;
    StatementPool* pool_ = nullptr;
};

template<class Sql, class... Args>
void execute(sqlite3* db, const Sql& sql, const Args&... args) {
    Statement statement(db, sql, args...);
    statement.stepDone();
}

// Rolls back just this operation if an exception escapes, leaving the caller's
// enclosing transaction usable. Nested instances use SQLite's savepoint stack.
class Savepoint {
public:
    explicit Savepoint(sqlite3* db);
    ~Savepoint();
    Savepoint(const Savepoint&) = delete;
    Savepoint& operator=(const Savepoint&) = delete;
    void release();
private:
    sqlite3* db_;
};
} // namespace vc::objects::detail
