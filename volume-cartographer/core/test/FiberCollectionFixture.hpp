#pragma once

#include "vc/core/types/FiberCollection.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::fibers::test
{
namespace fs = std::filesystem;

struct FixtureFiber {
    std::vector<Point> points;
    std::string family;
    std::string annotation = R"({"type":"vc3d_fiber","version":4})";
};

inline void exec(sqlite3* db, const std::string& sql)
{
    char* error = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error) != SQLITE_OK) {
        const std::string message = error ? error : "SQLite error";
        sqlite3_free(error);
        throw std::runtime_error(message);
    }
}

inline std::vector<double> boundsOf(const std::vector<Point>& points, size_t first, size_t last)
{
    std::vector<double> out;
    for (int axis = 0; axis < 3; ++axis) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (size_t i = first; i <= last; ++i) {
            lo = std::min(lo, points[i][axis]);
            hi = std::max(hi, points[i][axis]);
        }
        out.push_back(lo);
        out.push_back(hi);
    }
    return out;
}

// Writes an Automated Fiber Volume as described in docs/fiber-collections.md,
// the way an external producer would.
class FixtureFile
{
public:
    FixtureFile(const std::vector<FixtureFiber>& fibers, const std::string& extension = ".afv", bool complete = true, int version = 1)
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("vc_test_fiber_collection_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                 std::to_string(counter++) + extension);
        sqlite3* db = nullptr;
        if (sqlite3_open_v2(path_.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
            sqlite3_close(db);
            throw std::runtime_error("Could not create the fixture collection");
        }
        try {
            write(db, fibers, complete, version);
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    }
    ~FixtureFile()
    {
        std::error_code ignored;
        fs::remove(path_, ignored);
    }
    FixtureFile(const FixtureFile&) = delete;
    FixtureFile& operator=(const FixtureFile&) = delete;
    const fs::path& path() const { return path_; }

private:
    static void write(sqlite3* db, const std::vector<FixtureFiber>& fibers, bool complete, int version)
    {
        exec(
            db,
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,"
            "point_count INTEGER NOT NULL CHECK(point_count>=2),length REAL NOT NULL,"
            "min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,"
            "min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);"
            "CREATE INDEX fibers_length ON fibers(length DESC,id);"
            "CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL REFERENCES fibers(id),"
            "first_segment INTEGER NOT NULL,points BLOB NOT NULL,UNIQUE(fiber_id,first_segment));"
            "CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);");
        exec(db, "PRAGMA application_id=" + std::to_string(0x56434643) + "; PRAGMA user_version=" + std::to_string(version) + ";");
        exec(db, "BEGIN");
        sqlite3_stmt* fiber = nullptr;
        sqlite3_stmt* block = nullptr;
        sqlite3_stmt* bounds = nullptr;
        sqlite3_prepare_v2(db, "INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", -1, &fiber, nullptr);
        sqlite3_prepare_v2(db, "INSERT INTO blocks VALUES(?,?,?,?)", -1, &block, nullptr);
        sqlite3_prepare_v2(db, "INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)", -1, &bounds, nullptr);
        auto step = [db](sqlite3_stmt* statement) {
            if (sqlite3_step(statement) != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(db));
            sqlite3_reset(statement);
        };
        int64_t blockId = 0, points = 0;
        for (size_t f = 0; f < fibers.size(); ++f) {
            const auto& p = fibers[f].points;
            const int64_t id = int64_t(f) + 1;
            double length = 0;
            for (size_t i = 1; i < p.size(); ++i)
                length += std::hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1], p[i][2] - p[i - 1][2]);
            const auto box = boundsOf(p, 0, p.size() - 1);
            sqlite3_bind_int64(fiber, 1, id);
            sqlite3_bind_text(fiber, 2, ("fiber_" + std::to_string(id)).c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(fiber, 3, fibers[f].family.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(fiber, 4, int64_t(p.size()));
            sqlite3_bind_double(fiber, 5, length);
            for (int i = 0; i < 6; ++i)
                sqlite3_bind_double(fiber, 6 + i, box[i]);
            sqlite3_bind_text(fiber, 12, fibers[f].annotation.c_str(), -1, SQLITE_TRANSIENT);
            step(fiber);
            points += int64_t(p.size());
            for (size_t start = 0; start + 1 < p.size(); start += 256) {
                const size_t last = std::min(start + 256, p.size() - 1);
                std::vector<unsigned char> blob;
                for (size_t i = start; i <= last; ++i)
                    for (double x : p[i])
                        for (int b = 0; b < 8; ++b)
                            blob.push_back(static_cast<unsigned char>(std::bit_cast<uint64_t>(x) >> (8 * b)));
                sqlite3_bind_int64(block, 1, ++blockId);
                sqlite3_bind_int64(block, 2, id);
                sqlite3_bind_int64(block, 3, int64_t(start));
                sqlite3_bind_blob(block, 4, blob.data(), int(blob.size()), SQLITE_TRANSIENT);
                step(block);
                const auto blockBox = boundsOf(p, start, last);
                sqlite3_bind_int64(bounds, 1, blockId);
                for (int i = 0; i < 6; ++i)
                    sqlite3_bind_double(bounds, 2 + i, blockBox[i]);
                step(bounds);
            }
        }
        sqlite3_finalize(fiber);
        sqlite3_finalize(block);
        sqlite3_finalize(bounds);
        exec(
            db,
            "INSERT INTO metadata VALUES('complete','" + std::string(complete ? "true" : "false") +
                "'),('uuid','\"7f3c1a52-8f0e-4b8e-9d1a-2f6b0c4e5a91\"'),"
                "('frame','{\"vc_open_data_coordinate_space\":\"test\"}'),('root','{}'),"
                "('fiber_count','" +
                std::to_string(fibers.size()) + "'),('point_count','" + std::to_string(points) + "')");
        exec(db, "COMMIT");
    }

    fs::path path_;
};

}  // namespace vc::fibers::test
