#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/FiberCollection.hpp"

#include <sqlite3.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace vc::fibers;
namespace fs = std::filesystem;

struct FixtureFiber {
    std::vector<Point> points;
    std::string family;
    std::string annotation = R"({"type":"vc3d_fiber","version":4})";
};

void exec(sqlite3* db, const std::string& sql)
{
    char* error = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error) != SQLITE_OK) {
        const std::string message = error ? error : "SQLite error";
        sqlite3_free(error);
        throw std::runtime_error(message);
    }
}

std::vector<double> boundsOf(const std::vector<Point>& points, size_t first, size_t last)
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
        exec(db,
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
        exec(db,
             "INSERT INTO metadata VALUES('complete','" + std::string(complete ? "true" : "false") +
                 "'),('uuid','\"7f3c1a52-8f0e-4b8e-9d1a-2f6b0c4e5a91\"'),"
                 "('frame','{\"vc_open_data_coordinate_space\":\"test\"}'),('root','{}'),"
                 "('fiber_count','" + std::to_string(fibers.size()) + "'),('point_count','" + std::to_string(points) + "')");
        exec(db, "COMMIT");
    }

    fs::path path_;
};

// Fibers 1 and 3 are identical and 20 voxels long (H); fiber 2 has 600 points
// far from the origin (V); fiber 4 is 2 voxels long with a repeated point (V);
// fiber 5 has no family and only overlaps the unit box through its bounds.
std::vector<FixtureFiber> standardFibers()
{
    std::vector<Point> line;
    for (int i = 0; i < 600; ++i)
        line.push_back({double(i), 20, 30});
    return {
        {{{-10, 0, 0}, {10, 0, 0}}, "H"},
        {line, "V"},
        {{{-10, 0, 0}, {10, 0, 0}}, "H"},
        {{{-1, 0, 0}, {-1, 0, 0}, {1, 0, 0}}, "V", R"({"type":"vc3d_fiber","version":4,"producer_extra":{"sources":[5,9]}})"},
        {{{-10, 10, 0}, {10, 10, 0}, {10, -10, 0}}, ""},
    };
}

const fs::path& standardCollection()
{
    static const FixtureFile file(standardFibers());
    return file.path();
}

std::set<int64_t> visibleIds(const RegionPage& page)
{
    std::set<int64_t> ids;
    for (const auto& block : page.blocks)
        ids.insert(block.fiberId);
    return ids;
}

bool samePage(const RegionPage& a, const RegionPage& b)
{
    if (a.complete != b.complete || a.decodedBytes != b.decodedBytes || a.blocks.size() != b.blocks.size())
        return false;
    for (size_t i = 0; i < a.blocks.size(); ++i) {
        const auto& x = a.blocks[i];
        const auto& y = b.blocks[i];
        if (x.id != y.id || x.fiberId != y.fiberId || x.firstSegment != y.firstSegment || x.family != y.family || x.points != y.points)
            return false;
    }
    return true;
}

ViewRegion axisView(const Bounds& bounds)
{
    ViewRegion view;
    view.box = view.local = bounds;
    return view;
}

constexpr size_t kBudget = 8 * 1024 * 1024;
const Bounds kUnit{{-1, -1, -1}, {1, 1, 1}};
const double kDiagonal = std::sqrt(.5);
}  // namespace

TEST_CASE("FiberCollection opens only complete version 1 .afv files")
{
    const FiberCollection collection(standardCollection());
    CHECK(collection.metadata("fiber_count") == "5");
    CHECK(collection.metadata("missing").empty());

    const FixtureFile wrongExtension(standardFibers(), ".sqlite");
    CHECK_THROWS(FiberCollection(wrongExtension.path()));
    const FixtureFile incomplete(standardFibers(), ".afv", false);
    CHECK_THROWS(FiberCollection(incomplete.path()));
    const FixtureFile futureVersion(standardFibers(), ".afv", true, 2);
    CHECK_THROWS(FiberCollection(futureVersion.path()));
}

TEST_CASE("clipSegment clips exactly and handles degenerate segments")
{
    double lo, hi;
    CHECK((clipSegment({-2, 1, 0}, {2, 1, 0}, kUnit, lo, hi) && lo == .25 && hi == .75));
    CHECK(clipSegment({0, 0, 0}, {0, 0, 0}, kUnit, lo, hi));
    CHECK_FALSE(clipSegment({0, 2, 0}, {0, 2, 0}, kUnit, lo, hi));
}

TEST_CASE("viewRegion returns the fibers crossing the slice, with quota and selection")
{
    const FiberCollection c(standardCollection());
    const auto view = axisView(kUnit);
    const std::set<int64_t> crossing{1, 3, 4};
    CHECK(visibleIds(c.viewRegion(view)) == crossing);
    CHECK(visibleIds(c.viewRegion(view, 2, 1000)) == crossing);
    const auto one = visibleIds(c.viewRegion(view, 2, 1));
    CHECK(one.size() == 1);
    CHECK(one == visibleIds(c.viewRegion(view, 2, 1)));
    CHECK(visibleIds(c.viewRegion(view, 2, 1, 4)) == std::set<int64_t>{4});
    CHECK(visibleIds(c.viewRegion(view, 2, 1, 2)) == one);  // An offscreen selection does not use the quota.
    CHECK(visibleIds(c.viewRegion(view, 3, 1000)) == std::set<int64_t>{4});
    CHECK(c.viewRegion(axisView({{10000, 10000, 10000}, {10001, 10001, 10001}})).blocks.empty());

    // Broad-phase boxes overlap this oblique prism, but no actual segment does.
    auto oblique = view;
    oblique.box = {{-20, -20, -1}, {20, 20, 1}};
    oblique.origin = {0, 3, 0};
    oblique.u = {kDiagonal, kDiagonal, 0};
    oblique.v = {-kDiagonal, kDiagonal, 0};
    oblique.local = {{-.1, -.1, -.1}, {.1, .1, .1}};
    CHECK(c.viewRegion(oblique, 2, 1000).blocks.empty());

    std::atomic_bool cancelled{true};
    CHECK_THROWS(c.viewRegion(axisView({{-100, -100, -100}, {1000, 1000, 1000}}), 2, 0, 0, kBudget, &cancelled));
}

TEST_CASE("viewRegion filters by complete fiber length and family")
{
    const FiberCollection c(standardCollection());
    const auto view = axisView(kUnit);
    const std::set<int64_t> longCrossings{1, 3};
    CHECK(visibleIds(c.viewRegion(view, 2, 0, 0, kBudget, nullptr, 20)) == longCrossings);
    CHECK(visibleIds(c.viewRegion(view, 2, 1000, 4, kBudget, nullptr, 20)) == longCrossings);
    const auto filteredOne = visibleIds(c.viewRegion(view, 2, 1, 4, kBudget, nullptr, 20));
    REQUIRE(filteredOne.size() == 1);
    CHECK(longCrossings.contains(*filteredOne.begin()));
    CHECK(c.viewRegion(view, 2, 0, 0, kBudget, nullptr, 20.01).blocks.empty());
    CHECK(visibleIds(c.viewRegion(view, 2, 1000, 0, kBudget, nullptr, 0, FamilyFilter::Horizontal)) == longCrossings);
    CHECK(visibleIds(c.viewRegion(view, 2, 0, 0, kBudget, nullptr, 0, FamilyFilter::Vertical)) == std::set<int64_t>{4});
    CHECK(c.viewRegion(view, 2, 1000, 0, kBudget, nullptr, 20, FamilyFilter::Vertical).blocks.empty());
    CHECK(c.viewRegion(view, 2, 0, 0, kBudget, nullptr, 0, FamilyFilter::None).blocks.empty());
}

TEST_CASE("visible runs keep original coordinates and indices across storage blocks")
{
    const FiberCollection c(standardCollection());
    const auto view = axisView({{255.5, 19, 29}, {256.5, 21, 31}});
    const auto page = c.viewRegion(view, 2, 1);
    CHECK(page.complete);
    REQUIRE(page.blocks.size() == 2);
    for (const auto& block : page.blocks)
        for (size_t i = 0; i < block.points.size(); ++i)
            CHECK(block.points[i] == c.point(block.fiberId, block.firstSegment + int64_t(i)));
    CHECK_FALSE(c.viewRegion(view, 2, 1, 0, 48).complete);

    const auto line = c.fiberBlocks(2);
    REQUIRE(line.size() == 3);
    size_t count = 0;
    int64_t expectedStart = 0;
    for (const auto& block : line) {
        CHECK(block.firstSegment == expectedStart);
        for (size_t i = 0; i < block.points.size(); ++i)
            CHECK(block.points[i] == Point{double(expectedStart + int64_t(i)), 20, 30});
        count += block.points.size() - (count ? 1 : 0);
        expectedStart += int64_t(block.points.size()) - 1;
    }
    CHECK(count == 600);
    CHECK(c.point(2, 256) == Point{256, 20, 30});
    CHECK(c.point(2, 599) == Point{599, 20, 30});
    CHECK_THROWS(c.point(2, 600));
    CHECK(c.summary(1).id != c.summary(3).id);  // Identical geometries remain distinct fibers.
    CHECK(c.annotation(4).find("producer_extra") != std::string::npos);
}

TEST_CASE("ViewRegion::contains accepts only slices inside the loaded prism")
{
    ViewRegion loaded;
    loaded.local = {{0, 0, -1}, {10, 10, 1}};
    auto inside = loaded;
    inside.origin = {1, 2, 0};
    inside.local.upper = {8, 7, 1};
    CHECK(loaded.contains(inside));
    inside.origin[2] = .01;
    CHECK_FALSE(loaded.contains(inside));
    inside = loaded;
    inside.origin[0] = 1;
    CHECK_FALSE(loaded.contains(inside));
    loaded.u = {kDiagonal, kDiagonal, 0};
    loaded.v = {-kDiagonal, kDiagonal, 0};
    inside = loaded;
    inside.local.upper = {5, 5, 1};
    inside.origin = {0, 2, 0};
    CHECK(loaded.contains(inside));
    inside.origin = {0, 8, 0};
    CHECK_FALSE(loaded.contains(inside));
    inside = loaded;
    inside.n = {0, .1, 1};
    CHECK_FALSE(loaded.contains(inside));
}

TEST_CASE("the view cache returns exactly the uncached result")
{
    const FiberCollection c(standardCollection());
    const auto view = axisView(kUnit);
    for (size_t cacheBytes : {size_t(1024), size_t(8192), size_t(1024 * 1024)}) {
        const FiberCollection cached(standardCollection(), cacheBytes);
        for (int step = 0; step < 36; ++step) {
            auto query = axisView({{-.5 + (step % 6) * .1, -.5, -.5}, {1.5 + (step % 6) * .1, 1.5, 1.5}});
            if (step % 3 == 1) {
                query.u = {kDiagonal, kDiagonal, 0};
                query.v = {-kDiagonal, kDiagonal, 0};
            }
            const auto family = step % 5 == 0 ? FamilyFilter::Horizontal : FamilyFilter::All;
            const double length = step % 7 == 0 ? 20 : 0;
            const size_t limit = step % 4;
            const size_t bytes = step % 9 == 0 ? 48 : kBudget;
            const int64_t selected = step % 2 ? 1 : 4;
            CHECK(samePage(
                c.viewRegion(query, 2, limit, selected, bytes, nullptr, length, family),
                cached.viewRegion(query, 2, limit, selected, bytes, nullptr, length, family)));
            CHECK(cached.viewCacheStats().retainedBytes <= cacheBytes);
        }
        const auto first = cached.viewRegion(view, 2, 0);
        const auto stats = cached.viewCacheStats();
        CHECK(samePage(first, cached.viewRegion(view, 2, 0)));
        if (cacheBytes > 8192) {
            CHECK(cached.viewCacheStats().candidateHits > stats.candidateHits);
            CHECK(cached.viewCacheStats().blockReads == stats.blockReads);
        }
        std::atomic_bool stop{true};
        CHECK_THROWS(cached.viewRegion(view, 2, 0, 0, kBudget, &stop));
        CHECK(samePage(first, cached.viewRegion(view, 2, 0)));  // A cancelled query leaves no partial cache.
    }
}

TEST_CASE("catalogByLength pages through all fibers by descending length")
{
    const FiberCollection c(standardCollection());
    const double infinity = std::numeric_limits<double>::infinity();
    double beforeLength = infinity;
    int64_t afterId = 0;
    std::vector<int64_t> ordered;
    for (int page = 0; page < 10; ++page) {
        const auto rows = c.catalogByLength(beforeLength, afterId, 1);
        if (rows.empty())
            break;
        ordered.push_back(rows.front().id);
        beforeLength = rows.front().length;
        afterId = rows.front().id;
    }
    CHECK(ordered == std::vector<int64_t>({2, 5, 1, 3, 4}));
    const auto onlyVertical = c.catalogByLength(infinity, 0, 200, 20, FamilyFilter::Vertical);
    REQUIRE(onlyVertical.size() == 1);
    CHECK(onlyVertical.front().id == 2);
    CHECK(c.catalogByLength(infinity, 0, 200, 0, FamilyFilter::None).empty());
}
