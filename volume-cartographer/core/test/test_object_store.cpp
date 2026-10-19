#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/objects/ObjectStore.hpp"
#include <sqlite3.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace vc::objects;
namespace fs = std::filesystem;

struct TempDir {
    fs::path dir;
    explicit TempDir(const std::string& name)
    {
        dir = fs::temp_directory_path() / ("vc_object_test_" + name);
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    ~TempDir() { fs::remove_all(dir); }
};

OpenOptions testOptions()
{
    OpenOptions o;
    o.author = "tester";
    o.hostnameOverride = "testhost";
    o.readerPoolSize = 4;
    return o;
}

ObjectRecord makeObject(const std::string& kind, const std::string& name = "")
{
    ObjectRecord o;
    o.kind = kind;
    o.name = name;
    return o;
}

std::string addObject(ObjectStore& db, const std::string& kind, const std::string& name = "")
{
    auto w = db.write();
    std::string uuid = db.upsertObject(w, makeObject(kind, name));
    w.commit();
    return uuid;
}

} // namespace

TEST_CASE("geometry cache invalidation token is pinned to the SQLite read snapshot")
{
    TempDir tmp("cache_version_snapshot");
    auto open = [&] {
        sqlite3* raw = nullptr;
        const auto rc = sqlite3_open((tmp.dir / "version.sqlite").string().c_str(), &raw);
        std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db(raw, sqlite3_close);
        REQUIRE(rc == SQLITE_OK);
        return db;
    };
    auto writer = open();
    REQUIRE(sqlite3_exec(writer.get(), "PRAGMA journal_mode=WAL; CREATE TABLE t(x); INSERT INTO t VALUES(1)",
                        nullptr, nullptr, nullptr) == SQLITE_OK);
    auto reader = open();
    auto scalar = [&](const char* sql) {
        sqlite3_stmt* raw = nullptr;
        const auto rc = sqlite3_prepare_v2(reader.get(), sql, -1, &raw, nullptr);
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(raw, sqlite3_finalize);
        REQUIRE(rc == SQLITE_OK);
        REQUIRE(sqlite3_step(stmt.get()) == SQLITE_ROW);
        return sqlite3_column_int(stmt.get(), 0);
    };
    const auto before = scalar("PRAGMA data_version");
    REQUIRE(sqlite3_exec(reader.get(), "BEGIN", nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(scalar("SELECT x FROM t") == 1); // pin first, as ObjectStore::read does
    REQUIRE(sqlite3_exec(writer.get(), "UPDATE t SET x=2", nullptr, nullptr, nullptr) == SQLITE_OK);
    // A commit between pinning and reading the token must not label old cached
    // bytes with the newer snapshot's version.
    CHECK(scalar("PRAGMA data_version") == before);
    CHECK(scalar("SELECT x FROM t") == 1);
    REQUIRE(sqlite3_exec(reader.get(), "COMMIT; BEGIN", nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(scalar("SELECT x FROM t") == 2);
    CHECK(scalar("PRAGMA data_version") != before);
    REQUIRE(sqlite3_exec(reader.get(), "COMMIT", nullptr, nullptr, nullptr) == SQLITE_OK);
}

TEST_CASE("geometry revisions preserve read snapshots and reject foreign transactions")
{
    TempDir tmp("geometry_snapshot");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    auto other = ObjectStore::create(tmp.dir / "other.vcobj", testOptions());
    const auto id = addObject(db, "track");
    vc::objects::GeometryData data;
    data.kind = "track";
    data.arrays.push_back({"points", "<i4", {1, 3}, std::vector<uint8_t>(12, 0)});
    std::string first, second, edge;
    {
        auto w = db.write();
        first = db.putGeometry(w, id, data);
        const auto otherId = db.upsertObject(w, makeObject("note"));
        EdgeRecord link;
        link.kind = "connected"; link.aUuid = id; link.bUuid = otherId;
        link.aAnchor.wholeGeometry = true; link.bAnchor.wholeGeometry = true;
        edge = db.addEdge(w, link);
        db.setApproval(w, edge, "approved");
        w.commit();
    }
    auto snapshot = db.read();
    {
        auto foreign = other.write();
        CHECK_THROWS(db.putGeometry(foreign, id, data));
        CHECK_THROWS(db.currentGeometry(foreign, id));
        CHECK_THROWS(db.objectByUuid(foreign, id));
        CHECK_THROWS(db.edgeByUuid(foreign, edge));
        CHECK_THROWS(db.edgeGeometryState(foreign, edge));
        CHECK_THROWS(db.affectedGeometryEdges(foreign, id, data));
        CHECK_THROWS(db.repairEdgeGeometry(foreign, {}));
        CHECK_THROWS(db.upsertObject(foreign, makeObject("note")));
    }
    data.arrays[0].bytes[0] = 3;
    {
        auto w = db.write();
        CHECK_THROWS_AS(db.putGeometry(w, id, data, first), std::runtime_error);
        const auto link = *db.edgeByUuid(w, edge);
        GeometryEdgeEdit decision;
        decision.edgeUuid = edge; decision.expectedGeneration = link.generation; decision.action = "retain";
        decision.aRevision = first;
        second = db.putGeometry(w, id, data, first, {decision});
        w.commit();
    }
    CHECK(db.currentGeometry(snapshot, id)->uuid == first);
    CHECK_FALSE(db.geometryRevision(snapshot, second));
    CHECK_FALSE(db.edgeGeometryState(snapshot, edge).stale);
    CHECK(db.edgeByUuid(snapshot, edge)->approvalState == "approved");
    auto current = db.read();
    CHECK(db.currentGeometry(current, id)->uuid == second);
    CHECK_FALSE(db.edgeGeometryState(current, edge).stale);
    CHECK(db.edgeByUuid(current, edge)->approvalState == "pending");
    CHECK(db.readGeometryArray(current, first, "points")[0] == 0);
    CHECK(db.readGeometryArray(current, second, "points")[0] == 3);
    CHECK(db.geometryHistory(current, id).size() == 2);
    ObjectSelection atOrigin;
    atOrigin.bbox = std::array<double, 6>{0, 0, 0, 0, 0, 0};
    const auto projected = db.selectObjects(snapshot, atOrigin, std::vector<std::string>{"generation"});
    REQUIRE(projected.size() == 1);
    CHECK(projected.front().uuid == id);
    CHECK(projected.front().generation == db.objectByUuid(snapshot, id)->generation);
    CHECK(projected.front().meta == "{}");
    CHECK_FALSE(projected.front().track);
    CHECK(db.selectObjects(current, atOrigin).empty());
    const auto regions = db.spatialRegions(snapshot, atOrigin);
    REQUIRE(regions.size() == 1);
    CHECK(regions.front().objectUuid == id);
    CHECK(regions.front().revisionUuid == first);
    CHECK(regions.front().start == std::vector<int64_t>{0});
    CHECK(regions.front().shape == std::vector<int64_t>{1});
    CHECK(db.spatialRegions(current, atOrigin).empty());
    CHECK_THROWS_AS(db.spatialRegions(current, {}), std::invalid_argument);
    auto foreign = other.read();
    CHECK_THROWS(db.spatialRegions(foreign, atOrigin));
}

TEST_CASE("object store serializes independent writer transactions")
{
    TempDir tmp("object_writers");
    auto store = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    std::atomic_bool failed{false};
    std::vector<std::thread> writers;
    for (int worker = 0; worker < 3; ++worker) {
        writers.emplace_back([&] {
            try {
                for (int i = 0; i < 10; ++i) addObject(store, "note");
            } catch (...) { failed.store(true); }
        });
    }
    for (auto& writer : writers) writer.join();
    CHECK_FALSE(failed.load());
    auto read = store.read();
    CHECK(store.objectCount(read, "note") == 30);
}

TEST_CASE("tiled patch rectangles preserve revision snapshots and logical byte order")
{
    TempDir tmp("patch_rectangle");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    auto other = ObjectStore::create(tmp.dir / "other.vcobj", testOptions());
    const auto id = addObject(db, "patch");
    GeometryData data;
    data.kind = "patch"; data.attributes = R"({"scale":[1,1]})";
    data.arrays.push_back({"points", "<f4", {2, 130, 3}, std::vector<uint8_t>(2 * 130 * 12, 0)});
    auto& bytes = data.arrays[0].bytes;
    for (size_t point = 0; point < 260; ++point) {
        bytes[point * 12 + 10] = 0x80; bytes[point * 12 + 11] = 0x3f; // z=1, little endian
    }
    std::string first, second;
    {
        auto w = db.write(); first = db.putGeometry(w, id, data); w.commit();
    }
    auto snapshot = db.read();
    const auto old = db.readGeometryRectangle(snapshot, first, "points", 0, 127, 2, 3);
    REQUIRE(old.size() == 72);
    bytes[(130 + 128) * 12 + 11] = 0x40; // z=4 in the second physical tile
    {
        auto w = db.write(); second = db.putGeometry(w, id, data, first); w.commit();
    }
    CHECK(db.currentGeometry(snapshot, id)->uuid == first);
    CHECK(db.readGeometryRectangle(snapshot, first, "points", 0, 127, 2, 3) == old);
    CHECK_THROWS(db.readGeometryRectangle(snapshot, second, "points", 0, 127, 2, 3));
    auto current = db.read();
    CHECK(db.readGeometryArray(current, second, "points") == bytes);
    CHECK_THROWS(db.readGeometryRectangle(snapshot, second, "points", 0, 0, 1, 1));
    CHECK(db.readGeometryRectangles(snapshot, first, "points", {{0, 127, 2, 3}, {2, 130, 0, 0}})
          == std::vector<std::vector<uint8_t>>{old, {}});
    const auto changed = db.readGeometryRectangle(current, second, "points", 0, 127, 2, 3);
    CHECK(changed[36 + 12 + 11] == 0x40);
    CHECK(old[36 + 12 + 11] == 0x3f);
    std::string rolledBack;
    {
        auto w = db.write();
        bytes[0] = 1;
        rolledBack = db.putGeometry(w, id, data, second);
        CHECK(db.readGeometryRectangles(w, rolledBack, "points", {{0, 0, 1, 1}})[0][0] == 1);
        // Rollback must not leave a readable, cached unpublished revision.
    }
    auto fresh = db.read();
    CHECK_THROWS(db.readGeometryRectangles(fresh, rolledBack, "points", {{0, 0, 1, 1}}));
    CHECK(db.readGeometryRectangle(fresh, second, "points", 0, 0, 1, 1)[0] == 0);
    CHECK_THROWS(db.readGeometryRectangle(snapshot, second, "points", 0, 0, 1, 1));
    fresh.commit();
    CHECK_THROWS(db.readGeometryRectangle(fresh, second, "points", 0, 0, 1, 1));
    auto foreign = other.read();
    CHECK_THROWS(db.readGeometryRectangle(foreign, first, "points", 0, 0, 1, 1));
    CHECK_THROWS(db.readGeometryRectangles(foreign, first, "points", {{0, 0, 1, 1}}));
}
