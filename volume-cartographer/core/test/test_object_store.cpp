#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/objects/ObjectStore.hpp"

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
