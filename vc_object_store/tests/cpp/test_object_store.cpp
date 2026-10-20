#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc_object_store/objects/ObjectStore.hpp"

#include <atomic>
#include <filesystem>
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

TEST_CASE("object history publishes a whole checked action and preserves reader snapshots")
{
    TempDir tmp("unified_history");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    const auto a = addObject(db, "note"), b = addObject(db, "note");
    auto before = db.read();
    const auto a0 = db.currentObjectRevision(before, a)->uuid;
    const auto b0 = db.currentObjectRevision(before, b)->uuid;
    std::string edge;
    {
        auto write = db.write(RevisionSelection{{a, a0}, {b, b0}});
        db.setMeta(write, a, "winding", "2");
        EdgeRecord link;
        link.kind = "connected"; link.aUuid = a; link.bUuid = b;
        edge = db.addEdge(write, link);
        // Revision reads inside a write describe the last published state.
        CHECK(db.currentObjectRevision(write, a)->uuid == a0);
        write.commit();
    }
    CHECK(db.currentObjectRevision(before, a)->uuid == a0);
    CHECK(db.currentObjectRevision(before, b)->uuid == b0);
    CHECK(db.recordedRelationships(before, a0).empty());
    auto after = db.read();
    const auto a1 = *db.currentObjectRevision(after, a);
    const auto b1 = *db.currentObjectRevision(after, b);
    CHECK(a1.parentUuid == a0);
    CHECK(b1.parentUuid == b0);
    CHECK(a1.record.meta == R"({"winding":2})");
    const auto links = db.relationshipsAt(after, {{a, a1.uuid}, {b, b1.uuid}});
    REQUIRE(links.size() == 1);
    CHECK(links.front().applicable);
    CHECK(db.recordedRelationships(after, a1.uuid).front().uuid ==
          db.recordedRelationships(after, b1.uuid).front().uuid);
    CHECK_THROWS(db.write(RevisionSelection{{a, a0}, {b, b0}}));
    {
        auto write = db.write(RevisionSelection{{a, a1.uuid}});
        db.setMeta(write, a, "abort", "true");
        db.removeEdge(write, edge); // shared owner b lacks an expectation
        CHECK_THROWS(write.commit());
        CHECK_FALSE(write.active());
    }
    auto unchanged = db.read();
    CHECK(db.currentObjectRevision(unchanged, a)->uuid == a1.uuid);
    CHECK(db.currentObjectRevision(unchanged, b)->uuid == b1.uuid);
    CHECK(db.edgeByUuid(unchanged, edge).has_value());
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
    ObjectSelection atOrigin;
    atOrigin.bbox = std::array<double, 6>{0, 0, 0, 0, 0, 0};
    // Populate the resident index before another connection publishes a head.
    REQUIRE(db.spatialRegions(snapshot, atOrigin).size() == 1);
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

TEST_CASE("resident spatial queries keep writer changes out of reader caches")
{
    TempDir tmp("spatial_cache_writer");
    auto options = testOptions();
    options.geometryCacheBytes = 0;
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", options);
    auto object = makeObject("track");
    object.track.emplace();
    object.track->bbox = std::array<double, 6>{0, 0, 0, 0, 0, 0};
    ObjectSelection selection;
    selection.bbox = object.track->bbox;
    auto old = db.read();
    CHECK(db.spatialRegions(old, selection).empty()); // retain an empty index
    {
        auto w = db.write();
        db.upsertObject(w, object);
        CHECK(db.spatialRegions(w, selection).size() == 1);
        CHECK(db.spatialRegions(old, selection).empty());
        // Roll back the inserted region.
    }
    {
        auto r = db.read();
        CHECK(db.spatialRegions(r, selection).empty());
    }
    {
        auto w = db.write();
        const auto id = db.upsertObject(w, object);
        REQUIRE(db.spatialRegions(w, selection).size() == 1);
        db.deleteObject(w, id);
        CHECK(db.spatialRegions(w, selection).empty());
        db.upsertObject(w, object);
        w.commit();
    }
    CHECK(db.spatialRegions(old, selection).empty());
    old.commit();
    CHECK_THROWS(db.spatialRegions(old, selection));
    auto current = db.read(); // reuse the old snapshot's connection
    CHECK(db.spatialRegions(current, selection).size() == 1);
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

TEST_CASE("sparse geometry revisions preserve pinned snapshots and transaction ownership")
{
    TempDir tmp("delta_snapshot");
    auto store = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    auto foreign = ObjectStore::create(tmp.dir / "foreign.vcobj", testOptions());
    const auto uid = addObject(store, "track");
    GeometryData data;
    data.kind = "track";
    data.arrays.push_back({"points", "<i4", {2, 3}, std::vector<uint8_t>(24, 0)});
    std::string first, second, third;
    {
        auto w = store.write();
        first = store.putGeometry(w, uid, data); w.commit();
    }
    {
        auto w = store.write();
        second = store.editGeometry(w, uid, {{"points", 0, {1, 0, 0, 0}}}, first); w.commit();
    }
    auto snapshot = store.read();
    CHECK(store.readGeometryArray(snapshot, second, "points", 0, 4) == std::vector<uint8_t>{1, 0, 0, 0});
    {
        auto w = store.write();
        CHECK_THROWS(foreign.editGeometry(w, uid, {}, second));
        third = store.editGeometry(w, uid, {{"points", 0, {2, 0, 0, 0}}}, second); w.commit();
    }
    CHECK(store.currentGeometry(snapshot, uid)->uuid == second);
    CHECK(store.readGeometryArray(snapshot, second, "points", 0, 4) == std::vector<uint8_t>{1, 0, 0, 0});
    CHECK_FALSE(store.geometryRevision(snapshot, third));
    snapshot.commit();
    {
        auto w = store.write();
        CHECK_THROWS(store.editGeometry(w, uid, {}, second));
        const auto reverted = store.editGeometry(w, uid, {{"points", 0, {0, 0, 0, 0}}}, third);
        CHECK(store.readGeometryArray(w, reverted, "points") == data.arrays[0].bytes);
        // Roll back a published delta and its materialized current chunks.
    }
    auto r = store.read();
    CHECK(store.currentGeometry(r, uid)->uuid == third);
    CHECK(store.geometryHistory(r, uid).size() == 3);
    CHECK(store.readGeometryArray(r, second, "points", 0, 4) == std::vector<uint8_t>{1, 0, 0, 0});
    CHECK(store.readGeometryArray(r, third, "points", 0, 4) == std::vector<uint8_t>{2, 0, 0, 0});
}
