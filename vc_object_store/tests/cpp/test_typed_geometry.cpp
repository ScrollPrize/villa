#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/Uuid.hpp"
#include <bit>
#include <limits>

namespace {
namespace vg = vc::geometry;
using namespace vc::objects;
struct Database {
    std::filesystem::path path = std::filesystem::temp_directory_path() / ("vc-typed-" + util::newUuid());
    ObjectStore db = ObjectStore::create(path, options());
    static OpenOptions options() { OpenOptions o; o.author = "test"; return o; }
    ~Database() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(path.string() + "-wal", ignored);
        std::filesystem::remove(path.string() + "-shm", ignored);
    }
    std::string put(const vg::Geometry& g, int64_t tile = 128) {
        const auto data = vg::encode(g);
        auto write = db.write();
        ObjectRecord record; record.kind = data.kind;
        const auto id = db.upsertObject(write, record);
        db.putGeometry(write, id, data, std::nullopt, {}, GeometryStorageOptions{tile});
        write.commit();
        return id;
    }
};
}

TEST_CASE("typed owned geometry preserves dtype, order, IDs and exact bytes")
{
    vg::PointCollection collection{{42, 7}, {{-0.0, 2.25, 3}, {4, 5, 6}}};
    const auto data = vg::encode(collection);
    const auto copy = std::get<vg::PointCollection>(vg::decode(data));
    CHECK(copy.ids == collection.ids);
    CHECK(copy.points == collection.points);
    CHECK(std::bit_cast<uint64_t>(copy.points[0][0]) == std::bit_cast<uint64_t>(-0.0));
    CHECK(copy.point(7) == collection.points[1]);
    CHECK_FALSE(copy.point(99));
    CHECK(data.arrays[0].dtype == "<f8");
    CHECK(data.arrays[1].dtype == "<i8");
    vg::Track track{{{INT32_MIN, 0, 1}, {INT32_MAX, 0, 1}}};
    CHECK(track.length() == 4294967295.0);
    CHECK(std::get<vg::Track>(vg::decode(vg::encode(track))).points == track.points);
    vg::Fiber fiber{{{0, 0, 0}, {3, 4, 0}}, {}};
    CHECK(fiber.length() == 5);
    fiber.linePoints = {{0, 0, 0}, {0, 0, 2}};
    CHECK(fiber.length() == 2);
    CHECK(std::get<vg::Fiber>(vg::decode(vg::encode(fiber))).linePoints == fiber.linePoints);
    CHECK(std::get<vg::PointCollection>(vg::decode(vg::encode(vg::PointCollection{}))).points.empty());
}

TEST_CASE("typed geometry rejects invalid values and lossy conversions")
{
    vg::PointCollection g{{1, 1}, {{0, 0, 0}, {1, 1, 1}}};
    CHECK_THROWS_AS(vg::encode(g), std::invalid_argument);
    g.ids = {1, -2};
    CHECK_THROWS_AS(g.validate(), std::invalid_argument);
    g.ids = {1};
    CHECK_THROWS_AS(g.validate(), std::invalid_argument);
    g.points.resize(1);
    g.points[0][0] = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(g.validate(), std::invalid_argument);
    auto data = vg::encode(vg::Track{{{1, 2, 3}}});
    data.arrays[0].shape = {INT64_MAX, 3};
    CHECK_THROWS_AS(vg::decode(data), std::invalid_argument);
    data = vg::encode(vg::Track{{{1, 2, 3}}});
    data.attributes = R"({"custom":true})";
    CHECK_THROWS_AS(vg::decode(data), std::invalid_argument);
    data.attributes = "{}";
    data.arrays.push_back({"extra", "|u1", {1}, {0}});
    CHECK_THROWS_AS(vg::decode(data), std::invalid_argument);
    vg::Patch patch{1, 1, {1, 1}, {{-1, -1, -1}}};
    patch.validate();
    CHECK_FALSE(patch.valid(0, 0));
    patch.points[0] = {1, 2, 0};
    CHECK_THROWS_AS(patch.validate(), std::invalid_argument);
    patch.points[0] = {1, 2, 3};
    patch.scale[0] = 0;
    CHECK_THROWS_AS(patch.validate(), std::invalid_argument);
    CHECK_THROWS_AS(patch.vertex(-1, 0), std::out_of_range);
}

TEST_CASE("typed snapshots pin revisions and edits resolve stable IDs")
{
    Database fixture;
    auto& db = fixture.db;
    const auto id = fixture.put(vg::PointCollection{{42, 7}, {{1, 2, 3}, {4, 5, 6}}});
    auto snapshot = db.geometrySnapshot();
    auto view = snapshot.pointCollection(id);
    CHECK(std::holds_alternative<PointCollectionView>(snapshot.geometry(id)));
    CHECK(view.size() == 2);
    CHECK(view.ids(0, 2) == std::vector<int64_t>{42, 7});
    CHECK(view.point(7) == vg::Point{4, 5, 6});
    CHECK_FALSE(view.point(100));
    CHECK(view.points(2, 0).empty());
    CHECK_THROWS_AS(view.points(-1, 1), std::out_of_range);
    CHECK_THROWS_AS(view.points(1, INT64_MAX), std::out_of_range);
    CHECK_THROWS_AS(snapshot.patch(id), std::invalid_argument);
    auto edit = view.edit();
    CHECK_THROWS_AS(edit.movePoint(100, {0, 0, 0}), std::out_of_range);
    CHECK_THROWS_AS(edit.moveTrackPoint(0, {0, 0, 0}), std::invalid_argument);
    edit.movePoint(7, {7, 8, 9});
    const auto next = edit.commit();
    CHECK(next != view.revision().uuid);
    CHECK(view.point(7) == vg::Point{4, 5, 6});
    auto fresh = db.geometrySnapshot();
    CHECK(fresh.pointCollection(id).point(7) == vg::Point{7, 8, 9});
    CHECK(fresh.pointCollection(id).point(42) == vg::Point{1, 2, 3});
    CHECK_THROWS(edit.commit());
    auto stale = view.edit();
    stale.movePoint(42, {0, 0, 0});
    CHECK_THROWS(stale.commit());
    snapshot.close();
    CHECK_FALSE(snapshot.active());
    CHECK_THROWS(view.point(7));
    CHECK_THROWS(view.revision());
    CHECK_THROWS(stale.affectedRelationships());
    snapshot.close(); // idempotent
}

TEST_CASE("typed editors check metadata revisions and preserve no-op history")
{
    Database f;
    const auto id = f.put(vg::Track{{{1, 2, 3}}});
    auto snapshot = f.db.geometrySnapshot();
    auto view = snapshot.track(id);
    auto noop = view.edit();
    noop.moveTrackPoint(0, {1, 2, 3});
    CHECK(noop.commit() == view.revision().uuid);
    auto edit = view.edit();
    edit.moveTrackPoint(0, {4, 5, 6});
    {
        auto w = f.db.write();
        f.db.setMeta(w, id, "reviewed", "true");
        w.commit();
    }
    CHECK_THROWS(edit.commit());
}

TEST_CASE("typed patch tiles preserve exact coordinates in tiled and linear storage")
{
    for (int64_t tile : {int64_t(0), int64_t(32)}) {
        Database f;
        vg::Patch patch{35, 36, {0.5, 0.25}, {}};
        for (int r = 0; r < 35; ++r) for (int c = 0; c < 36; ++c)
            patch.points.push_back({float(c), float(r), float(r+c+1)});
        patch.points[0] = {-1, -1, -1};
        const auto id = f.put(patch, tile);
        auto snapshot = f.db.geometrySnapshot();
        auto view = snapshot.patch(id);
        CHECK(view.rows() == 35);
        CHECK(view.columns() == 36);
        CHECK_FALSE(view.valid(0, 0));
        CHECK(view.load().points == patch.points);
        const auto rect = view.readTile(30, 31, 4, 4);
        CHECK(rect.scale == patch.scale);
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c)
            CHECK(rect.vertex(r, c) == patch.vertex(r + 30, c + 31));
        CHECK_THROWS(view.readTile(34, 35, 2, 2));
        auto edit = view.edit();
        edit.moveVertex(32, 33, {1, 2, 100});
        CHECK_THROWS(edit.moveVertex(0, 0, {0, 0, 0}));
        edit.commit();
        auto current = f.db.geometrySnapshot();
        CHECK(current.patch(id).vertex(32, 33) == vg::PatchPoint{1, 2, 100});
        CHECK(view.vertex(32, 33) == patch.vertex(32, 33));
    }
}

TEST_CASE("typed edits require explicit relationship decisions")
{
    Database f;
    const auto a = f.put(vg::PointCollection{{42}, {{1, 2, 3}}});
    const auto b = f.put(vg::Track{{{1, 2, 3}}});
    std::string edgeId;
    {
        auto w = f.db.write();
        EdgeRecord edge;
        edge.kind = "contact"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"point_ids", {42}}};
        edge.bAnchor.points = {{"points", {0}}};
        edgeId = f.db.addEdge(w, edge);
        w.commit();
    }
    auto snapshot = f.db.geometrySnapshot();
    auto edit = snapshot.pointCollection(a).edit();
    edit.movePoint(42, {9, 8, 7});
    const auto affected = edit.affectedRelationships();
    REQUIRE(affected.size() == 1);
    CHECK(affected[0].uuid == edgeId);
    CHECK_THROWS(edit.commit());
    GeometryEdgeEdit decision;
    {
        auto read = f.db.read();
        const auto edge = f.db.edgeByUuid(read, edgeId);
        decision.edgeUuid = edgeId;
        decision.expectedGeneration = edge->generation;
        decision.action = "remove";
        decision.aRevision = f.db.currentGeometry(read, edge->aUuid)->uuid;
        decision.bRevision = f.db.currentGeometry(read, edge->bUuid)->uuid;
    }
    CHECK_NOTHROW(edit.commit({decision}));
    auto read = f.db.read();
    CHECK_FALSE(f.db.edgeByUuid(read, edgeId));
}

TEST_CASE("owned snapshot lifetime and fiber edits")
{
    Database f;
    const auto id = f.put(vg::Fiber{{{0, 0, 0}}, {{1, 2, 3}}});
    auto view = [&] { auto snapshot = f.db.geometrySnapshot(); return snapshot.fiber(id); }();
    auto edit = view.edit();
    edit.moveControlPoint(0, {4, 5, 6});
    edit.moveLinePoint(0, {7, 8, 9});
    edit.commit();
    auto fresh = f.db.geometrySnapshot();
    CHECK(fresh.fiber(id).controlPoints(0, 1)[0] == vg::Point{4, 5, 6});
    CHECK(fresh.fiber(id).linePoints(0, 1)[0] == vg::Point{7, 8, 9});
}
