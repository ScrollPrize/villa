#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../../apps/src/FiberConstraintGeometry.hpp"
#include <vc_object_store/objects/FiberJson.hpp>
#include <vc_object_store/objects/GeometryEdits.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/Uuid.hpp>
#include <cstdlib>
#include <iostream>

TEST_CASE("constraint inputs use edited geometry rather than ingest counts")
{
    using namespace vc::objects;
    const auto path = std::filesystem::temp_directory_path() / (util::newUuid() + ".vcobj");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    OpenOptions options; options.author = "constraint-geometry-test";
    auto store = ObjectStore::create(path, options);
    const auto uuid = util::newUuid();
    auto source = decodeFiberJson(nlohmann::json{
        {"type", "vc3d_fiber"}, {"version", 4}, {"uuid", uuid},
        {"control_points", {{{"position", {1,2,3}}}, {{"position", {5,2,3}}}}},
        {"line_points", {{1,2,3}}}}.dump());
    std::string initial;
    { auto txn = store.write(); ingestGeometryBatch(store, txn, {source});
      initial = store.currentGeometry(txn, uuid)->uuid; txn.commit(); }
    const std::map<std::string, VolumeAffine> members{{uuid, VolumeAffine::scale(0.25)}};
    { auto txn = store.read(); CHECK(selectFiberConstraintGeometry(store, txn, members).empty()); }
    std::string current = initial;
    for (const size_t count : {3u, 2u, 1u, 0u}) {
        vc::geometry::Fiber fiber{{{1,2,3},{5,2,3}}, {}};
        for (size_t i = 0; i < count; ++i) fiber.linePoints.push_back({1.0 + i, 2, 3});
        RevisionSelection expected;
        { auto txn = store.read(); expected[uuid] = store.currentObjectRevision(txn, uuid)->uuid; }
        { auto txn = store.write(expected);
          current = updateGeometryArrays(store, txn, uuid, vc::geometry::encode(fiber), current);
          txn.commit(); }
        auto txn = store.read();
        REQUIRE(store.objectByUuid(txn, uuid)->fiber->linePointCount == 1);
        const auto selected = selectFiberConstraintGeometry(store, txn, members);
        if (count < 2) { CHECK(selected.empty()); continue; }
        REQUIRE(selected.size() == 1);
        CHECK(selected[0].uuid == uuid);
        CHECK(selected[0].revision == current);
        CHECK(selected[0].mapping == VolumeAffine::scale(0.25));
        const auto packed = store.readGeometryArrays(txn, {selected[0].revision}, "line_points");
        CHECK(packed.offsets.back() == count);
    }
}

