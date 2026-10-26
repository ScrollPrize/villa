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
    const std::map<std::string, double> members{{uuid, 0.25}};
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
        CHECK(selected[0].scale == 0.25);
        const auto packed = store.readGeometryArrays(txn, {selected[0].revision}, "line_points");
        CHECK(packed.offsets.back() == count);
    }
}

TEST_CASE("constraint selection on an optional native fiber collection")
{
    const auto fixture = std::getenv("VC_FIBER_CONSTRAINT_FIXTURE");
    if (!fixture) return;
    using namespace vc::objects;
    auto batch = prepareGeometrySource("fiber", fixture);
    REQUIRE(!batch.empty());
    const auto sample = std::find_if(batch.begin(), batch.end(), [](const auto& entry) {
        return entry.source.value("branches", nlohmann::json::array()).empty() &&
               entry.source.value("adjacent_branches", nlohmann::json::array()).empty() &&
               entry.object.fiber && entry.object.fiber->linePointCount > 3;
    });
    REQUIRE(sample != batch.end());
    const auto path = std::filesystem::temp_directory_path() / (util::newUuid() + ".vcobj");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    OpenOptions options; options.author = "constraint-real-fixture-test";
    auto store = ObjectStore::create(path, options);
    { auto txn = store.write(); ingestGeometryBatch(store, txn, batch); txn.commit(); }
    auto geometry = std::get<vc::geometry::Fiber>(vc::geometry::decode(sample->geometry));
    const auto originalCount = geometry.linePoints.size();
    std::string revision;
    { auto txn = store.read(); revision = store.currentGeometry(txn, sample->object.uuid)->uuid; }
    for (const auto count : {originalCount - 1, size_t{2}, size_t{1}}) {
        geometry.linePoints.resize(count);
        RevisionSelection expected;
        { auto txn = store.read(); expected[sample->object.uuid] = store.currentObjectRevision(txn, sample->object.uuid)->uuid; }
        { auto txn = store.write(expected);
          revision = updateGeometryArrays(store, txn, sample->object.uuid, vc::geometry::encode(geometry), revision);
          txn.commit(); }
        auto txn = store.read();
        REQUIRE(store.objectByUuid(txn, sample->object.uuid)->fiber->linePointCount == originalCount);
        const auto selected = selectFiberConstraintGeometry(store, txn, {{sample->object.uuid, 1.0}});
        if (count < 2) { CHECK(selected.empty()); continue; }
        REQUIRE(selected.size() == 1);
        const auto packed = store.readGeometryArrays(txn, {selected[0].revision}, "line_points");
        CHECK(packed.offsets.back() == count);
    }
    std::cout << "real fixture: " << batch.size() << " fibers; edited " << sample->object.uuid
              << " from " << originalCount << " points across the two-point threshold\n";
}
