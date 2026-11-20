// A manual fiber (>= 2 controls) is its line between the first and last
// control; the extrapolated tails past them are ignored by length, bounds,
// the spatial index and region queries. Fewer controls: the whole line counts.
#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <iostream>
using namespace vc::objects;
namespace vg = vc::geometry;
int main() try
{
    // Controls at x=0 and x=10 are exact line vertices 2 and 4.
    const std::vector<vg::Point> line{{-20,0,0},{-10,0,0},{0,0,0},{5,0,0},{10,0,0},{20,0,0},{30,0,0}};
    const std::vector<vg::Point> controls{{0,0,0},{10,0,0}};
    {
        const auto r = vg::annotatedLineRange(controls, line);
        require(r.begin == 2 && r.end == 5, "tails on both sides are cut");
        require(vg::annotatedLineRange({{{-20,0,0},{30,0,0}}}, line).size() == line.size(), "controls at the ends keep the line");
        require(vg::annotatedLineRange({{{0,0,0}}}, line).size() == line.size(), "one control: no tails");
        require(vg::annotatedLineRange({}, line).size() == line.size(), "automatic trace: no tails");
        require(vg::annotatedLineRange(controls, {}).size() == 0, "empty line");
        require(vg::annotatedLineRange({{{0.4,0.1,0},{9.7,0,0.2}}}, line).begin == 2 &&
                vg::annotatedLineRange({{{0.4,0.1,0},{9.7,0,0.2}}}, line).end == 5, "nearest vertex");
        require(vg::annotatedLineRange({{{9,0,0},{1,0,0}}}, line).begin == 2 &&
                vg::annotatedLineRange({{{9,0,0},{1,0,0}}}, line).end == 5, "control order does not matter");
        require(vg::Fiber{controls, line, {}}.length() == 10, "Fiber::length excludes the tails");
        require(vg::Fiber{{}, line, {}}.length() == 50, "automatic fiber length is the whole line");
    }
    {
        const nlohmann::json root{{"type","vc3d_fiber"},{"uuid",util::newUuid()},{"version",4},{"name","tails"},
            {"control_points",{{{"position",{0,0,0}}},{{"position",{10,0,0}}}}},
            {"line_points",{{-20,0,0},{-10,0,0},{0,0,0},{5,0,0},{10,0,0},{20,0,0},{30,0,0}}}};
        require(decodeFiberJson(root.dump()).object.fiber->length == 10, "stored fiber length excludes the tails");
    }
    TemporaryDirectory directory;
    OpenOptions open; open.author = "tails-test";
    auto store = ObjectStore::create(directory.path / "tails.vcobj", open);
    std::string manual, automatic;
    {
        auto txn = store.write();
        for (auto* id : {&manual, &automatic}) {
            ObjectRecord record; record.kind = "fiber"; record.fiber.emplace();
            *id = store.upsertObject(txn, record);
            store.putGeometry(txn, *id, vg::encode(vg::Fiber{id == &manual ? controls : std::vector<vg::Point>{}, line, {}}));
        }
        txn.commit();
    }
    auto txn = store.read();
    ObjectSelection tail; tail.kind = "fiber"; tail.bbox = std::array<double,6>{-25,-1,-1,-5,1,1};
    const auto inTail = store.selectObjects(txn, tail);
    require(inTail.size() == 1 && inTail[0].uuid == automatic, "object bounds exclude the tails");
    for (const auto& region : store.spatialRegions(txn, tail))
        require(region.objectUuid == automatic, "the spatial index excludes the tails");
    ObjectSelection body = tail; body.bbox = std::array<double,6>{1,-1,-1,9,1,1};
    require(store.selectObjects(txn, body).size() == 2, "both fibers are indexed between the controls");

    RegionSlice slice; slice.origin = {-15,0,0}; slice.localBounds = {-2,-2,-1,2,2,1};
    auto region = store.geometryRegion(txn, slice, RegionOptions{});
    require(region.intersectingObjects == 1 && region.runs.at(0).objectUuid == automatic, "region skips the tails");
    slice.origin = {5,0,0};
    region = store.geometryRegion(txn, slice, RegionOptions{});
    require(region.intersectingObjects == 2, "region hits the annotated line");
    for (const auto& run : region.runs)
        if (run.objectUuid == manual) require(run.firstSegment >= 2 && run.firstSegment + int64_t(run.points.size()) - 1 <= 4,
                                              "manual runs stay within the annotated rows");
    RegionOptions longer; longer.minLength = 10.5;
    slice.localBounds = {-100,-2,-1,100,2,1};
    region = store.geometryRegion(txn, slice, longer);
    require(region.intersectingObjects == 1 && region.runs.at(0).objectUuid == automatic, "measured length excludes the tails");
    longer.minLength = 9.5;
    require(store.geometryRegion(txn, slice, longer).intersectingObjects == 2, "annotated length is measured");
    {
        // Bulk packed reads cut the tails unless asked for the raw rows.
        ObjectSelection fibers; fibers.kind = "fiber";
        const auto view = store.geometryReadView(txn, fibers);
        const auto& selected = *view.selectedObjects;
        std::vector<std::string> revisions;
        for (const auto& object : selected) revisions.push_back(*object.geometryRevision);
        const auto expected = [&](const PackedGeometryArray& packed, bool tails) {
            std::vector<uint64_t> offsets{0};
            for (const auto& object : selected) offsets.push_back(offsets.back() + (tails || object.objectUuid == automatic ? 7 : 3));
            return packed.offsets == offsets && packed.shape[0] == int64_t(offsets.back()) && packed.data.size() == offsets.back() * 24;
        };
        require(expected(store.readSelectedGeometryArrays(txn, selected, "line_points"), false), "selected packed read cuts the tails");
        require(expected(store.readSelectedGeometryArrays(txn, selected, "line_points", std::nullopt, true), true), "includeTails keeps them");
        require(expected(store.readGeometryArrays(txn, revisions, "line_points"), false), "revision packed read cuts the tails");
        require(expected(store.readGeometryArrays(txn, revisions, "line_points", true), true), "includeTails keeps them");
        const auto packed = store.readGeometryArrays(txn, revisions, "line_points");
        for (size_t i = 0; i < selected.size(); ++i)
            if (selected[i].objectUuid == manual) {
                std::array<double, 3> first;
                std::memcpy(first.data(), packed.data.data() + packed.offsets[i] * 24, 24);
                require(first == std::array<double, 3>{0, 0, 0}, "annotated rows start at the first control");
            }
    }
    std::cout << "fiber tails ok\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
