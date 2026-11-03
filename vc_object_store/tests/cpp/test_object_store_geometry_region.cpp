#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
using namespace vc::objects;
int main() try
{
    TemporaryDirectory directory;
    OpenOptions open; open.author = "region-test";
    auto store = ObjectStore::create(directory.path / "region.vcobj", open);
    std::vector<std::string> ids;
    {
        auto txn = store.write();
        for (int i = 0; i < 3; ++i) {
            ObjectRecord record; record.kind = "fiber"; record.fiber.emplace(); record.fiber->length = 20;
            if (i != 2) record.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":2}})";
            const auto id = store.upsertObject(txn, record); ids.push_back(id);
            vc::geometry::Fiber fiber; fiber.controlPoints = {{-10,0,0}, {10,0,0}};
            store.putGeometry(txn, id, vc::geometry::encode(fiber));
        }
        txn.commit();
    }
    std::sort(ids.begin(), ids.end());
    auto txn = store.read();
    RegionSlice slice;
    RegionOptions options;
    auto all = store.geometryRegion(txn, slice, options);
    require(all.complete() && all.intersectingObjects == 3, "boundary crossing segments");
    // The Fibers panel pages equal volume lengths by UUID, and the overlay's
    // display limit keeps exact geometry. The undeclared fiber is not bound.
    const std::string volume = "20260411134726";
    ObjectSelection page; page.kind = "fiber"; page.volume = volume; page.orderBy = "length_desc"; page.limit = 1;
    const auto first = store.selectObjects(txn, page);
    require(first.size() == 1, "volume catalog page");
    page.afterUuid = first[0].uuid; page.afterLength = first[0].fiber->length / 2;
    const auto second = store.selectObjects(txn, page);
    require(second.size() == 1 && second[0].uuid != first[0].uuid, "cursor keeps equal-length rows without repeats");
    page.afterUuid = second[0].uuid;
    require(store.selectObjects(txn, page).empty(), "the unbound fiber is not in the bound pages");
    RegionOptions overlay; overlay.volume = volume;
    const auto region = store.geometryRegion(txn, slice, overlay);
    require(region.complete() && region.intersectingObjects == 2, "region spans the volume's bound fibers");
    overlay.maximumObjects = 1;
    const auto limited = store.geometryRegion(txn, slice, overlay);
    require(limited.intersectingObjects == 1, "display limit applies to the region");
    require(limited.runs.front().points == region.runs.front().points, "bounded query preserves exact geometry");
    options.maximumObjects = 1; options.selected = ids.front();
    options.uuids = std::vector<std::string>{ids.back()};
    auto filtered = store.geometryRegion(txn, slice, options);
    require(filtered.intersectingObjects == 1 && filtered.runs[0].objectUuid == ids.back(), "filter before selected and limit");
    require(filtered.runs[0].points == all.runs.back().points, "filter preserves exact geometry");
    require(!filtered.complete() && filtered.stop == RegionStop::ObjectLimit, "limit reported incomplete");
    options.uuids = std::vector<std::string>{};
    require(store.geometryRegion(txn, slice, options).complete() && store.geometryRegion(txn, slice, options).runs.empty(), "empty inclusion");
    require(store.geometryRegion(txn, slice, options, [](const auto&) { return true; }).stop == RegionStop::Cancelled, "empty inclusion cancellation");
    options.uuids = std::vector<std::string>{ids.back(), ids.back()}; options.maximumObjects = 0;
    filtered = store.geometryRegion(txn, slice, options);
    require(filtered.complete() && filtered.intersectingObjects == 1, "duplicate inclusion no duplicate result");
    options.candidateWorkBudget = 0;
    require(store.geometryRegion(txn, slice, options).stop == RegionStop::CandidateBudget, "filter setup obeys work budget");
    options.candidateWorkBudget = 1000000; options.processingByteBudget = 0;
    require(store.geometryRegion(txn, slice, options).stop == RegionStop::ProcessingBudget, "filter setup obeys memory budget");
    options.processingByteBudget = 64*1024*1024; options.sqlStepBudget = 0;
    require(store.geometryRegion(txn, slice, options).stop == RegionStop::SqlBudget, "filter setup obeys SQL budget");
        {
        auto write = store.write();
        ObjectRecord record; record.kind = "track"; record.track.emplace(); record.track->length = 20;
        record.tags = {"reviewed", "track-only"};
        const auto id = store.upsertObject(write, record);
        vc::geometry::Track track; track.points = {{-10,0,0}, {10,0,0}, {10,10,0}};
        store.putGeometry(write, id, vc::geometry::encode(track));
        write.commit();
        auto read = store.read();
        RegionOptions tracks; tracks.kind = "track";
        const auto region = store.geometryRegion(read, slice, tracks);
        require(region.complete() && region.intersectingObjects == 1 && region.runs.size() == 1 &&
                region.runs[0].objectUuid == id && region.runs[0].arrayName == "points", "track kind uses its own geometry");
        require(region.runs[0].points == std::vector<std::array<double,3>>{{-10,0,0},{10,0,0}}, "integer track points widen exactly");
        const auto fibers = store.geometryRegion(read, slice, RegionOptions{});
        require(fibers.intersectingObjects == 3 &&
                std::none_of(fibers.runs.begin(), fibers.runs.end(), [&](const auto& run) { return run.objectUuid == id; }),
                "kind filter excludes other kinds");
        require(store.objectTags(read, std::string{"track"}) == std::vector<std::string>{"reviewed", "track-only"} &&
                store.objectTags(read, std::string{"fiber"}).empty() &&
                store.objectTags(read) == std::vector<std::string>{"reviewed", "track-only"}, "distinct tags filter by kind");
    }
    {
        // A surface tilted so z = x + 1, spanning three tile rows, cut by the plane z = 31.25.
        vc::geometry::Surface surface{70, 40, {1, 1}, {}};
        for (int r = 0; r < 70; ++r)
            for (int c = 0; c < 40; ++c) surface.points.push_back({float(c), float(r), float(c + 1)});
        std::string id;
        {
            auto write = store.write();
            ObjectRecord record; record.kind = "surface";
            id = store.upsertObject(write, record);
            store.putGeometry(write, id, vc::geometry::encode(surface));
            write.commit();
        }
        RegionSlice plane; plane.origin = {0, 0, 31.25}; plane.localBounds = {-1000, -1000, -0.5, 1000, 1000, 0.5};
        RegionOptions surfaces; surfaces.kind = "surface";
        auto read = store.read();
        auto region = store.geometryRegion(read, plane, surfaces);
        require(region.complete() && region.intersectingObjects == 1 && region.runs.size() == 1, "surface cut chains across tiles");
        const auto& run = region.runs[0].points;
        // Each row crossing plus each diagonal crossing (at row + 0.75), all exactly on the plane.
        require(run.size() == 139 && std::all_of(run.begin(), run.end(), [](const auto& p) { return p[0] == 30.25 && p[2] == 31.25; }),
                "surface cut follows the triangle mesh exactly");
        const auto [low, high] = std::minmax(run.front()[1], run.back()[1]);
        require(low == 0 && high == 69, "surface cut spans the grid");
        plane.localBounds = {-1000, 10, -0.5, 1000, 20, 0.5};
        region = store.geometryRegion(read, plane, surfaces);
        require(region.runs.size() == 1 && std::min(region.runs[0].points.front()[1], region.runs[0].points.back()[1]) == 10 &&
                std::max(region.runs[0].points.front()[1], region.runs[0].points.back()[1]) == 20, "surface cut is clipped to the slice");
        surfaces.family = "any";
        mustFail([&] { (void)store.geometryRegion(read, plane, surfaces); });
        surfaces.family.reset();
        {
            auto write = store.write();
            std::vector<uint8_t> valid(70 * 40, 1);
            valid[40 * 40 + 30] = 0;
            store.updateSurfaceChannels(write, id, {{"valid", "|u1", {70, 40}, valid}}, {}, store.currentGeometry(write, id)->uuid);
            write.commit();
        }
        auto masked = store.read();
        plane.localBounds = {-1000, -1000, -0.5, 1000, 1000, 0.5};
        region = store.geometryRegion(masked, plane, surfaces);
        require(region.runs.size() == 2 && region.intersectingObjects == 1, "invalid vertices break the cut");
    }
    {
        // The store's vertex validity rule, shared by readers, spatial indexes and linkers.
        using vc::geometry::SurfacePoint;
        using vc::geometry::validSurfacePoint;
        require(validSurfacePoint(SurfacePoint{1, 2, 3}) && validSurfacePoint(SurfacePoint{-2, 0, 0.5f}), "vertices with coordinates");
        for (const auto& missing : {SurfacePoint{-1, -1, -1}, SurfacePoint{5, -1, 7}, SurfacePoint{1, 2, 0}, SurfacePoint{1, 2, -3},
                                    SurfacePoint{std::nanf(""), 1, 2}, SurfacePoint{1, std::numeric_limits<float>::infinity(), 2}})
            require(!validSurfacePoint(missing), "a -1 or non-finite component, or z <= 0, is missing");
    }
    {
        // Nearest surfaces: pick, restrict by UUID, a volume target, and holes.
        auto sheets = ObjectStore::create(directory.path / "nearest.vcobj", open);
        auto sheet = [&](float z) {
            vc::geometry::Surface surface{40, 40, {1, 1}, {}};
            for (int r = 0; r < 40; ++r) for (int c = 0; c < 40; ++c) surface.points.push_back({float(c), float(r), z});
            auto write = sheets.write();
            ObjectRecord record; record.kind = "surface";
            record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":2}})";
            const auto uuid = sheets.upsertObject(write, record);
            sheets.putGeometry(write, uuid, vc::geometry::encode(surface));
            write.commit();
            return uuid;
        };
        const auto near = sheet(10), far = sheet(14);
        auto nearest = [&](std::array<double, 3> point, double tolerance, std::optional<std::vector<std::string>> uuids = std::nullopt,
                           std::optional<std::string> volume = std::nullopt) {
            SurfaceProximityQuery query; query.point = point; query.tolerance = tolerance; query.uuids = std::move(uuids);
            query.volume = std::move(volume);
            auto read = sheets.read();
            return sheets.nearestSurfaces(read, query);
        };
        auto hits = nearest({12.25, 20.5, 11}, 5);
        require(hits.size() == 2 && hits[0].objectUuid == near && hits[1].objectUuid == far, "nearest first");
        require(hits[0].distance == 1 && hits[0].grid == std::array<double, 2>{12.25, 20.5} &&
                hits[0].point == std::array<double, 3>{12.25, 20.5, 10}, "hit point, grid and distance");
        require(nearest({12.25, 20.5, 11}, 0.5).empty(), "tolerance is a bound");
        hits = nearest({12.25, 20.5, 11}, 5, std::vector<std::string>{far});
        require(hits.size() == 1 && hits[0].objectUuid == far, "restricted to the selected surfaces");
        require(nearest({12.25, 20.5, 11}, 5, std::vector<std::string>{}).empty(), "an empty selection matches nothing");
        // Volume voxels are half the stored coordinates.
        hits = nearest({6.125, 10.25, 5.5}, 2.5, std::nullopt, std::string("vol"));
        require(hits.size() == 2 && hits[0].objectUuid == near && hits[0].distance == 0.5 &&
                hits[0].point == std::array<double, 3>{6.125, 10.25, 5} && hits[0].grid == std::array<double, 2>{12.25, 20.5},
                "volume target");
        {
            auto write = sheets.write();
            std::vector<uint8_t> valid(40 * 40, 1);
            for (int r = 19; r < 23; ++r) for (int c = 11; c < 15; ++c) valid[size_t(r * 40 + c)] = 0;
            sheets.updateSurfaceChannels(write, near, {{"valid", "|u1", {40, 40}, valid}}, {}, sheets.currentGeometry(write, near)->uuid);
            write.commit();
        }
        hits = nearest({12.25, 20.5, 11}, 5);
        require(hits[0].objectUuid == near && hits[0].distance > 2, "invalid vertices are holes");
    }
    std::cout << "geometry region tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
