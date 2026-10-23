#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
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
        RegionOptions patches; patches.kind = "patch";
        bool rejected = false;
        try { (void)store.geometryRegion(read, slice, patches); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unsupported kinds are rejected");
    }
    std::cout << "geometry region tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
