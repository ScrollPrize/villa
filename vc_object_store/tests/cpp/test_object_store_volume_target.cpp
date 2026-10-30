#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <nlohmann/json.hpp>
using namespace vc::objects;

namespace {
template <class F> bool rejects(F f)
{
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
}

int main() try
{
    TemporaryDirectory directory;
    OpenOptions open; open.author = "volume-target-test";
    auto store = ObjectStore::create(directory.path / "volume.vcobj", open);
    const std::string volume = "20260411134726";
    // name -> uuid. Each fiber lies along x at volume y = 10 * row.
    std::map<std::string, std::string> ids;
    auto add = [&](const std::string& name, std::optional<double> scale, double storedLength, int row,
                   const std::string& declaredVolume) {
        auto txn = store.write();
        ObjectRecord record; record.kind = "fiber"; record.name = name;
        record.fiber.emplace(); record.fiber->length = storedLength;
        if (scale) record.meta = nlohmann::json{{"coordinates", {{"volume_uuid", declaredVolume}, {"scale_factor", *scale}}}}.dump();
        const auto id = store.upsertObject(txn, record);
        const double y = 10.0 * row * scale.value_or(1.0);
        vc::geometry::Fiber fiber; fiber.controlPoints = {{0, y, 0}, {storedLength, y, 0}};
        store.putGeometry(txn, id, vc::geometry::encode(fiber));
        txn.commit();
        ids[name] = id;
    };
    // Volume lengths: a40 -> 20, b72 -> 18, b60 -> 15, a30 -> 15, a10 -> 5.
    add("a40", 2.0, 40, 1, volume);
    add("a30", 2.0, 30, 2, volume);
    add("a10", 2.0, 10, 3, volume);
    add("b72", 4.0, 72, 4, volume);
    add("b60", 4.0, 60, 5, volume);
    add("loose1", std::nullopt, 100, 6, volume);
    add("loose2", std::nullopt, 100, 7, volume);
    add("elsewhere", 1.0, 1000, 8, "20990101000000");
    auto names = [&](const std::vector<ObjectRecord>& records) {
        std::vector<std::string> result;
        for (const auto& record : records) result.push_back(record.name);
        return result;
    };
    auto txn = store.read();
    ObjectSelection byLength; byLength.kind = "fiber"; byLength.volume = volume; byLength.orderBy = "length_desc";
    const std::vector<std::string> ordered{"a40", "b72", "b60", "a30", "a10"};
    require(names(store.selectObjects(txn, byLength)) == ordered,
            "bound frames merge by volume length, ties by stored length");

    // Pages resume exactly, including inside the cross-frame tie at 15.
    for (const int64_t size : {1, 2, 3}) {
        std::vector<std::string> paged;
        auto page = byLength; page.limit = size;
        for (;;) {
            const auto rows = store.selectObjects(txn, page, std::vector<std::string>{"name", "fiber", "meta"});
            if (rows.empty()) break;
            for (const auto& row : rows) paged.push_back(row.name);
            const auto& last = rows.back();
            const auto toVolume = geometryScaleToVolume(nlohmann::json::parse(last.meta), volume);
            require(last.volumeLength == last.fiber->length * *toVolume, "rows carry the volume length they were ordered by");
            page.afterUuid = last.uuid; page.afterLength = *last.volumeLength;
        }
        require(paged == ordered, "volume cursor pages without repeats or gaps");
    }
    auto atLeast = byLength; atLeast.minLength = 15.0;
    require(names(store.selectObjects(txn, atLeast)) == std::vector<std::string>{"a40", "b72", "b60", "a30"},
            "minimum length is inclusive in volume voxels");
    atLeast.minLength = std::nextafter(15.0, 16.0);
    require(names(store.selectObjects(txn, atLeast)) == std::vector<std::string>{"a40", "b72"},
            "minimum length is exact in volume voxels");
    // At a working level, half the volume's voxels: lengths are measured there.
    auto atHalf = byLength; atHalf.targetScale = 0.5; atHalf.minLength = 7.5;
    const auto half = store.selectObjects(txn, atHalf, std::vector<std::string>{"name", "fiber"});
    require(names(half) == std::vector<std::string>{"a40", "b72", "b60", "a30"} && half.front().volumeLength == 10.0,
            "a target scale measures lengths in its voxels");
    require(rejects([&] { auto bad = byLength; bad.targetScale = 0; (void)store.selectObjects(txn, bad); }),
            "a target scale must be positive");

    // Unbound objects (including other volumes' frames) follow in UUID order.
    auto withUnbound = byLength; withUnbound.includeUnbound = true;
    auto all = names(store.selectObjects(txn, withUnbound));
    require(all.size() == 8 && std::equal(ordered.begin(), ordered.end(), all.begin()), "bound rows come first");
    std::vector<std::string> tail{ids["loose1"], ids["loose2"], ids["elsewhere"]};
    std::sort(tail.begin(), tail.end());
    const auto records = store.selectObjects(txn, withUnbound, std::vector<std::string>{});
    for (size_t i = 0; i < tail.size(); ++i) require(records[5 + i].uuid == tail[i], "unbound tail is UUID ordered");
    auto inTail = withUnbound; inTail.afterUuid = tail[0];
    const auto rest = store.selectObjects(txn, inTail, std::vector<std::string>{});
    require(rest.size() == 2 && rest[0].uuid == tail[1] && rest[1].uuid == tail[2], "a cursor without length resumes the tail");
    auto boundCursor = withUnbound; boundCursor.afterUuid = ids["a10"]; boundCursor.afterLength = 5.0;
    require(store.selectObjects(txn, boundCursor).size() == 3, "the last bound row continues into the tail");
    withUnbound.minLength = 1.0;
    require(names(store.selectObjects(txn, withUnbound)) == ordered, "a minimum length excludes unbound objects");

    ObjectSelection byUuid; byUuid.kind = "fiber"; byUuid.volume = volume;
    auto uuids = store.selectObjects(txn, byUuid, std::vector<std::string>{});
    require(uuids.size() == 5 && std::is_sorted(uuids.begin(), uuids.end(),
            [](const auto& a, const auto& b) { return a.uuid < b.uuid; }), "UUID order spans the frames");

    // A volume box selects by mapped bounds; regions keep their identity, in volume units.
    ObjectSelection box; box.kind = "fiber"; box.volume = volume; box.bbox = std::array<double, 6>{-1, 39.5, -1, 100, 40.5, 1};
    require(names(store.selectObjects(txn, box)) == std::vector<std::string>{"b72"}, "box is in volume voxels");
    const auto regions = store.spatialRegions(txn, box);
    require(!regions.empty() && std::all_of(regions.begin(), regions.end(), [&](const auto& region) {
        return region.objectUuid == ids["b72"] && region.bbox[1] <= 40 && region.bbox[4] >= 40;
    }), "spatial regions report volume bounds");

    RegionSlice slice; slice.localBounds = {-1, -1, -0.5, 1000, 1000, 0.5};
    RegionOptions region; region.volume = volume;
    const auto everything = store.geometryRegion(txn, slice, region);
    require(everything.complete() && everything.intersectingObjects == 5, "region spans the bound frames only");
    for (const auto& run : everything.runs) {
        const auto name = std::find_if(ids.begin(), ids.end(), [&](const auto& id) { return id.second == run.objectUuid; })->first;
        const std::map<std::string, double> rows{{"a40", 10}, {"a30", 20}, {"a10", 30}, {"b72", 40}, {"b60", 50}};
        require(run.points.front()[1] == rows.at(name), "region points are in volume voxels");
    }
    region.maximumObjects = 2; region.selected = ids["b60"];
    const auto limited = store.geometryRegion(txn, slice, region);
    std::vector<std::string> limitedIds;
    for (const auto& run : limited.runs)
        if (limitedIds.empty() || limitedIds.back() != run.objectUuid) limitedIds.push_back(run.objectUuid);
    require(limited.stop == RegionStop::ObjectLimit && limitedIds.size() == 2 && limitedIds.front() == ids["b60"],
            "object limit applies across frames after the selected object");
    region.maximumObjects = 0; region.selected.reset(); region.minLength = 18.0;
    require(store.geometryRegion(txn, slice, region).intersectingObjects == 2, "region minimum length is in volume voxels");

    auto conflicting = byLength; conflicting.coordinates = R"({"volume_uuid":"20260411134726","scale_factor":2})";
    require(rejects([&] { store.selectObjects(txn, conflicting); }), "volume and coordinates are exclusive");
    ObjectSelection orphan; orphan.kind = "fiber"; orphan.includeUnbound = true;
    require(rejects([&] { store.selectObjects(txn, orphan); }), "unbound listing needs a volume");
    auto unboundRegions = box; unboundRegions.includeUnbound = true;
    require(rejects([&] { store.spatialRegions(txn, unboundRegions); }), "regions cannot place unbound objects");

    require(store.unboundObjectCount(txn, "fiber", volume) == 3 && store.unboundObjectCount(txn, "track", volume) == 0,
            "unbound count matches the unbound tail");

    // Deleting fibers retires them: they and any frame left without active
    // objects leave the catalog, the unbound count and the overlay regions.
    {
        const std::vector<std::string> retire{ids["a10"], ids["b72"], ids["b60"], ids["loose1"]};
        RevisionSelection expected;
        for (const auto& uuid : retire) expected[uuid] = store.currentObjectRevision(txn, uuid)->uuid;
        auto write = store.write(expected);
        for (const auto& uuid : retire) store.retireObject(write, uuid);
        write.commit();
    }
    auto after = store.read();
    require(names(store.selectObjects(after, byLength)) == std::vector<std::string>{"a40", "a30"},
            "retired objects and their frames leave the volume selection");
    auto activeUnbound = byLength; activeUnbound.includeUnbound = true;
    const auto active = names(store.selectObjects(after, activeUnbound));
    require(active.size() == 4 && active[0] == "a40" && active[1] == "a30" &&
            std::set<std::string>(active.begin() + 2, active.end()) == std::set<std::string>{"loose2", "elsewhere"},
            "the unbound tail lists active objects of other or no volumes");
    require(store.unboundObjectCount(after, "fiber", volume) == 2, "the unbound count skips retired objects");
    RegionOptions activeRegion; activeRegion.volume = volume;
    require(store.geometryRegion(after, slice, activeRegion).intersectingObjects == 2,
            "regions skip the frames of retired objects");
    std::cout << "volume target tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
