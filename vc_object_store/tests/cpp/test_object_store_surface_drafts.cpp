#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace vc::objects;
using vc::geometry::Surface;
using vc::geometry::SurfacePoint;

namespace {
const std::string volume = "20260411134726";

// Tilted so z = x + 1 + lift, spanning three tile rows, with a hole.
Surface tilted(float lift)
{
    Surface surface{70, 40, {1, 1}, {}};
    for (int r = 0; r < 70; ++r)
        for (int c = 0; c < 40; ++c) {
            const bool hole = r >= 40 && r < 44 && c >= 10 && c < 14;
            surface.points.push_back(hole ? SurfacePoint{-1, -1, -1} : SurfacePoint{float(c), float(r), float(c + 1) + lift});
        }
    return surface;
}

RegionSlice plane(double z)
{
    RegionSlice slice;
    slice.origin = {0, 0, z};
    slice.localBounds = {-100, -100, -0.5, 100, 100, 0.5};
    return slice;
}

RegionOptions surfaces()
{
    RegionOptions options;
    options.kind = "surface";
    options.volume = volume;
    return options;
}

std::vector<VisibleRun> runsOf(const std::vector<AttachedRegion>& regions)
{
    std::vector<VisibleRun> runs;
    for (const auto& region : regions) {
        require(region.result.complete(), "complete region");
        for (const auto& run : region.result.runs) runs.push_back(run);
    }
    return runs;
}

bool sameRuns(const std::vector<VisibleRun>& a, const std::vector<VisibleRun>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].objectUuid != b[i].objectUuid || a[i].firstSegment != b[i].firstSegment || a[i].points != b[i].points ||
            a[i].grid != b[i].grid)
            return false;
    return true;
}
}

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "draft-test";
    const auto path = directory.path / "drafts.vcobj";
    std::string stored;
    {
        auto store = ObjectStore::create(path, options);
        auto write = store.write();
        ObjectRecord record; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":2}})";
        stored = store.upsertObject(write, record);
        store.putGeometry(write, stored, vc::geometry::encode(tilted(0)));
        write.commit();
    }
    ObjectStoreSession session;
    const auto attachment = session.attach(path, options);
    (void)session.takeEvents();
    const auto order = std::vector{attachment};
    const auto coordinates = volumeCoordinates(volume, 2);
    // The volume is at half the stored scale: z = 10.25 in it is z = 20.5 stored.
    const auto slice = plane(10.25);

    const auto committed = runsOf(session.geometryRegion(order, slice, surfaces()));
    require(!committed.empty() && committed[0].revisionUuid.size(), "committed surface crosses the plane");
    // Stored x and y are the grid column and row, at twice the volume's voxels.
    for (const auto& run : committed) {
        require(run.grid.size() == run.points.size(), "a grid position per point");
        for (size_t i = 0; i < run.points.size(); ++i)
            require(std::abs(run.grid[i][0] - 2 * run.points[i][0]) < 1e-9 &&
                    std::abs(run.grid[i][1] - 2 * run.points[i][1]) < 1e-9, "runs carry where they cross the grid");
    }
    SurfaceProximityQuery near; near.volume = volume; near.point = {10, 20, 10.7}; near.tolerance = 1;
    const auto committedHits = session.nearestSurfaces(order, near);
    require(committedHits.size() == 1, "committed surface is near the point");

    // A draft of the same geometry answers exactly as the stored surface does.
    session.setDraft(stored, tilted(0), coordinates, attachment);
    require(session.hasDraft(stored), "draft is held");
    auto events = session.takeEvents();
    require(events.size() == 1 && events[0].kind == StoreSessionEvent::Kind::Draft && events[0].attachment == attachment &&
            events[0].changes.size() == 1 && events[0].changes[0].recordUuid == stored, "setting a draft is a Draft event");
    require(events[0].touches({"surface"}) && !events[0].touches({"fiber"}), "draft events touch surfaces only");
    auto regions = session.geometryRegion(order, slice, surfaces());
    auto drafted = runsOf(regions);
    require(sameRuns(drafted, committed), "a draft cuts exactly like its stored geometry");
    require(std::all_of(drafted.begin(), drafted.end(), [](const auto& run) { return run.revisionUuid.empty(); }),
            "draft runs have no revision");
    require(regions.back().attachment == attachment && regions.back().result.intersectingObjects == 1,
            "draft runs report the draft's attachment");
    auto hits = session.nearestSurfaces(order, near);
    require(hits.size() == 1 && hits[0].hit.revisionUuid.empty() && hits[0].hit.grid == committedHits[0].hit.grid &&
            hits[0].hit.point == committedHits[0].hit.point && hits[0].hit.distance == committedHits[0].hit.distance,
            "a draft is found exactly where its stored geometry is");

    // The draft replaces the stored geometry: lift it out of the slab.
    session.setDraft(stored, tilted(5), coordinates, attachment);
    drafted = runsOf(session.geometryRegion(order, slice, surfaces()));
    require(!drafted.empty() && !sameRuns(drafted, committed), "the draft, not the stored surface, is cut");
    hits = session.nearestSurfaces(order, near);
    require(hits.empty(), "the stored surface is hidden by its draft");

    // Rectangle edits equal setting the whole edited grid; unchanged tiles stay put.
    session.setDraft(stored, tilted(0), coordinates, attachment);
    auto edited = tilted(0);
    std::vector<SurfacePoint> patch;
    std::vector<uint8_t> validity;
    for (int r = 30; r < 36; ++r)
        for (int c = 5; c < 25; ++c) {
            auto& p = edited.points[size_t(r * 40 + c)];
            p[2] += 3;
            patch.push_back(p);
            validity.push_back(c != 7);
        }
    edited.validity.assign(edited.points.size(), 1);
    for (int r = 30; r < 36; ++r) edited.validity[size_t(r * 40 + 7)] = 0;
    session.updateDraft(stored, 30, 5, 6, 20, patch, validity);
    const auto updated = runsOf(session.geometryRegion(order, slice, surfaces()));
    session.setDraft(stored, edited, coordinates, attachment);
    require(sameRuns(updated, runsOf(session.geometryRegion(order, slice, surfaces()))), "edits are rectangles of the grid");
    mustFail([&] { session.updateDraft(stored, 68, 0, 3, 1, std::vector<SurfacePoint>(3, {1, 1, 1})); });
    mustFail([&] { session.updateDraft(stored, 0, 0, 1, 1, std::vector<SurfacePoint>{{1, 1, -5}}); });
    mustFail([&] { session.updateDraft(directory.path.string(), 0, 0, 1, 1, std::vector<SurfacePoint>{{1, 1, 1}}); });

    // Clearing restores the stored geometry.
    session.clearDraft(stored);
    require(!session.hasDraft(stored) && sameRuns(runsOf(session.geometryRegion(order, slice, surfaces())), committed),
            "cleared drafts leave the stored surface");
    session.clearDraft(stored);

    // A new object's draft: no attachment, filtered like a stored surface.
    const auto fresh = util::newUuid();
    (void)session.takeEvents();
    session.setDraft(fresh, tilted(0), volumeCoordinates(volume, 1));
    events = session.takeEvents();
    require(events.size() == 1 && events[0].attachment == AttachmentIdentity{}, "unattached draft event");
    regions = session.geometryRegion(order, plane(20.5), surfaces());
    require(regions.back().attachment == AttachmentIdentity{} && regions.back().result.runs.size() == 1 &&
            regions.back().result.runs[0].objectUuid == fresh, "an unattached draft is cut in its own volume frame");
    auto only = surfaces(); only.uuids = std::vector{stored};
    require(runsOf(session.geometryRegion(order, plane(20.5), only)).size() == 0, "uuid filters apply to drafts");
    auto elsewhere = surfaces(); elsewhere.volume = "another-volume";
    require(runsOf(session.geometryRegion(order, plane(20.5), elsewhere)).empty(), "drafts stay in their volume");
    require(runsOf(session.geometryRegion({}, plane(20.5), surfaces())).size() == 1, "drafts need no attachment");
    auto declared = surfaces(); declared.volume.reset(); declared.coordinates = volumeCoordinates(volume, 1).dump();
    require(runsOf(session.geometryRegion(order, plane(20.5), declared)).size() == 1, "coordinate filters match drafts");
    declared.coordinates = coordinates.dump();
    const auto byStored = runsOf(session.geometryRegion(order, plane(20.5), declared));
    require(byStored.size() && std::all_of(byStored.begin(), byStored.end(), [&](const auto& run) { return run.objectUuid == stored; }),
            "other coordinates exclude the draft");
    auto family = surfaces(); family.family = "any";
    mustFail([&] { (void)session.geometryRegion({}, plane(20.5), family); });
    session.clearDraft(fresh);
    std::cout << "surface draft tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
