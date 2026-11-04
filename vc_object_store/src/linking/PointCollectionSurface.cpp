#include "linking/StoredGeometry.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "spatial/SurfaceGrid.hpp"
#include "spatial/SurfaceIndex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace vc::object_links {
namespace {
using Json = nlohmann::json;

void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("point-collection-surface: cancelled; no changes committed");
}

struct Point {
    int64_t id;
    std::array<double, 3> source;
    cv::Vec3f working;
    Json winding;
};
struct Collection {
    objects::ObjectRecord object;
    std::vector<Point> points;
    bool absoluteWinding = false;
    double sourceScale = 1.0;
};
struct Surface {
    objects::ObjectRecord object;
    std::shared_ptr<SurfaceGrid> grid;
    double area = 0;
    double sourceScale = 1.0;
};

// Load every selected object, recording each one that fails and leaving it out
// of the run (and so of stale-edge removal). Cancellation and coordinate-frame
// errors (invalid_argument, e.g. unrepresentable scaled XYZ) still abort the run
// before any write.
template <class Load>
void loadEach(const std::vector<objects::ObjectRecord>& objects, DetectorReport& report,
    const std::atomic_bool* cancel, Load load)
{
    for (const auto& object : objects) {
        checkCancelled(cancel);
        try {
            load(object);
        } catch (const std::invalid_argument&) {
            throw;
        } catch (const std::exception& e) {
            checkCancelled(cancel);
            detail::reportFailure(report, "load", object.uuid, e.what(), object.uuid + ": " + e.what() + ", skipped");
        }
    }
}

// A collection with a role is data attached to an object (e.g. fit results),
// never a source of surface links. Naming one explicitly is an error.
std::vector<objects::ObjectRecord> linkableCollections(detail::StoredGeometry& inputs,
    const PointCollectionSurfaceParams& params)
{
    std::vector<objects::ObjectRecord> result;
    for (auto& object : inputs.select("point_collection", params.collectionUuids)) {
        if (const auto meta = Json::parse(object.meta.empty() ? std::string("{}") : object.meta);
            meta.is_object() && meta.contains("role")) {
            if (!params.collectionUuids.empty())
                throw std::invalid_argument("point collection " + object.uuid + " has role " +
                                            meta.at("role").dump() + " and is not linked to surfaces");
            continue;
        }
        result.push_back(std::move(object));
    }
    return result;
}

std::vector<Collection> loadCollections(detail::StoredGeometry& inputs,
    const std::vector<objects::ObjectRecord>& objects, const PointCollectionSurfaceParams& params,
    const std::map<std::string, double>& scales, DetectorReport& report, const std::atomic_bool* cancel)
{
    std::vector<Collection> result;
    loadEach(objects, report, cancel, [&](const objects::ObjectRecord& object) {
        const auto geometry = inputs.load(object);
        const auto xyz = geometry.values<double>("points");
        const auto ids = geometry.values<int64_t>("point_ids");
        const auto attributes = Json::parse(object.meta).value("source", Json::object());
        const auto extras = Json::parse(inputs.pointAnnotations(geometry.revision, "points"));
        if (!extras.is_null() && (!extras.is_array() || extras.size() != ids.size()))
            throw std::runtime_error("stored point annotations do not match geometry");
        Collection collection;
        collection.object = object;
        collection.sourceScale = scales.at(object.uuid);
        collection.absoluteWinding = params.absoluteWinding.value_or(attributes.value("winding_is_absolute", false));
        std::set<int64_t> seen;
        for (size_t i = 0; i < ids.size(); ++i) {
            checkCancelled(cancel);
            Point parsed;
            parsed.id = ids[i];
            if (parsed.id < 0 || !seen.insert(parsed.id).second)
                throw std::runtime_error("stored point IDs must be nonnegative and unique");
            parsed.source = {xyz[3*i], xyz[3*i+1], xyz[3*i+2]};
            for (int axis = 0; axis < 3; ++axis) {
                parsed.working[axis] = float(parsed.source[axis] * collection.sourceScale);
                if (!std::isfinite(parsed.working[axis])) throw std::runtime_error("non-finite scaled point");
            }
            parsed.winding = extras.is_null() || extras[i].is_null() ? Json() : extras[i].value("wind_a", Json());
            if (!parsed.winding.is_null() && !parsed.winding.is_number())
                throw std::runtime_error("wind_a must be numeric or null");
            collection.points.push_back(std::move(parsed));
        }
        std::sort(collection.points.begin(), collection.points.end(), [](const Point& a, const Point& b) { return a.id < b.id; });
        result.push_back(std::move(collection));
    });
    return result;
}

std::vector<Surface> loadSurfaces(detail::StoredGeometry& inputs, const std::vector<objects::ObjectRecord>& objects,
    const PointCollectionSurfaceParams& params, const std::map<std::string, double>& scales,
    DetectorReport& report, const std::atomic_bool* cancel)
{
    std::vector<Surface> surfaces;
    size_t loaded = 0;
    loadEach(objects, report, cancel, [&](const objects::ObjectRecord& object) {
        objects::util::reportProgress(params.progress, "loading PCL target surfaces", loaded++, objects.size(), object.uuid);
        Surface surface;
        surface.object = object;
        surface.sourceScale = scales.at(object.uuid);
        const auto original = detail::surfaceGrid(inputs.load(object));
        surface.grid = detail::scaledSurfaceGrid(original, surface.sourceScale, params.volumeUuid.has_value());
        const auto& source = *original->rawPointsPtr();
        for (int row = 0; row + 1 < source.rows; ++row) {
            checkCancelled(cancel);
            for (int col = 0; col + 1 < source.cols; ++col)
                if (vc::geometry::validSurfacePoint(source(row, col)) && vc::geometry::validSurfacePoint(source(row + 1, col)) &&
                    vc::geometry::validSurfacePoint(source(row, col + 1)) && vc::geometry::validSurfacePoint(source(row + 1, col + 1)))
                    surface.area += 1;
        }
        const auto scale = surface.grid->scale();
        surface.area /= double(scale[0]) * scale[1];
        surfaces.push_back(std::move(surface));
    });
    return surfaces;
}

template <class Value>
std::set<Value> betweenTargets(const std::string& name, const std::map<std::string, Value>& surfaces)
{
    const std::string prefix = "between_surfaces__";
    if (!name.starts_with(prefix)) return {};
    const auto rest = name.substr(prefix.size());
    for (auto split = rest.find("__"); split != std::string::npos; split = rest.find("__", split + 1)) {
        const auto a = surfaces.find(rest.substr(0, split)), b = surfaces.find(rest.substr(split + 2));
        if (a != surfaces.end() && b != surfaces.end()) return {a->second, b->second};
    }
    return {}; // Spiral falls back to the general policy if either name is absent.
}

template <class Keep>
std::vector<objects::ObjectRecord> keepOnly(const std::vector<objects::ObjectRecord>& objects, Keep keep)
{
    std::vector<objects::ObjectRecord> result;
    for (const auto& object : objects)
        if (keep(object.uuid)) result.push_back(object);
    return result;
}

std::array<double, 3> projection(const Surface& surface, const SurfaceIndex::LookupResult& hit, double surfaceScale)
{
    const auto& points = *surface.grid->rawPointsPtr();
    const int r = hit.cell[1], c = hit.cell[0];
    const double row = hit.grid[1], col = hit.grid[0];
    const float dr = std::clamp(float(row - r), 0.0f, 1.0f), dc = std::clamp(float(col - c), 0.0f, 1.0f);
    const auto p00 = points(r, c), p10 = points(r, c + 1), p01 = points(r + 1, c), p11 = points(r + 1, c + 1);
    const auto foot = dr + dc <= 1 ? p00 + dc * (p10 - p00) + dr * (p01 - p00)
        : p11 + (1 - dr) * (p10 - p11) + (1 - dc) * (p01 - p11);
    return {foot[0] / surfaceScale, foot[1] / surfaceScale, foot[2] / surfaceScale};
}
} // namespace

DetectorReport linkPointCollectionsToSurfaces(objects::ObjectStore& db,
    const PointCollectionSurfaceParams& params, const std::atomic_bool* cancel)
{
    if (!(params.tolerance > 0) || !std::isfinite(params.tolerance) || params.tolerance > std::numeric_limits<float>::max() ||
        !(params.pointScale > 0) || !std::isfinite(params.pointScale) ||
        !(params.surfaceScale > 0) || !std::isfinite(params.surfaceScale) || params.edgeKind.empty() ||
        params.windowPoints < 1 || params.windowMinPoints < 1 || params.windowMinPoints > params.windowPoints)
        throw std::invalid_argument("point-collection-surface: invalid tolerance, scale, window or edge kind");
    checkCancelled(cancel);
    objects::util::reportProgress(params.progress, "loading point collections");
    detail::StoredGeometry inputs(db);
    DetectorReport loading;
    const auto collectionObjects = linkableCollections(inputs, params);
    const auto surfaceObjects = inputs.select("surface", params.surfaceUuids);
    const auto fresh = detail::newAmong(params.newUuids, {&collectionObjects, &surfaceObjects},
                                        "point-collection-surface");
    const auto pointScales = detail::coordinateScales(collectionObjects, params, loading, params.pointScale);
    const auto surfaceScales = detail::coordinateScales(surfaceObjects, params, loading, params.surfaceScale);
    if (loading.failed()) return loading;
    // Named-pair targets of an incremental run come from every selected
    // surface (the pair may not be loaded); a full run's from the loaded ones.
    std::map<std::string, std::string> selectedByName;
    auto collectionsToLoad = collectionObjects;
    auto surfacesToLoad = surfaceObjects;
    if (fresh) {
        objects::util::reportProgress(params.progress, "finding partners of new objects");
        std::set<std::string> surfaces, collections, allSurfaces, allCollections;
        for (const auto& object : surfaceObjects) {
            allSurfaces.insert(object.uuid);
            selectedByName[object.name] = object.uuid;
        }
        for (const auto& object : collectionObjects) allCollections.insert(object.uuid);
        auto near = [&](const objects::ObjectRecord& object, double scale, const std::string& kind, double otherScale,
                        const std::set<std::string>& among, std::set<std::string>& into) {
            const auto box = inputs.workingBounds(object, params, scale);
            if (!box) { into.insert(among.begin(), among.end()); return; }
            for (const auto& uuid : inputs.objectsNear(kind, *box, params.tolerance, params, otherScale))
                if (among.contains(uuid)) into.insert(uuid);
        };
        for (const auto& object : collectionObjects) {
            if (!fresh->contains(object.uuid)) continue;
            collections.insert(object.uuid);
            near(object, params.pointScale, "surface", params.surfaceScale, allSurfaces, surfaces);
            for (const auto& uuid : betweenTargets(object.name, selectedByName)) surfaces.insert(uuid);
        }
        for (const auto& object : surfaceObjects) {
            if (!fresh->contains(object.uuid)) continue;
            surfaces.insert(object.uuid);
            near(object, params.surfaceScale, "point_collection", params.pointScale, allCollections, collections);
        }
        collectionsToLoad = keepOnly(collectionObjects, [&](const std::string& uuid) { return collections.contains(uuid); });
        surfacesToLoad = keepOnly(surfaceObjects, [&](const std::string& uuid) { return surfaces.contains(uuid); });
    }
    auto collections = loadCollections(inputs, collectionsToLoad, params, pointScales, loading, cancel);
    auto surfaces = loadSurfaces(inputs, surfacesToLoad, params, surfaceScales, loading, cancel);
    std::map<std::string, size_t> byName, byUuid;
    std::unordered_map<SurfaceGrid*, size_t> byGrid;
    std::vector<SurfaceIndex::SurfacePtr> grids;
    StaleEdgeScope scope{{params.edgeKind}, {}};
    for (size_t i = 0; i < surfaces.size(); ++i) {
        byName[surfaces[i].object.name] = i;
        byUuid[surfaces[i].object.uuid] = i;
        byGrid[surfaces[i].grid.get()] = i;
        grids.push_back(surfaces[i].grid);
        if (!fresh) scope.objectUuids.push_back(surfaces[i].object.uuid);
    }
    if (fresh) {
        // Edges of a new object, both endpoints selected and not failed.
        const auto failed = detail::failedObjects(loading);
        for (const auto* objects : {&collectionObjects, &surfaceObjects})
            for (const auto& object : *objects)
                if (!failed.contains(object.uuid)) scope.objectUuids.push_back(object.uuid);
        for (const auto& uuid : *fresh)
            if (!failed.contains(uuid)) scope.touching.push_back(uuid);
        if (scope.touching.empty()) scope.objectUuids.clear();
    }
    // Surfaces an incremental run may link an existing collection to: the new ones.
    std::vector<char> newSurface(surfaces.size(), 0);
    if (fresh)
        for (size_t i = 0; i < surfaces.size(); ++i) newSurface[i] = fresh->contains(surfaces[i].object.uuid);
    SurfaceIndex index;

    objects::util::reportProgress(params.progress, "indexing PCL target surfaces");
    index.rebuild(grids);
    std::vector<LinkCandidate> candidates;
    for (size_t ci = 0; ci < collections.size(); ++ci) {
        checkCancelled(cancel);
        const auto& collection = collections[ci];
        const bool absolute = collection.absoluteWinding;
        if (!fresh) scope.objectUuids.push_back(collection.object.uuid);
        objects::util::reportProgress(params.progress, "linking point collections", ci, collections.size(), collection.object.name);
        std::set<size_t> targets;
        if (!fresh) {
            targets = betweenTargets(collection.object.name, byName);
        } else {
            // As in a full run, a named surface that failed to load drops the pair policy.
            const auto named = betweenTargets(collection.object.name, selectedByName);
            for (const auto& uuid : named)
                if (const auto found = byUuid.find(uuid); found != byUuid.end()) targets.insert(found->second);
            if (targets.size() != named.size()) targets.clear();
        }
        // An existing collection is linked to the new surfaces alone.
        const bool existing = fresh && !fresh->contains(collection.object.uuid);
        if (existing && !targets.empty()) {
            std::erase_if(targets, [&](size_t surface) { return !newSurface[surface]; });
            if (targets.empty()) continue;  // neither named surface is new
        }
        const auto count = collection.points.size();
        std::vector<std::map<size_t, SurfaceIndex::LookupResult>> hits(count);
        std::map<size_t, std::vector<size_t>> hitPoints;
        for (size_t i = 0; i < count; ++i) {
            checkCancelled(cancel);
            for (const auto& hit : index.locateAll({collection.points[i].working, float(params.tolerance), {}})) {
                const auto surface = byGrid.at(hit.surface.get());
                if (!targets.empty() && !targets.contains(surface)) continue;
                if (existing && !newSurface[surface]) continue;
                hits[i].emplace(surface, hit);
                hitPoints[surface].push_back(i);
            }
        }
        const size_t half = absolute ? 0 : size_t(params.windowPoints / 2);
        for (size_t i = 0; i < count; ++i) {
            checkCancelled(cancel);
            const size_t lo = i > half ? i - half : 0, hi = std::min(count, i + half + 1);
            const size_t required = absolute ? 1 : std::min(size_t(params.windowMinPoints), hi - lo);
            std::optional<size_t> winner;
            for (const auto& [pi, hit] : hits[i]) {
                const auto& members = hitPoints.at(pi);
                const auto support = std::lower_bound(members.begin(), members.end(), hi) - std::lower_bound(members.begin(), members.end(), lo);
                if (size_t(support) < required) continue;
                if (!winner || (targets.empty() && surfaces[pi].area > surfaces[*winner].area) ||
                    ((!targets.empty() || surfaces[pi].area == surfaces[*winner].area) && hit.distance < hits[i].at(*winner).distance))
                    winner = pi;
            }
            if (!winner) continue;
            const auto& point = collection.points[i];
            const auto& surface = surfaces[*winner];
            const auto& hit = hits[i].at(*winner);
            const double col = hit.grid[0], row = hit.grid[1];
            LinkCandidate candidate;
            candidate.kind = params.edgeKind;
            candidate.aUuid = collection.object.uuid; candidate.bUuid = surface.object.uuid;
            candidate.aAnchor.points.push_back({"point_ids", {point.id}});
            candidate.aAnchor.position = point.source;
            candidate.bAnchor.position = projection(surface, hit, surface.sourceScale);
            const int r = hit.cell[1], c = hit.cell[0];
            const double dr = std::clamp(row - r, 0.0, 1.0), dc = std::clamp(col - c, 0.0, 1.0);
            candidate.bAnchor.points = dr + dc <= 1
                ? std::vector<objects::PointReference>{{"points", {r, c}}, {"points", {r, c+1}}, {"points", {r+1, c}}}
                : std::vector<objects::PointReference>{{"points", {r+1, c+1}}, {"points", {r, c+1}}, {"points", {r+1, c}}};
            candidate.meta = Json{{"anchor_index_space", "point_id"}, {"coordinate_frame", nullptr},
                {"point_collection_surface", {{"ij", {row, col}}, {"distance", hit.distance},
                    {"winding_annotation", point.winding}, {"winding_is_absolute", absolute},
                    {"selection", targets.empty() ? "largest_area" : "nearest_named_pair"}}},
                {"settings", {{"tolerance", params.tolerance}, {"point_scale", collection.sourceScale}, {"surface_scale", surface.sourceScale},
                    {"window_points", params.windowPoints}, {"window_min_points", params.windowMinPoints}}}}.dump();
            if (params.volumeUuid) {
                auto metadata = Json::parse(candidate.meta);
                metadata["coordinate_frame"] = {{"volume_uuid", *params.volumeUuid},
                    {"objects", {{collection.object.uuid, detail::coordinateProvenance(collection.object, params)},
                                 {surface.object.uuid, detail::coordinateProvenance(surface.object, params)}}}};
                candidate.meta = metadata.dump();
            }
            candidates.push_back(std::move(candidate));
        }
    }
    checkCancelled(cancel);
    auto txn = db.write();
    auto report = detail::commitStoredCandidates(db, inputs, txn, kPointCollectionSurfaceDetector, kPointCollectionSurfaceVersion, candidates,
        params.replaceStalePending, &scope, params.progress);
    checkCancelled(cancel);
    objects::util::reportProgress(params.progress, "committing point-collection links");
    checkCancelled(cancel);
    txn.commit();
    report.committed = true;
    report.objectsConsidered = int64_t(collections.size() + surfaces.size());
    report.issues = std::move(loading.issues);
    return report;
}
} // namespace vc::object_links
