#include "linking/StoredGeometry.hpp"
#include "vc/core/object_links/ObjectLinks.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

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
    if (cancel && cancel->load()) throw std::runtime_error("point-collection-patch: cancelled; no changes committed");
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
};
struct Patch {
    objects::ObjectRecord object;
    std::shared_ptr<QuadSurface> surface;
    double area = 0;
};

bool validPoint(const cv::Vec3f& p)
{
    return p[0] != -1 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

std::vector<Collection> loadCollections(detail::StoredGeometry& inputs,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    std::vector<Collection> result;
    for (const auto& object : inputs.select("point_collection", params.collectionUuids)) {
        checkCancelled(cancel);
        const auto geometry = inputs.load(object);
        const auto xyz = geometry.values<double>("points");
        const auto ids = geometry.values<int64_t>("point_ids");
        const auto attributes = Json::parse(geometry.revision.attributes);
        const auto extras = attributes.value("point_attributes", Json());
        if (!extras.is_null() && (!extras.is_array() || extras.size() != ids.size()))
            throw std::runtime_error("stored point annotations do not match geometry");
        Collection collection;
        collection.object = object;
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
                parsed.working[axis] = float(parsed.source[axis] * params.pointScale);
                if (!std::isfinite(parsed.working[axis])) throw std::runtime_error("non-finite scaled point");
            }
            parsed.winding = extras.is_null() || extras[i].is_null() ? Json() : extras[i].value("wind_a", Json());
            if (!parsed.winding.is_null() && !parsed.winding.is_number())
                throw std::runtime_error("wind_a must be numeric or null");
            collection.points.push_back(std::move(parsed));
        }
        std::sort(collection.points.begin(), collection.points.end(), [](const Point& a, const Point& b) { return a.id < b.id; });
        result.push_back(std::move(collection));
    }
    return result;
}

std::vector<Patch> loadPatches(detail::StoredGeometry& inputs,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    std::vector<Patch> patches;
    const auto objects = inputs.select("patch", params.patchUuids);
    for (const auto& object : objects) {
        checkCancelled(cancel);
        util::reportProgress(params.progress, "loading PCL target patches", patches.size(), objects.size(), object.uuid);
        Patch patch;
        patch.object = object;
        const auto original = detail::patchSurface(inputs.load(object));
        const auto scale = original->scale();
        const auto& source = *original->rawPointsPtr();
        auto working = source.clone();
        for (int row = 0; row < source.rows; ++row) {
            checkCancelled(cancel);
            for (int col = 0; col < source.cols; ++col) {
                if (row + 1 < source.rows && col + 1 < source.cols && validPoint(source(row, col)) &&
                    validPoint(source(row + 1, col)) && validPoint(source(row, col + 1)) && validPoint(source(row + 1, col + 1)))
                    patch.area += 1;
                if (validPoint(source(row, col))) {
                    working(row, col) *= params.patchScale;
                    for (int axis = 0; axis < 3; ++axis)
                        if (!std::isfinite(working(row, col)[axis])) throw std::runtime_error("scaled patch coordinate overflow");
                }
            }
        }
        patch.area /= double(scale[0]) * scale[1];
        patch.surface = std::make_shared<QuadSurface>(working, scale);
        patches.push_back(std::move(patch));
    }
    return patches;
}

std::set<size_t> betweenTargets(const std::string& name, const std::map<std::string, size_t>& patches)
{
    const std::string prefix = "between_patches__";
    if (!name.starts_with(prefix)) return {};
    const auto rest = name.substr(prefix.size());
    for (auto split = rest.find("__"); split != std::string::npos; split = rest.find("__", split + 1)) {
        const auto a = patches.find(rest.substr(0, split)), b = patches.find(rest.substr(split + 2));
        if (a != patches.end() && b != patches.end()) return {a->second, b->second};
    }
    return {}; // Spiral falls back to the general policy if either name is absent.
}

std::array<double, 3> projection(const Patch& patch, double row, double col, double patchScale)
{
    const auto& points = *patch.surface->rawPointsPtr();
    const int r = std::clamp(int(std::floor(row)), 0, points.rows - 2);
    const int c = std::clamp(int(std::floor(col)), 0, points.cols - 2);
    const float dr = std::clamp(float(row - r), 0.0f, 1.0f), dc = std::clamp(float(col - c), 0.0f, 1.0f);
    const auto p00 = points(r, c), p10 = points(r, c + 1), p01 = points(r + 1, c), p11 = points(r + 1, c + 1);
    const auto foot = dr + dc <= 1 ? p00 + dc * (p10 - p00) + dr * (p01 - p00)
        : p11 + (1 - dr) * (p10 - p11) + (1 - dc) * (p01 - p11);
    return {foot[0] / patchScale, foot[1] / patchScale, foot[2] / patchScale};
}
} // namespace

DetectorReport linkPointCollectionsToPatches(objects::ObjectStore& db,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    if (!(params.tolerance > 0) || !std::isfinite(params.tolerance) || params.tolerance > std::numeric_limits<float>::max() ||
        !(params.pointScale > 0) || !std::isfinite(params.pointScale) ||
        !(params.patchScale > 0) || !std::isfinite(params.patchScale) || params.edgeKind.empty() ||
        params.windowPoints < 1 || params.windowMinPoints < 1 || params.windowMinPoints > params.windowPoints)
        throw std::invalid_argument("point-collection-patch: invalid tolerance, scale, window or edge kind");
    checkCancelled(cancel);
    util::reportProgress(params.progress, "loading point collections");
    detail::StoredGeometry inputs(db);
    auto collections = loadCollections(inputs, params, cancel);
    auto patches = loadPatches(inputs, params, cancel);
    std::map<std::string, size_t> byName;
    std::unordered_map<QuadSurface*, size_t> bySurface;
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    StaleEdgeScope scope{{params.edgeKind}, {}};
    for (size_t i = 0; i < patches.size(); ++i) {
        byName[patches[i].object.name] = i;
        bySurface[patches[i].surface.get()] = i;
        surfaces.push_back(patches[i].surface);
        scope.objectUuids.push_back(patches[i].object.uuid);
    }
    SurfacePatchIndex index;
    index.setSamplingStride(1);
    util::reportProgress(params.progress, "indexing PCL target patches");
    index.rebuild(surfaces, float(params.tolerance), false);
    std::vector<LinkCandidate> candidates;
    for (size_t ci = 0; ci < collections.size(); ++ci) {
        checkCancelled(cancel);
        const auto& collection = collections[ci];
        const bool absolute = collection.absoluteWinding;
        scope.objectUuids.push_back(collection.object.uuid);
        util::reportProgress(params.progress, "linking point collections", ci, collections.size(), collection.object.name);
        const auto targets = betweenTargets(collection.object.name, byName);
        const auto count = collection.points.size();
        std::vector<std::map<size_t, SurfacePatchIndex::LookupResult>> hits(count);
        std::map<size_t, std::vector<size_t>> hitPoints;
        for (size_t i = 0; i < count; ++i) {
            checkCancelled(cancel);
            for (const auto& hit : index.locateAll({collection.points[i].working, float(params.tolerance), {}})) {
                const auto patch = bySurface.at(hit.surface.get());
                if (!targets.empty() && !targets.contains(patch)) continue;
                hits[i].emplace(patch, hit);
                hitPoints[patch].push_back(i);
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
                if (!winner || (targets.empty() && patches[pi].area > patches[*winner].area) ||
                    ((!targets.empty() || patches[pi].area == patches[*winner].area) && hit.distance < hits[i].at(*winner).distance))
                    winner = pi;
            }
            if (!winner) continue;
            const auto& point = collection.points[i];
            const auto& patch = patches[*winner];
            const auto& hit = hits[i].at(*winner);
            const auto center = patch.surface->center();
            const auto scale = patch.surface->scale();
            const double col = hit.ptr[0] + center[0] * scale[0], row = hit.ptr[1] + center[1] * scale[1];
            LinkCandidate candidate;
            candidate.kind = params.edgeKind;
            candidate.aUuid = collection.object.uuid; candidate.bUuid = patch.object.uuid;
            candidate.aAnchor.index = point.id;
            candidate.aAnchor.position = point.source;
            candidate.bAnchor.position = projection(patch, row, col, params.patchScale);
            candidate.meta = Json{{"anchor_index_space", "point_id"},
                {"point_collection_patch", {{"ij", {row, col}}, {"distance", hit.distance},
                    {"winding_annotation", point.winding}, {"winding_is_absolute", absolute},
                    {"selection", targets.empty() ? "largest_area" : "nearest_named_pair"}}},
                {"settings", {{"tolerance", params.tolerance}, {"point_scale", params.pointScale}, {"patch_scale", params.patchScale},
                    {"window_points", params.windowPoints}, {"window_min_points", params.windowMinPoints}}}}.dump();
            candidates.push_back(std::move(candidate));
        }
    }
    checkCancelled(cancel);
    auto txn = db.write();
    auto report = detail::commitStoredCandidates(db, inputs, txn, kPointCollectionPatchDetector, 1, candidates,
        params.replaceStalePending, &scope, params.progress);
    checkCancelled(cancel);
    util::reportProgress(params.progress, "committing point-collection links");
    checkCancelled(cancel);
    txn.commit();
    report.objectsConsidered = int64_t(collections.size() + patches.size());
    return report;
}
} // namespace vc::object_links
