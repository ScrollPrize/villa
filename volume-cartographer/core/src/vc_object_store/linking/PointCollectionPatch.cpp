#include "vc/core/object_links/ObjectLinks.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Uuid.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace vc::object_links {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("point-collection-patch: cancelled; no changes committed");
}

Json readJson(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("point-collection-patch: cannot read " + path.string());
    return Json::parse(input);
}

std::string sourceUuid(const Json& record)
{
    const auto uuid = record.value("uuid", "");
    if (!util::isUuid(uuid))
        throw std::runtime_error("point-collection-patch: every collection and patch must have a valid source UUID; re-save legacy collections in VC3D");
    std::string canonical = uuid;
    std::transform(canonical.begin(), canonical.end(), canonical.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return canonical;
}

int64_t pointId(const std::string& key)
{
    int64_t value;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), value);
    if (error != std::errc{} || end != key.data() + key.size() || value < 0)
        throw std::runtime_error("point-collection-patch: point IDs must be nonnegative signed 64-bit integers");
    return value;
}

std::string storageRef(const objects::ObjectStore& db, const fs::path& path)
{
    return fs::absolute(path).lexically_normal().lexically_relative(fs::absolute(db.path()).parent_path()).string();
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

std::vector<Collection> loadCollections(const objects::ObjectStore& db, const fs::path& path,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    const auto root = readJson(path);
    if (root.value("vc_pointcollections_json_version", "") != "1" || !root.at("collections").is_object())
        throw std::runtime_error("point-collection-patch: expected vc_pointcollections JSON version 1");
    std::vector<Collection> result;
    std::set<std::string> uuids;
    for (const auto& [id, data] : root.at("collections").items()) {
        checkCancelled(cancel);
        Collection collection;
        auto& object = collection.object;
        object.uuid = sourceUuid(data);
        if (!uuids.insert(object.uuid).second) throw std::runtime_error("point-collection-patch: duplicate collection UUID");
        object.kind = "point_collection";
        object.name = data.value("name", "");
        object.formatVersion = 1;
        object.storageKind = "file";
        object.storageRef = storageRef(db, path);
        object.dataGeneration = 1;
        const auto& points = data.at("points");
        if (!points.is_object()) throw std::runtime_error("point-collection-patch: points must be an object");
        Json metadata{{"source_collection_id", id}, {"point_count", points.size()},
            {"metadata", data.value("metadata", Json::object())}, {"color", data.value("color", Json::array())},
            {"tags", data.value("tags", Json::object())}};
        object.meta = Json{{"point_collection", metadata}}.dump();
        std::set<int64_t> ids;
        for (const auto& [key, point] : points.items()) {
            checkCancelled(cancel);
            Point parsed;
            parsed.id = pointId(key);
            if (!ids.insert(parsed.id).second) throw std::runtime_error("point-collection-patch: duplicate numeric point ID");
            const auto& xyz = point.at("p");
            if (!xyz.is_array() || xyz.size() != 3)
                throw std::runtime_error("point-collection-patch: p must contain exactly three XYZ coordinates");
            parsed.source = xyz.get<std::array<double, 3>>();
            for (int axis = 0; axis < 3; ++axis) {
                parsed.working[axis] = float(parsed.source[axis] * params.pointScale);
                if (!std::isfinite(parsed.source[axis]) || !std::isfinite(parsed.working[axis]))
                    throw std::runtime_error("point-collection-patch: non-finite point coordinate");
            }
            parsed.winding = point.value("wind_a", Json(nullptr));
            if (!parsed.winding.is_null() && !parsed.winding.is_number())
                throw std::runtime_error("point-collection-patch: wind_a must be numeric or null");
            collection.points.push_back(std::move(parsed));
        }
        std::sort(collection.points.begin(), collection.points.end(), [](const Point& a, const Point& b) { return a.id < b.id; });
        result.push_back(std::move(collection));
    }
    return result;
}

std::vector<Patch> loadPatches(const objects::ObjectStore& db, const fs::path& directory,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory))
        if (entry.is_directory() && !entry.path().filename().string().starts_with('.') && fs::exists(entry.path() / "meta.json"))
            paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    std::vector<Patch> patches;
    std::set<std::string> uuids;
    for (const auto& path : paths) {
        checkCancelled(cancel);
        util::reportProgress(params.progress, "loading PCL target patches", patches.size(), paths.size(), path.filename().string());
        Patch patch;
        auto& object = patch.object;
        object.uuid = sourceUuid(readJson(path / "meta.json"));
        if (!uuids.insert(object.uuid).second) throw std::runtime_error("point-collection-patch: duplicate patch UUID");
        object.kind = "patch";
        object.name = path.filename().string();
        object.storageKind = "file";
        object.storageRef = storageRef(db, path);
        object.dataGeneration = 1;
        QuadSurface original(path);
        original.ensureLoaded();
        const auto scale = original.scale();
        if (!(scale[0] > 0 && scale[1] > 0) || !std::isfinite(scale[0]) || !std::isfinite(scale[1]))
            throw std::runtime_error("point-collection-patch: invalid patch scale");
        const auto& source = *original.rawPointsPtr();
        objects::PatchDetail detail;
        detail.gridWidth = source.cols; detail.gridHeight = source.rows;
        detail.scale = {scale[0], scale[1]};
        const auto bounds = original.bbox();
        detail.bbox = std::array<double, 6>{bounds.low[0], bounds.low[1], bounds.low[2], bounds.high[0], bounds.high[1], bounds.high[2]};
        object.patch = detail;
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
                        if (!std::isfinite(working(row, col)[axis])) throw std::runtime_error("point-collection-patch: scaled patch coordinate overflow");
                }
            }
        }
        // Same grid-area convention as tifxyz.Patch.area, not 3D triangle area.
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

DetectorReport linkPointCollectionsToPatches(objects::ObjectStore& db, const fs::path& file, const fs::path& patchesDir,
    const PointCollectionPatchParams& params, const std::atomic_bool* cancel)
{
    if (!(params.tolerance > 0) || !std::isfinite(params.tolerance) || params.tolerance > std::numeric_limits<float>::max() ||
        !(params.pointScale > 0) || !std::isfinite(params.pointScale) ||
        !(params.patchScale > 0) || !std::isfinite(params.patchScale) || params.edgeKind.empty() ||
        params.windowPoints < 1 || params.windowMinPoints < 1 || params.windowMinPoints > params.windowPoints)
        throw std::invalid_argument("point-collection-patch: invalid tolerance, scale, window or edge kind");
    checkCancelled(cancel);
    util::reportProgress(params.progress, "loading point collections");
    auto collections = loadCollections(db, file, params, cancel);
    auto patches = loadPatches(db, patchesDir, params, cancel);
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
    const bool absolute = params.absoluteWinding.value_or(file.filename() == "abs_winding.json");
    std::vector<LinkCandidate> candidates;
    for (size_t ci = 0; ci < collections.size(); ++ci) {
        checkCancelled(cancel);
        const auto& collection = collections[ci];
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
    auto registerObject = [&](const objects::ObjectRecord& object) {
        checkCancelled(cancel);
        if (auto existing = db.objectByUuid(txn, object.uuid)) {
            if (existing->kind != object.kind) throw std::runtime_error("point-collection-patch: object UUID kind conflict");
        } else db.upsertObject(txn, object);
    };
    for (const auto& patch : patches) registerObject(patch.object);
    for (const auto& collection : collections) registerObject(collection.object);
    auto report = commitCandidates(db, txn, kPointCollectionPatchDetector, 1, candidates,
        params.replaceStalePending, &scope, params.progress);
    checkCancelled(cancel);
    util::reportProgress(params.progress, "committing point-collection links");
    checkCancelled(cancel);
    txn.commit();
    report.objectsConsidered = int64_t(collections.size() + patches.size());
    return report;
}
} // namespace vc::object_links
