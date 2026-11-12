#include "linking/StoredGeometry.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/ArrayLayout.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <climits>
#include <set>

namespace vc::object_links::detail {
void validateCoordinateParams(const CoordinateParams& params, double explicitScale)
{
    if (!std::isfinite(explicitScale) || explicitScale <= 0)
        throw std::invalid_argument("coordinate scale must be finite and positive");
    if (params.volumeUuid) {
        if (params.volumeUuid->empty()) throw std::invalid_argument("volume_uuid must be nonempty");
        if (explicitScale != 1.0)
            throw std::invalid_argument("explicit scales must be identity when volume_uuid is supplied");
    }
}
nlohmann::json coordinateProvenance(const objects::ObjectRecord& object, const CoordinateParams& params)
{
    if (!params.volumeUuid) return nlohmann::json::object();
    return {{"coordinates", objects::resolveGeometryCoordinates(nlohmann::json::parse(object.meta), *params.volumeUuid)}};
}
void CoordinateScales::check(const CoordinateParams& params) const
{
    if (!failures.empty())
        throw std::invalid_argument("objects without a coordinate frame in volume " + params.volumeUuid.value_or("") + ": " + failures);
}
CoordinateScales resolveCoordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale)
{
    validateCoordinateParams(params, explicitScale);
    CoordinateScales result;
    for (const auto& object : objects) {
        try {
            result.scales[object.uuid] = params.volumeUuid
                ? vc::objects::geometryScaleToFrame(coordinateProvenance(object, params), *params.volumeUuid)
                : explicitScale;
        } catch (const std::exception& e) {
            result.failures += (result.failures.empty() ? "" : "; ") + object.uuid + ": " + e.what();
        }
    }
    return result;
}
void appendCoordinateScales(CoordinateScales& into, const CoordinateScales& more)
{
    into.scales.insert(more.scales.begin(), more.scales.end());
    if (!more.failures.empty()) into.failures += (into.failures.empty() ? "" : "; ") + more.failures;
}
std::map<std::string, double> coordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale)
{
    const auto resolved = resolveCoordinateScales(objects, params, explicitScale);
    resolved.check(params);
    return resolved.scales;
}
std::shared_ptr<SurfaceGrid> scaledSurfaceGrid(const std::shared_ptr<SurfaceGrid>& source, double scale,
    bool scaleParameterization)
{
    if (scale == 1.0) return source;
    auto points = source->rawPointsPtr()->clone();
    for (auto& point : points) {
        if (point[0] == -1.0f) continue; // preserve invalid vertices verbatim
        const cv::Vec3d scaled = cv::Vec3d(point) * scale;
        point = cv::Vec3f(scaled);
        for (float value : point.val)
            if (!std::isfinite(value)) throw std::invalid_argument("scaled surface coordinate overflow");
        if (point[0] == -1.0f) throw std::invalid_argument("scaled surface coordinate collides with invalid sentinel");
    }
    auto gridScale = source->scale();
    if (scaleParameterization) {
        for (auto& value : gridScale.val) {
            value = float(double(value) / scale);
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("scaled surface grid spacing is not representable");
        }
    }
    return std::make_shared<SurfaceGrid>(points, gridScale);
}
const objects::ReadTxn& StoredGeometry::txn() const
{
    if (!read_) throw std::logic_error("object links: the stored geometry snapshot was closed");
    return *read_;
}
std::filesystem::path StoredGeometry::sourcePath(const objects::ObjectRecord& object) const
{
    const auto path = db_.resolveStoragePath(txn(), object.uuid);
    if (!path) throw std::invalid_argument("object has no external source: " + object.uuid);
    return std::filesystem::weakly_canonical(*path);
}
std::vector<objects::ObjectRecord> StoredGeometry::select(const std::string& kind,
    const std::vector<std::string>& uuids)
{
    std::set<std::string> requested;
    for (const auto& uuid : uuids) requested.insert(objects::util::canonicalUuid(uuid));
    objects::ObjectSelection selection;
    selection.kind = kind;
    selection.uuids.assign(requested.begin(), requested.end());
    auto objects = db_.selectObjects(txn(), selection);
    if (!requested.empty() && objects.size() != requested.size())
        throw std::invalid_argument("requested objects are missing or not of kind " + kind);
    for (const auto& object : objects) {
        if (!object.version) throw std::runtime_error("object " + object.uuid + " has no numbered version");
        versions_[object.uuid] = object.reference();
    }
    return objects;
}
objects::RecordSelection StoredGeometry::expectations() const
{
    objects::RecordSelection result;
    for (const auto& [uuid, version] : versions_) result.emplace(uuid, version);
    return result;
}
namespace {
// The arrays a detector reads of `object`'s geometry at `revision` (its
// current one, or the one `named`), through `readArray`; `checked` sees the
// revision once it is accepted, before its arrays are read.
template <class ReadArray, class Checked>
objects::detail::GeometrySnapshot snapshotOf(const objects::ObjectRecord& object,
    std::optional<objects::GeometryRevision> revision, const std::optional<std::string>& named,
    const ReadArray& readArray, const Checked& checked)
{
    if (named && (!revision || revision->objectUuid != object.uuid))
        throw std::runtime_error("object " + object.uuid + " has no geometry revision " + *named + " in this store");
    if (!revision) throw std::runtime_error("object " + object.uuid + " has no stored geometry; ingest it before discovery");
    if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz" || revision->kind != object.kind)
        throw std::runtime_error("unsupported stored geometry for " + object.uuid);
    checked(*revision);
    objects::detail::GeometrySnapshot result;
    result.revision = *revision;
    const std::vector<std::string> names = object.kind == "fiber" ? std::vector<std::string>{"control_points", "line_points"}
        : object.kind == "point_collection" ? std::vector<std::string>{"points", "point_ids"} : std::vector<std::string>{"points"};
    for (const auto& name : names) result.arrays[name] = readArray(revision->uuid, name);
    if (object.kind == "surface")
        for (const auto& array : revision->arrays)
            if (array.name == objects::detail::surfaceValidChannel)
                result.arrays[array.name] = readArray(revision->uuid, array.name);
    return result;
}
}  // namespace
objects::detail::GeometrySnapshot StoredGeometry::load(const objects::ObjectRecord& object,
    const std::optional<std::string>& named)
{
    return snapshotOf(object, named ? db_.geometryRevision(txn(), *named) : db_.currentGeometry(txn(), object.uuid),
        named, [&](const std::string& revision, const std::string& name) { return db_.readGeometryArray(txn(), revision, name); },
        [&](const objects::GeometryRevision& revision) { revisions_[object.uuid] = revision.uuid; });
}
objects::detail::GeometrySnapshot loadGeometrySnapshot(const objects::StoreReader& reader,
    const objects::ObjectRecord& object, const std::optional<std::string>& named)
{
    return snapshotOf(object, named ? reader.geometryRevision(*named) : reader.currentGeometry(object.uuid), named,
        [&](const std::string& revision, const std::string& name) { return reader.readGeometryArray(revision, name); },
        [](const objects::GeometryRevision&) {});
}
void StoredGeometry::validate(const objects::WriteTxn& txn) const
{
    for (const auto& [uuid, version] : versions_) {
        const auto object = db_.objectByUuid(txn, uuid);
        if (!object || objects::RecordExpectation(object->reference()) != version)
            throw std::runtime_error("object changed during discovery; retry: " + uuid);
    }
    for (const auto& [uuid, revision] : revisions_) {
        const auto current = db_.currentGeometry(txn, uuid);
        if (!current || current->uuid != revision)
            throw std::runtime_error("geometry changed during discovery; retry: " + uuid);
    }
}
std::optional<Box> StoredGeometry::workingBounds(const objects::ObjectRecord& object, const CoordinateParams& params,
    double explicitScale)
{
    objects::ObjectSelection selection;
    selection.kind = object.kind;
    selection.uuids = {object.uuid};
    selection.bbox = Box{-1e30, -1e30, -1e30, 1e30, 1e30, 1e30};
    if (params.volumeUuid) selection.volume = *params.volumeUuid;
    std::optional<Box> result;
    for (const auto& region : db_.spatialRegions(txn(), selection)) {
        if (!result) { result = region.bbox; continue; }
        for (int axis = 0; axis < 3; ++axis) {
            (*result)[axis] = std::min((*result)[axis], region.bbox[axis]);
            (*result)[axis + 3] = std::max((*result)[axis + 3], region.bbox[axis + 3]);
        }
    }
    if (result && !params.volumeUuid)
        for (auto& value : *result) value *= explicitScale;
    return result;
}
std::set<std::string> StoredGeometry::objectsNear(const std::string& kind, const Box& box, double margin,
    const CoordinateParams& params, double explicitScale)
{
    objects::ObjectSelection selection;
    selection.kind = kind;
    Box query = box;
    for (int axis = 0; axis < 3; ++axis) { query[axis] -= margin; query[axis + 3] += margin; }
    if (params.volumeUuid) selection.volume = *params.volumeUuid;
    else for (auto& value : query) value /= explicitScale;
    selection.bbox = query;
    std::set<std::string> result;
    for (const auto& object : db_.selectObjects(txn(), selection, std::vector<std::string>{})) result.insert(object.uuid);
    return result;
}
std::optional<std::set<std::string>> newAmong(const std::optional<std::vector<std::string>>& newUuids,
    const std::vector<const std::vector<objects::ObjectRecord>*>& selected, const char* detector)
{
    if (!newUuids) return std::nullopt;
    std::set<std::string> all, result;
    for (const auto* objects : selected)
        for (const auto& object : *objects) all.insert(object.uuid);
    for (const auto& uuid : *newUuids) {
        const auto canonical = objects::util::canonicalUuid(uuid);
        if (!all.contains(canonical))
            throw std::invalid_argument(std::string(detector) + ": new object " + canonical + " is not selected");
        result.insert(canonical);
    }
    return result;
}
StaleEdgeScope incrementalScope(const std::string& edgeKind, const std::vector<std::string>& selected,
    const std::set<std::string>& fresh, const std::set<std::string>& failed)
{
    StaleEdgeScope scope{{edgeKind}, {}, {}};
    for (const auto& uuid : selected)
        if (!failed.contains(uuid)) scope.objectUuids.push_back(uuid);
    for (const auto& uuid : fresh)
        if (!failed.contains(uuid)) scope.touching.push_back(uuid);
    if (scope.touching.empty()) scope.objectUuids.clear();
    return scope;
}
std::shared_ptr<SurfaceGrid> surfaceGrid(const objects::detail::GeometrySnapshot& geometry)
{
    const auto descriptor = std::find_if(geometry.revision.arrays.begin(), geometry.revision.arrays.end(),
        [](const auto& a) { return a.name == "points"; });
    if (descriptor == geometry.revision.arrays.end() || descriptor->shape.size() != 3 ||
        descriptor->shape[0] < 2 || descriptor->shape[1] < 2 ||
        descriptor->shape[0] > INT_MAX || descriptor->shape[1] > INT_MAX)
        throw std::runtime_error("surface discovery requires a grid of at least 2 by 2 points");
    cv::Mat_<cv::Vec3f> points(int(descriptor->shape[0]), int(descriptor->shape[1]));
    geometry.copyValues<float>("points", {reinterpret_cast<float*>(points.data), points.total() * 3});
    for (const auto& point : points)
        for (float value : point.val)
            if (!std::isfinite(value)) throw std::runtime_error("geometry requires finite coordinates");
    // Discovery sees vertices marked invalid as holes, like missing coordinates.
    if (const auto valid = geometry.arrays.find(objects::detail::surfaceValidChannel); valid != geometry.arrays.end()) {
        if (valid->second.size() != points.total()) throw std::runtime_error("surface validity does not match points");
        for (size_t i = 0; i < points.total(); ++i)
            if (!valid->second[i]) points(int(i / size_t(points.cols)), int(i % size_t(points.cols))) = {-1, -1, -1};
    }
    const auto scale = nlohmann::json::parse(geometry.revision.attributes).at("scale").get<std::array<float, 2>>();
    if (!(scale[0] > 0 && scale[1] > 0) || !std::isfinite(scale[0]) || !std::isfinite(scale[1]))
        throw std::runtime_error("invalid stored surface scale");
    return std::make_shared<SurfaceGrid>(points, cv::Vec2f(scale[0], scale[1]));
}
} // namespace vc::object_links::detail
