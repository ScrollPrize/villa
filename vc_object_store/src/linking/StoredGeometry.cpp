#include "linking/StoredGeometry.hpp"
#include "vc_object_store/Uuid.hpp"
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
std::map<std::string, double> coordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale)
{
    validateCoordinateParams(params, explicitScale);
    std::map<std::string, double> result;
    for (const auto& object : objects) {
        try {
            result[object.uuid] = params.volumeUuid
                ? vc::objects::geometryScaleToFrame(coordinateProvenance(object, params), *params.volumeUuid)
                : explicitScale;
        } catch (const std::exception& e) {
            throw std::invalid_argument(object.uuid + ": " + e.what());
        }
    }
    return result;
}
std::shared_ptr<PatchGrid> scaledPatchSurface(const std::shared_ptr<PatchGrid>& source, double scale,
    bool scaleParameterization)
{
    if (scale == 1.0) return source;
    auto points = source->rawPointsPtr()->clone();
    for (auto& point : points) {
        if (point[0] == -1.0f) continue; // preserve invalid vertices verbatim
        const cv::Vec3d scaled = cv::Vec3d(point) * scale;
        point = cv::Vec3f(scaled);
        for (float value : point.val)
            if (!std::isfinite(value)) throw std::invalid_argument("scaled patch coordinate overflow");
        if (point[0] == -1.0f) throw std::invalid_argument("scaled patch coordinate collides with invalid sentinel");
    }
    auto gridScale = source->scale();
    if (scaleParameterization) {
        for (auto& value : gridScale.val) {
            value = float(double(value) / scale);
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("scaled patch grid spacing is not representable");
        }
    }
    return std::make_shared<PatchGrid>(points, gridScale);
}
std::vector<objects::ObjectRecord> StoredGeometry::select(const std::string& kind,
    const std::vector<std::string>& uuids)
{
    std::set<std::string> requested;
    for (const auto& uuid : uuids) requested.insert(objects::util::canonicalUuid(uuid));
    objects::ObjectSelection selection;
    selection.kind = kind;
    selection.uuids.assign(requested.begin(), requested.end());
    auto objects = db_.selectObjects(read_, selection);
    if (!requested.empty() && objects.size() != requested.size())
        throw std::invalid_argument("requested objects are missing or not of kind " + kind);
    for (const auto& object : objects) generations_[object.uuid] = object.generation;
    return objects;
}
objects::detail::GeometrySnapshot StoredGeometry::load(const objects::ObjectRecord& object)
{
    const auto revision = db_.currentGeometry(read_, object.uuid);
    if (!revision) throw std::runtime_error("object " + object.uuid + " has no stored geometry; ingest it before discovery");
    if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz" || revision->kind != object.kind)
        throw std::runtime_error("unsupported stored geometry for " + object.uuid);
    revisions_[object.uuid] = revision->uuid;
    objects::detail::GeometrySnapshot result;
    result.revision = *revision;
    const std::vector<std::string> names = object.kind == "fiber" ? std::vector<std::string>{"control_points", "line_points"}
        : object.kind == "point_collection" ? std::vector<std::string>{"points", "point_ids"} : std::vector<std::string>{"points"};
    for (const auto& name : names) result.arrays[name] = db_.readGeometryArray(read_, revision->uuid, name);
    return result;
}
void StoredGeometry::validate(const objects::WriteTxn& txn) const
{
    for (const auto& [uuid, generation] : generations_) {
        const auto object = db_.objectByUuid(txn, uuid);
        if (!object || object->generation != generation)
            throw std::runtime_error("object changed during discovery; retry: " + uuid);
    }
    for (const auto& [uuid, revision] : revisions_) {
        const auto current = db_.currentGeometry(txn, uuid);
        if (!current || current->uuid != revision)
            throw std::runtime_error("geometry changed during discovery; retry: " + uuid);
    }
}
std::shared_ptr<PatchGrid> patchSurface(const objects::detail::GeometrySnapshot& geometry)
{
    const auto descriptor = std::find_if(geometry.revision.arrays.begin(), geometry.revision.arrays.end(),
        [](const auto& a) { return a.name == "points"; });
    if (descriptor == geometry.revision.arrays.end() || descriptor->shape.size() != 3 ||
        descriptor->shape[0] < 2 || descriptor->shape[1] < 2 ||
        descriptor->shape[0] > INT_MAX || descriptor->shape[1] > INT_MAX)
        throw std::runtime_error("patch discovery requires a grid of at least 2 by 2 points");
    cv::Mat_<cv::Vec3f> points(int(descriptor->shape[0]), int(descriptor->shape[1]));
    geometry.copyValues<float>("points", {reinterpret_cast<float*>(points.data), points.total() * 3});
    for (const auto& point : points)
        for (float value : point.val)
            if (!std::isfinite(value)) throw std::runtime_error("geometry requires finite coordinates");
    const auto scale = nlohmann::json::parse(geometry.revision.attributes).at("scale").get<std::array<float, 2>>();
    if (!(scale[0] > 0 && scale[1] > 0) || !std::isfinite(scale[0]) || !std::isfinite(scale[1]))
        throw std::runtime_error("invalid stored patch scale");
    return std::make_shared<PatchGrid>(points, cv::Vec2f(scale[0], scale[1]));
}
} // namespace vc::object_links::detail
