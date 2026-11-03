#include "linking/StoredGeometry.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/ArrayLayout.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <climits>
#include <set>

namespace vc::object_links::detail {
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
    if (object.kind == "surface")
        for (const auto& array : revision->arrays)
            if (array.name == objects::detail::surfaceValidChannel)
                result.arrays[array.name] = db_.readGeometryArray(read_, revision->uuid, array.name);
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
