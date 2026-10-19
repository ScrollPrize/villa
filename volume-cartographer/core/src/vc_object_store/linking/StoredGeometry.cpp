#include "linking/StoredGeometry.hpp"
#include "vc/core/util/Uuid.hpp"
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
    for (auto uuid : uuids) {
        if (!util::isUuid(uuid)) throw std::invalid_argument("invalid object UUID: " + uuid);
        std::transform(uuid.begin(), uuid.end(), uuid.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        requested.insert(std::move(uuid));
    }
    auto objects = db_.objectsByKind(read_, kind);
    if (!requested.empty()) {
        std::erase_if(objects, [&](const auto& o) { return !requested.contains(o.uuid); });
        if (objects.size() != requested.size())
            throw std::invalid_argument("requested objects are missing or not of kind " + kind);
    }
    std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a.uuid < b.uuid; });
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
std::shared_ptr<QuadSurface> patchSurface(const objects::detail::GeometrySnapshot& geometry)
{
    const auto descriptor = std::find_if(geometry.revision.arrays.begin(), geometry.revision.arrays.end(),
        [](const auto& a) { return a.name == "points"; });
    if (descriptor == geometry.revision.arrays.end() || descriptor->shape.size() != 3 ||
        descriptor->shape[0] < 2 || descriptor->shape[1] < 2 ||
        descriptor->shape[0] > INT_MAX || descriptor->shape[1] > INT_MAX)
        throw std::runtime_error("patch discovery requires a grid of at least 2 by 2 points");
    const auto xyz = geometry.values<float>("points");
    cv::Mat_<cv::Vec3f> points(int(descriptor->shape[0]), int(descriptor->shape[1]));
    for (size_t i = 0; i < points.total(); ++i) points.ptr<cv::Vec3f>()[i] = {xyz[3*i], xyz[3*i+1], xyz[3*i+2]};
    const auto scale = nlohmann::json::parse(geometry.revision.attributes).at("scale").get<std::array<float, 2>>();
    if (!(scale[0] > 0 && scale[1] > 0) || !std::isfinite(scale[0]) || !std::isfinite(scale[1]))
        throw std::runtime_error("invalid stored patch scale");
    return std::make_shared<QuadSurface>(points, cv::Vec2f(scale[0], scale[1]));
}
} // namespace vc::object_links::detail
