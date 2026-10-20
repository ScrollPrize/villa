#include "formats/GeometryInput.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "internal/GeometryCoordinates.hpp"
#include <fstream>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Json readJson(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return Json::parse(input);
}
std::string uuid(const Json& data)
{
    return util::canonicalUuid(data.at("uuid").get<std::string>());
}
ObjectRecord sourceObject(const Json& root, const std::string& kind, const fs::path& source)
{
    ObjectRecord result;
    result.uuid = uuid(root); result.kind = kind; result.name = source.stem().string();
    result.storageKind = "file"; result.storageRef = fs::absolute(source).lexically_normal().string();
    result.formatVersion = root.value("version", int64_t(1));
    result.name = root.value("name", result.name);
    if (root.contains("tags") && root.at("tags").is_array())
        for (const auto& tag : root.at("tags")) result.tags.push_back(tag.get<std::string>());
    if (kind == "point_collection" && root.contains("tags") && root.at("tags").is_object())
        for (const auto& [name, value] : root.at("tags").items()) result.tags.push_back(name);
    return result;
}
}

namespace vc::objects {
std::string readGeometrySourceCoordinates(const std::string& kind, const std::filesystem::path& source,
    const std::string& objectUuid, const std::string& coordinateContext)
{
    if (kind != "point_collection" && kind != "fiber" && kind != "patch")
        throw std::invalid_argument("coordinate source must be a point_collection, fiber or patch");
    const auto root = formats::readJson(kind == "patch" ? source / "meta.json" : source);
    const auto uuid = util::canonicalUuid(objectUuid);
    auto coordinates = detail::mergeGeometryCoordinates(
        detail::sourceGeometryCoordinates(root), nlohmann::json::parse(coordinateContext));
    if (kind != "point_collection") {
        if (kind == "fiber" && root.value("type", "") != "vc3d_fiber")
            throw std::runtime_error("expected vc3d_fiber JSON");
        if (util::canonicalUuid(root.at("uuid").get<std::string>()) != uuid)
            throw std::runtime_error("source UUID does not match object");
        return coordinates.dump();
    }
    if (root.value("vc_pointcollections_json_version", "") != "1" || !root.at("collections").is_object())
        throw std::runtime_error("expected vc_pointcollections JSON version 1");
    std::optional<nlohmann::json> found;
    for (const auto& [id, collection] : root.at("collections").items()) {
        if (util::canonicalUuid(collection.at("uuid").get<std::string>()) != uuid) continue;
        if (found) throw std::runtime_error("duplicate collection UUID");
        found = detail::mergeGeometryCoordinates(coordinates, detail::sourceGeometryCoordinates(collection));
    }
    if (!found) throw std::runtime_error("collection UUID not found in source");
    return found->dump();
}
}
