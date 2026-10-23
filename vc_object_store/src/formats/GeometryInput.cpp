#include "formats/GeometryInput.hpp"
#include "vc_object_store/Uuid.hpp"
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
