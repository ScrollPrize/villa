#include "formats/GeometryInput.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
namespace vc::objects::formats {
Prepared fiber(const std::filesystem::path& source)
{
    const auto root = readJson(source);
    auto decoded = decodeFiberJson(root.dump());
    Prepared p;
    p.object = sourceObject(root, "fiber", source);
    p.object.fiber = decoded.object.fiber;
    p.geometry = std::move(decoded.geometry);
    p.source = nlohmann::json::parse(decoded.source);
    return p;
}
}
