#include "formats/GeometryInput.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
// The store's umbilicus document: control points as {x, y, z} objects in the
// coordinates it declares, which Umbilicus::LoadFileInfo also reads.
Prepared umbilicus(const fs::path& source)
{
    const auto root = readJson(source);
    if (root.value("vc_umbilicus_json_version", "") != "1" || !root.contains("points") || !root.at("points").is_array())
        throw std::runtime_error("expected vc_umbilicus JSON version 1");
    Prepared p;
    p.object = sourceObject(root, "umbilicus", source);
    vc::geometry::Umbilicus geometry;
    for (const auto& entry : root.at("points")) {
        if (!entry.is_object() || !entry.contains("x") || !entry.contains("y") || !entry.contains("z"))
            throw std::runtime_error("umbilicus points must be {x, y, z} objects");
        geometry.points.push_back({entry.at("x").get<double>(), entry.at("y").get<double>(), entry.at("z").get<double>()});
    }
    auto attributes = root;
    for (const auto* key : {"vc_umbilicus_json_version", "points", "uuid", "name", "tags", "version"}) attributes.erase(key);
    const auto coordinates = sourceGeometryCoordinates(root);
    eraseSourceGeometryCoordinates(attributes);
    attributes["coordinates"] = coordinates;
    attributes["origin"] = p.object.storageRef;
    p.geometry = vc::geometry::encode(geometry);
    p.source = std::move(attributes);
    return p;
}
}
