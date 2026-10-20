#include "formats/GeometryInput.hpp"
#include "geometry/ArrayBytes.hpp"
#include "internal/GeometryCoordinates.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <charconv>
#include <set>
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
std::vector<Prepared> collections(const fs::path& source)
{
    const auto root = readJson(source);
    if (root.value("vc_pointcollections_json_version", "") != "1" || !root.at("collections").is_object())
        throw std::runtime_error("expected vc_pointcollections JSON version 1");
    std::vector<Prepared> result;
    const auto documentCoordinates = detail::sourceGeometryCoordinates(root);
    std::set<std::string> seen;
    for (const auto& [collectionId, original] : root.at("collections").items()) {
        Prepared p;
        p.object = sourceObject(original, "point_collection", source);
        if (!seen.insert(p.object.uuid).second) throw std::runtime_error("duplicate collection UUID");
        p.object.name = original.value("name", ""); p.geometry.kind = "point_collection";
        std::map<int64_t, Json> ordered;
        if (!original.at("points").is_object()) throw std::runtime_error("points must be an object");
        for (const auto& [key, value] : original.at("points").items()) {
            int64_t id;
            const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
            if (error != std::errc{} || end != key.data() + key.size() || id < 0 || !ordered.emplace(id, value).second)
                throw std::runtime_error("invalid or duplicate numeric point ID");
        }
        std::vector<int64_t> ids;
        std::vector<std::array<double, 3>> coordinates;
        auto attributes = original; attributes.erase("points");
        const auto localCoordinates = detail::mergeGeometryCoordinates(
            documentCoordinates, detail::sourceGeometryCoordinates(original));
        detail::eraseSourceGeometryCoordinates(attributes);
        attributes["coordinates"] = localCoordinates;
        attributes["source_collection_id"] = collectionId;
        attributes["origin"] = p.object.storageRef;
        if (!attributes.contains("winding_is_absolute") && source.filename() == "abs_winding.json")
            attributes["winding_is_absolute"] = true;
        attributes["point_attributes"] = Json::array();
        for (const auto& [id, entry] : ordered) {
            if (!entry.at("p").is_array() || entry.at("p").size() != 3) throw std::runtime_error("invalid XYZ point");
            auto xyz = entry.at("p").get<std::array<double, 3>>();
            for (double value : xyz) if (!std::isfinite(value)) throw std::runtime_error("non-finite point");
            ids.push_back(id); coordinates.push_back(xyz);
            auto extra = entry; extra.erase("p"); attributes["point_attributes"].push_back(std::move(extra));
        }
        p.geometry = vc::geometry::encode(vc::geometry::PointCollection{std::move(ids), std::move(coordinates)});
        p.source = std::move(attributes);
        result.push_back(std::move(p));
    }
    return result;
}
}
