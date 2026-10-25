#include "formats/GeometryInput.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
#include <charconv>
#include <iterator>
#include <map>
#include <set>
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
namespace {
int64_t pointId(const std::string& key)
{
    int64_t id;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
    if (error != std::errc{} || end != key.data() + key.size() || id < 0)
        throw std::runtime_error("invalid numeric point ID");
    return id;
}
// Collections sharing a point ID with another collection of the document. Point IDs
// are document-scoped identities, so these cannot be ingested as written.
std::set<std::string> collidingCollections(const Json& collections, int64_t& largest)
{
    std::map<int64_t, std::vector<std::string>> owners;
    for (const auto& [collectionId, document] : collections.items()) {
        if (!document.at("points").is_object()) throw std::runtime_error("points must be an object");
        for (const auto& [key, value] : document.at("points").items()) {
            const auto id = pointId(key);
            largest = std::max(largest, id);
            owners[id].push_back(collectionId);
        }
    }
    std::set<std::string> result;
    for (const auto& [id, collections] : owners)
        if (collections.size() > 1) result.insert(collections.begin(), collections.end());
    return result;
}
}
std::vector<Prepared> collections(const fs::path& source, const SourceIngestOptions& options)
{
    const auto root = readJson(source);
    if (root.value("vc_pointcollections_json_version", "") != "1" || !root.at("collections").is_object())
        throw std::runtime_error("expected vc_pointcollections JSON version 1");
    int64_t nextId = -1;
    const auto colliding = collidingCollections(root.at("collections"), nextId);
    ++nextId;
    if (!colliding.empty()) {
        if (!options.renumberCollidingPointIds) {
            std::string named;
            for (auto it = colliding.begin(); it != colliding.end() && std::distance(colliding.begin(), it) < 5; ++it)
                named += (named.empty() ? "" : ", ") + *it;
            throw std::runtime_error(std::to_string(colliding.size()) + " collections (" + named +
                (colliding.size() > 5 ? ", ..." : "") + ") share point IDs; point IDs must be unique within a document");
        }
        for (const auto& [collectionId, document] : root.at("collections").items())
            for (const auto& [key, value] : document.at("points").items())
                if (value.is_object() && value.contains("links"))
                    throw std::runtime_error("cannot renumber point IDs in a document with point links");
    }
    std::vector<Prepared> result;
    const auto documentCoordinates = sourceGeometryCoordinates(root);
    std::set<std::string> seen;
    for (const auto& [collectionId, document] : root.at("collections").items()) {
        auto original = document;
        deriveSourceUuid(original, "point_collection", source.stem().string() + ":" + collectionId, options.uuidNamespace);
        Prepared p;
        p.object = sourceObject(original, "point_collection", source);
        if (!seen.insert(p.object.uuid).second) throw std::runtime_error("duplicate collection UUID");
        p.object.name = original.value("name", ""); p.geometry.kind = "point_collection";
        std::map<int64_t, Json> ordered;
        if (!original.at("points").is_object()) throw std::runtime_error("points must be an object");
        for (const auto& [key, value] : original.at("points").items())
            if (!ordered.emplace(pointId(key), value).second) throw std::runtime_error("duplicate numeric point ID");
        const bool renumber = colliding.contains(collectionId);
        std::vector<int64_t> ids;
        std::vector<std::array<double, 3>> coordinates;
        auto attributes = original; attributes.erase("points");
        const auto localCoordinates = mergeGeometryCoordinates(
            documentCoordinates, sourceGeometryCoordinates(original));
        eraseSourceGeometryCoordinates(attributes);
        attributes["coordinates"] = localCoordinates;
        attributes["source_collection_id"] = collectionId;
        attributes["origin"] = p.object.storageRef;
        if (!attributes.contains("winding_is_absolute") && source.filename() == "abs_winding.json")
            attributes["winding_is_absolute"] = true;
        if (renumber) attributes["point_ids_renumbered"] = true;
        attributes["point_attributes"] = Json::array();
        for (const auto& [id, entry] : ordered) {
            if (!entry.at("p").is_array() || entry.at("p").size() != 3) throw std::runtime_error("invalid XYZ point");
            auto xyz = entry.at("p").get<std::array<double, 3>>();
            for (double value : xyz) if (!std::isfinite(value)) throw std::runtime_error("non-finite point");
            auto extra = entry; extra.erase("p");
            if (renumber) extra["source_point_id"] = id;
            ids.push_back(renumber ? nextId++ : id); coordinates.push_back(xyz);
            attributes["point_attributes"].push_back(std::move(extra));
        }
        p.geometry = vc::geometry::encode(vc::geometry::PointCollection{std::move(ids), std::move(coordinates)});
        p.source = std::move(attributes);
        result.push_back(std::move(p));
    }
    return result;
}
}
