#include "vc/core/StoredCollection.hpp"

#include <nlohmann/json.hpp>
#include <vc_object_store/geometry/PointMap.hpp>

namespace vc {

vc::objects::AttachedCollection storedCollection(const nlohmann::json& json)
{
    const auto collection = json;
    vc::objects::AttachedCollection stored;
    stored.uuid = collection.at("uuid").get<std::string>();
    stored.name = collection.value("name", std::string{});
    stored.source = collection;
    for (const auto* key : {"points", "uuid", "name"}) stored.source.erase(key);
    auto decoded=vc::geometry::decodePointMap(collection.at("points"));
    stored.points=std::move(decoded.points);
    stored.annotations=std::move(decoded.annotations);
    return stored;
}

nlohmann::json collectionJson(const vc::objects::AttachedCollection& stored)
{
    auto collection = stored.source;
    collection["uuid"] = stored.uuid;
    collection["name"] = stored.name;
    auto entries = nlohmann::json::object();
    for (size_t i = 0; i < stored.points.ids.size(); ++i) {
        auto entry = i < stored.annotations.size() && stored.annotations[i].is_object() ? stored.annotations[i]
                                                                                         : nlohmann::json::object();
        entry["p"] = stored.points.points[i];
        entries[std::to_string(stored.points.ids[i])] = std::move(entry);
    }
    collection["points"] = std::move(entries);
    return collection;
}

} // namespace vc
