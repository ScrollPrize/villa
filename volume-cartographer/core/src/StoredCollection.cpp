#include "vc/core/StoredCollection.hpp"

#include <nlohmann/json.hpp>

#include <map>

namespace vc {

vc::objects::AttachedCollection storedCollection(const utils::Json& json)
{
    const auto collection = nlohmann::json::parse(json.dump());
    vc::objects::AttachedCollection stored;
    stored.uuid = collection.at("uuid").get<std::string>();
    stored.name = collection.value("name", std::string{});
    stored.source = collection;
    for (const auto* key : {"points", "uuid", "name"}) stored.source.erase(key);
    std::map<int64_t, nlohmann::json> ordered;
    for (const auto& [key, value] : collection.at("points").items()) ordered.emplace(std::stoll(key), value);
    for (const auto& [pointId, entry] : ordered) {
        stored.points.ids.push_back(pointId);
        stored.points.points.push_back(entry.at("p").get<std::array<double, 3>>());
        auto extra = entry;
        extra.erase("p");
        stored.annotations.push_back(std::move(extra));
    }
    return stored;
}

utils::Json collectionJson(const vc::objects::AttachedCollection& stored)
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
    return utils::Json::parse(collection.dump());
}

} // namespace vc
