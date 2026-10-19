#include "imports/SourceRelationships.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <map>
#include <set>

namespace vc::objects::detail {
namespace {
using Json = nlohmann::json;
struct Points {
    std::vector<double> xyz;
    std::vector<int64_t> ids;
    Anchor anchor(int64_t index, bool pointId = false) const {
        auto offset = index;
        if (pointId) {
            auto it = std::find(ids.begin(), ids.end(), index);
            if (it == ids.end()) throw std::runtime_error("source link refers to missing point ID");
            offset = it - ids.begin();
        }
        if (offset < 0 || static_cast<size_t>(offset) >= xyz.size() / 3)
            throw std::runtime_error("source branch control point index out of range");
        Anchor result;
        result.index = index;
        result.position = {xyz[3*offset], xyz[3*offset+1], xyz[3*offset+2]};
        return result;
    }
};
}
int64_t importSourceRelationships(ObjectStore& db, WriteTxn& txn, const std::vector<SourceRecord>& sources)
{
    std::map<std::string, Points> points;
    auto load = [&](const std::string& id, const std::string& kind) -> const Points& {
        if (points.contains(id)) return points.at(id);
        auto object = db.objectByUuid(txn, id);
        auto revision = db.currentGeometry(txn, id);
        if (!object || object->kind != kind || !revision)
            throw std::runtime_error("source relationship endpoint needs stored " + kind + " geometry: " + id);
        GeometrySnapshot snapshot; snapshot.revision = *revision;
        const auto name = kind == "fiber" ? "control_points" : "points";
        snapshot.arrays[name] = db.readGeometryArray(txn, revision->uuid, name);
        Points p; p.xyz = snapshot.values<double>(name);
        if (kind == "point_collection") {
            snapshot.arrays["point_ids"] = db.readGeometryArray(txn, revision->uuid, "point_ids");
            p.ids = snapshot.values<int64_t>("point_ids");
        }
        return points.emplace(id, std::move(p)).first->second;
    };
    int64_t inserted = 0;
    auto add = [&](EdgeRecord edge) {
        for (const auto& current : db.edgesOf(txn, edge.aUuid)) {
            if (current.kind != edge.kind) continue;
            const bool forward = current.aUuid == edge.aUuid && current.bUuid == edge.bUuid &&
                current.aAnchor.index == edge.aAnchor.index && current.bAnchor.index == edge.bAnchor.index;
            const bool reverse = current.bUuid == edge.aUuid && current.aUuid == edge.bUuid &&
                current.bAnchor.index == edge.aAnchor.index && current.aAnchor.index == edge.bAnchor.index;
            if (forward || reverse) return; // Existing DB annotations/reviews are authoritative.
        }
        db.addEdge(txn, edge); ++inserted;
    };
    // Resolve source filenames only during ingestion; stored edges always use UUIDs.
    std::map<std::string, std::set<std::string>> fiberNames;
    for (const auto& object : db.objectsByKind(txn, "fiber")) {
        fiberNames[std::filesystem::path(object.storageRef).filename().string()].insert(object.uuid);
        fiberNames[object.name + ".json"].insert(object.uuid);
    }
    for (const auto& entry : sources)
        if (entry.object.kind == "fiber")
            fiberNames[std::filesystem::path(entry.object.storageRef).filename().string()].insert(entry.object.uuid);
    for (const auto& entry : sources) {
        if (entry.object.kind != "fiber") continue;
        for (const auto* key : {"branches", "adjacent_branches"}) {
            for (const auto& branch : entry.source.value(key, Json::array())) {
                auto target = branch.value("target_uuid", std::string());
                if (target.empty()) {
                    const auto file = std::filesystem::path(branch.at("branch_file").get<std::string>()).filename().string();
                    const auto found = fiberNames.find(file);
                    if (found == fiberNames.end() || found->second.size() != 1)
                        throw std::runtime_error("missing or ambiguous branch target: " + file + "; ingest connected fibers together or supply target_uuid");
                    target = *found->second.begin();
                }
                EdgeRecord edge; edge.kind = "fiber_branch";
                edge.aUuid = entry.object.uuid; edge.bUuid = target;
                edge.aAnchor = load(edge.aUuid, "fiber").anchor(branch.at("control_point_index").get<int64_t>());
                edge.bAnchor = load(edge.bUuid, "fiber").anchor(branch.at("branch_control_point_index").get<int64_t>());
                for (auto [anchor, field] : {std::pair{&edge.aAnchor, "control_point_direction"}, {&edge.bAnchor, "branch_control_point_direction"}})
                    if (branch.contains(field)) anchor->direction = branch.at(field).get<std::array<double, 3>>();
                edge.meta = Json{{"source_import", "object-source"}, {"adjacent", std::string(key) == "adjacent_branches"}}.dump();
                add(std::move(edge));
            }
        }
    }
    if (std::none_of(sources.begin(), sources.end(), [](const auto& s) { return s.object.kind == "point_collection"; })) return inserted;
    // Point IDs and collection IDs are scoped to the originating document.
    using Key = std::pair<std::string, std::string>;
    std::map<Key, std::string> collections, pointOwners;
    for (const auto& object : db.objectsByKind(txn, "point_collection")) {
        if (!db.currentGeometry(txn, object.uuid)) continue;
        auto source = Json::parse(object.meta).value("source", Json::object());
        if (!source.contains("source_collection_id")) continue;
        auto origin = source.value("origin", object.storageRef);
        if (!collections.emplace(Key{origin, source.at("source_collection_id").get<std::string>()}, object.uuid).second)
            throw std::runtime_error("duplicate source collection ID");
        for (auto id : load(object.uuid, "point_collection").ids)
            if (!pointOwners.emplace(Key{origin, std::to_string(id)}, object.uuid).second)
                throw std::runtime_error("duplicate point ID in source document");
    }
    for (const auto& entry : sources) {
        if (entry.object.kind != "point_collection") continue;
        const auto origin = entry.source.value("origin", entry.object.storageRef);
        const auto& p = load(entry.object.uuid, "point_collection");
        const auto extras = entry.source.at("point_attributes");
        for (size_t i = 0; i < p.ids.size(); ++i) {
            for (const auto& link : extras.at(i).value("links", Json::array())) {
                const auto id = link.get<int64_t>();
                auto found = pointOwners.find({origin, std::to_string(id)});
                if (found == pointOwners.end()) throw std::runtime_error("point link target not ingested: " + std::to_string(id));
                EdgeRecord edge; edge.kind = "point_link"; edge.aUuid = entry.object.uuid; edge.bUuid = found->second;
                edge.aAnchor = p.anchor(p.ids[i], true);
                edge.bAnchor = load(edge.bUuid, "point_collection").anchor(id, true);
                edge.meta = R"({"source_import":"object-source"})";
                add(std::move(edge));
            }
        }
        for (const auto& link : entry.source.value("windings_linked", Json::array())) {
            const auto id = link.is_string() ? link.get<std::string>() : std::to_string(link.get<int64_t>());
            auto found = collections.find({origin, id});
            if (found == collections.end()) throw std::runtime_error("winding link target not ingested: " + id);
            EdgeRecord edge; edge.kind = "collection_winding"; edge.aUuid = entry.object.uuid; edge.bUuid = found->second;
            edge.meta = R"({"source_import":"object-source"})";
            add(std::move(edge));
        }
    }
    return inserted;
}
} // namespace vc::objects::detail
