#include "imports/SourceRelationships.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/Dependencies.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include <map>
#include <set>

namespace vc::objects::detail {
namespace {
using Json = nlohmann::json;
struct Points {
    std::vector<double> xyz;
    Anchor anchor(int64_t index) const {
        const auto offset = index;
        if (offset < 0 || static_cast<size_t>(offset) >= xyz.size() / 3)
            throw std::runtime_error("source branch control point index out of range");
        Anchor result;
        result.points.push_back({"control_points", {index}});
        result.position = {xyz[3*offset], xyz[3*offset+1], xyz[3*offset+2]};
        return result;
    }
};
}
int64_t importSourceRelationships(ObjectStore& db, WriteTxn& txn, const std::vector<SourceRecord>& sources)
{
    GeometryPhase phase("importSourceRelationships");
    DependencyScope dependencies;
    auto* connection=ObjectStoreImplAccess::conn(txn);
    refreshSourceIdentityIndex(connection);
    std::map<std::string, Points> points;
    auto load = [&](const std::string& id) -> const Points& {
        if (points.contains(id)) return points.at(id);
        auto object = db.objectByUuid(txn, id);
        auto revision = db.currentGeometry(txn, id);
        if (!object || object->kind != "fiber" || !revision)
            throw std::runtime_error("source relationship endpoint needs stored fiber geometry: " + id);
        GeometrySnapshot snapshot; snapshot.revision = *revision;
        snapshot.arrays["control_points"] = db.readGeometryArray(txn, revision->uuid, "control_points");
        Points p; p.xyz = snapshot.values<double>("control_points");
        return points.emplace(id, std::move(p)).first->second;
    };
    int64_t inserted = 0;
    auto add = [&](EdgeRecord edge) {
        for (const auto& current : db.edgesOf(txn, edge.aUuid)) {
            if (current.kind != edge.kind) continue;
            const bool forward = current.aUuid == edge.aUuid && current.bUuid == edge.bUuid &&
                current.aAnchor.points == edge.aAnchor.points && current.bAnchor.points == edge.bAnchor.points;
            const bool reverse = current.bUuid == edge.aUuid && current.aUuid == edge.bUuid &&
                current.bAnchor.points == edge.aAnchor.points && current.aAnchor.points == edge.bAnchor.points;
            if (forward || reverse) {
                if (edge.kind == "fiber_branch" &&
                    Json::parse(current.meta).value("adjacent", false) != Json::parse(edge.meta).value("adjacent", false))
                    throw std::runtime_error("conflicting ordinary/adjacent fiber branch declarations: " +
                                             edge.aUuid + " <-> " + edge.bUuid);
                return; // Consistent reciprocal declarations represent one edge.
            }
        }
        db.addEdge(txn, edge); ++inserted;
    };
    // Resolve source filenames only during ingestion; stored edges always use UUIDs.
    std::map<std::string, std::set<std::string>> fiberNames;
    if (std::any_of(sources.begin(),sources.end(),[](const auto& s) { return s.object.kind=="fiber"; }))
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
                edge.aAnchor = load(edge.aUuid).anchor(branch.at("control_point_index").get<int64_t>());
                edge.bAnchor = load(edge.bUuid).anchor(branch.at("branch_control_point_index").get<int64_t>());
                for (auto [anchor, field] : {std::pair{&edge.aAnchor, "control_point_direction"}, {&edge.bAnchor, "branch_control_point_direction"}})
                    if (branch.contains(field)) anchor->direction = branch.at(field).get<std::array<double, 3>>();
                edge.meta = Json{{"source_import", "object-source"}, {"adjacent", std::string(key) == "adjacent_branches"}}.dump();
                add(std::move(edge));
            }
        }
    }
    if (std::none_of(sources.begin(), sources.end(), [](const auto& s) { return s.object.kind == "point_collection"; })) return inserted;
    // Validate only the incoming identities. Indexed probes include existing
    // owners in the same document, without reading unrelated geometry.
    for(const auto& entry:sources)
        if(entry.object.kind=="point_collection") validateSourceIdentities(connection,entry.object.uuid);
    auto anchor=[&](const std::string& object,int64_t id) {
        const auto revision=db.currentGeometry(txn,object);
        if(!revision) throw std::runtime_error("source relationship endpoint needs stored point_collection geometry: "+object);
        const PointReference reference{"point_ids",{id}};
        const auto bytes=pointValue(dependencyGeometry(db,txn,*revision),reference);
        if(!bytes) throw std::runtime_error("source link refers to missing point ID");
        GeometrySnapshot snapshot; snapshot.arrays["point"]=*bytes;
        const auto xyz=snapshot.values<double>("point");
        Anchor result; result.points.push_back(reference); result.position={xyz[0],xyz[1],xyz[2]}; return result;
    };
    for (const auto& entry : sources) {
        if (entry.object.kind != "point_collection") continue;
        const auto origin = entry.source.value("origin", entry.object.storageRef);
        const auto& extras = entry.source.at("point_attributes");
        Statement ids(connection,"SELECT point_id FROM source_points WHERE object_uuid=? ORDER BY point_offset");
        ids.bindBlob(1,parseUuid(entry.object.uuid));
        size_t i=0;
        while(ids.row()) {
            const auto point=ids.i64(0);
            const auto& attributes=extras.at(i++);
            for (const auto& link : attributes.value("links", Json::array())) {
                const auto id = link.get<int64_t>();
                const auto found=sourcePoint(connection,origin,id);
                if(!found) throw std::runtime_error("point link target not ingested: " + std::to_string(id));
                EdgeRecord edge; edge.kind = "point_link"; edge.aUuid = entry.object.uuid; edge.bUuid = found->object;
                edge.aAnchor = anchor(edge.aUuid,point);
                edge.bAnchor = anchor(edge.bUuid,id);
                edge.meta = R"({"source_import":"object-source"})";
                add(std::move(edge));
            }
        }
        for (const auto& link : entry.source.value("windings_linked", Json::array())) {
            const auto id = link.is_string() ? link.get<std::string>() : std::to_string(link.get<int64_t>());
            const auto found=sourceCollection(connection,origin,id);
            if(!found) throw std::runtime_error("winding link target not ingested: " + id);
            EdgeRecord edge; edge.kind = "collection_winding"; edge.aUuid = entry.object.uuid; edge.bUuid = *found;
            edge.aAnchor.wholeGeometry = true; edge.bAnchor.wholeGeometry = true;
            edge.meta = R"({"source_import":"object-source"})";
            add(std::move(edge));
        }
    }
    return inserted;
}
} // namespace vc::objects::detail
