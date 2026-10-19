#include "geometry/SourceMetadata.hpp"
#include "geometry/GeometrySnapshot.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
Json nativeMetadata(ObjectStore& db, const Txn& txn, const ObjectRecord& object, const GeometryRevision& revision)
{
    const auto metadata = Json::parse(object.meta);
    auto result = metadata.value("source", Json::object());
    if (object.kind == "patch") result["scale"] = Json::parse(revision.attributes).at("scale");
    result.erase("origin");
    result["name"] = object.name;
    if (object.kind == "point_collection" && metadata.contains("source") && result.contains("tags") && result.at("tags").is_object()) {
        auto tags = Json::object();
        for (const auto& tag : object.tags) tags[tag] = result.at("tags").value(tag, Json(""));
        result["tags"] = std::move(tags);
    } else if (!object.tags.empty() || (result.contains("tags") && result.at("tags").is_array())) result["tags"] = object.tags;
    auto requireCurrentBinding = [&](const EdgeRecord& edge) {
        const auto state = db.edgeGeometryState(txn, edge.uuid);
        const auto binding = edge.aUuid == object.uuid ? state.aRevision : state.bRevision;
        if (state.stale || binding != revision.uuid)
            throw std::runtime_error("cannot export topology against a different or stale geometry revision; revalidate first");
    };
    if (object.kind == "fiber") {
        result["version"] = object.formatVersion > 0 ? object.formatVersion : result.value("version", int64_t(1));
        if (object.fiber) {
            const auto& f = *object.fiber;
            result["width"] = f.width; result["width_gap_fraction"] = f.widthGapFraction;
            result["optimization_mode"] = f.optimizationMode;
            result["hv_classification"]["manual_tag"] = f.manualFamily ? f.family : "";
            if (!f.manualFamily) result["hv_classification"]["automatic_tag"] = f.family;
        }
        result["branches"] = Json::array(); result["adjacent_branches"] = Json::array();
        for (const auto& edge : db.edgesOf(txn, object.uuid)) {
            if (edge.kind != "fiber_branch") continue;
            requireCurrentBinding(edge);
            const bool forward = edge.aUuid == object.uuid;
            const auto& a = forward ? edge.aAnchor : edge.bAnchor;
            const auto& b = forward ? edge.bAnchor : edge.aAnchor;
            const auto target = db.objectByUuid(txn, forward ? edge.bUuid : edge.aUuid);
            if (!a.index || !b.index) throw std::runtime_error("native fiber branch needs control point indices");
            Json branch{{"control_point_index", *a.index}, {"branch_control_point_index", *b.index},
                {"target_uuid", target->uuid}, {"branch_file", std::filesystem::path(target->storageRef).filename().string()},
                {"pending", edge.approvalState != "approved"}};
            if (branch["branch_file"] == "") branch["branch_file"] = target->name + ".json";
            if (a.position) branch["control_point_position"] = *a.position;
            if (b.position) branch["branch_control_point_position"] = *b.position;
            if (a.direction) branch["control_point_direction"] = *a.direction;
            if (b.direction) branch["branch_control_point_direction"] = *b.direction;
            result[Json::parse(edge.meta).value("adjacent", false) ? "adjacent_branches" : "branches"].push_back(std::move(branch));
        }
    } else if (object.kind == "point_collection") {
        GeometrySnapshot snapshot; snapshot.revision = revision;
        snapshot.arrays["point_ids"] = db.readGeometryArray(txn, revision.uuid, "point_ids");
        const auto ids = snapshot.values<int64_t>("point_ids");
        auto& extras = result["point_attributes"];
        if (extras.is_null()) extras = Json(std::vector<Json>(ids.size(), Json::object()));
        if (!extras.is_array() || extras.size() != ids.size()) throw std::runtime_error("point annotations do not match geometry");
        for (auto& point : extras) { if (point.is_null()) point = Json::object(); point["links"] = Json::array(); }
        result["windings_linked"] = Json::array();
        for (const auto& edge : db.edgesOf(txn, object.uuid)) {
            if (edge.kind != "point_link" && edge.kind != "collection_winding") continue;
            requireCurrentBinding(edge);
            if (edge.kind == "point_link") {
                auto link = [&](const std::string& owner, const Anchor& a, const Anchor& b) {
                    if (owner != object.uuid) return;
                    if (!a.index || !b.index) throw std::runtime_error("point link needs point ID anchors");
                    const auto found = std::find(ids.begin(), ids.end(), *a.index);
                    if (found == ids.end()) throw std::runtime_error("point link ID missing from exported geometry");
                    extras[found - ids.begin()]["links"].push_back(*b.index);
                };
                link(edge.aUuid, edge.aAnchor, edge.bAnchor); link(edge.bUuid, edge.bAnchor, edge.aAnchor);
            } else {
                const auto target = db.objectByUuid(txn, edge.aUuid == object.uuid ? edge.bUuid : edge.aUuid);
                const auto targetMeta = Json::parse(target->meta).at("source");
                result["windings_linked"].push_back(std::stoll(targetMeta.at("source_collection_id").get<std::string>()));
            }
        }
        for (auto& point : extras) std::sort(point["links"].begin(), point["links"].end());
        std::sort(result["windings_linked"].begin(), result["windings_linked"].end());
    } else if (object.kind == "track" && object.track) {
        result["external_key"] = object.track->externalKey;
        result["family"] = object.track->family;
    }
    return result;
}
} // namespace vc::objects::detail
