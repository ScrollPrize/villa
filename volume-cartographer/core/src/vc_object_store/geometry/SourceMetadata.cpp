#include "geometry/SourceMetadata.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/Dependencies.hpp"

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
        const auto& anchor = edge.aUuid == object.uuid ? edge.aAnchor : edge.bAnchor;
        if (state.stale || detail::endpointChanged(db, txn, anchor, binding, revision.uuid))
            throw std::runtime_error("cannot export topology against a different or stale geometry revision; update the relationship first");
    };
    auto pointIdentity = [](const Anchor& anchor, const std::string& array) {
        if (anchor.points.empty() || anchor.points.front().array != array || anchor.points.front().index.size() != 1)
            throw std::runtime_error("native relationship requires an explicit " + array + " point reference");
        return anchor.points.front().index.front();
    };
    auto controlPosition = [&](const Anchor& anchor, const GeometryRevision& geometry) {
        const auto bytes = pointValue(dependencyGeometry(db, txn, geometry), anchor.points.front());
        if (!bytes) throw std::runtime_error("native relationship control point is missing");
        GeometrySnapshot snapshot; snapshot.arrays["point"] = *bytes;
        const auto values = snapshot.values<double>("point");
        return std::array<double, 3>{values[0], values[1], values[2]};
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
            const auto aIndex = pointIdentity(a, "control_points"), bIndex = pointIdentity(b, "control_points");
            const auto targetRevision = db.currentGeometry(txn, target->uuid);
            if (!targetRevision) throw std::runtime_error("native fiber branch target has no geometry");
            Json branch{{"control_point_index", aIndex}, {"branch_control_point_index", bIndex},
                {"target_uuid", target->uuid}, {"branch_file", std::filesystem::path(target->storageRef).filename().string()},
                {"pending", edge.approvalState != "approved"}};
            if (branch["branch_file"] == "") branch["branch_file"] = target->name + ".json";
            branch["control_point_position"] = controlPosition(a, revision);
            branch["branch_control_point_position"] = controlPosition(b, *targetRevision);
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
                    const auto aId = pointIdentity(a, "point_ids"), bId = pointIdentity(b, "point_ids");
                    const auto found = std::find(ids.begin(), ids.end(), aId);
                    if (found == ids.end()) throw std::runtime_error("point link ID missing from exported geometry");
                    extras[found - ids.begin()]["links"].push_back(bId);
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
