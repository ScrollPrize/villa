#include "geometry/SourceMetadata.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/Dependencies.hpp"
#include "formats/FiberMetadata.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
Json nativeMetadata(ObjectStore& db, const Txn& txn, const ObjectRecord& object, const GeometryRevision& revision)
{
    const auto metadata = Json::parse(object.meta);
    auto result = geometryTransferMetadata(metadata);
    if (object.kind == "patch") result["scale"] = Json::parse(revision.attributes).at("scale");
    result.erase("origin");
    GeometryReader structure(txn,revision.uuid);
    if (structure.manifest().contains("sequences"))
        for (const auto& [name,state]:structure.manifest().at("sequences").items())
            result[object.kind=="point_collection" ? "point_attributes" : name+"_attributes"] =
                Json::parse(db.geometryPointAnnotations(txn,revision.uuid,name));
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
    if (object.kind == "fiber") {
        result = fiberMetadata(db, txn, object, revision);
        for (const auto& edge : db.edgesOf(txn, object.uuid)) {
            if (edge.kind != "fiber_branch") continue;
            requireCurrentBinding(edge);
            const bool forward = edge.aUuid == object.uuid;
            const auto target = db.currentGeometry(txn, forward ? edge.bUuid : edge.aUuid);
            if (!target) throw std::runtime_error("native fiber branch target has no geometry");
            const auto key = Json::parse(edge.meta).value("adjacent", false) ? "adjacent_branches" : "branches";
            result[key].push_back(fiberBranchJson(db, txn, edge,
                forward ? edge.aAnchor : edge.bAnchor, forward ? edge.bAnchor : edge.aAnchor, revision, *target));
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
