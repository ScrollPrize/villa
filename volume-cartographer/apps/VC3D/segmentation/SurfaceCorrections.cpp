#include "segmentation/SurfaceCorrections.hpp"

#include <vc_object_store/objects/AttachedDocument.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include "vc/core/types/Volume.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
// Corrections are data attached to the surface, never graph links.
constexpr const char* kRole = "corrections";

// A VCCollection as an attached collection: its points, one annotation per
// point (the point's other fields), and the rest of it as meta.source.
AttachedCollection toStore(uint64_t id, const Json& collection)
{
    AttachedCollection stored;
    stored.uuid = collection.at("uuid").get<std::string>();
    stored.name = collection.value("name", std::string{});
    stored.source = collection;
    for (const auto* key : {"points", "uuid", "name"}) stored.source.erase(key);
    stored.source["source_collection_id"] = std::to_string(id);
    std::map<int64_t, Json> ordered;
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
}

std::map<uint64_t, utils::Json> vc3d::corrections::read(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                        const std::string& surfaceUuid, const std::optional<vc::objects::ViewFrame>& view)
{
    std::map<uint64_t, utils::Json> result;
    for (const auto& stored : attachedCollections(session, attachment, surfaceUuid, kRole, view)) {
        auto collection = stored.source;
        const auto id = std::stoull(collection.value("source_collection_id", std::string("0")));
        collection.erase("source_collection_id");
        collection["uuid"] = stored.uuid;
        collection["name"] = stored.name;
        auto entries = Json::object();
        for (size_t i = 0; i < stored.points.ids.size(); ++i) {
            auto entry = i < stored.annotations.size() && stored.annotations[i].is_object() ? stored.annotations[i] : Json::object();
            entry["p"] = stored.points.points[i];
            entries[std::to_string(stored.points.ids[i])] = std::move(entry);
        }
        collection["points"] = std::move(entries);
        result[id] = utils::Json::parse(collection.dump());
    }
    return result;
}

void vc3d::corrections::write(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& surfaceUuid,
                              const std::map<uint64_t, utils::Json>& collections, const Volume& volume,
                              const std::optional<vc::objects::ViewFrame>& view)
{
    std::vector<AttachedCollection> stored;
    for (const auto& [id, json] : collections) stored.push_back(toStore(id, Json::parse(json.dump())));
    replaceAttachedCollections(session, attachment, surfaceUuid, kRole, stored,
                               vc::objects::volumeCoordinates(volume.id(), 1), view);
}

std::string vc3d::corrections::writeGrowthAnnotation(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                     const GrowthAnnotation& annotation, const Volume& volume)
{
    auto points = annotation.points;
    std::stable_sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.creationTime < b.creationTime; });
    vc::geometry::PointCollection geometry;
    auto pointAttributes = Json::array();
    for (const auto& point : points) {
        geometry.ids.push_back(int64_t(point.pointId));
        geometry.points.push_back(point.position);
        pointAttributes.push_back({{"collection_id", point.collectionId}, {"collection_name", point.collectionName},
                                   {"creation_time", point.creationTime}});
    }
    Json meta{{"coordinates", vc::objects::volumeCoordinates(volume.id(), 1)}};
    meta["role"] = "growth_annotation";
    meta["source"] = {{"growth_annotation", {{"surface", annotation.surfaceUuid},
                                             {"before_geometry", annotation.beforeGeometry},
                                             {"after_geometry", annotation.afterGeometry},
                                             {"volume_used", annotation.volumeUsed},
                                             {"grid_region", annotation.gridRegion},
                                             {"bounds", {{"min", annotation.boundsMin}, {"max", annotation.boundsMax}}}}},
                      {"point_attributes", std::move(pointAttributes)}};
    const auto uuid = util::newUuid();
    std::optional<std::string> surface;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        if (const auto head = db.currentObjectRevision(txn, annotation.surfaceUuid)) surface = head->uuid;
    });
    if (!surface) throw std::runtime_error("surface not found: " + annotation.surfaceUuid);
    session.act(attachment, {{uuid, std::nullopt}, {annotation.surfaceUuid, surface}}, [&](auto& db, auto& txn) {
        ObjectRecord record;
        record.uuid = uuid;
        record.kind = "point_collection";
        record.name = "growth corrections";
        record.tags = {"growth_annotation"};
        record.meta = meta.dump();
        db.upsertObject(txn, record);
        db.putGeometry(txn, uuid, vc::geometry::encode(geometry));
        db.attach(txn, uuid, annotation.surfaceUuid, "growth_annotation");
    });
    return uuid;
}
