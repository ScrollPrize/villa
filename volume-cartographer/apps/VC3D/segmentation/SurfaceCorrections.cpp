#include "segmentation/SurfaceCorrections.hpp"

#include "ObjectStoreCoordinates.hpp"
#include "vc/core/types/Volume.hpp"

#include <vc_object_store/objects/GeometryEdits.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
// Corrections are data attached to the surface, never graph links.
constexpr const char* kRole = "corrections";

// The correction relationships of a surface, by collection UUID.
std::map<std::string, EdgeRecord> correctionEdges(const ObjectStore& db, const Txn& txn, const std::string& surfaceUuid)
{
    std::map<std::string, EdgeRecord> edges;
    for (auto& edge : db.attachments(txn, surfaceUuid, std::string(kRole))) {
        const auto collection = edge.aUuid == surfaceUuid ? edge.bUuid : edge.aUuid;
        edges.emplace(collection, std::move(edge));
    }
    return edges;
}

// A collection in store terms: its points, one annotation per point (the
// point's other fields), and the rest of it as meta.source.
struct Stored {
    vc::geometry::PointCollection points;
    Json annotations = Json::array();
    Json source;
};
Stored toStore(uint64_t id, const Json& collection)
{
    Stored stored;
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
                                                        const std::string& surfaceUuid, const std::optional<vc3d::VolumeView>& view)
{
    std::map<uint64_t, utils::Json> result;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& [uuid, edge] : correctionEdges(db, txn, surfaceUuid)) {
            const auto object = db.objectByUuid(txn, uuid);
            const auto head = object && !object->retired ? db.currentGeometry(txn, uuid) : std::nullopt;
            if (!head) continue;
            auto points = std::get<vc::geometry::PointCollection>(vc::geometry::decode(db.readGeometryData(txn, head->uuid)));
            if (view) {
                const auto shown = GeometrySnapshot(db, txn).viewTransformedAt(uuid, view->volume, view->scale);
                points.points = shown.points("points", 0, int64_t(points.points.size()));
            }
            const auto attributes = Json::parse(db.geometryPointAnnotations(txn, head->uuid, "points"));
            auto collection = Json::parse(object->meta).value("source", Json::object());
            const auto id = std::stoull(collection.value("source_collection_id", std::string("0")));
            collection.erase("source_collection_id");
            collection["uuid"] = object->uuid;
            collection["name"] = object->name;
            auto entries = Json::object();
            for (size_t i = 0; i < points.ids.size(); ++i) {
                auto entry = i < attributes.size() && attributes[i].is_object() ? attributes[i] : Json::object();
                entry["p"] = points.points[i];
                entries[std::to_string(points.ids[i])] = std::move(entry);
            }
            collection["points"] = std::move(entries);
            result[id] = utils::Json::parse(collection.dump());
        }
    });
    return result;
}

void vc3d::corrections::write(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& surfaceUuid,
                              const std::map<uint64_t, utils::Json>& collections, const Volume& volume,
                              const std::optional<vc3d::VolumeView>& view)
{
    std::map<std::string, std::pair<uint64_t, Json>> desired;
    for (const auto& [id, json] : collections) {
        auto collection = Json::parse(json.dump());
        const auto uuid = util::canonicalUuid(collection.at("uuid").get<std::string>());
        desired[uuid] = {id, std::move(collection)};
    }
    std::map<std::string, EdgeRecord> linked;
    RevisionSelection pins, related;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        linked = correctionEdges(db, txn, surfaceUuid);
        const auto surface = db.currentObjectRevision(txn, surfaceUuid);
        if (!surface) throw std::runtime_error("surface not found: " + surfaceUuid);
        related[surfaceUuid] = surface->uuid;
        auto pin = [&](const std::string& uuid) {
            const auto head = db.currentObjectRevision(txn, uuid);
            pins[uuid] = head ? std::optional(head->uuid) : std::nullopt;
        };
        for (const auto& [uuid, edge] : linked) pin(uuid);
        for (const auto& [uuid, collection] : desired) pin(uuid);
    });
    const auto binding = vc3d::coordinateBindingForVolume(volume, 1);
    session.editKeepingRelationships(attachment, pins, related, [&](auto& db, auto& txn, const auto&) {
        for (const auto& [uuid, entry] : desired) {
            const auto& [id, collection] = entry;
            auto stored = toStore(id, collection);
            const auto current = db.objectByUuid(txn, uuid);
            if (current && current->retired) db.restoreObject(txn, uuid);
            auto record = current ? *db.objectByUuid(txn, uuid) : ObjectRecord{};
            auto meta = Json::parse(record.meta);
            if (!current) {
                record.uuid = uuid;
                record.kind = "point_collection";
                for (const auto* key : {"coordinates", "coordinate_binding"}) meta[key] = binding.at(key);
                meta["role"] = kRole;
            }
            record.name = collection.value("name", std::string{});
            const auto head = current ? db.currentGeometry(txn, uuid) : std::nullopt;
            if (!head) {
                // A new geometry consumes the points' annotations from meta.source.
                meta["source"] = stored.source;
                meta["source"]["point_attributes"] = stored.annotations;
                record.meta = meta.dump();
                db.upsertObject(txn, record);
                db.putGeometry(txn, uuid, vc::geometry::encode(stored.points));
            } else {
                meta["source"] = stored.source;
                record.meta = meta.dump();
                db.upsertObject(txn, record);
                if (view) {
                    const auto shown = GeometrySnapshot(db, txn).viewTransformedAt(uuid, view->volume, view->scale);
                    stored.points = shown.canonicalPointCollection(stored.points);
                }
                const auto previous = std::get<vc::geometry::PointCollection>(vc::geometry::decode(db.readGeometryData(txn, head->uuid)));
                const auto previousAnnotations = Json::parse(db.geometryPointAnnotations(txn, head->uuid, "points"));
                if (previous.ids == stored.points.ids && previous.points == stored.points.points &&
                    previousAnnotations == stored.annotations) {
                    // Only the record changed: the correction relationship still applies.
                    for (const auto& edge : db.edgesOf(txn, uuid)) db.retainRelationship(txn, edge.uuid);
                } else {
                    // Points that stay keep their identity (stable IDs); the correction
                    // relationship applies to the edited geometry as before.
                    auto retain = [&](const std::vector<EdgeRecord>& edges) {
                        std::vector<GeometryEdgeEdit> decisions;
                        for (const auto& edge : edges) {
                            GeometryEdgeEdit decision;
                            decision.edgeUuid = edge.uuid;
                            decision.expectedGeneration = edge.generation;
                            decision.action = "retain";
                            if (const auto r = db.currentGeometry(txn, edge.aUuid)) decision.aRevision = r->uuid;
                            if (const auto r = db.currentGeometry(txn, edge.bUuid)) decision.bRevision = r->uuid;
                            decisions.push_back(std::move(decision));
                        }
                        return decisions;
                    };
                    updateGeometryArrays(db, txn, uuid, vc::geometry::encode(stored.points), head->uuid,
                                         {{"points", stored.annotations.dump()}}, retain);
                }
            }
            if (!linked.contains(uuid)) db.attach(txn, uuid, surfaceUuid, kRole);
        }
        for (const auto& [uuid, edge] : linked) {
            if (desired.contains(uuid)) continue;
            db.removeEdge(txn, edge.uuid);
            db.retireObject(txn, uuid);
        }
    });
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
    auto meta = vc3d::coordinateBindingForVolume(volume, 1);
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
