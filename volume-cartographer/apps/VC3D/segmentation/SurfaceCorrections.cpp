#include "segmentation/SurfaceCorrections.hpp"

#include "ObjectStoreCoordinates.hpp"
#include "vc/core/types/Volume.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <nlohmann/json.hpp>
#include <cstring>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
constexpr const char* kEdgeKind = "point-surface";
constexpr const char* kRole = "corrections";

// The correction relationships of a surface, by collection UUID.
std::map<std::string, EdgeRecord> correctionEdges(const ObjectStore& db, const Txn& txn, const std::string& surfaceUuid)
{
    std::map<std::string, EdgeRecord> edges;
    EdgeSelection selection;
    selection.kind = kEdgeKind;
    selection.endpointUuids = std::vector<std::string>{surfaceUuid};
    for (auto& edge : db.selectEdges(txn, selection)) {
        if (Json::parse(edge.meta).value("role", "") != kRole) continue;
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
                                                        const std::string& surfaceUuid)
{
    std::map<uint64_t, utils::Json> result;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& [uuid, edge] : correctionEdges(db, txn, surfaceUuid)) {
            const auto object = db.objectByUuid(txn, uuid);
            const auto head = object && !object->retired ? db.currentGeometry(txn, uuid) : std::nullopt;
            if (!head) continue;
            const auto points = std::get<vc::geometry::PointCollection>(vc::geometry::decode(db.readGeometryData(txn, head->uuid)));
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
                              const std::map<uint64_t, utils::Json>& collections, const Volume& volume)
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
                const auto previous = std::get<vc::geometry::PointCollection>(vc::geometry::decode(db.readGeometryData(txn, head->uuid)));
                const auto previousAnnotations = Json::parse(db.geometryPointAnnotations(txn, head->uuid, "points"));
                if (previous.ids == stored.points.ids && previous.points == stored.points.points &&
                    previousAnnotations == stored.annotations) {
                    // Only the record changed: the correction relationship still applies.
                    for (const auto& edge : db.edgesOf(txn, uuid)) db.retainRelationship(txn, edge.uuid);
                } else {
                    // The new sequence keeps the points that stay (referenced from the previous
                    // revision, so they keep their identity), adds new ones, then moves the kept
                    // points that moved. The correction relationship applies to it as before.
                    std::map<int64_t, int64_t> previousRow;
                    for (size_t i = 0; i < previous.ids.size(); ++i) previousRow[previous.ids[i]] = int64_t(i);
                    GeometrySequenceEdit sequence;
                    sequence.array = "points";
                    sequence.eraseCount = int64_t(previous.ids.size());
                    sequence.transformedAnnotations = stored.annotations.dump();
                    std::vector<std::pair<int64_t, std::array<double, 3>>> moved;
                    for (size_t i = 0; i < stored.points.ids.size(); ++i) {
                        const auto id = stored.points.ids[i];
                        const auto& xyz = stored.points.points[i];
                        const auto found = previousRow.find(id);
                        auto& last = sequence.parts;
                        if (found != previousRow.end()) {
                            if (previous.points[size_t(found->second)] != xyz) moved.emplace_back(int64_t(i), xyz);
                            if (!last.empty() && last.back().revision && last.back().start + last.back().count == found->second) {
                                ++last.back().count;
                                continue;
                            }
                            GeometrySequencePart part;
                            part.revision = head->uuid;
                            part.start = found->second;
                            part.count = 1;
                            last.push_back(std::move(part));
                            continue;
                        }
                        if (last.empty() || last.back().revision) last.emplace_back();
                        auto& part = last.back();
                        ++part.count;
                        part.ids.push_back(id);
                        const auto offset = part.bytes.size();
                        part.bytes.resize(offset + sizeof xyz);
                        std::memcpy(part.bytes.data() + offset, xyz.data(), sizeof xyz);
                    }
                    for (auto& part : sequence.parts)
                        if (!part.revision) {
                            auto annotations = Json::array();
                            for (const auto id : part.ids)
                                annotations.push_back(stored.annotations[std::find(stored.points.ids.begin(), stored.points.ids.end(), id) -
                                                                         stored.points.ids.begin()]);
                            part.annotations = annotations.dump();
                        }
                    auto retain = [&](const std::string& geometry, const std::vector<EdgeRecord>& edges) {
                        std::vector<GeometryEdgeEdit> decisions;
                        for (const auto& edge : edges) {
                            GeometryEdgeEdit decision;
                            decision.edgeUuid = edge.uuid;
                            decision.expectedGeneration = edge.generation;
                            decision.action = "retain";
                            decision.aRevision = edge.aUuid == uuid ? std::optional(geometry) : std::nullopt;
                            decision.bRevision = edge.bUuid == uuid ? std::optional(geometry) : std::nullopt;
                            if (!decision.aRevision) if (const auto r = db.currentGeometry(txn, edge.aUuid)) decision.aRevision = r->uuid;
                            if (!decision.bRevision) if (const auto r = db.currentGeometry(txn, edge.bUuid)) decision.bRevision = r->uuid;
                            decisions.push_back(std::move(decision));
                        }
                        return decisions;
                    };
                    // Only moved points: their coordinates change in place.
                    const bool sameSequence = previous.ids == stored.points.ids && previousAnnotations == stored.annotations;
                    auto geometry = sameSequence ? head->uuid
                        : db.editGeometrySequence(txn, uuid, sequence, head->uuid, retain(head->uuid, db.edgesOf(txn, uuid)));
                    if (!moved.empty()) {
                        std::vector<GeometryArrayEdit> edits;
                        for (const auto& [row, xyz] : moved) {
                            std::vector<uint8_t> bytes(sizeof xyz);
                            std::memcpy(bytes.data(), xyz.data(), sizeof xyz);
                            edits.push_back({"points", row * 3, std::move(bytes)});
                        }
                        db.editGeometry(txn, uuid, edits, geometry,
                                        retain(geometry, db.affectedGeometryEditEdges(txn, uuid, edits, geometry)));
                    }
                }
            }
            if (!linked.contains(uuid)) {
                EdgeRecord edge;
                edge.kind = kEdgeKind;
                edge.aUuid = uuid;
                edge.bUuid = surfaceUuid;
                edge.aAnchor.wholeGeometry = true;
                edge.bAnchor.wholeGeometry = true;
                edge.meta = Json{{"role", kRole}}.dump();
                db.addEdge(txn, edge);
            }
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
        EdgeRecord edge;
        edge.kind = kEdgeKind;
        edge.aUuid = uuid;
        edge.bUuid = annotation.surfaceUuid;
        edge.aAnchor.wholeGeometry = true;
        edge.bAnchor.wholeGeometry = true;
        edge.meta = Json{{"role", "growth_annotation"}}.dump();
        db.addEdge(txn, edge);
    });
    return uuid;
}
