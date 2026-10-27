#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <cmath>
#include <map>

namespace vc::objects {
namespace {
geometry::Point position(const nlohmann::json& value)
{
    if (value.is_array() && value.size() >= 3 && value[0].is_number() && value[1].is_number() && value[2].is_number()) {
        const geometry::Point xyz{value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
        if (std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2])) return xyz;
    }
    return {-1, -1, -1};
}

// The relationships attaching collections to `objectUuid` in `role`, by collection UUID.
std::map<std::string, EdgeRecord> attachedEdges(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
                                                const std::string& role)
{
    std::map<std::string, EdgeRecord> edges;
    for (auto& edge : store.attachments(txn, objectUuid, role)) {
        const auto collection = edge.aUuid == objectUuid ? edge.bUuid : edge.aUuid;
        edges.emplace(collection, std::move(edge));
    }
    return edges;
}

// A live collection with geometry `head`, its points in `frame` when given.
AttachedCollection readCollection(const ObjectStore& store, const Txn& txn, const ObjectRecord& object,
                                  const std::string& head, const std::optional<ViewFrame>& frame)
{
    AttachedCollection collection;
    collection.uuid = object.uuid;
    collection.name = object.name;
    collection.points = std::get<geometry::PointCollection>(geometry::decode(store.readGeometryData(txn, head)));
    if (frame) {
        const auto shown = GeometrySnapshot(store, txn).viewTransformedAt(object.uuid, frame->volume, frame->scale);
        collection.points.points = shown.points("points", 0, int64_t(collection.points.points.size()));
    }
    collection.annotations = nlohmann::json::parse(store.geometryPointAnnotations(txn, head, "points"));
    collection.source = nlohmann::json::parse(object.meta).value("source", nlohmann::json::object());
    return collection;
}

// Make collection `uuid` hold `wanted`: created (or restored) with record
// meta `created` (coordinates, role) when new; otherwise a new geometry
// revision whose remaining points keep their identity, keeping its
// relationships. Points are in `frame` when given.
void writeCollection(ObjectStore& store, WriteTxn& txn, const std::string& uuid, const AttachedCollection& wanted,
                     const nlohmann::json& created, const std::optional<ViewFrame>& frame)
{
    auto points = wanted.points;
    const auto current = store.objectByUuid(txn, uuid);
    if (current && current->retired) store.restoreObject(txn, uuid);
    auto record = current ? *store.objectByUuid(txn, uuid) : ObjectRecord{};
    auto meta = nlohmann::json::parse(record.meta.empty() ? std::string("{}") : record.meta);
    if (!current) {
        record.uuid = uuid;
        record.kind = "point_collection";
        meta.update(created);
    }
    record.name = wanted.name;
    meta["source"] = wanted.source;
    const auto head = current ? store.currentGeometry(txn, uuid) : std::nullopt;
    if (!head) {
        // A new geometry takes its points' annotations from meta.source.
        meta["source"]["point_attributes"] = wanted.annotations;
        record.meta = meta.dump();
        store.upsertObject(txn, record);
        store.putGeometry(txn, uuid, geometry::encode(points));
        return;
    }
    record.meta = meta.dump();
    store.upsertObject(txn, record);
    if (frame) {
        const auto shown = GeometrySnapshot(store, txn).viewTransformedAt(uuid, frame->volume, frame->scale);
        points = shown.canonicalPointCollection(points);
    }
    const auto previous = std::get<geometry::PointCollection>(geometry::decode(store.readGeometryData(txn, head->uuid)));
    const auto previousAnnotations = nlohmann::json::parse(store.geometryPointAnnotations(txn, head->uuid, "points"));
    if (previous.ids == points.ids && previous.points == points.points && previousAnnotations == wanted.annotations) {
        // Only the record changed: its relationships still apply.
        for (const auto& edge : store.edgesOf(txn, uuid)) store.retainRelationship(txn, edge.uuid);
        return;
    }
    // Points that stay keep their identity (stable ids); relationships apply
    // to the edited geometry as before.
    auto retain = [&](const std::vector<EdgeRecord>& edges) {
        std::vector<GeometryEdgeEdit> decisions;
        for (const auto& edge : edges) {
            GeometryEdgeEdit decision;
            decision.edgeUuid = edge.uuid;
            decision.expectedGeneration = edge.generation;
            decision.action = "retain";
            if (const auto r = store.currentGeometry(txn, edge.aUuid)) decision.aRevision = r->uuid;
            if (const auto r = store.currentGeometry(txn, edge.bUuid)) decision.bRevision = r->uuid;
            decisions.push_back(std::move(decision));
        }
        return decisions;
    };
    updateGeometryArrays(store, txn, uuid, geometry::encode(points), head->uuid, {{"points", wanted.annotations.dump()}},
                         retain);
}
}

void attachDocument(ObjectStore& store, WriteTxn& txn, const std::string& objectUuid, const std::string& collectionUuid,
                    const nlohmann::json& document, const DocumentLayout& layout, const std::string& name,
                    const nlohmann::json& coordinates)
{
    auto records = nlohmann::json::array();
    for (const auto& record : document.value(layout.listKey, nlohmann::json::array()))
        if (record.is_object()) records.push_back(record);
    geometry::PointCollection points;
    for (const auto& record : records) {
        points.ids.push_back(geometry::PointId(points.ids.size()));
        points.points.push_back(position(record.value(layout.positionKey, nlohmann::json())));
    }
    auto rest = document;
    rest.erase(layout.listKey);
    ObjectRecord record;
    record.uuid = collectionUuid;
    record.kind = "point_collection";
    record.name = name;
    record.meta = nlohmann::json{{"role", layout.role},
                                 {"coordinates", coordinates},
                                 {"source", {{"document", rest},
                                             {"list_key", layout.listKey},
                                             {"position_key", layout.positionKey},
                                             {"point_attributes", records}}}}.dump();
    store.upsertObject(txn, record);
    store.putGeometry(txn, collectionUuid, geometry::encode(points));
    store.attach(txn, collectionUuid, objectUuid, layout.role);
}

std::optional<nlohmann::json> attachedDocument(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
                                               const std::string& role)
{
    for (const auto& edge : store.attachments(txn, objectUuid, role)) {
        const auto uuid = edge.aUuid == objectUuid ? edge.bUuid : edge.aUuid;
        const auto object = store.objectByUuid(txn, uuid);
        const auto head = object && !object->retired ? store.currentGeometry(txn, uuid) : std::nullopt;
        if (!head) continue;
        const auto source = nlohmann::json::parse(object->meta).value("source", nlohmann::json::object());
        auto document = source.value("document", nlohmann::json::object());
        document[source.value("list_key", std::string("records"))] =
            nlohmann::json::parse(store.geometryPointAnnotations(txn, head->uuid, "points"));
        return document;
    }
    return std::nullopt;
}

std::optional<nlohmann::json> attachedDocument(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const std::string& objectUuid, const std::string& role)
{
    std::optional<nlohmann::json> document;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        document = attachedDocument(store, txn, objectUuid, role);
    });
    return document;
}

std::vector<AttachedCollection> attachedCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                    const std::string& objectUuid, const std::string& role,
                                                    const std::optional<ViewFrame>& frame)
{
    std::vector<AttachedCollection> result;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        for (const auto& [uuid, edge] : attachedEdges(store, txn, objectUuid, role)) {
            const auto object = store.objectByUuid(txn, uuid);
            const auto head = object && !object->retired ? store.currentGeometry(txn, uuid) : std::nullopt;
            if (!head) continue;
            result.push_back(readCollection(store, txn, *object, head->uuid, frame));
        }
    });
    return result;
}

void replaceAttachedCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& objectUuid,
                                const std::string& role, const std::vector<AttachedCollection>& collections,
                                const nlohmann::json& coordinates, const std::optional<ViewFrame>& frame)
{
    std::map<std::string, const AttachedCollection*> desired;
    for (const auto& collection : collections) desired[util::canonicalUuid(collection.uuid)] = &collection;
    std::map<std::string, EdgeRecord> linked;
    RevisionSelection pins, related;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        linked = attachedEdges(store, txn, objectUuid, role);
        const auto object = store.currentObjectRevision(txn, objectUuid);
        if (!object) throw std::runtime_error("object not found: " + objectUuid);
        related[objectUuid] = object->uuid;
        auto pin = [&](const std::string& uuid) {
            const auto head = store.currentObjectRevision(txn, uuid);
            pins[uuid] = head ? std::optional(head->uuid) : std::nullopt;
        };
        for (const auto& [uuid, edge] : linked) pin(uuid);
        for (const auto& [uuid, collection] : desired) pin(uuid);
    });
    session.editKeepingRelationships(attachment, pins, related, [&](auto& store, auto& txn, const auto&) {
        for (const auto& [uuid, wanted] : desired) {
            writeCollection(store, txn, uuid, *wanted, {{"coordinates", coordinates}, {"role", role}}, frame);
            if (!linked.contains(uuid)) store.attach(txn, uuid, objectUuid, role);
        }
        for (const auto& [uuid, edge] : linked) {
            if (desired.contains(uuid)) continue;
            store.removeEdge(txn, edge.uuid);
            store.retireObject(txn, uuid);
        }
    });
}

std::vector<AttachedCollection> volumeCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                  const std::string& volume)
{
    std::vector<AttachedCollection> result;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        ObjectSelection selection;
        selection.kind = "point_collection";
        selection.volume = volume;
        for (const auto& object : store.selectObjects(txn, selection)) {
            if (nlohmann::json::parse(object.meta).contains("role")) continue;
            if (const auto head = store.currentGeometry(txn, object.uuid))
                result.push_back(readCollection(store, txn, object, head->uuid, ViewFrame{volume, 1}));
        }
    });
    return result;
}

void writeVolumeCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& volume,
                            const std::vector<AttachedCollection>& changed, const std::vector<std::string>& removed)
{
    RevisionSelection pins;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        auto pin = [&](const std::string& uuid) {
            const auto head = store.currentObjectRevision(txn, uuid);
            pins[util::canonicalUuid(uuid)] = head ? std::optional(head->uuid) : std::nullopt;
        };
        for (const auto& collection : changed) pin(collection.uuid);
        for (const auto& uuid : removed) pin(uuid);
    });
    session.editKeepingRelationships(attachment, pins, {}, [&](auto& store, auto& txn, const auto&) {
        for (const auto& collection : changed)
            writeCollection(store, txn, util::canonicalUuid(collection.uuid), collection,
                            {{"coordinates", volumeCoordinates(volume, 1)}}, ViewFrame{volume, 1});
        for (const auto& uuid : removed)
            if (const auto object = store.objectByUuid(txn, util::canonicalUuid(uuid)); object && !object->retired)
                store.retireObject(txn, object->uuid);
    });
}
}
