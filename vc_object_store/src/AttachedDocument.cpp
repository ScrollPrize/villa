#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
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
std::map<std::string, EdgeRecord> attachedEdges(const StoreReader& reader, const std::string& objectUuid,
                                                const std::string& role)
{
    std::map<std::string, EdgeRecord> edges;
    for (auto& edge : attachments(reader, objectUuid, role)) {
        const auto collection = edge.otherEndpoint(objectUuid);
        edges.emplace(collection, std::move(edge));
    }
    return edges;
}

// A live collection with geometry `head`, its points in `frame` when given.
AttachedCollection readCollection(const StoreReader& reader, const ObjectRecord& object,
                                  const std::string& head, const std::optional<ViewFrame>& frame)
{
    AttachedCollection collection;
    collection.uuid = object.uuid;
    collection.name = object.name;
    collection.points = std::get<geometry::PointCollection>(geometry::decode(reader.readGeometryData(head)));
    if (frame) {
        const auto shown = GeometrySnapshot(reader).viewTransformedAt(object.uuid, frame->volume, frame->scale);
        collection.points.points = shown.points("points", 0, int64_t(collection.points.points.size()));
    }
    collection.annotations = reader.geometryPointAnnotations(head, "points");
    collection.source = nlohmann::json::parse(object.meta).value("source", nlohmann::json::object());
    return collection;
}

// Operations making collection `uuid` hold `wanted`, read through `reader`:
// created (or restored) with record meta `created` (coordinates, role) when
// new; otherwise a new geometry revision whose remaining points keep their
// identity and whose relationships still apply. Points are in `frame` when
// given. Expectations of the collection and its relationships' endpoints are
// added to `request`.
void writeCollection(const StoreReader& reader, TransactionRequest& request, const std::string& uuid,
                     const AttachedCollection& wanted, const nlohmann::json& created, const std::optional<ViewFrame>& frame)
{
    auto& operations = request.operations;
    auto points = wanted.points;
    const auto current = reader.currentObject(uuid);
    request.expectedObjects[uuid] = currentVersionOf(reader, uuid);
    if (current && current->record.retired) operations.push_back(RestoreObjectOp{uuid, std::nullopt});
    auto record = current ? current->record : ObjectRecord{};
    auto meta = nlohmann::json::parse(record.meta.empty() ? std::string("{}") : record.meta);
    if (!current) {
        record.uuid = uuid;
        record.kind = "point_collection";
        meta.update(created);
    }
    record.name = wanted.name;
    meta["source"] = wanted.source;
    const auto head = current && current->geometryRevision ? reader.geometryRevision(*current->geometryRevision) : std::nullopt;
    if (!head) {
        // A new geometry takes its points' annotations from meta.source.
        meta["source"]["point_attributes"] = wanted.annotations;
        record.meta = meta.dump();
        if (current) operations.push_back(UpdateObjectOp{objectRecordJson(record)});
        else operations.push_back(CreateObjectOp{objectRecordJson(record)});
        operations.push_back(PutGeometryOp{uuid, geometry::encode(points), {}, false});
        return;
    }
    record.meta = meta.dump();
    if (frame) {
        const auto shown = GeometrySnapshot(reader).viewTransformedAt(uuid, frame->volume, frame->scale);
        points = shown.canonicalPointCollection(points);
        operations.push_back(shown.checkMappingOperation());
    }
    operations.push_back(UpdateObjectOp{objectRecordJson(record)});
    const auto previous = std::get<geometry::PointCollection>(geometry::decode(reader.readGeometryData(head->uuid)));
    const auto previousAnnotations = reader.geometryPointAnnotations(head->uuid, "points");
    if (previous.ids == points.ids && previous.points == points.points && previousAnnotations == wanted.annotations) {
        // Only the record changed: its relationships still apply.
        EdgeSelection edges;
        edges.endpointUuids = std::vector<std::string>{uuid};
        for (const auto& edge : reader.selectEdges(edges, std::vector<std::string>{}))
            operations.push_back(RetainRelationshipOp{edge.uuid});
        return;
    }
    // Points that stay keep their identity (stable ids); relationships apply
    // to the edited geometry as before.
    operations.push_back(UpdateGeometryArraysOp{uuid, geometry::encode(points), {{"points", wanted.annotations.dump()}}, "retain"});
}
}

std::string attachDocument(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& objectUuid,
                           const nlohmann::json& document, const DocumentLayout& layout, const std::string& name,
                           const nlohmann::json& coordinates)
{
    AttachedCollection collection;
    collection.uuid = util::newUuid();
    collection.name = name;
    for (const auto& record : document.value(layout.listKey, nlohmann::json::array())) {
        if (!record.is_object()) continue;
        collection.points.ids.push_back(geometry::PointId(collection.points.ids.size()));
        collection.points.points.push_back(position(record.value(layout.positionKey, nlohmann::json())));
        collection.annotations.push_back(record);
    }
    auto rest = document;
    rest.erase(layout.listKey);
    collection.source = {{"document", rest}, {"list_key", layout.listKey}, {"position_key", layout.positionKey}};
    auto request = session.newRequest(attachment);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        const auto object = reader.currentObject(objectUuid);
        if (!object || object->deleted) throw std::invalid_argument("object not found: " + objectUuid);
        request.expectedObjects[object->record.uuid] = object->version;
        writeCollection(reader, request, collection.uuid, collection, {{"coordinates", coordinates}, {"role", layout.role}},
                        std::nullopt);
        auto edge = attachmentEdge(collection.uuid, object->record.uuid, layout.role);
        edge.uuid = util::newUuid();
        request.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});
        expectRelationshipEndpoints(reader, request.expectedObjects);
    });
    session.submit(attachment, request);
    return collection.uuid;
}

std::optional<nlohmann::json> attachedDocument(const StoreReader& reader, const std::string& objectUuid,
                                               const std::string& role)
{
    for (const auto& edge : attachments(reader, objectUuid, role)) {
        const auto uuid = edge.otherEndpoint(objectUuid);
        const auto object = reader.currentObject(uuid);
        if (!object || object->record.retired || !object->geometryRevision) continue;
        const auto source = nlohmann::json::parse(object->record.meta).value("source", nlohmann::json::object());
        auto document = source.value("document", nlohmann::json::object());
        document[source.value("list_key", std::string("records"))] =
            reader.geometryPointAnnotations(*object->geometryRevision, "points");
        return document;
    }
    return std::nullopt;
}

std::optional<nlohmann::json> attachedDocument(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const std::string& objectUuid, const std::string& role)
{
    std::optional<nlohmann::json> document;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        document = attachedDocument(reader, objectUuid, role);
    });
    return document;
}

std::vector<AttachedCollection> attachedCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                    const std::string& objectUuid, const std::string& role,
                                                    const std::optional<ViewFrame>& frame)
{
    std::vector<AttachedCollection> result;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        for (const auto& [uuid, edge] : attachedEdges(reader, objectUuid, role)) {
            const auto object = reader.currentObject(uuid);
            if (!object || object->record.retired || !object->geometryRevision) continue;
            result.push_back(readCollection(reader, object->record, *object->geometryRevision, frame));
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
    auto request = session.newRequest(attachment);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        const auto linked = attachedEdges(reader, objectUuid, role);
        const auto object = reader.currentObject(objectUuid);
        if (!object) throw std::runtime_error("object not found: " + objectUuid);
        request.expectedObjects[object->record.uuid] = object->version;
        for (const auto& [uuid, wanted] : desired) {
            writeCollection(reader, request, uuid, *wanted, {{"coordinates", coordinates}, {"role", role}}, frame);
            if (!linked.contains(uuid)) {
                auto edge = attachmentEdge(uuid, object->record.uuid, role);
                edge.uuid = util::newUuid();
                request.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});
            }
        }
        for (const auto& [uuid, edge] : linked) {
            if (desired.contains(uuid)) continue;
            request.expectedObjects[uuid] = currentVersionOf(reader, uuid);
            request.operations.push_back(RemoveEdgeOp{edge.uuid});
            request.operations.push_back(RetireObjectOp{uuid});
        }
        expectRelationshipEndpoints(reader, request.expectedObjects);
    });
    session.submit(attachment, request);
}

std::vector<AttachedCollection> volumeCollections(const StoreReader& reader, const VolumeVoxels& volume)
{
    std::vector<AttachedCollection> result;
    ObjectSelection selection;
    selection.kind = "point_collection";
    selection.volume = volume.volume;
    selection.targetScale = volume.scale;
    for (const auto& object : reader.selectObjects(selection)) {
        if (nlohmann::json::parse(object.meta).contains("role")) continue;
        if (const auto head = reader.currentGeometry(object.uuid))
            result.push_back(readCollection(reader, object, head->uuid, ViewFrame{volume.volume, volume.scale}));
    }
    return result;
}

std::vector<AttachedCollection> pointCollections(const StoreReader& reader, const std::vector<std::string>& uuids,
                                                 const VolumeVoxels& volume)
{
    std::vector<AttachedCollection> result;
    for (const auto& uuid : uuids) {
        const auto object = reader.currentObject(util::canonicalUuid(uuid));
        if (!object || object->record.retired || object->record.kind != "point_collection")
            throw std::runtime_error("point collection not found: " + uuid);
        if (!object->geometryRevision) throw std::runtime_error("point collection has no geometry: " + uuid);
        result.push_back(readCollection(reader, object->record, *object->geometryRevision, ViewFrame{volume.volume, volume.scale}));
    }
    return result;
}

std::vector<AttachedCollection> volumeCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                  const VolumeVoxels& volume)
{
    std::vector<AttachedCollection> result;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { result = volumeCollections(reader, volume); });
    return result;
}

TransactionReceipt writeVolumeCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                          const VolumeVoxels& volume, const std::vector<AttachedCollection>& changed,
                                          const std::vector<std::string>& removed, const VersionSelection& expected)
{
    auto request = session.newRequest(attachment);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        for (const auto& collection : changed)
            writeCollection(reader, request, util::canonicalUuid(collection.uuid), collection,
                            {{"coordinates", volumeCoordinates(volume)}}, ViewFrame{volume.volume, volume.scale});
        for (const auto& removedUuid : removed) {
            const auto uuid = util::canonicalUuid(removedUuid);
            request.expectedObjects[uuid] = currentVersionOf(reader, uuid);
            if (const auto object = reader.currentObject(uuid); object && !object->record.retired)
                request.operations.push_back(RetireObjectOp{uuid});
        }
        // The caller's pins win over the current versions read here.
        for (const auto& [uuid, version] : expected) request.expectedObjects[util::canonicalUuid(uuid)] = version;
        expectRelationshipEndpoints(reader, request.expectedObjects);
    });
    return session.submit(attachment, request);
}
}
