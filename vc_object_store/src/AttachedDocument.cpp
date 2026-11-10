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
AttachedCollection readCollection(const StoreReader& reader, const ObjectRevision& selected,
                                  const std::optional<ViewFrame>& frame)
{
    const auto& object = selected.record;
    const auto& head = *selected.geometryRevision;
    AttachedCollection collection;
    collection.reference = selected.reference();
    collection.nextPointId = reader.nextGeometryPointId(object.uuid, "points");
    collection.uuid = object.uuid;
    collection.name = object.name;
    collection.points = std::get<geometry::PointCollection>(geometry::decode(reader.readGeometryData(head)));
    if (frame) {
        const auto shown = GeometrySnapshot(reader).viewTransformedAt(selected, frame->volume, frame->scale);
        collection.mapping = shown.checkMappingOperation();
        collection.points.points = shown.points("points", 0, int64_t(collection.points.points.size()));
    }
    collection.annotations = reader.geometryPointAnnotations(head, "points");
    collection.source = nlohmann::json::parse(object.meta).value("source", nlohmann::json::object());
    return collection;
}

std::optional<nlohmann::json> documentOf(const StoreReader& reader, const ObjectRevision& object)
{
    if (object.deleted || object.record.retired || !object.geometryRevision) return std::nullopt;
    const auto source = nlohmann::json::parse(object.record.meta).value("source", nlohmann::json::object());
    auto document = source.value("document", nlohmann::json::object());
    document[source.value("list_key", std::string("records"))] = reader.geometryPointAnnotations(*object.geometryRevision, "points");
    return document;
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
    if (wanted.mapping) operations.push_back(*wanted.mapping);
    auto points = wanted.points;
    const auto current = reader.currentObject(uuid);
    request.expectedObjects[uuid] = currentVersionOf(reader, uuid);
    if (current) expectObject(request, current->reference());
    if (wanted.reference) {
        if (util::canonicalUuid(wanted.reference->objectUuid) != uuid) throw std::invalid_argument("collection reference names another object");
        expectObject(request, *wanted.reference);
    }
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
        const auto shown = GeometrySnapshot(reader).viewTransformedAt(*current, frame->volume, frame->scale);
        points = shown.canonicalPointCollection(points);
        operations.push_back(shown.checkMappingOperation());
    }
    operations.push_back(UpdateObjectOp{objectRecordJson(record)});
    const auto previous = std::get<geometry::PointCollection>(geometry::decode(reader.readGeometryData(head->uuid)));
    const auto previousAnnotations = reader.geometryPointAnnotations(head->uuid, "points");
    // Points that stay keep their identity (stable ids); relationships apply
    // to the edited document, including annotation-only changes which do not
    // affect spatial dependencies but still publish new attached evidence.
    if (previous.ids != points.ids || previous.points != points.points || previousAnnotations != wanted.annotations)
        operations.push_back(UpdateGeometryArraysOp{uuid, geometry::encode(points), {{"points", wanted.annotations.dump()}}, "retain"});
    EdgeSelection edges;
    edges.endpointUuids = std::vector<std::string>{uuid};
    for (const auto& edge : reader.selectEdges(edges, std::vector<std::string>{}))
        operations.push_back(RetainRelationshipOp{edge.uuid});
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
        expectRelationshipEndpoints(reader, request);
    });
    session.submit(attachment, request);
    return collection.uuid;
}

std::optional<nlohmann::json> attachedDocument(const StoreReader& reader, const std::string& objectUuid,
                                               const std::string& role)
{
    for (const auto& edge : attachments(reader, objectUuid, role))
        if (const auto object = reader.currentObject(edge.otherEndpoint(objectUuid)))
            if (auto document = documentOf(reader, *object)) return document;
    return std::nullopt;
}

std::optional<nlohmann::json> attachedDocument(const StoreReader& reader, const ObjectVersion& owner,
                                               const std::string& role)
{
    EdgeSelection selection;
    selection.kind = kAttachmentEdgeKind;
    selection.applicableOnly = true;
    const auto graph = reader.graphReadViewAround({owner}, selection);
    const auto ownerUuid = util::canonicalUuid(owner.objectUuid);
    for (size_t i = 0; i < graph.edgeUuids.size(); ++i) {
        const auto& edge = graph.selectedRelationships->at(i).record;
        if (edge.aUuid != ownerUuid && edge.bUuid != ownerUuid) continue;
        if (nlohmann::json::parse(edge.meta).value("role", std::string{}) != role) continue;
        const auto uuid = edge.otherEndpoint(ownerUuid);
        const auto index = graph.objectUuids.at(graph.endpoints[i][0]) == uuid ? graph.endpoints[i][0] : graph.endpoints[i][1];
        if (auto document = documentOf(reader, graph.selectedObjects->at(index))) return document;
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

std::optional<nlohmann::json> attachedDocument(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const ObjectVersion& owner, const std::string& role)
{
    std::optional<nlohmann::json> document;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { document = attachedDocument(reader, owner, role); });
    return document;
}

AttachedCollectionsSnapshot attachedCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                    const std::string& objectUuid, const std::string& role,
                                                    const std::optional<ViewFrame>& frame)
{
    AttachedCollectionsSnapshot result;
    result.role = role;
    result.frame = frame;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        result.snapshot = reader.status();
        const auto owner = reader.currentObject(objectUuid);
        if (!owner || owner->deleted) throw std::runtime_error("object not found: " + objectUuid);
        result.owner = owner->reference();
        for (const auto& [uuid, edge] : attachedEdges(reader, objectUuid, role)) {
            const auto object = reader.currentObject(uuid);
            if (!object || object->record.retired || !object->geometryRevision) continue;
            result.collections.push_back(readCollection(reader, *object, frame));
        }
        const auto after = reader.status();
        if (after.storeUuid != result.snapshot.storeUuid || after.generation != result.snapshot.generation ||
            after.commitSequence != result.snapshot.commitSequence)
            throw TransactionConflict(TransactionConflict::Reason::CommitSequence, {}, "Attached collections changed while loading; reload");
    });
    return result;
}

TransactionReceipt replaceAttachedCollections(ObjectStoreSession& session, const AttachmentIdentity& attachment, AttachedCollectionsSnapshot& baseline,
                                const std::vector<AttachedCollection>& collections, const nlohmann::json& coordinates)
{
    auto next = baseline;
    next.collections = collections;
    const auto& objectUuid = baseline.owner.objectUuid;
    const auto& role = baseline.role;
    const auto& frame = baseline.frame;
    std::map<std::string, const AttachedCollection*> loaded;
    for (const auto& collection : baseline.collections) loaded.emplace(util::canonicalUuid(collection.uuid), &collection);
    std::map<std::string, const AttachedCollection*> desired;
    for (auto& collection : next.collections) {
        collection.uuid = util::canonicalUuid(collection.uuid);
        if (const auto found = loaded.find(collection.uuid); found != loaded.end()) {
            collection.reference = found->second->reference;
            collection.mapping = found->second->mapping;
        }
        if (!desired.emplace(collection.uuid, &collection).second) throw std::invalid_argument("duplicate attached collection");
    }
    auto request = session.newRequest(attachment);
    request.storeUuid = baseline.snapshot.storeUuid;
    request.generation = baseline.snapshot.generation;
    request.expectedCommitSequence = baseline.snapshot.commitSequence;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        const auto linked = attachedEdges(reader, objectUuid, role);
        const auto object = reader.currentObject(objectUuid);
        if (!object) throw std::runtime_error("object not found: " + objectUuid);
        request.expectedObjects[object->record.uuid] = object->version;
        for (const auto& [uuid, wanted] : desired) {
            // New points are already expressed in the displayed frame.
            const auto createdCoordinates = frame ? volumeCoordinates(frame->volume, frame->scale) : coordinates;
            writeCollection(reader, request, uuid, *wanted, {{"coordinates", createdCoordinates}, {"role", role}}, frame);
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
        expectRelationshipEndpoints(reader, request);
        // These loaded guards must win over current-state expectations above.
        // Membership is guarded by the query's commit sequence: inherited
        // endpoints need not change identity when local attachments change.
        expectObject(request, baseline.owner);
        for (const auto& loaded : baseline.collections) {
            if (!loaded.reference) throw std::invalid_argument("attached collection has no loaded reference");
            expectObject(request, *loaded.reference);
        }
        for (const auto& [uuid, unused] : desired) if (!loaded.contains(uuid)) {
            request.expectedObjects[uuid] = std::nullopt;
            request.expectedObjectReferences.erase(uuid);
        }
    });
    const auto receipt = session.submit(attachment, request);
    next.snapshot.commitSequence = receipt.commitSequence.value_or(baseline.snapshot.commitSequence);
    next.owner = *receipt.objectReference(objectUuid);
    for (auto& collection : next.collections) {
        collection.reference = receipt.objectReference(collection.uuid);
        if (frame && !collection.mapping)
            collection.mapping = CheckVolumeMappingOp{collection.uuid, frame->volume, frame->scale, VolumeAffine::identity()};
    }
    baseline = std::move(next);
    return receipt;
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
        if (const auto selected = reader.selectedObject(object.uuid, {}, {}, object.reference()); selected && selected->geometryRevision)
            result.push_back(readCollection(reader, *selected, ViewFrame{volume.volume, volume.scale}));
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
        result.push_back(readCollection(reader, *object, ViewFrame{volume.volume, volume.scale}));
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
                                          const std::vector<std::string>& removed, const VersionSelection& expected,
                                          const std::vector<ObjectVersion>& references)
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
        for (const auto& reference : references) expectObject(request, reference);
        expectRelationshipEndpoints(reader, request);
    });
    return session.submit(attachment, request);
}
}
