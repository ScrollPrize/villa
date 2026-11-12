#include "segmentation/SurfaceCorrections.hpp"
#include "segmentation/SurfaceStoreEdit.hpp"

#include "vc/core/StoredCollection.hpp"
#include <vc_object_store/objects/AttachedDocument.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
// Corrections are data attached to the surface, never graph links.
constexpr const char* kRole = "corrections";

// A VC collection as an attached collection, remembering its VC id.
AttachedCollection toStore(uint64_t id, const nlohmann::json& collection)
{
    auto stored = vc::storedCollection(collection);
    stored.source["source_collection_id"] = std::to_string(id);
    return stored;
}

std::map<uint64_t, Json> fromStore(const std::vector<AttachedCollection>& collections)
{
    std::map<uint64_t, Json> result;
    for (auto stored : collections) {
        const auto id = std::stoull(stored.source.value("source_collection_id", std::string("0")));
        stored.source.erase("source_collection_id");
        result[id] = vc::collectionJson(stored);
    }
    return result;
}
}

vc3d::corrections::Snapshot vc3d::corrections::read(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                        const std::string& surfaceUuid, const std::optional<vc::objects::ViewFrame>& view)
{
    Snapshot result;
    result.baseline = attachedCollections(session, attachment, surfaceUuid, kRole, view);
    result.collections = fromStore(result.baseline.collections);
    return result;
}

void vc3d::corrections::write(ObjectStoreSession& session, const AttachmentIdentity& attachment, Snapshot& snapshot,
                              const std::map<uint64_t, nlohmann::json>& collections, const vc::objects::VolumeVoxels& volume,
                              SurfaceStoreEdit* editor)
{
    // Compare with the retained baseline, not the editable presentation map:
    // callers can edit snapshot.collections in place. A genuine no-op neither
    // submits a stale edit nor advances its baseline.
    if (collections == fromStore(snapshot.baseline.collections)) return;
    std::vector<AttachedCollection> stored;
    for (const auto& [id, json] : collections) stored.push_back(toStore(id, json));
    if (editor) editor->saveAttachedCollections(snapshot.baseline, stored, vc::objects::volumeCoordinates(volume));
    else replaceAttachedCollections(session, attachment, snapshot.baseline, stored, vc::objects::volumeCoordinates(volume));
    snapshot.collections = collections;
}

std::string vc3d::corrections::writeGrowthAnnotation(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                                     const GrowthAnnotation& annotation, const vc::objects::VolumeVoxels& volume,
                                                     SurfaceStoreEdit* editor)
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
    Json meta{{"coordinates", vc::objects::volumeCoordinates(volume)}};
    meta["role"] = "growth_annotation";
    meta["source"] = {{"growth_annotation", {{"surface", annotation.surfaceUuid},
                                             {"before_geometry", annotation.beforeGeometry},
                                             {"after_geometry", annotation.afterGeometry},
                                             {"volume_used", annotation.volumeUsed},
                                             {"grid_region", annotation.gridRegion},
                                             {"bounds", {{"min", annotation.boundsMin}, {"max", annotation.boundsMax}}}}},
                      {"point_attributes", std::move(pointAttributes)}};
    const auto uuid = util::newUuid();
    ObjectRecord record;
    record.uuid = uuid;
    record.kind = "point_collection";
    record.name = "growth corrections";
    record.tags = {"growth_annotation"};
    record.meta = meta.dump();
    auto edge = attachmentEdge(uuid, annotation.surfaceUuid, "growth_annotation");
    edge.uuid = util::newUuid();
    auto request = session.newRequest(attachment);
    if (!editor) session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        const auto surface = reader.currentObject(annotation.surfaceUuid);
        if (!surface) throw std::runtime_error("surface not found: " + annotation.surfaceUuid);
        expectObject(request, *surface);
    });
    request.expectedObjects[uuid] = std::nullopt;
    request.operations = {CreateObjectOp{objectRecordJson(record)}, PutGeometryOp{uuid, vc::geometry::encode(geometry), {}},
                          AddEdgeOp{edgeRecordJson(edge)}};
    if (editor) (void)editor->submitAttached(std::move(request));
    else (void)session.submit(attachment, request);
    return uuid;
}
