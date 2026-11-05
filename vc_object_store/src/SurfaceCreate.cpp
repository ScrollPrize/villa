#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/Uuid.hpp"

namespace vc::objects {
const std::vector<std::string>& tifxyzFileKeys()
{
    // The area keys too: the store keeps the surface's area itself (SurfaceDetail::area).
    static const std::vector<std::string> keys{"uuid", "name", "format", "type", "bbox", "scale", "tiff_dimensions",
                                               "area_vx2", "area_cm2"};
    return keys;
}

ObjectRecord surfaceRecord(const std::string& uuid, const std::string& name, const std::vector<std::string>& tags,
                           const nlohmann::json& coordinates, const SurfaceProvenance& provenance, const nlohmann::json& meta)
{
    auto recordMeta = meta.is_object() ? meta : nlohmann::json::object();
    for (const auto& key : tifxyzFileKeys()) recordMeta.erase(key);
    for (const auto* key : {"tags", "provenance"}) recordMeta.erase(key);
    eraseSourceGeometryCoordinates(recordMeta);
    recordMeta["coordinates"] = geometryCoordinates(coordinates);
    recordMeta["provenance"] = {{"tool", provenance.tool}, {"parameters", provenance.parameters}, {"sources", provenance.sources}};
    ObjectRecord record;
    record.uuid = uuid;
    record.kind = "surface";
    record.name = name;
    record.tags = tags;
    record.meta = recordMeta.dump();
    return record;
}

void createSurface(ObjectStore& store, WriteTxn& txn, const std::string& uuid, const std::string& name,
                   const std::vector<std::string>& tags, const nlohmann::json& coordinates,
                   const SurfaceProvenance& provenance, const nlohmann::json& meta, const GeometryData& geometry)
{
    if (geometry.kind != "surface") throw std::invalid_argument("createSurface needs surface geometry");
    store.upsertObject(txn, surfaceRecord(uuid, name, tags, coordinates, provenance, meta));
    store.putGeometry(txn, uuid, geometry);
}

std::vector<TransactionOperation> createSurfaceOperations(const std::string& uuid, const std::string& name,
                                                          const std::vector<std::string>& tags,
                                                          const nlohmann::json& coordinates, const SurfaceProvenance& provenance,
                                                          const nlohmann::json& meta, GeometryData geometry)
{
    if (geometry.kind != "surface") throw std::invalid_argument("createSurfaceOperations needs surface geometry");
    return {CreateObjectOp{objectRecordJson(surfaceRecord(uuid, name, tags, coordinates, provenance, meta))},
            PutGeometryOp{uuid, std::move(geometry), {}}};
}

std::string createSurface(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& name,
                          const std::vector<std::string>& tags, const nlohmann::json& coordinates,
                          const SurfaceProvenance& provenance, const nlohmann::json& meta, GeometryData geometry)
{
    const auto uuid = util::newUuid();
    auto request = session.newRequest(attachment);
    request.expectedObjects[uuid] = std::nullopt;
    request.operations = createSurfaceOperations(uuid, name, tags, coordinates, provenance, meta, std::move(geometry));
    session.submit(attachment, request);
    return uuid;
}
}
