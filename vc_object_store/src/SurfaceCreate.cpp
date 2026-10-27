#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"

namespace vc::objects {
const std::vector<std::string>& tifxyzFileKeys()
{
    static const std::vector<std::string> keys{"uuid", "name", "format", "type", "bbox", "scale", "tiff_dimensions"};
    return keys;
}

void createSurface(ObjectStore& store, WriteTxn& txn, const std::string& uuid, const std::string& name,
                   const std::vector<std::string>& tags, const nlohmann::json& coordinates,
                   const SurfaceProvenance& provenance, const nlohmann::json& meta, const GeometryData& geometry)
{
    if (geometry.kind != "surface") throw std::invalid_argument("createSurface needs surface geometry");
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
    store.upsertObject(txn, record);
    store.putGeometry(txn, uuid, geometry);
}
}
