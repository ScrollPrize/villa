#include "vc_object_store/objects/GeometryIngest.hpp"
#include "imports/SourceRelationships.hpp"
#include "internal/Database.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include <set>
namespace vc::objects {
namespace {
using Json = nlohmann::json;
void normalizeMetadata(GeometrySource& p, const std::string& coordinateContext)
{
    auto source = p.source;
    auto metadata = Json::parse(p.object.meta);
    const auto coordinates = mergeGeometryCoordinates(
        mergeGeometryCoordinates(sourceGeometryCoordinates(source), metadata.value("coordinates", Json::object())),
        Json::parse(coordinateContext));
    eraseSourceGeometryCoordinates(source);
    for (const auto* field : {"uuid", "name", "type", "version", "branches", "adjacent_branches", "windings_linked"}) source.erase(field);
    if (source.contains("tags") && source["tags"].is_array()) source.erase("tags");
    if (p.object.kind == "surface") for (const auto& field : tifxyzFileKeys()) source.erase(field);
    if (p.object.kind == "fiber") for (const auto* field : {"width", "width_gap_fraction", "optimization_mode"}) source.erase(field);
    if (p.object.kind == "fiber" && source.contains("hv_classification")) {
        source["hv_classification"].erase("manual_tag");
        if (!p.object.fiber || !p.object.fiber->manualFamily) source["hv_classification"].erase("automatic_tag");
        if (source["hv_classification"].empty()) source.erase("hv_classification");
    }
    if (p.object.kind == "point_collection" && source.contains("point_attributes"))
        for (auto& point : source["point_attributes"]) if (point.is_object()) point.erase("links");
    metadata["coordinates"] = coordinates;
    if (source.contains("coordinate_binding")) {
        metadata["coordinate_binding"] = source.at("coordinate_binding");
        source.erase("coordinate_binding");
    }
    metadata["source"] = std::move(source);
    p.object.meta = metadata.dump();
}
}
int64_t ingestGeometryBatch(ObjectStore& db, WriteTxn& write, std::vector<GeometrySource> batch,
    const std::string& coordinateContext)
{
    geometryCoordinates(nlohmann::json::parse(coordinateContext));
    auto* connection=ObjectStoreImplAccess::conn(write);
    std::set<std::string> incoming;
    // Check the entire batch under the writer lock before mutating any object.
    for(auto& p:batch) {
        p.object.uuid = util::canonicalUuid(p.object.uuid);
        if (p.object.kind != p.geometry.kind) throw std::invalid_argument("import object and geometry kinds differ");
        if(!incoming.insert(p.object.uuid).second)
            throw std::runtime_error("duplicate object UUID in import: "+p.object.uuid);
        detail::Statement prior(connection,"SELECT 1 FROM object_revisions WHERE object_uuid=? LIMIT 1", p.object.uuid);
        if(db.objectByUuid(write,p.object.uuid) || prior.row())
            throw std::runtime_error("object already exists or was previously imported; re-import is unsupported: "+p.object.uuid);
    }

    std::vector<detail::SourceRecord> sources;
    for (auto& p : batch) {
        normalizeMetadata(p, coordinateContext);
        db.upsertObject(write, p.object);
        db.putGeometry(write, p.object.uuid, p.geometry);
        p.geometry.arrays.clear();
        sources.push_back({p.object, std::move(p.source)});
    }
    return detail::importSourceRelationships(db, write, sources);
}
}
