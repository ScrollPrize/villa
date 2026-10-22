#include "geometry/GeometryIngestBatch.hpp"
#include "imports/SourceRelationships.hpp"
#include "internal/Database.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
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
    if (p.object.kind == "patch") for (const auto* field : {"scale", "bbox", "tiff_dimensions", "format"}) source.erase(field);
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
int64_t detail::ingestGeometryBatch(ObjectStore& db, WriteTxn& write, std::vector<GeometrySource> batch,
    const std::string& coordinateContext, bool registeredInitial)
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
        const auto existing=db.objectByUuid(write,p.object.uuid);
        detail::Statement history(connection,"SELECT 1 FROM geometry_revisions WHERE object_uuid=? LIMIT 1",
            detail::parseUuid(p.object.uuid));
        detail::Statement prior(connection,"SELECT 1 FROM object_revisions WHERE object_uuid=? LIMIT 1", p.object.uuid);
        if(registeredInitial ? (!existing || existing->retired || history.row()) : (existing.has_value() || prior.row()))
            throw std::runtime_error("object already exists or was previously imported; re-import is unsupported: "+p.object.uuid);
    }

    std::vector<detail::SourceRecord> sources;
    for (auto& p : batch) {
        const auto existing = db.objectByUuid(write, p.object.uuid);
        normalizeMetadata(p, coordinateContext);
        auto record = existing.value_or(p.object);
        auto metadata = Json::parse(record.meta);
        const auto incoming = Json::parse(p.object.meta);
        metadata["source"] = incoming["source"];
        if (incoming.contains("coordinate_binding")) metadata["coordinate_binding"] = incoming.at("coordinate_binding");
        metadata["coordinates"] = mergeGeometryCoordinates(
            metadata.value("coordinates", Json::object()), incoming["coordinates"]);
        record.meta = metadata.dump();
        if (p.object.fiber) record.fiber = p.object.fiber;
        if (p.object.patch) record.patch = p.object.patch;
        if (p.object.track) record.track = p.object.track;
        db.upsertObject(write, record);
        db.putGeometry(write, p.object.uuid, p.geometry);
        p.geometry.arrays.clear();
        sources.push_back({p.object, std::move(p.source)});

    }
    return detail::importSourceRelationships(db, write, sources);
}
int64_t ingestGeometryBatch(ObjectStore& db, WriteTxn& txn, std::vector<GeometrySource> batch,
    const std::string& coordinateContext)
{
    return detail::ingestGeometryBatch(db, txn, std::move(batch), coordinateContext, false);
}
}
