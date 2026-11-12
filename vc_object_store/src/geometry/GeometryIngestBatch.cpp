#include "geometry/GeometryIngestBatch.hpp"
#include "imports/SourceRelationships.hpp"
#include "internal/Database.hpp"
#include "internal/RecordResolver.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include <set>
namespace vc::objects {
namespace {
using Json = nlohmann::json;
void normalizeMetadata(GeometrySource& p, const std::string& coordinateContext)
{
    detail::validateSourceTopology(p.object.kind, p.source);
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
std::string detail::sourceUuid(const std::string& uuid)
{
    try { return util::canonicalUuid(uuid); }
    catch (const std::exception& error) { throw GeometrySourceRejected(error.what()); }
}
int64_t detail::ingestGeometryBatch(ObjectStore& db, WriteTxn& write, std::vector<GeometrySource> batch,
    const std::string& coordinateContext, bool registeredInitial)
{
    geometryCoordinates(nlohmann::json::parse(coordinateContext));
    auto* connection=ObjectStoreImplAccess::conn(write);
    detail::Savepoint savepoint(connection);
    std::set<std::string> incoming;
    std::vector<std::string> clashes;
    // Check the entire batch under the writer lock before mutating any object.
    for(auto& p:batch) {
        p.object.uuid = sourceUuid(p.object.uuid);
        if (p.object.kind != p.geometry.kind) throw GeometrySourceRejected("import object and geometry kinds differ");
        if(!incoming.insert(p.object.uuid).second)
            throw GeometrySourceRejected("duplicate object UUID in import: "+p.object.uuid);
        const auto existing=db.objectByUuid(write,p.object.uuid);
        const bool previouslyImported = detail::objectSource(connection,p.object.uuid,false,true)!=nullptr;
        if (registeredInitial && (!existing || existing->retired || db.currentGeometry(write,p.object.uuid)))
            throw GeometrySourceRejected("object already exists or was previously imported; re-import is unsupported: "+p.object.uuid);
        if (!registeredInitial && (existing.has_value() || previouslyImported)) clashes.push_back(p.object.uuid);
    }
    if (!clashes.empty()) throw GeometryIdentitiesExist(std::move(clashes));

    std::vector<detail::SourceRecord> sources;
    for (auto& p : batch) {
        const auto existing = db.objectByUuid(write, p.object.uuid);
        // Merging the source's declarations with the import's is source validation.
        try { normalizeMetadata(p, coordinateContext); }
        catch (const std::exception& error) { throw GeometrySourceRejected(error.what()); }
        auto record = existing.value_or(p.object);
        auto metadata = Json::parse(record.meta);
        const auto incoming = Json::parse(p.object.meta);
        metadata["source"] = incoming["source"];
        if (incoming.contains("coordinate_binding")) metadata["coordinate_binding"] = incoming.at("coordinate_binding");
        metadata["coordinates"] = mergeGeometryCoordinates(
            metadata.value("coordinates", Json::object()), incoming["coordinates"]);
        record.meta = metadata.dump();
        if (p.object.fiber) record.fiber = p.object.fiber;
        if (p.object.surface) record.surface = p.object.surface;
        if (p.object.track) record.track = p.object.track;
        db.upsertObject(write, record);
        db.putGeometry(write, p.object.uuid, p.geometry);
        p.geometry.arrays.clear();
        sources.push_back({p.object, std::move(p.source)});

    }
    // Source JSON that does not describe its relationships rejects the source.
    try {
        const auto count = detail::importSourceRelationships(db, write, sources);
        savepoint.release();
        return count;
    }
    catch (const nlohmann::json::exception& error) { throw GeometrySourceRejected(error.what()); }
}
GeometryIdentitiesExist::GeometryIdentitiesExist(std::vector<std::string> existing)
    : GeometrySourceRejected("object already exists or was previously imported; re-import is unsupported: " + existing.front()),
      uuids(std::move(existing)) {}
int64_t ingestGeometryBatch(ObjectStore& db, WriteTxn& txn, std::vector<GeometrySource> batch,
    const std::string& coordinateContext)
{
    return detail::ingestGeometryBatch(db, txn, std::move(batch), coordinateContext, false);
}
void addIngestOperation(TransactionRequest& request, const StoreReader& reader, std::vector<GeometrySource> batch,
    const std::string& coordinateContext)
{
    geometryCoordinates(Json::parse(coordinateContext));
    IngestGeometryOp op;
    op.coordinateContext = Json::parse(coordinateContext).dump();
    for (auto& p : batch) p.object.uuid = util::canonicalUuid(p.object.uuid);
    // An existing object is expected at its current version, so importing it
    // again is refused as a re-import rather than as a stale request.
    auto expect = [&](const std::string& uuid) {
        if (request.expectedObjects.contains(uuid)) return;
        const auto current = reader.currentObject(uuid);
        request.expectedObjects[uuid] = currentObjectExpectation(reader, uuid);
    };
    // Fiber branches name their existing endpoints; links resolved by source
    // identity (point and winding links) stay within the batch.
    for (const auto& p : batch) expect(p.object.uuid);
    for (const auto& target : fiberBatchEndpoints(batch)) expect(target);
    for (auto& p : batch) op.items.push_back({objectRecordJson(p.object), std::move(p.geometry), p.source.dump()});
    request.operations.push_back(std::move(op));
}

}
