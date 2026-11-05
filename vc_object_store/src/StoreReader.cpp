#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"

#include <nlohmann/json.hpp>

#include <set>

namespace vc::objects {

std::vector<vcr::catalog::TransformResolution> StoreReader::availableTransforms(const std::string& objectUuid) const
{
    const auto object = currentObject(objectUuid);
    if (!object) throw std::invalid_argument("object not found");
    const auto coordinates = geometryCoordinates(
        nlohmann::json::parse(object->record.meta).value("coordinates", nlohmann::json::object()));
    if (coordinates["volume_uuid"].is_null() || coordinates["scale_factor"].is_null()) return {};
    const auto source = coordinates["volume_uuid"].get<std::string>();
    const auto context = volumeTransforms();
    auto result = context.graph.available(source);
    for (auto& resolution : result) resolution = context.resolve(source, resolution.target);
    return result;
}

void expectRelationshipEndpoints(const StoreReader& reader, VersionSelection& expected)
{
    EdgeSelection edges;
    edges.endpointUuids.emplace();
    for (const auto& [uuid, version] : expected)
        if (version) edges.endpointUuids->push_back(uuid);
    if (edges.endpointUuids->empty()) return;
    edges.includeRetired = true;
    std::set<std::string> endpoints;
    for (const auto& edge : reader.selectEdges(edges))
        for (const auto& endpoint : {edge.aUuid, edge.bUuid})
            if (!expected.contains(endpoint)) endpoints.insert(endpoint);
    for (const auto& endpoint : endpoints) {
        const auto head = reader.currentObject(endpoint);
        expected.emplace(endpoint, head ? std::optional(head->version) : std::nullopt);
    }
}

VersionSelection transformDependencies(const StoreReader& reader)
{
    VersionSelection result;
    for (const auto* kind : {"volume_transform", "volume_transform_choice"}) {
        ObjectSelection selection;
        selection.kind = kind;
        selection.includeRetired = true;
        for (const auto& record : reader.selectObjects(selection)) {
            if (!record.version) throw std::runtime_error("transform record without a numbered version: " + record.uuid);
            result.emplace(record.uuid, record.version);
        }
    }
    return result;
}

StoreStatus LocalStoreReader::status() const { return store_->status(*txn_); }

std::optional<ObjectRevision> LocalStoreReader::currentObject(const std::string& objectUuid) const
{
    auto revision = store_->currentObjectRevision(*txn_, objectUuid);
    if (revision && revision->deleted) return std::nullopt;
    return revision;
}

std::vector<ObjectRevision> LocalStoreReader::objectHistory(const std::string& objectUuid) const
{
    return store_->objectHistory(*txn_, objectUuid);
}

std::vector<std::optional<ObjectRevision>> LocalStoreReader::objectsAtVersions(
    const std::vector<ObjectVersion>& references) const
{
    return store_->objectsAtVersions(*txn_, references);
}

std::optional<GeometryRevision> LocalStoreReader::currentGeometry(const std::string& objectUuid) const
{
    return store_->currentGeometry(*txn_, objectUuid);
}

std::optional<GeometryRevision> LocalStoreReader::geometryRevision(const std::string& revisionUuid) const
{
    return store_->geometryRevision(*txn_, revisionUuid);
}

std::vector<uint8_t> LocalStoreReader::readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                         int64_t byteOffset, const std::optional<int64_t>& byteCount) const
{
    return store_->readGeometryArray(*txn_, revisionUuid, arrayName, byteOffset, byteCount);
}

std::vector<uint8_t> LocalStoreReader::readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                                             int64_t row, int64_t column, int64_t rows,
                                                             int64_t columns) const
{
    return store_->readGeometryRectangle(*txn_, revisionUuid, arrayName, row, column, rows, columns);
}

VolumeTransforms LocalStoreReader::volumeTransforms() const { return store_->volumeTransforms(*txn_); }

std::vector<ObjectRecord> LocalStoreReader::selectObjects(const ObjectSelection& selection) const
{
    return store_->selectObjects(*txn_, selection);
}

std::vector<EdgeRecord> LocalStoreReader::selectEdges(const EdgeSelection& selection) const
{
    return store_->selectEdges(*txn_, selection);
}

} // namespace vc::objects
