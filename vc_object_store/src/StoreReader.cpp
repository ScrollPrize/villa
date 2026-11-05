#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "internal/TransformedLength.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
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

PackedGeometryArray StoreReader::readGeometryArrays(const std::vector<std::string>& revisionUuids,
                                                   const std::string& arrayName) const
{
    PackedGeometryArray packed;
    for (const auto& revision : revisionUuids) {
        const auto descriptor = geometryRevision(revision);
        if (!descriptor) throw std::invalid_argument("geometry: revision not found");
        const auto found = std::find_if(descriptor->arrays.begin(), descriptor->arrays.end(),
                                        [&](const auto& a) { return a.name == arrayName; });
        if (found == descriptor->arrays.end()) throw std::invalid_argument("geometry: array not found: " + arrayName);
        const auto& shape = found->shape;
        if (packed.offsets.size() == 1) { packed.dtype = found->dtype; packed.shape = shape; packed.shape[0] = 0; }
        if (packed.dtype != found->dtype || packed.shape.size() != shape.size() ||
            !std::equal(shape.begin() + 1, shape.end(), packed.shape.begin() + 1))
            throw std::invalid_argument("geometry: packed arrays require matching dtype and trailing dimensions");
        const auto bytes = readGeometryArray(revision, arrayName);
        packed.data.insert(packed.data.end(), bytes.begin(), bytes.end());
        packed.shape[0] += shape[0];
        packed.offsets.push_back(uint64_t(packed.shape[0]));
    }
    return packed;
}

double StoreReader::geometryLengthInVolume(const std::string& objectUuid, const std::string& target,
                                           double targetScale) const
{
    const auto object = currentObject(objectUuid);
    if (!object || (object->record.kind != "fiber" && object->record.kind != "track"))
        throw std::invalid_argument("length requires fiber or track");
    const auto& kind = object->record.kind;
    const auto mapping = volumeTransforms().toVolume(nlohmann::json::parse(object->record.meta), target, targetScale);
    if (!mapping) throw std::invalid_argument("volume mapping is unavailable or ambiguous");
    if (!object->geometryRevision) throw std::invalid_argument("object has no geometry");
    const auto descriptor = geometryRevision(*object->geometryRevision);
    if (!descriptor) throw std::invalid_argument("object has no geometry");
    const auto rows = [&](const std::string& name) {
        for (const auto& array : descriptor->arrays)
            if (array.name == name) return array.shape.at(0);
        throw std::invalid_argument("geometry: array not found: " + name);
    };
    const auto name = detail::lengthArrayName(kind, kind == "fiber" ? rows("line_points") : 0);
    return detail::polylineLength(
        [&](int64_t offset, int64_t bytes) { return readGeometryArray(descriptor->uuid, name, offset, bytes); },
        rows(name), kind, *mapping);
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

PackedGeometryArray LocalStoreReader::readGeometryArrays(const std::vector<std::string>& revisionUuids,
                                                        const std::string& arrayName) const
{
    return store_->readGeometryArrays(*txn_, revisionUuids, arrayName);
}

std::vector<uint8_t> LocalStoreReader::readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                                             int64_t row, int64_t column, int64_t rows,
                                                             int64_t columns) const
{
    return store_->readGeometryRectangle(*txn_, revisionUuid, arrayName, row, column, rows, columns);
}

std::string LocalStoreReader::geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName) const
{
    return store_->geometryPointAnnotations(*txn_, revisionUuid, arrayName);
}

VolumeTransforms LocalStoreReader::volumeTransforms() const { return store_->volumeTransforms(*txn_); }

std::vector<ObjectRecord> LocalStoreReader::selectObjects(const ObjectSelection& selection,
                                                          const std::optional<std::vector<std::string>>& fields) const
{
    return store_->selectObjects(*txn_, selection, fields);
}

std::vector<EdgeRecord> LocalStoreReader::selectEdges(const EdgeSelection& selection,
                                                      const std::optional<std::vector<std::string>>& fields) const
{
    return store_->selectEdges(*txn_, selection, fields);
}

std::vector<std::string> LocalStoreReader::componentOf(const std::string& objectUuid,
                                                       const std::vector<std::string>& edgeKinds,
                                                       const std::atomic_bool* cancel) const
{
    return store_->componentOf(*txn_, objectUuid, edgeKinds, cancel);
}

std::vector<std::string> LocalStoreReader::neighbours(const std::string& objectUuid,
                                                      const std::optional<std::string>& edgeKind,
                                                      const std::optional<std::string>& neighbourKind) const
{
    return store_->neighbours(*txn_, objectUuid, edgeKind, neighbourKind);
}

int64_t LocalStoreReader::objectCount(const std::string& kind, bool includeRetired) const
{
    return store_->objectCount(*txn_, kind, includeRetired);
}

std::vector<std::string> LocalStoreReader::objectTags(const std::optional<std::string>& kind) const
{
    return store_->objectTags(*txn_, kind);
}

} // namespace vc::objects
