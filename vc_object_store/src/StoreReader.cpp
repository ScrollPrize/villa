#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/TransformedLength.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>

namespace vc::objects {

std::vector<vcr::catalog::TransformResolution> StoreReader::availableTransforms(const std::string& objectUuid) const
{
    const auto object = currentObject(objectUuid);
    if (!object) throw std::invalid_argument("object not found");
    return availableTransforms(*object);
}

std::vector<vcr::catalog::TransformResolution> StoreReader::availableTransforms(const ObjectRevision& object) const
{
    const auto coordinates = geometryCoordinates(
        nlohmann::json::parse(object.record.meta).value("coordinates", nlohmann::json::object()));
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

GeometryData StoreReader::readGeometryData(const std::string& revisionUuid, uint64_t budget) const
{
    const auto revision = geometryRevision(revisionUuid);
    if (!revision) throw std::invalid_argument("geometry revision not found");
    uint64_t total = 0;
    for (const auto& array : revision->arrays) {
        if (array.byteLength < 0 || uint64_t(array.byteLength) > budget - total)
            throw std::runtime_error("geometry read exceeds byte budget; use bounded ranges");
        total += uint64_t(array.byteLength);
    }
    GeometryData data; data.kind = revision->kind; data.formatVersion = revision->formatVersion;
    data.coordinateSystem = revision->coordinateSystem; data.attributes = revision->attributes;
    for (const auto& info : revision->arrays) {
        GeometryArray array{info.name, info.dtype, info.shape, {}};
        array.bytes.reserve(size_t(info.byteLength));
        for (int64_t offset = 0; offset < info.byteLength;) {
            const auto count = std::min<int64_t>(1024 * 1024, info.byteLength - offset);
            auto bytes = readGeometryArray(revisionUuid, info.name, offset, count);
            array.bytes.insert(array.bytes.end(), bytes.begin(), bytes.end()); offset += count;
        }
        data.arrays.push_back(std::move(array));
    }
    return data;
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

void editCurrentRequest(const StoreReader& reader, TransactionRequest& request, const std::vector<std::string>& objectUuids,
                        std::vector<TransactionOperation> operations)
{
    buildCurrentEditRequest(reader, request, objectUuids, [&] { return std::move(operations); });
}

void buildCurrentEditRequest(const StoreReader& reader, TransactionRequest& request, const std::vector<std::string>& objectUuids,
                             const std::function<std::vector<TransactionOperation>()>& operations)
{
    std::vector<std::string> uuids;
    for (const auto& objectUuid : objectUuids) {
        const auto uuid = util::canonicalUuid(objectUuid);
        const auto current = reader.currentObject(uuid);
        if (!current) throw std::runtime_error("object is not in the store: " + uuid);
        request.expectedObjects[uuid] = current->version;
        uuids.push_back(uuid);
    }
    expectRelationshipEndpoints(reader, request.expectedObjects);
    for (auto& operation : operations()) request.operations.push_back(std::move(operation));
    if (uuids.empty()) return;
    EdgeSelection edges;
    edges.endpointUuids = uuids;
    edges.includeRetired = true;
    std::set<std::string> kept;
    const bool history = reader.status().historyEnabled;
    std::set<std::string> removedEdges, removedEndpoints;
    for (const auto& operation : request.operations) {
        if (const auto* remove = std::get_if<RemoveEdgeOp>(&operation))
            removedEdges.insert(util::canonicalUuid(remove->uuid));
        else if (const auto* retire = std::get_if<RetireObjectOp>(&operation))
            removedEndpoints.insert(util::canonicalUuid(retire->uuid));
        else if (const auto* restore = std::get_if<RestoreObjectOp>(&operation); restore && history && restore->source)
            removedEndpoints.insert(util::canonicalUuid(restore->uuid));
    }
    for (const auto& edge : reader.selectEdges(edges, std::vector<std::string>{}))
        if (!removedEdges.contains(edge.uuid) && !removedEndpoints.contains(edge.aUuid) &&
            !removedEndpoints.contains(edge.bUuid) && kept.insert(edge.uuid).second) {
            if (history) request.operations.push_back(KeepRecordedRelationshipOp{edge.uuid});
            else request.operations.push_back(RetainRelationshipOp{edge.uuid});
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

std::optional<int64_t> currentVersionOf(const StoreReader& reader, const std::string& objectUuid)
{
    if (const auto head = reader.currentObject(objectUuid)) return head->version;
    if (!reader.status().historyEnabled) return std::nullopt;
    const auto history = reader.objectHistory(objectUuid);
    return history.empty() ? std::nullopt : std::optional(history.back().version);
}

std::vector<EdgeRecord> attachments(const StoreReader& reader, const std::string& ownerUuid,
                                    const std::optional<std::string>& role)
{
    EdgeSelection selection;
    selection.kind = kAttachmentEdgeKind;
    selection.endpointUuids = std::vector<std::string>{ownerUuid};
    auto edges = reader.selectEdges(selection);
    if (role)
        std::erase_if(edges, [&](const EdgeRecord& edge) {
            return nlohmann::json::parse(edge.meta).value("role", std::string{}) != *role;
        });
    return edges;
}

bool changesTouchKinds(const StoreReader& reader, const FeedCursor& since, const std::set<std::string>& kinds)
{
    const auto status = reader.status();
    if (since.storeUuid != status.storeUuid || since.generation != status.generation) return true;
    if (!status.historyEnabled) return since.sequence != status.commitSequence || since.offset != 0;
    std::vector<ObjectVersion> objects;
    std::vector<std::string> edges;
    for (auto cursor = since;;) {
        const auto page = reader.changesSince(cursor);
        if (!page.cursorValid) return true;
        for (const auto& commit : page.commits)
            for (const auto& change : commit.changes)
                if (change.recordKind == "object") objects.push_back({status.storeUuid, status.generation, change.uuid, change.version});
                else edges.push_back(change.uuid);
        cursor = page.next;
        if (!page.more) break;
    }
    // Objects by their changed versions (a deletion's keeps the record it
    // removed); relationships by their current record.
    for (const auto& revision : objects.empty() ? std::vector<std::optional<ObjectRevision>>{} : reader.objectsAtVersions(objects))
        if (!revision || kinds.contains(revision->record.kind)) return true;
    if (edges.empty()) return false;
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    EdgeSelection selection;
    selection.uuids = edges;
    selection.includeRetired = true;
    const auto found = reader.selectEdges(selection, std::vector<std::string>{"kind"});
    if (found.size() != edges.size()) return true;
    return std::any_of(found.begin(), found.end(), [&](const auto& edge) { return kinds.contains(edge.kind); });
}

StoreStatus LocalStoreReader::status() const { return store_->status(*txn_); }

ChangePage LocalStoreReader::changesSince(const FeedCursor& cursor, int64_t maxChanges) const
{
    return store_->changesSince(*txn_, cursor, maxChanges);
}

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

std::vector<SelectedRelationship> LocalStoreReader::relationshipsAt(const std::vector<ObjectVersion>& versions) const
{
    return store_->relationshipsAt(*txn_, versions);
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

nlohmann::json LocalStoreReader::geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName) const
{
    return store_->geometryPointAnnotations(*txn_, revisionUuid, arrayName);
}

std::optional<ObjectRevision> LocalStoreReader::objectAtGeometry(const std::string& uuid) const
{ return store_->objectAtGeometry(*txn_,uuid); }
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

int64_t LocalStoreReader::unboundObjectCount(const std::string& kind, const std::string& volume) const
{
    return store_->unboundObjectCount(*txn_, kind, volume);
}

std::vector<std::string> LocalStoreReader::objectTags(const std::optional<std::string>& kind) const
{
    return store_->objectTags(*txn_, kind);
}

} // namespace vc::objects
