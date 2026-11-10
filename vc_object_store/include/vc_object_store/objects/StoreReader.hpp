#pragma once
// Reads of one attachment that work the same for a local store and a hosted
// one (ObjectStoreSession::readStore). Geometry views (store/GeometryViews.hpp)
// read through it, so code written against a StoreReader serves both backends.
//
// A local reader answers from one read transaction: every answer is from the
// same snapshot. A hosted reader answers each call from the host (or the
// attachment's cache of immutable data and of heads current as of the last
// check); answers about immutable identities (numbered versions, geometry
// revisions) are exact, current heads may be from different commits.

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Transactions.hpp"

#include <limits>
#include <set>
#include <functional>

namespace vc::objects {

class StoreReader {
public:
    virtual ~StoreReader() = default;
    // False once the reader can no longer answer (its transaction ended).
    virtual bool active() const = 0;
    virtual StoreStatus status() const = 0;
    // The change feed after `cursor` (ObjectStore::changesSince).
    virtual ChangePage changesSince(const FeedCursor& cursor, int64_t maxChanges = 4096) const = 0;
    // The live (or retired) current revision; nullopt when absent or deleted.
    virtual std::optional<ObjectRevision> currentObject(const std::string& objectUuid) const = 0;
    virtual std::vector<ObjectRevision> objectHistory(const std::string& objectUuid) const = 0;
    virtual std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&) const = 0;
    // The relationships among objects at these versions (ObjectStore::relationshipsAt).
    virtual std::vector<SelectedRelationship> relationshipsAt(const std::vector<ObjectVersion>&) const = 0;
    // The geometry of the object's current revision.
    virtual std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const = 0;
    virtual std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const = 0;
    virtual std::optional<ObjectRevision> objectAtGeometry(const std::string& revisionUuid) const = 0;
    // Exact logical (C-order) bytes, as ObjectStore::readGeometryArray.
    virtual std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                   int64_t byteOffset = 0,
                                                   const std::optional<int64_t>& byteCount = std::nullopt) const = 0;
    // One packed buffer in caller revision order, as
    // ObjectStore::readGeometryArrays. By default each array is read whole
    // through readGeometryArray (a hosted reader's cached arrays).
    virtual PackedGeometryArray readGeometryArrays(const std::vector<std::string>& revisionUuids,
                                                   const std::string& arrayName) const;
    // Dense row-major rectangle of a surface-aligned array, as
    // ObjectStore::readGeometryRectangle.
    virtual std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                                       int64_t row, int64_t column, int64_t rows,
                                                       int64_t columns) const = 0;
    // Per-point annotations of a sequence array (ObjectStore::geometryPointAnnotations).
    virtual nlohmann::json geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName) const = 0;
    virtual VolumeTransforms volumeTransforms() const = 0;
    // Complete listings (every page when the selection has no limit), with
    // the projections of ObjectStore::selectObjects/selectEdges.
    virtual std::vector<ObjectRecord> selectObjects(
        const ObjectSelection&, const std::optional<std::vector<std::string>>& fields = std::nullopt) const = 0;
    virtual std::vector<EdgeRecord> selectEdges(
        const EdgeSelection&, const std::optional<std::vector<std::string>>& fields = std::nullopt) const = 0;
    // As ObjectStore::componentOf (retired objects excluded; `cancel` stops a
    // local traversal) and ObjectStore::neighbours.
    virtual std::vector<std::string> componentOf(const std::string& objectUuid,
                                                 const std::vector<std::string>& edgeKinds = {},
                                                 const std::atomic_bool* cancel = nullptr) const = 0;
    virtual std::vector<std::string> neighbours(const std::string& objectUuid,
                                                const std::optional<std::string>& edgeKind = std::nullopt,
                                                const std::optional<std::string>& neighbourKind = std::nullopt) const = 0;
    // As ObjectStore::objectCount, unboundObjectCount and objectTags.
    virtual int64_t objectCount(const std::string& kind, bool includeRetired = false) const = 0;
    virtual int64_t unboundObjectCount(const std::string& kind, const std::string& volume) const = 0;
    virtual std::vector<std::string> objectTags(const std::optional<std::string>& kind = std::nullopt) const = 0;

    // The transforms from the object's recorded volume to every reachable
    // volume (ObjectStore::availableTransforms).
    std::vector<vcr::catalog::TransformResolution> availableTransforms(const std::string& objectUuid) const;
    std::vector<vcr::catalog::TransformResolution> availableTransforms(const ObjectRevision&) const;
    // Every array of a geometry revision (ObjectStore::readGeometryData);
    // throws when their bytes exceed `byteBudget`.
    GeometryData readGeometryData(const std::string& revisionUuid, uint64_t byteBudget = 64 * 1024 * 1024) const;
    // The length of a fiber or track's current geometry in `target` voxels at
    // `targetScale` (ObjectStore::geometryLengthInVolume).
    double geometryLengthInVolume(const std::string& objectUuid, const std::string& target,
                                  double targetScale = 1) const;
};

// The attachment relationships of `owner` (ObjectStore::attachments).
std::vector<EdgeRecord> attachments(const StoreReader&, const std::string& ownerUuid,
                                    const std::optional<std::string>& role = std::nullopt);

// The current head version, or the latest retained deletion version when
// history is enabled. Missing/deleted objects return nullopt without history.
std::optional<int64_t> currentVersionOf(const StoreReader&, const std::string& objectUuid);

// Request expectations read through a reader (objects/Transactions.hpp).
// Adds every endpoint of the relationships (retired included) of the objects
// `expected` expects present, at its current version (absent: nullopt), so a
// geometry change that keeps or edits those relationships is checked against
// both ends. Existing entries are kept.
void expectRelationshipEndpoints(const StoreReader&, VersionSelection& expected);
// The typed form of an edit of existing objects that keeps their
// relationships as recorded (ObjectStoreSession::editCurrent): expects each of
// `objectUuids` at its current version (throws when one is absent) and every
// endpoint of their relationships, appends `operations`, then keeps every
// surviving relationship of each object as recorded. Relationships removed
// explicitly or by retirement/historical restoration are not retained. With
// history disabled there is no archival binding; retain and validate the
// current relationships instead. Geometry-changing requests must make their
// affected-relationship decisions explicitly.
void editCurrentRequest(const StoreReader&, TransactionRequest&, const std::vector<std::string>& objectUuids,
                        std::vector<TransactionOperation> operations);
// Callback form: capture expected versions before computing the edits, then
// choose surviving relationships after the operations are known. This keeps
// hosted reads during the callback from advancing the request's expectations.
void buildCurrentEditRequest(const StoreReader&, TransactionRequest&, const std::vector<std::string>& objectUuids,
                             const std::function<std::vector<TransactionOperation>()>& operations);
// The operation that declares `record`'s coordinate frame
// (ObjectStore::bindGeometryCoordinates): stored at `storedUnitsPerVoxel`
// units per voxel of `volume`, merged into its meta.coordinates; throws for a
// kind without one or a conflict with what it declares.
SetMetaOp bindCoordinatesOperation(const ObjectRecord&, const std::string& volume, double storedUnitsPerVoxel);
// The current versions of the recorded volume transforms and route choices:
// read dependencies of a write computed through a transformed view, so a
// changed registration conflicts instead of mapping the edit wrongly.
VersionSelection transformDependencies(const StoreReader&);
// Whether commits after `since` may have changed records of `kinds`: true when
// one did, when a change's kind cannot be learned (a removed relationship) or
// when the feed position is gone. A cheap guard for caches keyed by such records.
bool changesTouchKinds(const StoreReader&, const FeedCursor& since, const std::set<std::string>& kinds);

// A reader over a caller's transaction, valid while it is.
class LocalStoreReader final : public StoreReader {
public:
    LocalStoreReader(const ObjectStore& store, const Txn& txn) : store_(&store), txn_(&txn) {}
    const ObjectStore& store() const { return *store_; }
    const Txn& txn() const { return *txn_; }

    bool active() const override { return txn_->active(); }
    StoreStatus status() const override;
    ChangePage changesSince(const FeedCursor& cursor, int64_t maxChanges = 4096) const override;
    std::optional<ObjectRevision> currentObject(const std::string& objectUuid) const override;
    std::vector<ObjectRevision> objectHistory(const std::string& objectUuid) const override;
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&) const override;
    std::vector<SelectedRelationship> relationshipsAt(const std::vector<ObjectVersion>&) const override;
    std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const override;
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const override;
    std::optional<ObjectRevision> objectAtGeometry(const std::string& revisionUuid) const override;
    std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                           int64_t byteOffset = 0,
                                           const std::optional<int64_t>& byteCount = std::nullopt) const override;
    PackedGeometryArray readGeometryArrays(const std::vector<std::string>& revisionUuids,
                                           const std::string& arrayName) const override;
    std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                               int64_t row, int64_t column, int64_t rows,
                                               int64_t columns) const override;
    nlohmann::json geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName) const override;
    VolumeTransforms volumeTransforms() const override;
    std::vector<ObjectRecord> selectObjects(
        const ObjectSelection&, const std::optional<std::vector<std::string>>& fields = std::nullopt) const override;
    std::vector<EdgeRecord> selectEdges(
        const EdgeSelection&, const std::optional<std::vector<std::string>>& fields = std::nullopt) const override;
    std::vector<std::string> componentOf(const std::string& objectUuid, const std::vector<std::string>& edgeKinds = {},
                                         const std::atomic_bool* cancel = nullptr) const override;
    std::vector<std::string> neighbours(const std::string& objectUuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const override;
    int64_t objectCount(const std::string& kind, bool includeRetired = false) const override;
    int64_t unboundObjectCount(const std::string& kind, const std::string& volume) const override;
    std::vector<std::string> objectTags(const std::optional<std::string>& kind = std::nullopt) const override;

private:
    const ObjectStore* store_;
    const Txn* txn_;
};

} // namespace vc::objects
