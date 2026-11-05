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

namespace vc::objects {

class StoreReader {
public:
    virtual ~StoreReader() = default;
    // False once the reader can no longer answer (its transaction ended).
    virtual bool active() const = 0;
    virtual StoreStatus status() const = 0;
    // The live (or retired) current revision; nullopt when absent or deleted.
    virtual std::optional<ObjectRevision> currentObject(const std::string& objectUuid) const = 0;
    virtual std::vector<ObjectRevision> objectHistory(const std::string& objectUuid) const = 0;
    virtual std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&) const = 0;
    // The geometry of the object's current revision.
    virtual std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const = 0;
    virtual std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const = 0;
    // Exact logical (C-order) bytes, as ObjectStore::readGeometryArray.
    virtual std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                   int64_t byteOffset = 0,
                                                   const std::optional<int64_t>& byteCount = std::nullopt) const = 0;
    // Dense row-major rectangle of a surface-aligned array, as
    // ObjectStore::readGeometryRectangle.
    virtual std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                                       int64_t row, int64_t column, int64_t rows,
                                                       int64_t columns) const = 0;
    virtual VolumeTransforms volumeTransforms() const = 0;
    // Complete listings (every page when the selection has no limit).
    virtual std::vector<ObjectRecord> selectObjects(const ObjectSelection&) const = 0;
    virtual std::vector<EdgeRecord> selectEdges(const EdgeSelection&) const = 0;

    // The transforms from the object's recorded volume to every reachable
    // volume (ObjectStore::availableTransforms).
    std::vector<vcr::catalog::TransformResolution> availableTransforms(const std::string& objectUuid) const;
};

// A reader over a caller's transaction, valid while it is.
class LocalStoreReader final : public StoreReader {
public:
    LocalStoreReader(const ObjectStore& store, const Txn& txn) : store_(&store), txn_(&txn) {}
    const ObjectStore& store() const { return *store_; }
    const Txn& txn() const { return *txn_; }

    bool active() const override { return txn_->active(); }
    StoreStatus status() const override;
    std::optional<ObjectRevision> currentObject(const std::string& objectUuid) const override;
    std::vector<ObjectRevision> objectHistory(const std::string& objectUuid) const override;
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&) const override;
    std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const override;
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const override;
    std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                           int64_t byteOffset = 0,
                                           const std::optional<int64_t>& byteCount = std::nullopt) const override;
    std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                               int64_t row, int64_t column, int64_t rows,
                                               int64_t columns) const override;
    VolumeTransforms volumeTransforms() const override;
    std::vector<ObjectRecord> selectObjects(const ObjectSelection&) const override;
    std::vector<EdgeRecord> selectEdges(const EdgeSelection&) const override;

private:
    const ObjectStore* store_;
    const Txn* txn_;
};

} // namespace vc::objects
