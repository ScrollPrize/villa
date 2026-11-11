#pragma once
// The hosted backend of an ObjectStoreSession attachment: a HostedStore
// client, a bounded cache of what it read, and a poller that turns the store's
// change feed into session events.
//
// Cache keys are immutable identities (geometry revision UUIDs, numbered
// versions) or current heads that the feed invalidates, so a cached answer is
// "current as of the last check". Exact-endpoint relationship queries also
// cache complete answers, scoped to the observed commit (membership can change
// without changing inherited endpoints). Listings, regions and proximity run
// on the host, so a partial object cache never claims complete coverage.

#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

#include <vc_reader/native/lru.hpp>
#include <nlohmann/json.hpp>

#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace vcr { class DiskCache; }

namespace vc::objects::detail {

class HostedAttachment {
public:
    // Committed changes in feed order (or a revalidation when the feed position
    // was lost), for the session's event queue.
    using Deliver = std::function<void(StoreSessionEvent)>;

    HostedAttachment(const std::string& endpoint, const HostedSessionOptions&);
    ~HostedAttachment();
    HostedAttachment(const HostedAttachment&) = delete;
    HostedAttachment& operator=(const HostedAttachment&) = delete;

    const std::string& storeUuid() const { return storeUuid_; }
    HostedStore& client() { return client_; }

    // Starts the poller; `identity` tags its events.
    void start(const AttachmentIdentity& identity, Deliver deliver);
    // Stops the poller (waits for a running check).
    void stop();
    // One check now, on the caller's thread.
    void refresh();
    HostedFreshness freshness() const;

    StoreStatus status();
    TransactionReceipt submit(const TransactionRequest&);
    std::optional<ObjectRevision> currentObject(const std::string& objectUuid);
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>&);
    std::vector<SelectedRelationship> relationshipsAt(const std::vector<ObjectVersion>&);
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid);
    std::shared_ptr<const std::vector<uint8_t>> geometryArray(const std::string& revisionUuid, const std::string& array);
    std::vector<uint8_t> geometryArrayRange(const std::string& revisionUuid, const std::string& array,
                                           int64_t offset, const std::optional<int64_t>& count);
    std::shared_ptr<const nlohmann::json> geometryPointAnnotations(const std::string& revisionUuid, const std::string& array);
    std::vector<uint8_t> artifactRange(const ArtifactFile&, uint64_t offset, uint64_t count);
    void writeArtifact(const ArtifactFile&, const std::filesystem::path&);
    void prefetch(ObjectPrefetchJob&, const std::atomic_bool& detached);
    bool persistentCacheEnabled() const { return bool(disk_); }
    VolumeTransforms volumeTransforms();
    // The UUIDs among `uuids` that the store holds (retired included).
    std::set<std::string> holds(const std::set<std::string>& uuids);
    std::vector<ObjectRecord> selectObjects(const ObjectSelection&,
                                            const std::optional<std::vector<std::string>>& fields = std::nullopt);
    std::vector<EdgeRecord> selectEdges(const EdgeSelection&,
                                        const std::optional<std::vector<std::string>>& fields = std::nullopt);

private:
    void poll();
    void check();
    void invalidate(const std::vector<std::string>& objectUuids);

    void publishPayload(const std::string&, const std::vector<uint8_t>&);
    std::optional<std::vector<uint8_t>> readCachedPayload(const std::string&, std::optional<size_t>);
    std::vector<uint8_t> cachedPayload(const std::string& key,
        const std::function<std::vector<uint8_t>()>& fetch,
        std::optional<size_t> length = std::nullopt);
    std::string payloadKey(const std::string&) const;
    std::shared_ptr<vcr::DiskCache> disk_;
    std::filesystem::path diskRoot_;
    std::atomic_uint interactivePayloads_{0};
    std::mutex priorityMutex_;
    std::condition_variable priorityWake_;
    std::mutex missesMutex_;
    std::map<std::string, std::shared_future<std::vector<uint8_t>>> misses_;
    HostedStore client_;
    std::string storeUuid_;
    std::chrono::milliseconds interval_;

    // Cache: immutable entries and current heads share one byte budget.
    struct Entry {
        std::optional<ObjectRevision> revision;
        std::optional<GeometryRevision> geometry;
        std::shared_ptr<const std::vector<uint8_t>> bytes;
        std::shared_ptr<const nlohmann::json> json;
    };
    mutable std::mutex cacheMutex_;
    vcr::Lru<Entry> cache_;
    std::optional<VolumeTransforms> transforms_;
    std::map<std::string, std::string> edgeKinds_; // relationship UUID -> kind, as last seen
    // Advances with every head invalidation: a head read that started before
    // one is not cached.
    uint64_t invalidations_ = 0;
    uint64_t transformInvalidations_ = 0;

    // Poller state.
    mutable std::mutex stateMutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::mutex checking_; // one check at a time (poller or refresh)
    std::optional<FeedCursor> cursor_;
    HostedFreshness freshness_;
    AttachmentIdentity identity_;
    Deliver deliver_;
    std::thread poller_;
};

// ObjectStoreSession::readStore for a hosted attachment: each answer comes
// from the attachment's cache or the host.
class HostedStoreReader final : public StoreReader {
public:
    explicit HostedStoreReader(HostedAttachment& attachment) : attachment_(&attachment) {}
    bool active() const override { return true; }
    StoreStatus status() const override { return attachment_->status(); }
    std::optional<ObjectRevision> currentObject(const std::string& objectUuid, bool includeDeleted = false) const override;
    std::vector<ObjectRevision> objectHistory(const std::string& objectUuid, bool ifRetained = false) const override;
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>& references) const override
    {
        return attachment_->objectsAtVersions(references);
    }
    std::vector<SelectedRelationship> relationshipsAt(const std::vector<ObjectVersion>& versions) const override
    {
        return attachment_->relationshipsAt(versions);
    }
    SelectedInputs selectedInputs(const std::vector<ObjectVersion>& versions, const FeedCursor& position) const override
    {
        return attachment_->client().selectedInputs(versions, position);
    }
    std::vector<ObjectVersion> versionsAt(const ObjectVersion& anchor) const override
    { return attachment_->client().versionsAt(anchor); }
    std::optional<std::filesystem::path> resolveStoragePath(const std::string& objectUuid,
        const std::optional<std::filesystem::path>& datasetRoot = std::nullopt) const override
    { return attachment_->client().resolveStoragePath(objectUuid, datasetRoot); }
    std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const override;
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const override
    {
        return attachment_->geometryRevision(revisionUuid);
    }
    std::optional<ObjectRevision> objectAtGeometry(const std::string& revisionUuid) const override
    { return attachment_->client().objectAtGeometry(revisionUuid); }
    std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                           int64_t byteOffset = 0,
                                           const std::optional<int64_t>& byteCount = std::nullopt) const override;
    std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                               int64_t row, int64_t column, int64_t rows,
                                               int64_t columns) const override;
    VolumeTransforms volumeTransforms() const override { return attachment_->volumeTransforms(); }
    std::vector<ObjectRecord> selectObjects(
        const ObjectSelection& selection,
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const override
    {
        return attachment_->selectObjects(selection, fields);
    }
    std::vector<EdgeRecord> selectEdges(
        const EdgeSelection& selection,
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const override
    {
        return attachment_->selectEdges(selection, fields);
    }
    nlohmann::json geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName) const override
    {
        return *attachment_->geometryPointAnnotations(revisionUuid, arrayName);
    }
    int64_t nextGeometryPointId(const std::string& objectUuid,const std::string& arrayName) const override
    {
        return attachment_->client().nextGeometryPointId(objectUuid,arrayName);
    }
    std::vector<std::string> componentOf(const std::string& objectUuid, const std::vector<std::string>& edgeKinds = {},
                                         const std::atomic_bool* = nullptr) const override
    {
        return attachment_->client().traverse({GraphQuery::Kind::Component, objectUuid, edgeKinds, std::nullopt, false});
    }
    std::vector<std::string> neighbours(const std::string& objectUuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const override
    {
        std::vector<std::string> kinds;
        if (edgeKind) kinds.push_back(*edgeKind);
        return attachment_->client().traverse({GraphQuery::Kind::Neighbours, objectUuid, kinds, neighbourKind, false});
    }
    int64_t objectCount(const std::string& kind, bool includeRetired = false) const override
    {
        return attachment_->client().objectCount(kind, includeRetired);
    }
    int64_t unboundObjectCount(const std::string& kind, const std::string& volume) const override
    {
        return attachment_->client().unboundObjectCount(kind, volume);
    }
    ChangePage changesSince(const FeedCursor& cursor, int64_t maxChanges = 4096) const override
    {
        return attachment_->client().changesSince(cursor, maxChanges);
    }
    std::vector<std::string> objectTags(const std::optional<std::string>& kind = std::nullopt) const override
    {
        return attachment_->client().objectTags(kind);
    }

private:
    HostedAttachment* attachment_;
};

} // namespace vc::objects::detail
