#pragma once
// The hosted backend of an ObjectStoreSession attachment: a HostedStore
// client, a bounded cache of what it read, and a poller that turns the store's
// change feed into session events.
//
// Cache keys are immutable identities (geometry revision UUIDs, numbered
// versions) or current heads that the feed invalidates, so a cached answer is
// "current as of the last check". The cache never answers queries: listings,
// regions and proximity always run on the host, so a partial cache never
// claims complete coverage.

#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

#include <vc_reader/native/lru.hpp>

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

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
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid);
    std::shared_ptr<const std::vector<uint8_t>> geometryArray(const std::string& revisionUuid, const std::string& array);
    VolumeTransforms volumeTransforms();
    // The UUIDs among `uuids` that the store holds (retired included).
    std::set<std::string> holds(const std::set<std::string>& uuids);
    std::vector<ObjectRecord> selectObjects(const ObjectSelection&);
    std::vector<EdgeRecord> selectEdges(const EdgeSelection&);

private:
    void poll();
    void check();
    void invalidate(const std::vector<std::string>& objectUuids);

    HostedStore client_;
    std::string storeUuid_;
    std::chrono::milliseconds interval_;

    // Cache: immutable entries and current heads share one byte budget.
    struct Entry {
        std::optional<ObjectRevision> revision;
        std::optional<GeometryRevision> geometry;
        std::shared_ptr<const std::vector<uint8_t>> bytes;
    };
    mutable std::mutex cacheMutex_;
    vcr::Lru<Entry> cache_;
    std::optional<VolumeTransforms> transforms_;
    std::map<std::string, std::string> edgeKinds_; // relationship UUID -> kind, as last seen
    // Advances with every head invalidation: a head read that started before
    // one is not cached.
    uint64_t invalidations_ = 0;

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
    std::optional<ObjectRevision> currentObject(const std::string& objectUuid) const override;
    std::vector<ObjectRevision> objectHistory(const std::string& objectUuid) const override;
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const std::vector<ObjectVersion>& references) const override
    {
        return attachment_->objectsAtVersions(references);
    }
    std::optional<GeometryRevision> currentGeometry(const std::string& objectUuid) const override;
    std::optional<GeometryRevision> geometryRevision(const std::string& revisionUuid) const override
    {
        return attachment_->geometryRevision(revisionUuid);
    }
    std::vector<uint8_t> readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                           int64_t byteOffset = 0,
                                           const std::optional<int64_t>& byteCount = std::nullopt) const override;
    std::vector<uint8_t> readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                               int64_t row, int64_t column, int64_t rows,
                                               int64_t columns) const override;
    VolumeTransforms volumeTransforms() const override { return attachment_->volumeTransforms(); }
    std::vector<ObjectRecord> selectObjects(const ObjectSelection& selection) const override
    {
        return attachment_->selectObjects(selection);
    }
    std::vector<EdgeRecord> selectEdges(const EdgeSelection& selection) const override
    {
        return attachment_->selectEdges(selection);
    }

private:
    HostedAttachment* attachment_;
};

} // namespace vc::objects::detail
