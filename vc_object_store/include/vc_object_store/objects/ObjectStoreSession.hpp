#pragma once

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace vc::objects {

struct ObjectIdentity {
    std::string storeUuid, objectUuid;
    bool operator==(const ObjectIdentity&) const = default;
};

struct AttachmentIdentity {
    std::string storeUuid;
    uint64_t generation = 0;
    bool operator==(const AttachmentIdentity&) const = default;
};

struct PinnedObject {
    AttachmentIdentity attachment;
    ObjectIdentity object;
    std::string objectRevision;
    std::optional<std::string> geometryRevision;
    bool operator==(const PinnedObject&) const = default;
};

struct StoreSessionEvent {
    // Attached/Detached: an attachment joined or left the session (replace()
    // reports the old one's Detached before the new one's Attached).
    // Hub: that attachment's hubStatus() changed.
    enum class Kind { Attached, Detached, Committed, Hub };
    Kind kind = Kind::Attached;
    // Generations are unique within a session and increase on every attach,
    // including reattachment of the same file/UUID.
    AttachmentIdentity attachment;
    int64_t journalPosition = 0;
    // Generic record UUIDs: objects AND relationships, including deleted records.
    // Consumers invalidate by identity, never by names or storage paths.
    std::vector<JournalEntry> changes;
};

// Hosted replicas (objects/Hub.hpp) attached to a session sync in the background.
struct HubSessionOptions {
    std::chrono::milliseconds syncInterval{15000};
    std::chrono::milliseconds pushDelay{2000};  // after a local action, to batch edits
};

struct HubSessionStatus {
    std::vector<std::string> pending;   // records changed here and not yet pushed
    std::vector<HubConflict> conflicts; // block pushing; see resolveHubConflicts
    std::string error;                  // last failed sync; empty once one succeeds
    std::optional<int64_t> syncedAtUs;  // last successful sync, UTC Unix microseconds
    std::vector<HubLease> leases;       // as of the last hub contact
};

namespace detail { struct HubLeaseCounts; class HubWorker; }

// Keeps objects leased while it lives. Leases are renewed in the background;
// once no hold remains and the object's changes are pushed, the lease is released.
class LeaseHold {
public:
    LeaseHold() = default;
    LeaseHold(LeaseHold&&) noexcept;
    LeaseHold& operator=(LeaseHold&&) noexcept;
    LeaseHold(const LeaseHold&) = delete;
    LeaseHold& operator=(const LeaseHold&) = delete;
    ~LeaseHold() { reset(); }
    void reset();

private:
    friend class detail::HubWorker;
    std::weak_ptr<detail::HubLeaseCounts> counts_;
    std::vector<std::string> objects_;
};

// Application-owned connection and lifetime boundary for any number of stores,
// independent of object kind and UI toolkit. Each attachment is independent:
// its own identity, access mode, writer, journal events and hub sync. All worker
// work belongs inside read(); its snapshot and store references must not escape
// the callback. Callbacks must not reenter the session. Cancellation is
// cooperative. Detaching or replacing an attachment waits for its accepted reads
// and checked writes to finish; the caller must not hold a worker's completion
// hostage to the thread performing that change.
class ObjectStoreSession {
public:
    using Read = std::function<void(const ObjectStore&, const ReadTxn&, const std::atomic_bool&)>;
    using Action = std::function<void(ObjectStore&, WriteTxn&)>;

    ObjectStoreSession() = default;
    ~ObjectStoreSession();
    ObjectStoreSession(const ObjectStoreSession&) = delete;
    ObjectStoreSession& operator=(const ObjectStoreSession&) = delete;

    // Existing database only; read-only attachments reject all actions. A store
    // is attached at most once. A writable hosted replica also starts background hub sync.
    AttachmentIdentity attach(const std::filesystem::path&, const OpenOptions&, const HubSessionOptions& = {});
    // Swap one attachment for another store. The new store is validated first
    // (failure preserves the old attachment); the old one drains and delivers its
    // last commits before the new one is announced.
    AttachmentIdentity replace(const AttachmentIdentity&, const std::filesystem::path&, const OpenOptions&,
                               const HubSessionOptions& = {});
    void detach(const AttachmentIdentity&);
    std::vector<AttachmentIdentity> attachments() const;
    bool writable(const AttachmentIdentity&) const;
    bool current(const AttachmentIdentity&) const;

    void read(const AttachmentIdentity&, const Read&);
    // All writes are immediate checked atomic actions. The library enforces
    // expectations for every affected object and relationship endpoint.
    RevisionSelection act(const AttachmentIdentity&, const RevisionSelection&, const Action&);
    // Copy an object's current record and geometry from one attachment into
    // another as a new object (objects/ObjectCopy.hpp); returns its UUID.
    std::string copyObject(const AttachmentIdentity& from, const AttachmentIdentity& to, const std::string& objectUuid);
    // Drain accepted writes on every attachment. No delayed persistence or new geometry publication.
    void finalize();
    // Poll from the application's event loop. Includes external commits and
    // every record kind, tagged by attachment, in order per attachment.
    std::vector<StoreSessionEvent> takeEvents();

    // Hosted replicas: lease objects before editing them, pulling once if the
    // hub is ahead. Blocks on the hub; throws HubLeaseRefused when another
    // replica holds one. Returns an empty hold for other stores.
    LeaseHold holdLeases(const AttachmentIdentity&, const std::vector<std::string>& objects);
    // nullopt unless the attachment is a writable hosted replica.
    std::optional<HubSessionStatus> hubStatus(const AttachmentIdentity&) const;
    // Asynchronous: sync now, or settle every conflict ("hub" or "local") then sync.
    void syncHub(const AttachmentIdentity&);
    void resolveHubConflicts(const AttachmentIdentity&, const std::string& keep);

private:
    struct Attachment;
    std::shared_ptr<Attachment> open(const std::filesystem::path&, const OpenOptions&, const HubSessionOptions&);
    std::shared_ptr<Attachment> find(const AttachmentIdentity&) const;
    std::shared_ptr<Attachment> acquire(const AttachmentIdentity&, bool write);
    void release(Attachment&, bool write);
    void retire(const std::shared_ptr<Attachment>&);
    void observe(Attachment&);
    void hubChanged(const Attachment*);

    // transition_ serializes lifecycle operations and is never held by read
    // callbacks. mutex_ protects the attachment list, admission and events.
    std::mutex transition_;
    mutable std::mutex mutex_;
    std::condition_variable idle_;
    std::vector<std::shared_ptr<Attachment>> attached_;
    uint64_t generation_ = 0;
    std::vector<StoreSessionEvent> events_;
};
}
