#pragma once

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include <chrono>
#include <initializer_list>
#include <string_view>
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

// Where an object's geometry is: in the store, published but not fetched yet,
// being fetched, or the last fetch failed (it is retried on a later read).
enum class GeometryState { Local, Remote, Fetching, Failed };

// An object's published geometry could not be fetched (offline, missing, invalid).
struct GeometryUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct StoreSessionEvent {
    // Attached/Detached: an attachment joined or left the session (replace()
    // reports the old one's Detached before the new one's Attached).
    // Hub: that attachment's hubStatus() changed.
    // Fetch: a fetched attachment's object changed state (objectUuid,
    // fetchState, error); its geometry arriving is also a Committed event.
    enum class Kind { Attached, Detached, Committed, Hub, Fetch };
    Kind kind = Kind::Attached;
    // Generations are unique within a session and increase on every attach,
    // including reattachment of the same file/UUID.
    AttachmentIdentity attachment;
    int64_t journalPosition = 0;
    // Generic record UUIDs: objects AND relationships, including deleted records.
    // Consumers invalidate by identity, never by names or storage paths.
    std::vector<JournalEntry> changes;
    std::string objectUuid;
    GeometryState fetchState = GeometryState::Local;
    std::string error;
    // Whether it may have changed records of these kinds: a commit touching
    // one, or any attachment or hub event.
    bool touches(std::initializer_list<std::string_view> kinds) const;
};

// A record, region or proximity hit and the attachment it came from.
struct AttachedRecord {
    AttachmentIdentity attachment;
    ObjectRecord record;
};
struct AttachedRegion {
    AttachmentIdentity attachment;
    RegionResult result;
};
struct AttachedHit {
    AttachmentIdentity attachment;
    SurfaceProximityHit hit;
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
    // Writes on a fetched attachment, by the code that owns its records.
    using FetchWrite = std::function<RevisionSelection(const RevisionSelection& expected, const Action&)>;
    // Fetches one object's geometry and writes it with `write`; throws when it
    // cannot. Runs on the attachment's fetch thread; must only read `store`.
    using GeometryFetcher = std::function<void(ObjectStore& store, const std::string& objectUuid, const FetchWrite& write)>;
    // A store whose objects without geometry are published elsewhere (e.g. an
    // Open Data sample, objects/CatalogStore.hpp). Callers see it read-only.
    // Geometry regions and nearest-surface queries that reach such an object
    // queue its fetch (Fetch events, then the Committed write).
    AttachmentIdentity attachFetched(const std::filesystem::path&, const OpenOptions&, GeometryFetcher);
    RevisionSelection fetchWrite(const AttachmentIdentity&, const RevisionSelection& expected, const Action&);
    // Fetch these again (their published geometry changed), read or not.
    void refetch(const AttachmentIdentity&, const std::vector<std::string>& objectUuids);
    GeometryState geometryState(const AttachmentIdentity&, const std::string& objectUuid);
    // Block until the object's geometry is local, fetching it now on a fetched
    // attachment. Throws GeometryUnavailable when the fetch fails or `cancel` returns true.
    void ensureGeometry(const AttachmentIdentity&, const std::string& objectUuid, const std::function<bool()>& cancel = {});
    // Drop queued fetches nobody waits for (e.g. the viewer closed).
    void cancelFetches(const AttachmentIdentity&);
    std::vector<AttachmentIdentity> attachments() const;
    bool writable(const AttachmentIdentity&) const;
    bool current(const AttachmentIdentity&) const;

    void read(const AttachmentIdentity&, const Read&);
    // All writes are immediate checked atomic actions. The library enforces
    // expectations for every affected object and relationship endpoint.
    RevisionSelection act(const AttachmentIdentity&, const RevisionSelection&, const Action&);
    // Decisions for relationships an edit affects: each is kept as recorded, with
    // its historical pins and no claim that it applies to the new geometry.
    using KeepRecorded = std::function<std::vector<GeometryEdgeEdit>(const std::vector<EdgeRecord>&)>;
    // A checked edit of objects pinned at the object revisions in `pins` (null:
    // the edit creates it) that keeps their relationships as recorded. `related`
    // objects are expected but not edited (new relationship endpoints). It also
    // expects every endpoint of their relationships, holds the existing pinned
    // objects' leases on a hosted replica (relationships are guarded by the
    // revision check alone), and hands the callback the decisions to pass to
    // geometry edits. Returns the receipt.
    RevisionSelection editKeepingRelationships(const AttachmentIdentity&, const RevisionSelection& pins,
        const RevisionSelection& related, const std::function<void(ObjectStore&, WriteTxn&, const KeepRecorded&)>&);
    // Copy an object's current record and geometry from one attachment into
    // another as a new object (objects/ObjectCopy.hpp) named `name` (empty: the
    // original's name), in one commit; returns its UUID.
    std::string copyObject(const AttachmentIdentity& from, const AttachmentIdentity& to, const std::string& objectUuid,
                           const std::string& name = {});
    // Bring an object into another attachment as the same object (same UUID):
    // its current record and geometry become its first revision there, so
    // restoring that revision returns to the source's version. Throws when the
    // destination already has the object.
    // An object of a fetched attachment is fetched first.
    void importObject(const AttachmentIdentity& from, const AttachmentIdentity& to, const std::string& objectUuid);
    // Restore an imported object to the version it was imported at (its first
    // revision here), e.g. "Reset to Open Data Version"; its relationships are
    // kept as recorded.
    void restoreImported(const AttachmentIdentity&, const std::string& objectUuid);
    // Queries across attachments in `order`: an object (by UUID) comes from the
    // first attachment holding it, so an imported copy hides the catalog one.
    std::vector<AttachedRecord> selectObjects(const std::vector<AttachmentIdentity>& order, const ObjectSelection&);
    std::vector<AttachedRegion> geometryRegion(const std::vector<AttachmentIdentity>& order, const RegionSlice&,
                                               const RegionOptions&, const RegionCancel& = {});
    std::vector<AttachedHit> nearestSurfaces(const std::vector<AttachmentIdentity>& order, const SurfaceProximityQuery&);
    // Files that belong to an object but are not geometry (fitted models, mesh
    // exports): <store>.artifacts/<object uuid>/, created unless `create` is
    // false (looking for a file there). A hosted replica uses the directory
    // beside its hub so every replica and service shares it; a hub addressed
    // by sftp:// URL has no shared directory.
    std::filesystem::path artifactDirectory(const AttachmentIdentity&, const std::string& objectUuid, bool create = true);
    // The store file of an attachment (a hosted store's local replica), e.g. to
    // hand to a tool that opens the same file.
    std::filesystem::path path(const AttachmentIdentity&);
    // Drain accepted writes on every attachment. No delayed persistence or new geometry publication.
    void finalize();
    // Poll from the application's event loop. Includes external commits and
    // every record kind, tagged by attachment, in order per attachment.
    std::vector<StoreSessionEvent> takeEvents();

    // Hosted replicas: lease objects before editing them, pulling once if the
    // hub is ahead. Blocks on the hub; throws HubLeaseRefused when another
    // replica holds one. Returns an empty hold for other stores.
    LeaseHold holdLeases(const AttachmentIdentity&, const std::vector<std::string>& objects);
    // Editing an object from the revision on screen: lease it (leasing may first
    // pull the hub's newer record), then check that the pin is still its current
    // revision. Throws when it is not.
    LeaseHold holdForEditing(const PinnedObject&);
    // nullopt unless the attachment is a writable hosted replica.
    std::optional<HubSessionStatus> hubStatus(const AttachmentIdentity&) const;
    // Asynchronous: sync now, or settle every conflict ("hub" or "local") then sync.
    void syncHub(const AttachmentIdentity&);
    // Sync now and block until that sync has finished: everything committed
    // before the call is pushed and the hub's changes are pulled, unless the
    // returned status reports an error or conflicts. nullopt when the attachment
    // is not a writable hosted replica. Throws when no sync finishes in time.
    std::optional<HubSessionStatus> syncHubAndWait(const AttachmentIdentity&, std::chrono::milliseconds timeout);
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
