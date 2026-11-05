#pragma once

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include <chrono>
#include <initializer_list>
#include <string_view>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <map>
#include <set>
#include <span>

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

// An object as some code holds it: its numbered version (0: not stored yet)
// in the store's history generation, and the geometry of that version.
// Checked requests editing it expect that version in that generation.
struct PinnedObject {
    AttachmentIdentity attachment;
    ObjectIdentity object;
    std::string generation;
    int64_t version = 0;
    std::optional<std::string> geometryRevision;
    bool stored() const { return version > 0; }
    ObjectVersion reference() const { return {object.storeUuid, generation, object.objectUuid, version}; }
    bool operator==(const PinnedObject&) const = default;
};
// The pin of a revision read from `attachment`.
PinnedObject pinOf(const AttachmentIdentity& attachment, const ObjectRevision& revision);

// An object's published geometry could not be fetched (offline, missing, invalid).
struct GeometryUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct StoreSessionEvent {
    // Attached/Detached: an attachment joined or left the session (replace()
    // reports the old one's Detached before the new one's Attached).
    // Hub: that attachment's hubStatus() changed.
    // A fetched object's geometry arriving is a Committed event.
    enum class Kind { Attached, Detached, Committed, Hub, Draft };
    Kind kind = Kind::Attached;
    // Draft: this session's drafts of the objects in `changes` (recordUuid,
    // kind) were set, edited or cleared; `attachment` is empty for drafts
    // without one.
    // Generations are unique within a session and increase on every attach,
    // including reattachment of the same file/UUID.
    AttachmentIdentity attachment;
    int64_t journalPosition = 0;
    // Generic record UUIDs: objects AND relationships, including deleted records.
    // Consumers invalidate by identity, never by names or storage paths.
    // Hosted attachments: journalPosition is the store's commit sequence, and
    // a change whose kind could not be learned has an empty kind.
    std::vector<JournalEntry> changes;
    // Committed, hosted: the attachment lost its place in the store's change
    // feed (another history generation, a restored store), so any record may
    // have changed; reread what is shown.
    bool revalidate = false;
    // Whether it may have changed records of these kinds: a commit or draft
    // touching one (or a change of unknown kind), a revalidation, or any
    // attachment or hub event.
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

// A store served by vc_object_server, attached through its endpoint.
struct HostedSessionOptions {
    HostedStoreOptions client;
    // Attribution of this session's requests (newRequest).
    std::string author;
    // How often the attachment checks the store for commits (one status
    // request; the change feed only when the commit sequence moved).
    std::chrono::milliseconds pollInterval{2000};
    // Bound of the attachment's memory cache: geometry descriptors and arrays
    // by revision, records at numbered versions, and current heads until the
    // feed reports them changed.
    std::size_t cacheBytes = std::size_t(256) << 20;
};

// What a hosted attachment knows about its store, for the UI: current as of
// the last successful check, and the last failure since.
struct HostedFreshness {
    std::optional<StoreStatus> status;
    std::optional<int64_t> checkedAtUs; // UTC Unix microseconds of that check
    std::string error;                  // empty when the last check succeeded
};

// Durable drafts (useDraftDirectory): what the session knows of one draft.
struct DraftState {
    uint64_t revision = 0;
    // The committed version the draft edits (none: a new object or not given).
    std::optional<ObjectVersion> base;
    // A durable submission of this revision is pending (its outcome unknown
    // or not yet returned).
    std::optional<uint64_t> submittedRevision;
    std::optional<std::string> pendingRequest;
    // The last save conflicted: reload, discard, or reapply by setting a new
    // base. Never resent automatically.
    std::string conflict;
    // This revision is in the last completed checkpoint (or needs none: saved).
    bool checkpointed = false;
    // This revision holds its base's committed geometry (markDraftSaved, or a
    // durable save that kept the draft): nothing to recover, so checkpoints
    // leave it out until it changes.
    bool saved = false;
};

// A draft's grid, as setDraft and updateDraft left it.
struct DraftContents {
    vc::geometry::Surface surface;
    CoordinateJson coordinates;
    std::optional<AttachmentIdentity> attachment;
    uint64_t revision = 0;
};

// A draft a durable save publishes: the draft revision whose grid the request
// holds (0: the current one). Edits after that revision are not in the save.
struct SavedDraft {
    std::string objectUuid;
    uint64_t revision = 0;
};

struct DurableSave {
    std::vector<SavedDraft> drafts;
    // An editor still has the drafts open: after success each stays, on the
    // committed version, and is marked saved when it did not change after its
    // saved revision (instead of being cleared). Not kept across a restart: a
    // pending save recovered by useDraftDirectory clears its drafts.
    bool keepDrafts = false;
};

struct PendingSubmission {
    std::string requestUuid, storeUuid;
    std::vector<std::string> drafts;
};

struct DraftRecovery {
    uint64_t checkpoint = 0;           // its sequence; 0 when the directory had none
    std::vector<std::string> drafts;   // drafts installed again
    std::vector<PendingSubmission> pending;
};

namespace detail { struct HubLeaseCounts; class HubWorker; struct SurfaceDraft; class HostedAttachment; struct DurableDrafts; }

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
    using StoreRead = std::function<void(const StoreReader&, const std::atomic_bool& cancelled)>;

    ObjectStoreSession();
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
    // A store hosted by vc_object_server (client/HostedStore.hpp). The session
    // never opens its database: read() and act() and the other callbacks that
    // take an ObjectStore throw for it; use the typed operations below. Its
    // commits arrive through takeEvents() after each check (pollInterval).
    // Throws when the endpoint does not answer or its store is already attached.
    AttachmentIdentity attachHosted(const std::string& endpoint, const HostedSessionOptions& = {});
    bool hosted(const AttachmentIdentity&) const;
    // nullopt for local attachments.
    std::optional<HostedFreshness> freshness(const AttachmentIdentity&) const;
    // Check a hosted attachment now (blocking); its commits then arrive
    // through takeEvents(). A failed check is reported by freshness().
    void refresh(const AttachmentIdentity&);

    // Typed operations, the same for local and hosted attachments.
    StoreStatus status(const AttachmentIdentity&);
    // A new request (fresh UUID) for the attachment's store and active history
    // generation, by the attachment's author (OpenOptions::author, or
    // HostedSessionOptions::author).
    TransactionRequest newRequest(const AttachmentIdentity&);
    // Checked atomic request (objects/Transactions.hpp); conflicts throw
    // TransactionConflict. Hosted: transport failures leave the outcome
    // unknown; recover it with receipt() or submit the identical request again.
    // A hub replica first leases the existing objects the request expects.
    TransactionReceipt submit(const AttachmentIdentity&, const TransactionRequest&);
    std::optional<TransactionReceipt> receipt(const AttachmentIdentity&, const std::string& requestUuid);
    // Hosted: current as of the last check (or a newer read).
    std::optional<ObjectRevision> currentObject(const AttachmentIdentity&, const std::string& objectUuid);
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const AttachmentIdentity&, const std::vector<ObjectVersion>&);
    std::optional<GeometryRevision> geometryRevision(const AttachmentIdentity&, const std::string& revisionUuid);
    std::vector<uint8_t> readGeometryArray(const AttachmentIdentity&, const std::string& revisionUuid, const std::string& arrayName);
    // The attachment's volume transforms: recorded registrations and choices
    // (and, locally, the client catalog's unrecorded registrations).
    VolumeTransforms volumeTransforms(const AttachmentIdentity&);
    // Artifacts (objects/Artifacts.hpp) on any attachment: the manifest of an
    // object's current version (or `version`, in the active generation).
    std::vector<ArtifactFile> artifacts(const AttachmentIdentity&, const std::string& objectUuid,
                                        const std::optional<int64_t>& version = std::nullopt);
    // Writes those files under `destination` (local to this process; names
    // are relative paths), each atomically and verified against the manifest.
    std::vector<ArtifactFile> materializeArtifacts(const AttachmentIdentity&, const std::string& objectUuid,
                                                   const std::filesystem::path& destination,
                                                   const std::optional<int64_t>& version = std::nullopt);
    // Stores each (name, local file) durably, then names them in the object's
    // manifest (replacing entries of the same names) in one checked request
    // that expects `expectedVersion` of the object; returns its receipt.
    TransactionReceipt publishArtifacts(const AttachmentIdentity&, const std::string& objectUuid, int64_t expectedVersion,
                                        const std::vector<std::pair<std::string, std::filesystem::path>>& files,
                                        const std::string& author);
    // Writes by a geometry fetcher (setGeometryFetcher) on a writable attachment.
    using FetchWrite = std::function<RevisionSelection(const RevisionSelection& expected, const Action&)>;
    // Fetches one object's geometry and writes it with `write`; throws when it
    // cannot. Runs on the attachment's fetch thread; must only read `store`.
    using GeometryFetcher = std::function<void(ObjectStore& store, const std::string& objectUuid, const FetchWrite& write)>;
    // Objects whose record meta has `metaKey` publish their geometry elsewhere
    // (e.g. Open Data catalog surfaces, objects/CatalogStore.hpp): `fetch`
    // gives them their geometry, in any writable attachment. Geometry regions
    // and nearest-surface queries that reach such an object without local
    // geometry queue its fetch (then the Committed write); refetch fetches it
    // again. An empty `fetch` removes the fetcher. Read-only attachments never fetch.
    void setGeometryFetcher(const std::string& metaKey, GeometryFetcher fetch);
    RevisionSelection fetchWrite(const AttachmentIdentity&, const RevisionSelection& expected, const Action&);
    // Fetch these again (their published geometry changed), read or not.
    void refetch(const AttachmentIdentity&, const std::vector<std::string>& objectUuids);
    // Block until the object's geometry is local, fetching it now when it has
    // a fetcher. Throws GeometryUnavailable when the fetch
    // fails or `cancel` returns true.
    void ensureGeometry(const AttachmentIdentity&, const std::string& objectUuid, const std::function<bool()>& cancel = {});
    std::vector<AttachmentIdentity> attachments() const;
    bool writable(const AttachmentIdentity&) const;
    bool current(const AttachmentIdentity&) const;

    // Reads of any attachment, local or hosted (objects/StoreReader.hpp): a
    // local one answers from one read transaction, a hosted one from its
    // cache and the host. Geometry views over the reader
    // (GeometrySnapshot(reader)) work on both. The reader and views must not
    // escape the callback; `cancelled` is set when the attachment detaches.
    void readStore(const AttachmentIdentity&, const StoreRead&);
    // Local attachments only (throws for hosted ones).
    void read(const AttachmentIdentity&, const Read&);
    // All writes are immediate checked atomic actions. The library enforces
    // expectations for every affected object and relationship endpoint.
    // `receipt`, when given, receives the commit's receipt (numbered versions).
    RevisionSelection act(const AttachmentIdentity&, const RevisionSelection&, const Action&,
                          TransactionReceipt* receipt = nullptr);
    // Edit these objects at their current versions in one checked request
    // that keeps their recorded relationships (editCurrentRequest), on any
    // attachment: `operations` gives each object's operations, read through
    // the attachment. Throws when one is gone, and TransactionConflict when
    // one changed between the read and the commit. nullopt for no objects.
    using EditOperations = std::function<std::vector<TransactionOperation>(const StoreReader&, const std::string& objectUuid)>;
    std::optional<TransactionReceipt> editCurrent(const AttachmentIdentity&, const std::vector<std::string>& objectUuids,
                                                  const EditOperations& operations);
    // The same over each object's current record (name, tags, meta).
    std::optional<TransactionReceipt> editRecords(const AttachmentIdentity&, const std::vector<std::string>& objectUuids,
                                                  const std::function<void(ObjectRecord&)>& change);
    // Add, then remove, tags of these objects.
    void tagObjects(const AttachmentIdentity&, const std::vector<std::string>& objectUuids,
                    const std::vector<std::string>& add, const std::vector<std::string>& remove = {});
    // Copy an object's current record and geometry from one attachment into
    // another as a new object (objects/ObjectCopy.hpp) named `name` (empty: the
    // original's name), in one commit; returns its UUID.
    std::string copyObject(const AttachmentIdentity& from, const AttachmentIdentity& to, const std::string& objectUuid,
                           const std::string& name = {});
    // Bring an object into another attachment as the same object (same UUID):
    // its current record and geometry become its first revision there. Throws
    // when the destination already has the object. An object without local
    // geometry is fetched first.
    void importObject(const AttachmentIdentity& from, const AttachmentIdentity& to, const std::string& objectUuid);
    // A long job on a writable attachment, e.g. ingestion (objects/GeometryIngest.hpp),
    // run on the caller's thread. It reads the store outside any transaction and
    // writes only through `commit`: each call is one transaction, serialized with
    // the attachment's actions and unchecked, so the job guards its own writes.
    // `cancelled` is set when the attachment detaches, which waits for the job.
    using Commit = std::function<void(const Action&)>;
    using Job = std::function<void(ObjectStore&, const Commit& commit, const std::atomic_bool& cancelled)>;
    void writeJob(const AttachmentIdentity&, const Job&);
    // Queries across attachments in `order`: an object (by UUID) comes from the
    // first attachment holding it, so an imported copy hides the catalog one.
    // With a selection limit, each attachment contributes up to `limit`
    // records that are not hidden: it is read in further pages when earlier
    // attachments hide some, so a short result means there are no more.
    std::vector<AttachedRecord> selectObjects(const std::vector<AttachmentIdentity>& order, const ObjectSelection&);
    std::vector<AttachedRegion> geometryRegion(const std::vector<AttachmentIdentity>& order, const RegionSlice&,
                                               const RegionOptions&, const RegionCancel& = {});
    std::vector<AttachedHit> nearestSurfaces(const std::vector<AttachmentIdentity>& order, const SurfaceProximityQuery&);
    // The same around each of `points` (the query's own point is ignored), in
    // one read per attachment: the hits per point.
    std::vector<std::vector<AttachedHit>> nearestSurfaces(const std::vector<AttachmentIdentity>& order,
                                                          const SurfaceProximityQuery&,
                                                          const std::vector<std::array<double, 3>>& points);
    // Drafts: surface geometry not committed anywhere, held in this session's
    // memory only (never written, synced or seen by another session). A draft
    // stands in for its object's committed geometry, in every attachment, in
    // this session's geometryRegion and nearestSurfaces, which treat it by the
    // same tile and triangle rules as stored surfaces; an object need not exist
    // yet. Runs and hits from a draft have an empty revisionUuid and report the
    // draft's attachment (an empty identity when it has none). Drafts are
    // filtered like stored surfaces by uuids, coordinates and volume; a volume
    // target places a draft through the query's transforms, else those of its
    // attachment, else of the first attachment queried. Each change is a Draft event.
    // Set (or replace) the draft of `objectUuid`: its whole grid, the coordinates
    // it is declared in (geometryCoordinates) and the attachment it belongs to.
    // `base`, when given, is the committed version the draft edits (and
    // clears a recorded conflict: an explicit reapplication); otherwise the
    // draft keeps the base it had.
    void setDraft(const std::string& objectUuid, const vc::geometry::Surface&, const CoordinateJson& coordinates,
                  const std::optional<AttachmentIdentity>& attachment = std::nullopt,
                  const std::optional<ObjectVersion>& base = std::nullopt);
    // Replace a rows x columns rectangle (row-major) of a draft's vertices at
    // row, column, and their valid channel unless `validity` is empty.
    void updateDraft(const std::string& objectUuid, int64_t row, int64_t column, int64_t rows, int64_t columns,
                     std::span<const vc::geometry::SurfacePoint> points, std::span<const uint8_t> validity = {});
    // Drop a draft (e.g. after committing it, or abandoning the edit). No-op without one.
    void clearDraft(const std::string& objectUuid);
    bool hasDraft(const std::string& objectUuid) const;
    // Changes with every set or update of the object's draft (0 without one),
    // so a reader can keep what it cut from a draft until the draft changes.
    uint64_t draftRevision(const std::string& objectUuid) const;
    std::optional<DraftState> draftState(const std::string& objectUuid) const;
    std::optional<DraftContents> draftContents(const std::string& objectUuid) const;
    // The draft at `revision` holds its base's committed geometry (e.g. a
    // surface just opened from the store): checkpoints leave it out until it
    // changes. No-op when the draft has moved past `revision` or is gone.
    void markDraftSaved(const std::string& objectUuid, uint64_t revision);

    // Durable drafts. Checkpoints of every draft and pending submission go to
    // `directory` (one session at a time); a checkpoint rewrites only drafts
    // changed since the last one and replaces the previous one atomically, so
    // a crash or failed write keeps the previous checkpoint. Call it once,
    // before making drafts: it installs the drafts of the directory's last
    // checkpoint (in the attachments of their stores, when attached) and its
    // pending submissions. Edits since that checkpoint are lost.
    DraftRecovery useDraftDirectory(const std::filesystem::path& directory);
    // Write a checkpoint now; returns its sequence. Throws when it cannot
    // (disk full, ...); the error stays in draftCheckpointError() until a
    // checkpoint succeeds. Scheduling is the application's.
    uint64_t checkpointDrafts();
    std::string draftCheckpointError() const;
    // A checked save that survives a crash: with a draft directory, the exact
    // request and its payloads are written, and a checkpoint completed, before
    // it is sent. Success clears each of `drafts` that did not change after
    // the call started; a changed one stays, based on its committed version.
    // A conflict keeps the drafts and records it (draftState().conflict);
    // nothing is resent automatically. When a hosted store cannot be reached
    // the outcome is unknown and the submission stays pending.
    TransactionReceipt submitDurably(const AttachmentIdentity&, const TransactionRequest&,
                                     const std::vector<std::string>& drafts = {});
    TransactionReceipt submitDurably(const AttachmentIdentity&, const TransactionRequest&, const DurableSave&);
    std::vector<PendingSubmission> pendingSubmissions() const;
    // A pending submission's outcome: its saved receipt when it committed,
    // else the identical request sent again (the same rules as submitDurably).
    TransactionReceipt resolvePending(const AttachmentIdentity&, const std::string& requestUuid);
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
    void submitCopy(const AttachmentIdentity& to, const ObjectCopy&);
    struct Attachment;
    std::shared_ptr<Attachment> open(const std::filesystem::path&, const OpenOptions&, const HubSessionOptions&);
    std::shared_ptr<Attachment> find(const AttachmentIdentity&) const;
    std::shared_ptr<Attachment> acquire(const AttachmentIdentity&, bool write);
    void release(Attachment&, bool write);
    void retire(const std::shared_ptr<Attachment>&);
    void observe(Attachment&);
    void hubChanged(const Attachment*);
    void startFetching(Attachment&);
    // Runs `f(attachment)` while the attachment is admitted.
    template <class F> auto withAttachment(const AttachmentIdentity&, bool write, F&& f);
    // The UUIDs among `uuids` that the attachment (or one of them) holds.
    std::set<std::string> holds(const AttachmentIdentity&, const std::set<std::string>& uuids);
    std::set<std::string> heldBy(const std::vector<AttachmentIdentity>&, const std::set<std::string>& uuids);

    // transition_ serializes lifecycle operations and is never held by read
    // callbacks. mutex_ protects the attachment list, admission and events.
    std::mutex transition_;
    mutable std::mutex mutex_;
    std::condition_variable idle_;
    std::vector<std::shared_ptr<Attachment>> attached_;
    uint64_t generation_ = 0;
    std::vector<StoreSessionEvent> events_;
    using Drafts = std::map<std::string, std::shared_ptr<const detail::SurfaceDraft>>;
    // Guarded by mutex_; queries take a snapshot.
    Drafts drafts_;
    uint64_t draftRevision_ = 0;
    // Guarded by mutex_.
    std::map<std::string, GeometryFetcher> fetchers_;
    Drafts drafts() const;
    void draftChanged(const std::string& objectUuid, const std::optional<AttachmentIdentity>&);
    std::unique_ptr<detail::DurableDrafts> durable_;
    TransactionReceipt sendPending(const AttachmentIdentity&, const TransactionRequest&, const std::string& requestUuid);
    // Applies a pending submission's outcome (receipt, or the reason nothing
    // committed) to its drafts and forgets it.
    void conclude(const std::string& requestUuid, const TransactionReceipt*, const std::string& conflict);
};
}
