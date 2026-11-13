#pragma once
#include "vc_object_store/objects/SurfaceTileCache.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"

// SQLite object database with immutable geometry and relationship metadata.

#include "vc_object_store/objects/Geometry.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/objects/Transactions.hpp"
#include "vc_object_store/objects/Extensions.hpp"
#include <nlohmann/json_fwd.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {

// Editable ingestion summaries; putGeometry() leaves them unchanged. For live
// shapes/scale use currentGeometry(); for bounds use selectObjects()/spatialRegions().
struct FiberDetail {
    std::string family;          // "H", "V", or "" (unset)
    bool manualFamily = false;
    int64_t controlPointCount = 0;
    int64_t linePointCount = 0;
    double length = 0.0;       // full polyline arclength in source voxels
    double width = 0.0;
    double widthGapFraction = 0.2;
    std::string optimizationMode;
    std::optional<std::array<double, 6>> bbox;  // minx,miny,minz,maxx,maxy,maxz
};

struct SurfaceDetail {
    int64_t gridWidth = 0;
    int64_t gridHeight = 0;
    std::array<double, 2> scale{0.0, 0.0};  // column and row scale, as in tifxyz meta.json
    // The grid's area in stored units squared (geometry::surfaceGridArea over
    // its valid vertices). The store keeps it with the geometry head; a record
    // write keeps the stored value.
    double area = 0.0;
    std::optional<std::array<double, 6>> bbox;
};

struct TrackDetail {
    std::string datasetRef;    // identity component: (datasetRef, externalKey) is unique
    std::string externalKey;   // key within the external dataset
    std::string family;
    int64_t pointCount = 0;
    double length = 0.0;       // polyline arclength in source dataset voxels
    std::optional<std::array<double, 6>> bbox;
};

// Versions, commit stamps and approval are library-managed and ignored on write.
struct RecordCommon {
    std::string uuid;  // canonical text; empty on create = mint a new one
    std::string kind;  // required, non-empty
    int64_t commitSequence = 0; // Store commit which last changed this record.
    std::string createdBy, createdAt, createdHost;
    std::string changedBy, changedAt, changedHost;
    std::string approvalState = "pending";  // pending | approved | rejected
    std::optional<std::string> approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedVersion;
    std::vector<std::string> tags;
    std::string meta = "{}";  // JSON object text
    // Current numbered version; present for every persisted record, including
    // history-off stores and records edited in the current transaction.
    std::optional<int64_t> version;
    std::string versionGeneration;
    // Runtime/wire owner of a selected state, not a persistent row field.
    std::string versionStoreUuid;
};

struct ObjectRecord : RecordCommon {
    // Library-managed lifecycle, independent of ordinary JSON metadata.
    bool retired = false;
    std::string lifecycleEpoch;
    std::string name;
    int64_t formatVersion = 0;
    std::string storageKind;  // e.g. "file", "dataset", "" (db-only)
    std::string storageRef;   // location hint relative to DB directory, or dataset URI; uuid is identity
    std::optional<std::string> dataChecksum;
    std::optional<FiberDetail> fiber;
    std::optional<SurfaceDetail> surface;
    std::optional<TrackDetail> track;
    // Measured query-frame length for length-ordered selections: native units
    // without a volume target, target voxels otherwise. Legacy field name;
    // use this for afterLength, not the persisted ingestion summary. Unbound
    // volume tails have no measured length.
    std::optional<double> volumeLength;
    ObjectVersion reference() const;
};

struct PointReference {
    // Geometry array and point coordinates, excluding the final XYZ axis:
    // control_points:[5], line_points:[12], points:[20,145] (surface row,col).
    // point_ids:[42] addresses stable point ID 42 in a point_collection.
    std::string array;
    std::vector<int64_t> index;
    bool operator==(const PointReference&) const = default;
};

struct Anchor {
    std::optional<std::array<double, 3>> position;
    std::optional<std::array<double, 3>> direction;
    // Include interpolation/direction neighbours. Empty means unknown dependencies.
    std::vector<PointReference> points;
    bool wholeGeometry = false; // explicit object-wide relationship
    // Dependent top-level metadata keys; edits require explicit relationship decisions.
    std::vector<std::string> metadataKeys;
    // Whole named surface channels, including presence, layout and values.
    std::vector<std::string> channels;
    bool operator==(const Anchor&) const = default;
};

struct EdgeRecord : RecordCommon {
    RelationshipVersion reference() const;
    std::string aUuid, bUuid;  // endpoint order is normalised by addEdge
    Anchor aAnchor, bAnchor;
    // Canonical opposite endpoint; rejects UUIDs not incident to this edge.
    std::string otherEndpoint(const std::string& endpoint) const;
};

// Object UUID -> internal object revision UUID (storage keys; public history
// references are numbered ObjectVersions). null means no object.
using ObjectRevisionKey = std::variant<std::string, ObjectVersion>;
using RevisionSelection = std::map<std::string, std::optional<ObjectRevisionKey>>;

struct ObjectRevision {
    std::string uuid, objectUuid;
    std::optional<std::string> parentUuid, geometryRevision;
    int64_t sequence = 0;
    // False for baselines imported without their original metadata history
    // (recorded by stores converted from hub replicas).
    bool metadataComplete = true;
    // UTC Unix microseconds assigned once per atomic publication, monotonically
    // increasing within this store. Missing only for those untimed baselines.
    std::optional<int64_t> publishedAtUs;
    bool deleted = false;
    // Numbered history: (store, generation, objectUuid, version) is this
    // revision's public reference. The generation is the lineage it was
    // published in. commitSequence is absent for history converted without a
    // recorded publication (e.g. untimed imported baselines).
    int64_t version = 0;
    std::string generation;
    // Origin of this exact state, independent of a newer visible overlay.
    // Runtime/wire identity, not another per-record storage column.
    std::string storeUuid;
    std::optional<int64_t> commitSequence;
    ObjectRecord record;
    // Construct a reference in this state's owning store/lineage, never the
    // reader's attachment. Optional numbers select another retained version
    // in that owner; use a full ObjectVersion to name a different owner.
    ObjectVersion reference(std::optional<int64_t> selectedVersion = std::nullopt,
                            const std::optional<std::string>& selectedGeneration = std::nullopt) const;
};

// Edge kind tying a data object to the object it describes (e.g. fit results
// attached to a surface). Attachments are not graph links: traversal
// (componentOf, reachableWithin, connectedTo), neighbours() and graph read views
// skip them unless the attachment kind is requested explicitly. Edge listings
// and exports include them. Point collections attached this way carry a record
// meta "role" and are never linked to surfaces by the point-collection linker.
inline constexpr const char* kAttachmentEdgeKind = "attachment";
// The relationship attaching object `data` to `owner` in `role` (ObjectStore::attach); no UUID.
EdgeRecord attachmentEdge(const std::string& dataUuid, const std::string& ownerUuid, const std::string& role);

// Edge changes revise both endpoints atomically; unchanged membership is inherited.
// Endpoint pins are complete immutable references, never moving heads.
struct RelationshipRevision {
    std::string uuid;
    EdgeRecord record;
    std::string aRevision, bRevision; // archive storage UUIDs only; empty for current-only endpoints
    std::optional<std::string> aGeometryRevision, bGeometryRevision;
    // Numbered history of the relationship and of its pinned endpoint states.
    int64_t version = 0;
    std::string generation;
    // Resolved endpoint identities for consumers; no parsing of internal revision tokens.
    ObjectVersion aReference, bReference;
    RelationshipVersion reference() const;
};

struct SelectedRelationship {
    RelationshipRevision relationship;
    // Both selected endpoints contain this version and its dependencies still match.
    bool applicable = false;
};

// One selected graph input: exact records plus the relationships between them.
struct SelectedInputs {
    std::vector<ObjectRevision> objects;
    std::vector<SelectedRelationship> relationships;
};

struct JournalEntry {
    int64_t seq = 0;
    std::string recordUuid;
    std::string kind; // the object or edge kind, recorded at the change
    int64_t commitSequence = 0;
    std::string author, hostname, timestamp, changeKind;
    // Approval snapshot after this change; absent for deletion events.
    std::optional<std::string> approvalState, approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedVersion;
};

enum class HistoryMode { Off, On };

struct StoreProvisioning {
    int schemaVersion = 22;
    int64_t commitSequence = 0;
    std::string generation; // Publication lineage UUID.
    HistoryMode history = HistoryMode::On;
    bool sealed = false;
    bool spatialIndex = true;
};

// Physical identities are distinct from the logical store UUID. Moving or
// copying a database preserves them; independently created stores do not.
struct TableIdentity {
    std::string uuid, name;
    std::vector<std::string> primaryKey;
};

struct DatabaseIdentity {
    std::string uuid;
    int schemaVersion = 0;
    std::vector<TableIdentity> tables;
};

struct OpenOptions {
    // Creation only. Opening reads the persistent policy, never overrides it.
    HistoryMode history = HistoryMode::On;
    // Creation only. Omit bounds/R-tree storage; spatial queries then throw.
    // Ordinary geometry, metadata and graph operations remain available.
    bool spatialIndex = true;
    // Creation only: optional adjacency accelerators and explicit edge layout.
    bool edgeAdjacencyIndex = true;
    // nullopt selects all columns; a list selects optional columns in addition
    // to uuid/a_uuid/b_uuid/kind/commit_sequence. Unsupported writes fail.
    std::optional<std::vector<std::string>> edgeColumns;
    // Creation-only provisioning for independently built shards. Integer key
    // ranges are reservations, not counts of currently populated records.
    std::optional<std::string> creationStoreUuid;
    int64_t initialCommitSequence = 0;
    std::optional<std::string> creationGeneration;
    std::array<int64_t,2> objectIdRange{1,INT64_MAX};
    // Explicit low-level access to one physical partition. Ordinary opens of
    // collections fail rather than silently return only locally stored data.
    bool physicalPartitionOnly = false;
    std::shared_ptr<const vcr::catalog::RegistrationGraph> catalogTransforms;
    bool readOnly = false;
    int readerPoolSize = 4;
    int busyTimeoutMs = 5000;
    std::string author;  // required for read-write opens
    std::optional<std::string> hostnameOverride;  // tests / reproducible imports only
    // Retained verified geometry per pooled read connection; zero disables it.
    // Payload bytes plus conservatively charged metadata, excluding active reads.
    std::size_t geometryCacheBytes = 16 * 1024 * 1024;
    // SQLite page-cache budget per connection (not total process memory).
    // Preserves the historical PRAGMA cache_size=-8000 default. Collections
    // apply this to dependency connections too; multiply by workers/stores.
    std::size_t sqliteCacheBytes = 8000 * 1024;
    // Resident spatial descriptors/index per read connection; zero uses SQL.
    // Built lazily; snapshots too large for this budget also use SQL.
    std::size_t spatialCacheBytes = 16 * 1024 * 1024;
    // Decoded surface tiles shared by all connections (and by other stores
    // given the same cache); none: every read decodes its tiles.
    std::shared_ptr<SurfaceTileCache> surfaceTiles;
    // An edit whose sparse delta would exceed this many raw bytes is published
    // as a full revision instead (unchanged chunks are shared by hash). Capped
    // at 256 MiB, the largest delta read back.
    std::size_t maxGeometryDeltaBytes = 256 * 1024 * 1024;
    // Open as the store's hosting service: exclusive host-local ownership.
    // Fails while any standalone writable handle is open, and standalone
    // writable opens fail while the service holds it (see ObjectStore).
    bool serviceOwner = false;
    // Called (on the reading thread, inside the read) with the surfaces a
    // geometryRegion or nearestSurfaces query reached whose geometry is not
    // local yet. Must only queue work; ObjectStoreSession sets it on writable attachments.
    std::function<void(const std::vector<std::string>&)> remoteGeometryReached;
    // Store_reference dependency relocation only; never changes the root locator.
    // After the recorded path, try these files in order, then regular files in
    // each nonrecursive directory in sorted filename order. Relative search
    // paths are relative to the process working directory. Candidates must
    // match the pinned database/store UUIDs, generation and readonly policy.
    // Packed container files are supported and select the pinned member UUID.
    std::vector<std::filesystem::path> storeReferenceFiles;
    std::vector<std::filesystem::path> storeReferenceDirectories;
};

// Validate projected top-level field names without selecting data. Exact graph
// materializers and SQL selections use the same native field policy.
void validateObjectFields(const std::optional<std::vector<std::string>>&);
// SHA-256 over domain + newline, followed by UUID-ordered complete owning
// references as "store generation object version\n". No geometry is read.
std::string objectReferencesDigest(std::span<const ObjectVersion>, const std::string& domain);
void validateEdgeFields(const std::optional<std::vector<std::string>>&);

struct ObjectSelection {
    std::optional<std::string> kind, tag, connectedTo;
    std::vector<std::string> uuids, edgeKinds;
    std::optional<int64_t> limit; // UUID-ordered page; absent means all matches
    int64_t offset = 0;
    // Inclusive object AABB overlap in source XYZ (minx,miny,minz,maxx,maxy,maxz).
    // Geometry heads determine bounds; un-ingested objects use annotated boxes.
    // Objects with no bounds do not match. This is not an exact mesh intersection.
    std::optional<std::array<double, 6>> bbox;
    // Typed filters require kind="fiber" or "track". Length uses source units
    // and the ingested summary, not a live geometry measurement.
    std::optional<std::string> family;
    std::optional<double> minLength;
    std::string orderBy = "uuid"; // or "length_desc", UUID tie break
    std::optional<std::string> afterUuid;
    std::optional<double> afterLength; // paired with afterUuid for length_desc
    bool includeRetired = false;
    // Exact declared coordinate system (JSON volume_uuid/scale_factor).
    // Missing declarations compare as null; absent filter matches all frames.
    std::optional<std::string> coordinates;
    // Explicit target volume, exclusive with coordinates. Includes compatible
    // registered frames. Bounds and length filters use target voxels; affine
    // lengths are measured from the transformed polyline. Returned object
    // records retain their canonical metadata. Spatial regions use target bounds.
    std::optional<std::string> volume;
    // With volume: target voxels at this scale of the volume's (as in RegionOptions).
    double targetScale = 1;
    // With volume: also list objects not bound to it, after the bound ones in
    // UUID order (a cursor into them has afterUuid without afterLength).
    // Excluded when minLength is positive. Not valid for spatialRegions.
    bool includeUnbound = false;
    std::shared_ptr<const VolumeTransforms> transforms;
    // Objects with at least one tag of every group (`tag` is a group of one)
    // and none of `withoutTags`.
    std::vector<std::vector<std::string>> tagGroups;
    std::vector<std::string> withoutTags;
};
struct EdgeSelection {
    std::optional<std::string> kind, tag;
    std::vector<std::string> approvalStates; // empty means all states
    // nullopt means unrestricted; an explicitly empty set matches no edges.
    std::optional<std::vector<std::string>> endpointUuids, uuids;
    bool bothEndpoints = false; // induced subgraph vs incident edges
    std::optional<int64_t> limit;
    std::optional<std::string> afterUuid; // exclusive, UUID-ordered cursor
    bool includeRetired = false; // endpoint filter; retirement removes current edges
    // nullopt: all kinds; false/true: ordinary/adjacent fiber_branch only.
    // A missing adjacent metadata flag is an ordinary fiber branch.
    std::optional<bool> adjacent;
    // Shared membership/dependencies against latest heads. Requires a ReadTxn.
    bool applicableOnly = false;
    // false skips attachment edges (graph read views do unless kind asks for them).
    bool attachments = true;
};

// Snapshot topology in selection order. Every edge, including self/parallel edges,
// retains its identity and has two CSR incidences.
struct GraphReadView {
    std::vector<std::string> objectUuids, edgeUuids;
    std::vector<std::optional<std::string>> geometryRevisions;
    std::vector<int64_t> versions;
    std::vector<uint64_t> offsets, neighbours, edgeIndices;
    std::vector<std::array<uint64_t, 2>> endpoints;
    int64_t journalSequence = 0;
    // Present for selected-revision graphs; records align with nodes/edges.
    // These retain historical metadata and pins, including deleted current records.
    std::optional<std::vector<ObjectRevision>> selectedObjects;
    std::optional<std::vector<RelationshipRevision>> selectedRelationships;
    std::vector<std::string> reachableWithin(const std::string& objectUuid, int maxDepth,
                                            const std::atomic_bool* cancel = nullptr) const;
    std::vector<std::string> componentOf(const std::string& objectUuid,
                                       const std::atomic_bool* cancel = nullptr) const;
};

// Homogeneous arrays concatenated along their first dimension. Offsets index
// that dimension (not bytes); values retain their stored dtype and coordinates.
struct PackedGeometryArray {
    std::string dtype = "|u1";
    std::vector<int64_t> shape{0};
    std::vector<uint64_t> offsets{0};
    std::vector<uint8_t> data;
};
// Conservative bounds from one snapshot. start/shape index vertices, excluding XYZ;
// adjacent tiles/blocks share boundary vertices. Empty arrayName/start/shape means
// an external object's annotated box.
struct SpatialRegion {
    std::string objectUuid, kind, arrayName;
    std::optional<std::string> revisionUuid;
    std::vector<int64_t> start, shape;
    std::array<double, 6> bbox;
};
struct JsonExportOptions {
    ObjectSelection selection;
    std::vector<std::string> fields; // top-level record fields; UUIDs/endpoints always retained
};

class ObjectStore;
class GeometrySnapshot;
struct ObjectStoreImplAccess;

class Txn {
public:
    Txn(const Txn&) = delete;
    Txn& operator=(const Txn&) = delete;
    Txn(Txn&& other) noexcept;
    Txn& operator=(Txn&&) = delete;
    ~Txn();
    void commit();
    bool active() const { return handle_ != nullptr; }
    // Numbered receipt of a committed write: commit sequence and versions,
    // captured before releasing the writer (a later commit cannot change it).
    const std::optional<TransactionReceipt>& receipt() const { return receipt_; }

protected:
    Txn(const ObjectStore* db, void* handle, bool write) : db_(db), handle_(handle), write_(write) {}
    void release(bool commit);

private:
    friend class ObjectStore;
    friend struct ObjectStoreImplAccess;
    const ObjectStore* db_ = nullptr;
    void* handle_ = nullptr;  // sqlite3*
    bool write_ = false;
    std::shared_ptr<void> referenceContext_;
    RevisionSelection expectedRevisions_;
    std::optional<TransactionReceipt> receipt_;
    VersionSelection expectedObjectVersions_, expectedRelationshipVersions_;
};

class ReadTxn : public Txn {
public:
    ReadTxn(const ObjectStore* db, void* handle) : Txn(db, handle, false) {}
};

class WriteTxn : public Txn {
public:
    WriteTxn(ObjectStore* db, void* handle) : Txn(db, handle, true) {}
};

struct EdgeGeometryState {
    std::optional<std::string> aRevision, bRevision;
    bool stale = false;
};

// Snapshot repair in edgeByUuid() endpoint order. Null revisions mean no geometry.
struct EdgeGeometryRepair {
    std::string edgeUuid;
    int64_t expectedVersion = 0;
    std::optional<std::string> aRevision, bRevision;
    Anchor aAnchor, bAnchor;
};

// A GeometryEdgeEdit with this edgeUuid and action "keep_recorded" decides
// every affected relationship without its own decision: each is kept as
// recorded (from the edit's checked state).
inline constexpr const char* kKeepAffectedRecorded = "*";
struct GeometryEdgeEdit {
    std::string edgeUuid;
    int64_t expectedVersion = 0;
    // "retain", "update", "remove", or "keep_recorded". Retain keeps the same
    // relationship and anchors; update supplies both complete anchors in
    // stored a/b order. Revisions are the heads observed BEFORE the edit.
    // keep_recorded preserves historical pins and removes the current edge;
    // it does NOT assert applicability to the new geometry and requires history.
    std::string action;
    std::optional<std::string> aRevision, bRevision;
    Anchor aAnchor, bAnchor;
};

// Primary database for objects, geometry, and relationships between objects.
// One writer and a bounded reader pool per handle. Transactions are thread-confined
// and must finish before their handle is moved/destroyed. Do not nest writes or
// checkpoint while a write is active; decode external files before starting writes.
class ObjectStore {
public:
    static ObjectStore create(const std::filesystem::path&, const OpenOptions&);
    StoreProvisioning provisioning(const Txn&) const;
    // Optional typed tables. Definitions/provenance are immutable; append new
    // result rows (e.g. keyed by run UUID) instead of rewriting old input pins.
    // The extension owns its SQL layout and semantics; core owns registration,
    // transactions and commit stamps. No automatic history of extension rows.
    void createExtensionTable(WriteTxn&, const ExtensionTable&);
    std::vector<ExtensionTableInfo> extensionTables(const Txn&) const;
    void appendExtensionRows(WriteTxn&, const std::string& table, std::span<const ExtensionRow>);
    // Stream in primary-key order; optional equality prefix uses the PK index.
    void scanExtensionRows(const Txn&, const std::string& table,
        const std::function<bool(const ExtensionRow&)>&, std::span<const ExtensionValue> keyPrefix = {},
        const ExtensionKeyFilter& keyFilter = {}) const;
    void replaceExtensionRows(WriteTxn&, const std::string& table, const std::string& ownerUuid, std::span<const ExtensionRow>);
    void scanExtensionRowsAtVersion(const Txn&, const std::string& table, const ObjectVersion&,
        const std::function<bool(const ExtensionRow&)>&) const;
    void scanExtensionRowsAtVersion(const Txn&, const std::string& table, const RelationshipVersion&,
        const std::function<bool(const ExtensionRow&)>&) const;
    DatabaseIdentity databaseIdentity(const Txn&) const;
    // Irreversible publication: future writes (including through already-open
    // handles) are rejected. Referenced stores must have been sealed, not merely
    // opened using readOnly=true.
    void seal(WriteTxn&);
    static ObjectStore open(const std::filesystem::path&, const OpenOptions&);
    ObjectStore(ObjectStore&&) noexcept;
    ObjectStore& operator=(ObjectStore&&) noexcept;
    ObjectStore(const ObjectStore&) = delete;
    ObjectStore& operator=(const ObjectStore&) = delete;
    ~ObjectStore();
    ReadTxn read() const;
    VolumeTransforms volumeTransforms(const Txn&) const;
    std::vector<vcr::catalog::TransformResolution> availableTransforms(const Txn&, const std::string& objectUuid) const;
    // Exact polyline length after mapping to target voxels; reads bounded chunks.
    double geometryLengthInVolume(const Txn&, const std::string& objectUuid, const std::string& target, double targetScale=1) const;
    std::string registerVolumeTransform(WriteTxn&, const std::string& source, const std::string& target,
        const VolumeAffine&, const std::string& name, const std::optional<std::string>& uuid = std::nullopt);
    // The route from `source` to `target` (registration keys as a resolution's
    // candidate path). Updates the pair's existing choice, else creates one
    // (as `uuid` when given).
    std::string chooseVolumeTransform(WriteTxn&, const std::string& source, const std::string& target,
        const std::vector<std::string>& path, const std::optional<std::string>& uuid = std::nullopt);
    // Record external catalog registrations as versioned store records
    // (catalogTransformUuid(id), exact coefficients, catalog id and revision
    // evidence). Unchanged registrations are left untouched; others need
    // their expected versions in a checked write. Once recorded, a
    // registration resolves from the store for every client: a client's own
    // catalog copy of that id is ignored by volumeTransforms(). Returns the
    // records created or changed.
    std::vector<std::string> recordCatalogTransforms(WriteTxn&, const std::vector<vcr::catalog::Registration>&);
    WriteTxn write();
    // Checked write expecting these object versions (null: absent); the same as
    // write(TransactionContext) with only expectedObjects. Every changed
    // object/endpoint must be expected. Commit publishes one revision per changed
    // object. Missing expectations or conflicts abort the whole transaction, even if caught.
    WriteTxn write(const RecordSelection& expectedObjects);
    // Explicit numeric input convenience; resolves under this write lock.
    WriteTxn writeNumbered(const VersionSelection&);
    // Numbered checked write (see TransactionContext). A request UUID already
    // committed is rejected here; use receipt() or submit() to recover it.
    WriteTxn write(const TransactionContext&);
    // Execute typed operations atomically. An identical retry of a committed
    // request returns its saved receipt without re-executing; the same request
    // UUID with different contents fails. Conflicts throw TransactionConflict.
    // Prepared operations (link_objects) are prepared first, outside the
    // writer lock, as prepare() then submit(request, prepared) do.
    TransactionReceipt submit(const TransactionRequest&);
    // The preparation of the request's prepared operations (see
    // Transactions.hpp): each runs under its own read snapshot through its
    // registered preparer. Empty when the request has none, or when its
    // receipt is already saved (submit then replays it). Throws
    // std::invalid_argument for a prepared operation with no preparer
    // registered.
    PreparedOperations prepare(const TransactionRequest&, const PrepareOptions& = {});
    // submit() with the preparation already made: the prepared versions join
    // the request's expectations and read dependencies, and each prepared
    // operation publishes through its `apply`.
    TransactionReceipt submit(const TransactionRequest&, const PreparedOperations&);
    // The same, from the canonical JSON form (transactionRequestToJson). A
    // saved receipt of the identical request answers before any payload is
    // needed, so a retry recovers a committed result even after its staged
    // bytes expired. Otherwise every payload must be available or
    // MissingPayloads names those that are not.
    TransactionReceipt submit(const std::string& requestJson, const PayloadLookup&);
    std::optional<TransactionReceipt> receipt(const Txn&, const std::string& requestUuid) const;
    StoreStatus status(const Txn&) const;
    // Changes after the cursor, whole commits in sequence order, at most
    // maxChanges records unless one commit alone exceeds it (then continued).
    ChangePage changesSince(const Txn&, const FeedCursor&, int64_t maxChanges = 4096) const;
    std::optional<int64_t> currentVersion(const Txn&, const std::string& objectUuid) const;
    std::optional<int64_t> currentRelationshipVersion(const Txn&, const std::string& relationshipUuid) const;
    // Exact numbered reference; nullopt when this store never published it in
    // that generation (never another generation's same-numbered version).
    std::optional<ObjectRevision> objectAtVersion(const Txn&, const ObjectVersion&) const;
    std::optional<RelationshipRevision> relationshipAtVersion(const Txn&, const RelationshipVersion&) const;
    // objectAtVersion for many references in one snapshot, in order (nullopt
    // where a reference is not recorded). Versions that are still the head
    // read the materialized head record without replaying history.
    std::vector<std::optional<ObjectRevision>> objectsAtVersions(const Txn&, const std::vector<ObjectVersion>&) const;
    // Restore tooling: start a fresh, never reused history generation for new
    // writes. Recorded versions keep their publication generations.
    std::string startHistoryGeneration(const std::string& reason);
    std::optional<ObjectRevision> currentObjectRevision(const Txn&, const std::string& objectUuid) const;
    // ifRetained returns no entries when the winning record's owner has history
    // disabled. The default explicit history request still rejects that case.
    std::vector<ObjectRevision> objectHistory(const Txn&, const std::string& objectUuid, bool ifRetained = false) const;
    // The version of every object live at/before this publication time
    // (UTC Unix microseconds), ordered by object UUID; actions are indivisible.
    // Empty before the first publication; excludes deletions.
    std::vector<ObjectVersion> versionsAt(const Txn&, int64_t publishedAtUs, bool includeRetired = false) const;
    std::vector<ObjectVersion> versionsAt(const Txn&, const ObjectVersion& anchor) const;
    // Relationships recorded between the selected objects, each at an exact
    // recorded version of this store (one per object; omitted endpoints are
    // not selected; deleted versions and unknown references fail). Results are
    // ordered by relationship UUID, then complete publication reference.
    std::vector<SelectedRelationship> relationshipsAt(const Txn&, const std::vector<ObjectVersion>&) const;
    // After endpoint edits, validate anchors/metadataKeys and retain the relationship.
    // The caller decides domain meaning.
    void retainRelationship(WriteTxn&, const std::string& edgeUuid);
    // Keep the relationship and its historical pins after an endpoint edit,
    // without asserting applicability to the edited endpoint. Current queries
    // can filter out the now-inapplicable relationship. Requires history.
    void keepRecordedRelationship(WriteTxn&, const std::string& edgeUuid);
    // keepRecordedRelationship for every current relationship of the object
    // (e.g. after a record-only edit of it).
    void keepRecordedRelationshipsOf(WriteTxn&, const std::string& objectUuid);
    const std::filesystem::path& path() const;
    std::string hostname() const;
    std::string author() const;

    // Stable store identity, independent of its filesystem location.
    std::string uuid(const Txn&) const;

    // Create/update by UUID (mint one if empty). Content changes advance the
    // ordinary record version once per transaction, stamp changed_*, reset
    // approval and journal when history is retained. Returns the UUID.
    std::string upsertObject(WriteTxn& txn, const ObjectRecord& record);
    // Import-only explicit local row ID, within this store's reserved range.
    // UUID remains the portable identity. Existing UUIDs cannot be renumbered.
    std::string insertObjectWithId(WriteTxn&, const ObjectRecord&, int64_t id);
    std::optional<ObjectRecord> objectByUuid(const Txn& txn, const std::string& uuid) const;
    std::vector<ObjectRecord> objectsByKind(const Txn& txn, const std::string& kind,
                                            const std::optional<std::string>& approvalState = std::nullopt, bool includeRetired = false) const;
    int64_t objectCount(const Txn& txn, const std::string& kind, bool includeRetired = false) const;
    // Active objects of kind that a volume selection lists as unbound (no
    // finite positive scale declared for volume). Reads only the unbound index
    // ranges, so a store bound to one volume is not scanned.
    int64_t unboundObjectCount(const Txn&, const std::string& kind, const std::string& volume) const;
    std::vector<ObjectRecord> objectsByTag(const Txn& txn, const std::string& tag, bool includeRetired = false) const;
    // Distinct tags on active objects, optionally of one kind, sorted.
    std::vector<std::string> objectTags(const Txn&, const std::optional<std::string>& kind = std::nullopt) const;
    // Require write(expected). Decisions use keepRecordedRelationship,
    // updateEdge/retainRelationship, or removeEdge before commit. UUID lookup
    // always includes retired objects. Restore copies content into a new revision;
    // omitted source selects the last active revision. Links need explicit validation.
    void retireObject(WriteTxn&, const std::string& uuid);
    void restoreObject(WriteTxn&, const std::string& uuid,
                       const std::optional<ObjectVersion>& source = std::nullopt);
    void deleteObject(WriteTxn& txn, const std::string& uuid);  // fails while edges or retained geometry reference it

    // Create geometry, or replace a surface; sequences use structural/sparse edits.
    // Creation consumes source point annotations into the immutable revision.
    // Null expectedRevision expects no geometry; stale writers fail. Every
    // affected edge requires an explicit retain/update/remove decision.
    std::string putGeometry(WriteTxn& txn, const std::string& objectUuid,
                            const GeometryData& data,
                            const std::optional<std::string>& expectedRevision = std::nullopt,
                            const std::vector<GeometryEdgeEdit>& edgeEdits = {},
                            const GeometryStorageOptions& storage = {});
    // Preview changed point dependencies without writing. Unknown dependencies
    // require explicit update or removal.
    std::vector<EdgeRecord> affectedGeometryEdges(const Txn&, const std::string& objectUuid,
                                                const GeometryData&) const;
    // Sparse, lossless, compressed edits. One call publishes one revision in
    // the caller's transaction. expectedRevision is mandatory, even for no-ops.
    // No automatic checkpoints. Use editGeometrySequence/resizeSurface for structure.
    std::string editGeometry(WriteTxn&, const std::string& objectUuid,
                             const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision,
                             const std::vector<GeometryEdgeEdit>& edgeEdits = {});
    // As above, choosing the decisions for the relationships the edit affects
    // from the same plan (affectedGeometryEditEdges + editGeometry plans twice).
    using EdgeDecider = std::function<std::vector<GeometryEdgeEdit>(const std::vector<EdgeRecord>&)>;
    std::string editGeometry(WriteTxn&, const std::string& objectUuid, const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision, const EdgeDecider& decide);
    std::vector<EdgeRecord> affectedGeometryEditEdges(const Txn&, const std::string& objectUuid,
                             const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision) const;
    std::optional<GeometryRevision> currentGeometry(const Txn& txn, const std::string& objectUuid) const;
    // The current geometry revision (payload identity; nullopt without
    // geometry) of an object checked to be at `expectedVersion`: the
    // expectation the geometry mutations take. Any other version is a history
    // conflict, which aborts a checked write.
    std::optional<std::string> geometryAtVersion(const Txn&, const std::string& objectUuid, int64_t expectedVersion) const;
    std::optional<GeometryRevision> geometryRevision(const Txn& txn, const std::string& revisionUuid) const;
    // Latest retained metadata for this geometry, preferring the visible head
    // when it still names it. Explicit ObjectVersion reads select metadata exactly.
    std::optional<ObjectRevision> objectAtGeometry(const Txn&, const std::string& revisionUuid) const;
    std::vector<GeometryRevision> geometryHistory(const Txn& txn, const std::string& objectUuid) const;
    // Byte ranges allow bounded loading without materializing a whole array.
    // Every touched chunk is verified against its SHA-256 content address.
    std::vector<uint8_t> readGeometryArray(const Txn& txn, const std::string& revisionUuid,
        const std::string& arrayName, int64_t byteOffset = 0,
        const std::optional<int64_t>& byteCount = std::nullopt) const;

    // Batched byte ranges [offset,count], in caller order, sharing verified chunks.
    std::vector<std::vector<uint8_t>> readGeometryRanges(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const;

    // Descriptor-only catalog in selection order. No coordinate chunks are read.
    // The UUID/count columns belong to this transaction, not edit references.
    struct GeometryArrayCounts {
        std::vector<std::array<unsigned char,16>> uuids;
        std::vector<int64_t> counts; // first dimension; zero for absent arrays
    };
    GeometryArrayCounts geometryArrayCounts(const Txn&, const std::string& objectKind,
                                           const std::string& arrayName,
                                           const std::optional<int64_t>& limit=std::nullopt) const;
    // Streaming current visible-reference digest in one snapshot, including
    // source ownership across overlays/shards. Explicit UUID requests must all
    // exist in the selection. Selection ordering must be UUID ascending.
    std::string selectedReferenceDigest(const Txn&, const ObjectSelection&,
                                        const std::string& domain) const;

    // One packed buffer, caller revision order (including duplicates).
    // Dtype and trailing dimensions must match. All chunks remain verified.
    PackedGeometryArray readGeometryArrays(const Txn&, const std::vector<std::string>& revisionUuids,
        const std::string& arrayName) const;
    // Captured selected metadata and payload identities stay paired. Optional
    // mapping currently supports fiber line_points and returns float64 XYZ.
    PackedGeometryArray readSelectedGeometryArrays(const Txn&, std::span<const ObjectRevision>,
        const std::string& arrayName, const std::optional<std::string>& volume = std::nullopt) const;

    // Owned pinned payload with an explicit aggregate byte bound. Reads are
    // chunked; callers needing larger objects should use ranges/rectangles.
    GeometryData readGeometryData(const Txn&, const std::string& revisionUuid,
        uint64_t byteBudget = 64 * 1024 * 1024) const;

    // Add/replace/remove channels with checked revisions and relationship decisions.
    // Never resize points or read/rehash unchanged arrays. "points" is reserved.
    std::string updateSurfaceChannels(WriteTxn&, const std::string& objectUuid,
        const std::vector<GeometryArray>& channels, const std::vector<std::string>& remove,
        const std::string& expectedRevision, const std::vector<GeometryEdgeEdit>& edgeEdits = {});
    // Dense C-order rectangle edit; dtype/components match the existing array.
    std::string editGeometryRectangle(WriteTxn&, const std::string& objectUuid,
        const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns,
        const std::vector<uint8_t>& bytes, const std::string& expectedRevision,
        const std::vector<GeometryEdgeEdit>& edgeEdits = {});

    // Structural mutations require write(expected object revisions). References
    // to another object's revision require that object's expected head too.
    // Append/insert/delete/replace are splices; reverse/reorder/split/merge use
    // immutable slices. Source retirement is explicit in the same transaction.
    std::string editGeometrySequence(WriteTxn&, const std::string& objectUuid,
        const GeometrySequenceEdit&, const std::string& expectedRevision,
        const std::vector<GeometryEdgeEdit>& edgeEdits = {});
    std::string resizeSurface(WriteTxn&, const std::string& objectUuid,
        const GeometrySurfaceResize&, const std::string& expectedRevision,
        const std::vector<GeometryEdgeEdit>& edgeEdits = {});
    // Aligned annotations for a geometry revision, derived from stable IDs: a
    // JSON array with one value per point.
    nlohmann::json geometryPointAnnotations(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName) const;
    // First unused ID above all retained allocations, including restored-away
    // points and referenced versions. A proposal, not a reservation: commit
    // under an expected object version and ordinary structural validation.
    int64_t nextGeometryPointId(const Txn&, const std::string& objectUuid, const std::string& arrayName) const;
    // The same, or nullopt as soon as the values read exceed maxBytes encoded
    // (reading stops there: at most one value beyond the budget is unpacked).
    std::optional<nlohmann::json> geometryPointAnnotationsWithin(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, size_t maxBytes) const;

    // Dense row-major rectangle from any aligned surface array. Exact
    // bytes; handles tiled and linear layouts. Coordinates/counts are vertices.
    // Rectangle/range reads may reuse verified bytes on a pooled reader until
    // its next snapshot observes a commit. Writes always use fresh readers.
    std::vector<uint8_t> readGeometryRectangle(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns) const;

    // [row,column,rows,columns], in caller order, sharing manifest/chunk reads.
    std::vector<std::vector<uint8_t>> readGeometryRectangles(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, const std::vector<std::array<int64_t, 4>>& rectangles) const;

    // Set pending, approved or rejected; a changed decision advances the
    // ordinary record version once per transaction and stamps approved_version.
    void setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state);

    // Open-ended metadata: single-key atomic updates via SQLite JSON1.
    // Geometry kinds use meta.coordinates = {volume_uuid, scale_factor}.
    // volume_uuid is a catalog volume identifier (or null); scale is positive finite (or
    // null), with stored XYZ = volume-voxel XYZ * scale_factor. Metadata only:
    // neither writes nor import transform geometry. Omission means unknown.
    // jsonValue is JSON text (e.g. "\"reviewed\"", "3", "true").
    // Changed values advance the ordinary version and reset approval; no-ops do neither.
    void setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue);
    void removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key);
    // Declare an object's coordinate frame: stored at `storedUnitsPerVoxel`
    // units per voxel of `volume`, merged into its meta.coordinates, refusing a
    // conflict with what it already declares.
    void bindGeometryCoordinates(WriteTxn&, const std::string& objectUuid, const std::string& volume,
                                 double storedUnitsPerVoxel);

    std::vector<JournalEntry> journalSince(const Txn& txn, int64_t afterSeq) const;
    int64_t journalPosition(const Txn& txn) const;

    EdgeGeometryState edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const;
    // Atomically repair anchors/bindings, checking edge version and endpoint heads.
    // Preserves other metadata, resets approval and journals when history is retained.
    // Identical input is a no-op.
    void repairEdgeGeometry(WriteTxn&, const EdgeGeometryRepair&);


    // Edges. addEdge normalises endpoint order (swapping anchors to
    // match) and enforces uniqueness of the edge's own UUID.
    std::string addEdge(WriteTxn& txn, const EdgeRecord& record);
    // Refresh anchors/tags/meta while retaining UUID, endpoints, kind and history.
    // Changes reset approval and validate the resulting current endpoint dependencies.
    void updateEdge(WriteTxn& txn, const EdgeRecord& record);
    std::optional<EdgeRecord> edgeByUuid(const Txn& txn, const std::string& uuid) const;
    void removeEdge(WriteTxn& txn, const std::string& uuid);
    std::vector<EdgeRecord> edgesOf(const Txn& txn, const std::string& objectUuid, bool includeRetired = false) const;
    std::vector<EdgeRecord> edgesByKind(const Txn& txn, const std::string& edgeKind, bool includeRetired = false) const;
    // Projection always includes uuid/kind/a/b; empty fields means identities.
    std::vector<EdgeRecord> selectEdges(const Txn&, const EdgeSelection& = {},
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const;
    // Attach object `data` to `owner` in `role` (kAttachmentEdgeKind, whole
    // geometry at both ends, the role in the edge meta). Returns the edge UUID.
    std::string attach(WriteTxn&, const std::string& dataUuid, const std::string& ownerUuid, const std::string& role);
    // The attachment relationships of `owner`, in `role` when given, in UUID
    // order; the attached object is the endpoint other than `owner`.
    std::vector<EdgeRecord> attachments(const Txn&, const std::string& ownerUuid,
                                        const std::optional<std::string>& role = std::nullopt) const;
    // Both endpoints must belong to the selected objects. Reads topology,
    // object versions and geometry heads from the caller's consistent snapshot.
    GraphReadView graphReadView(const Txn&, const ObjectSelection& = {},
                               const EdgeSelection& = {}) const;
    // Current geometry-only selection: captures selected records and their
    // geometry heads together in one read snapshot, without history or edges.
    // Supports UUID/kind/tag/family/coordinate filters and UUID pagination;
    // volume eligibility uses the same snapshot's native transform context.
    // Measured-length, bounds and connection filters are not accepted here.
    GraphReadView geometryReadView(const Txn&, const ObjectSelection& = {},
                                  const std::optional<StoreStatus>& expectedPosition = std::nullopt) const;
    // Requires a ReadTxn. Closed selection of exact recorded versions (one per
    // object; deleted versions and unknown references fail).
    // Always includes only applicable relationships. Retains isolated objects,
    // including explicitly selected retired objects (which have no applicable edges).
    GraphReadView graphReadViewAt(const Txn&, const std::vector<ObjectVersion>&,
                                 const EdgeSelection& = {}) const;
    // Neighbour object UUIDs, optionally filtered by edge kind and/or
    // neighbour object kind.
    std::vector<std::string> neighbours(const Txn& txn, const std::string& objectUuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt, bool includeRetired = false,
                                        std::optional<bool> adjacent = std::nullopt, bool applicableOnly = false) const;

    // Indexed traversal; empty edgeKindFilter means all kinds. adjacent=false
    // excludes adjacent fiber links. applicableOnly requires ReadTxn/latest heads.
    // Setting cancel from another thread aborts SQL/BFS with a "cancelled" runtime_error.
    std::vector<std::string> componentOf(const Txn& txn, const std::string& objectUuid,
                                         const std::vector<std::string>& edgeKindFilter = {},
                                         const std::atomic_bool* cancel = nullptr, bool includeRetired = false,
                                         std::optional<bool> adjacent = std::nullopt, bool applicableOnly = false) const;
    std::vector<std::string> reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                             const std::vector<std::string>& edgeKindFilter = {},
                                             const std::atomic_bool* cancel = nullptr, bool includeRetired = false,
                                             std::optional<bool> adjacent = std::nullopt, bool applicableOnly = false) const;

    // No fields argument loads complete records. With fields, only requested
    // top-level fields plus uuid/kind are loaded; other C++ members stay default.
    // An empty fields vector requests identities only.
    std::vector<ObjectRecord> selectObjects(const Txn&, const ObjectSelection& = {},
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const;
    // Exact slab intersection of one object kind's geometry. One read snapshot;
    // owned partial runs, no continuation or coverage cache.
    RegionResult geometryRegion(const ReadTxn&, const RegionSlice&,
        const RegionOptions& = {}, const RegionCancel& cancel = {}) const;
    // Nearest point of each surface near a point, from the persisted tile index.
    std::vector<SurfaceProximityHit> nearestSurfaces(const ReadTxn&, const SurfaceProximityQuery&) const;

    // Requires selection.bbox. Same filters as selectObjects, but pagination
    // counts regions, ordered by object UUID, array name and start indices.
    // Managed geometry returns tiles/blocks; external objects return boxes.
    std::vector<SpatialRegion> spatialRegions(const Txn&, const ObjectSelection&) const;
    // Metadata only, with geometry UUID/hash references; payloads stay in the store.
    std::string exportJson(const Txn&, const JsonExportOptions& = {}) const;
    void importJson(WriteTxn&, const std::string&);

    // Resolve surface/fiber/point_collection source UUIDs: verify storageRef, then
    // search datasetRoot (default: DB directory; also rebases relative hints).
    // Read JSON only; skip hidden/Zarr directories and directory symlinks. No cache
    // or writes. Return an absolute path or nullopt; ambiguity/traversal errors throw.
    std::optional<std::filesystem::path> resolveStoragePath(
        const Txn& txn, const std::string& objectUuid,
        const std::optional<std::filesystem::path>& datasetRoot = std::nullopt) const;
    void validateExportDestination(const Txn&, const std::filesystem::path&) const;

    // Flush the WAL into the main DB file (safe copying). Must not be
    // called while a WriteTxn is live on this handle.
    void checkpoint();

    // One WAL checkpoint, for services that schedule their own (SQLite's
    // automatic passive checkpoints still run on commit). Passive never
    // waits; Truncate waits up to `wait` for readers, then empties the WAL
    // file only if every frame was copied. Same restriction as checkpoint().
    // Artifact bytes (objects/Artifacts.hpp). putArtifact copies `file` (or
    // moves it, when `move` and it is on the same filesystem) into the
    // store's content-addressed directory and syncs it to disk before
    // returning its SHA-256, so a transaction may then reference it. Content
    // already held is kept. Writable handles only (under service ownership,
    // the service). Reads work on any handle; ranges past the end throw.
    std::string putArtifact(const std::filesystem::path& file, bool move = false);
    std::optional<uint64_t> artifactLength(const std::string& sha256) const;
    std::vector<uint8_t> readArtifact(const std::string& sha256, uint64_t offset = 0,
                                      const std::optional<uint64_t>& count = std::nullopt) const;

    enum class WalCheckpointMode { Passive, Truncate };
    struct WalCheckpoint {
        int64_t logFrames = 0;           // frames in the WAL after the call
        int64_t checkpointedFrames = 0;  // of those, copied into the database
        bool busy = false;               // readers or a writer prevented completion
        uint64_t walBytes = 0;           // WAL file size after the call
        bool complete() const { return !busy && checkpointedFrames == logFrames; }
    };
    WalCheckpoint checkpointWal(WalCheckpointMode, std::chrono::milliseconds wait = std::chrono::milliseconds(0));

private:
    // Lookups by internal revision UUID (storage keys, not public history
    // references); src/ reaches them through ObjectStoreImplAccess.
    std::optional<ObjectRevision> objectRevision(const Txn&, const std::string& revisionUuid) const;
    RevisionSelection revisionSelectionAt(const Txn&, int64_t publishedAtUs, bool includeRetired) const;
    // Explicit, closed selection: omitted endpoints are not selected. null
    // selects the latest head (unlike write(), where null expects absence).
    std::vector<SelectedRelationship> relationshipsAtSelected(const Txn&, const std::vector<ObjectRevision>&) const;
    struct Impl;
    void configureWrite(WriteTxn&, const TransactionContext&);
    static std::unique_ptr<Impl> openImpl(const std::filesystem::path&, const OpenOptions&, bool create, bool immutableInitial = false);
    explicit ObjectStore(std::unique_ptr<Impl>);
    void* connection(const Txn&) const;
    std::string editGeometry(WriteTxn&, const std::string& objectUuid, const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision, const std::vector<GeometryEdgeEdit>& edgeEdits,
                             const EdgeDecider* decide);
    std::unique_ptr<Impl> impl_;
    friend class Txn;
    friend struct ObjectStoreImplAccess;
};

} // namespace vc::objects
