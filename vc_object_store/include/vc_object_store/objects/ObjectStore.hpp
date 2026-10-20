#pragma once

// SQLite object database with immutable geometry and relationship metadata.

#include "vc_object_store/objects/Geometry.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {

// Typed detail fields are editable source/import annotations. Counts, lengths,
// dimensions and boxes are summaries captured at ingestion, not live views of
// managed geometry. Use currentGeometry() for current array shapes/patch scale,
// and selectObjects()/spatialRegions() for revision-consistent spatial bounds.
// putGeometry() deliberately does not overwrite these annotations.
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

struct PatchDetail {
    int64_t gridWidth = 0;
    int64_t gridHeight = 0;
    std::array<double, 2> scale{0.0, 0.0};  // column and row scale, as in tifxyz meta.json
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

// Fields shared by every record (object or edge). Library-managed
// fields (generation, stamps, approval) are read-only to callers:
// values set on input structs are ignored on write.
struct RecordCommon {
    std::string uuid;  // canonical text; empty on create = mint a new one
    std::string kind;  // required, non-empty
    int64_t generation = 0;
    std::string createdBy, createdAt, createdHost;
    std::string changedBy, changedAt, changedHost;
    std::string approvalState = "pending";  // pending | approved | rejected
    std::optional<std::string> approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedGeneration;
    std::vector<std::string> tags;
    std::string meta = "{}";  // JSON object text
};

struct ObjectRecord : RecordCommon {
    std::string name;
    int64_t formatVersion = 0;
    std::string storageKind;  // e.g. "file", "dataset", "" (db-only)
    std::string storageRef;   // location hint relative to DB directory, or dataset URI; uuid is identity
    int64_t dataGeneration = 0;
    std::optional<std::string> dataChecksum;
    std::optional<FiberDetail> fiber;
    std::optional<PatchDetail> patch;
    std::optional<TrackDetail> track;
};

struct PointReference {
    // Geometry array and point coordinates, excluding the final XYZ axis:
    // control_points:[5], line_points:[12], points:[20,145] (patch row,col).
    // point_ids:[42] addresses stable point ID 42 in a point_collection.
    std::string array;
    std::vector<int64_t> index;
    bool operator==(const PointReference&) const = default;
};

struct Anchor {
    std::optional<std::array<double, 3>> position;
    std::optional<std::array<double, 3>> direction;
    // Exact geometry dependencies, including any neighbours used by a
    // direction/interpolated anchor. Empty means unknown, not independent.
    std::vector<PointReference> points;
    bool wholeGeometry = false; // explicit object-wide relationship
    // Top-level ordinary metadata keys that affect this relationship.
    // Changing these requires an explicit retain/update/remove decision.
    std::vector<std::string> metadataKeys;
    bool operator==(const Anchor&) const = default;
};

struct EdgeRecord : RecordCommon {
    std::string aUuid, bUuid;  // endpoint order is normalised by addEdge
    Anchor aAnchor, bAnchor;
};

// Object UUID -> immutable object revision UUID. null means expect no object.
// These are object revisions, not geometry payload revisions.
using RevisionSelection = std::map<std::string, std::optional<std::string>>;

struct ObjectRevision {
    std::string uuid, objectUuid;
    std::optional<std::string> parentUuid, geometryRevision;
    int64_t sequence = 0;
    // False only for migration baselines: pre-v9 metadata history did not exist.
    bool metadataComplete = true;
    // UTC Unix microseconds assigned once per atomic publication, monotonically
    // increasing within this store. Missing only for untimed migration baselines.
    std::optional<int64_t> publishedAtUs;
    bool deleted = false;
    ObjectRecord record;
};

// Shared edges are owned by both endpoints. Adding, changing or removing one
// revises both objects in the same transaction. Unchanged membership is inherited.
// Pins never follow heads: resolve aRevision/bRevision with objectRevision().
struct RelationshipRevision {
    std::string uuid;
    EdgeRecord record;
    std::string aRevision, bRevision;
    std::optional<std::string> aGeometryRevision, bGeometryRevision;
};

struct SelectedRelationship {
    RelationshipRevision relationship;
    // Both selected endpoints must contain this version and its declared
    // dependencies must still match. Unrelated metadata edits remain applicable.
    bool applicable = false;
};

struct JournalEntry {
    int64_t seq = 0;
    std::string recordUuid;
    int64_t generation = 0;
    std::string author, hostname, timestamp, changeKind;
    // Approval snapshot after this change; absent for deletion events.
    std::optional<std::string> approvalState, approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedGeneration;
};

struct OpenOptions {
    bool readOnly = false;
    int readerPoolSize = 4;
    int busyTimeoutMs = 5000;
    std::string author;  // required for read-write opens
    std::optional<std::string> hostnameOverride;  // tests / reproducible imports only
    // Retained verified geometry per pooled read connection; zero disables it.
    // Payload bytes plus conservatively charged metadata, excluding active reads.
    std::size_t geometryCacheBytes = 16 * 1024 * 1024;
    // Resident spatial descriptors/index per read connection; zero uses SQL.
    // Built lazily; snapshots too large for this budget also use SQL.
    std::size_t spatialCacheBytes = 16 * 1024 * 1024;
};

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
};
struct EdgeSelection {
    std::optional<std::string> kind, tag;
    std::vector<std::string> approvalStates; // empty means all states
    // nullopt means unrestricted; an explicitly empty set matches no edges.
    std::optional<std::vector<std::string>> endpointUuids, uuids;
    bool bothEndpoints = false; // induced subgraph vs incident edges
    std::optional<int64_t> limit;
    std::optional<std::string> afterUuid; // exclusive, UUID-ordered cursor
};

// Compact topology read from one store transaction. No separate persistent
// representation. Parallel edges retain identity; each has two CSR incidences,
// including self-links. Node order follows the object selection.
struct GraphReadView {
    std::vector<std::string> objectUuids, edgeUuids;
    std::vector<std::optional<std::string>> geometryRevisions;
    std::vector<int64_t> generations;
    std::vector<uint64_t> offsets, neighbours, edgeIndices;
    std::vector<std::array<uint64_t, 2>> endpoints;
    int64_t journalSequence = 0;
};

// Homogeneous arrays concatenated along their first dimension. Offsets index
// that dimension (not bytes); values retain their stored dtype and coordinates.
struct PackedGeometryArray {
    std::string dtype = "|u1";
    std::vector<int64_t> shape{0};
    std::vector<uint64_t> offsets{0};
    std::vector<uint8_t> data;
};
// A conservative candidate, not an exact geometry hit. start/shape index
// vertices (excluding XYZ); adjacent surface tiles and polyline blocks share
// their boundary vertices. Empty arrayName/start/shape denotes an external
// object's annotated box. Bounds and revision come from the same snapshot.
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

struct SyncOptions {
    bool dryRun = false;
    std::map<std::string, std::string> resolutions; // record UUID -> "local" or "remote"
    std::optional<std::string> expectedPlan; // preview token; required with resolutions
};
struct SyncConflict {
    std::string uuid, reason;
    std::string base, local, remote; // canonical record JSON, or "null" for absence/deletion
};
struct SyncReport {
    bool applied = false;
    std::string localReplica, remoteReplica, planToken;
    int64_t localChanges = 0, remoteChanges = 0;
    std::vector<SyncConflict> conflicts;
};

class ObjectStore;
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

protected:
    Txn(ObjectStore* db, void* handle, bool write) : db_(db), handle_(handle), write_(write) {}
    void release(bool commit);

private:
    friend class ObjectStore;
    friend struct ObjectStoreImplAccess;
    ObjectStore* db_ = nullptr;
    void* handle_ = nullptr;  // sqlite3*
    bool write_ = false;
};

class ReadTxn : public Txn {
public:
    ReadTxn(ObjectStore* db, void* handle) : Txn(db, handle, false) {}
};

class WriteTxn : public Txn {
public:
    WriteTxn(ObjectStore* db, void* handle) : Txn(db, handle, true) {}
};

struct EdgeGeometryState {
    std::optional<std::string> aRevision, bRevision;
    bool stale = false;
};

// A repair computed from one read snapshot. Anchors and revisions use the
// stored edge's a/b order (edgeByUuid), not caller-selected endpoint order.
// Null revisions explicitly mean the endpoint has no managed geometry.
struct EdgeGeometryRepair {
    std::string edgeUuid;
    int64_t expectedGeneration = 0;
    std::optional<std::string> aRevision, bRevision;
    Anchor aAnchor, bAnchor;
};

struct GeometryEdgeEdit {
    std::string edgeUuid;
    int64_t expectedGeneration = 0;
    // "retain", "update", "remove", or "keep_recorded". Retain keeps the same
    // relationship and anchors; update supplies both complete anchors in
    // stored a/b order. Revisions are the heads observed BEFORE the edit.
    // keep_recorded checks the resulting anchor identities but preserves the
    // historical pins; it does NOT assert applicability to the new geometry.
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
    static ObjectStore open(const std::filesystem::path&, const OpenOptions&);
    ObjectStore(ObjectStore&&) noexcept;
    ObjectStore& operator=(ObjectStore&&) noexcept;
    ObjectStore(const ObjectStore&) = delete;
    ObjectStore& operator=(const ObjectStore&) = delete;
    ~ObjectStore();
    ReadTxn read();
    WriteTxn write();
    // Optimistic action: check every supplied head before making changes and
    // require an expectation for every modified object and relationship endpoint.
    // One commit publishes one object revision per changed object. Any missing
    // expectation or conflict rolls back the ENTIRE transaction, even if caught.
    // Use this overload for edits prepared outside the writer transaction.
    WriteTxn write(const RevisionSelection& expected);
    std::optional<ObjectRevision> currentObjectRevision(const Txn&, const std::string& objectUuid) const;
    std::optional<ObjectRevision> objectRevision(const Txn&, const std::string& revisionUuid) const;
    std::vector<ObjectRevision> objectHistory(const Txn&, const std::string& objectUuid) const;
    // Latest revision of each live object at or before this publication time.
    // All objects changed by an action have the same timestamp; no partial action
    // can be selected. Empty before the first publication. Deletions are excluded.
    RevisionSelection revisionSelectionAt(const Txn&, int64_t publishedAtUs) const;
    std::vector<RelationshipRevision> recordedRelationships(const Txn&, const std::string& objectRevisionUuid) const;
    // Explicit, closed selection: omitted endpoints are not selected. null
    // selects the latest head (unlike write(), where null expects absence).
    std::vector<SelectedRelationship> relationshipsAt(const Txn&, const RevisionSelection&) const;
    // Explicitly validate and retain an existing relationship against the
    // resulting endpoint state, including metadataKeys. Call after endpoint edits.
    // Domain-specific meaning is the caller's decision; the store checks anchors.
    void retainRelationship(WriteTxn&, const std::string& edgeUuid);
    // Keep historical pins after an endpoint edit without asserting that the
    // relationship applies to the edited endpoint. Current queries may be stale.
    void keepRecordedRelationship(WriteTxn&, const std::string& edgeUuid);
    const std::filesystem::path& path() const;
    std::string hostname() const;
    std::string author() const;

    // Stable store identity, independent of its filesystem location.
    std::string uuid(const Txn&) const;

    // Two-store three-way merge. Conflicts/dry runs leave both stores untouched.
    // Both stores must be writable, with no active caller write transactions.
    // Commits are sequential; retained common baselines make interrupted sync retryable.
    SyncReport synchronize(ObjectStore& peer, const SyncOptions& options = {});
    std::string replicaUuid(const Txn&) const;
    // Assign a copied database its own replica identity before editing/synchronizing it.
    // Clears sync history; synchronize with the original before editing either copy.
    std::string forkReplica();

    // Objects. upsertObject creates (minting a UUID when record.uuid is
    // empty) or updates by UUID; updates bump generation, restamp
    // changed_*, reset approval, and journal. Returns the UUID.
    std::string upsertObject(WriteTxn& txn, const ObjectRecord& record);
    std::optional<ObjectRecord> objectByUuid(const Txn& txn, const std::string& uuid) const;
    std::vector<ObjectRecord> objectsByKind(const Txn& txn, const std::string& kind,
                                            const std::optional<std::string>& approvalState = std::nullopt) const;
    int64_t objectCount(const Txn& txn, const std::string& kind) const;
    std::vector<ObjectRecord> objectsByTag(const Txn& txn, const std::string& tag) const;
    void deleteObject(WriteTxn& txn, const std::string& uuid);  // fails while edges or retained geometry reference it

    // Immutable geometry. A null expectedRevision means "expect no geometry";
    // pass the current revision UUID to replace it. Stale writers fail, even
    // when their proposed bytes match. Identical current content is a no-op.
    // Every affected edge requires an explicit retain/update/remove decision.
    // The write is savepoint-protected. Identical content is always a no-op.
    std::string putGeometry(WriteTxn& txn, const std::string& objectUuid,
                            const GeometryData& data,
                            const std::optional<std::string>& expectedRevision = std::nullopt,
                            const std::vector<GeometryEdgeEdit>& edgeEdits = {},
                            const GeometryStorageOptions& storage = {});
    // Preview the relationships whose declared point dependencies change.
    // Unknown dependencies are included and require update (with explicit
    // dependencies) or removal. Does not publish geometry or modify edges.
    std::vector<EdgeRecord> affectedGeometryEdges(const Txn&, const std::string& objectUuid,
                                                const GeometryData&) const;
    // Sparse, lossless, compressed edits. One call publishes one revision in
    // the caller's transaction. expectedRevision is mandatory, even for no-ops.
    // No automatic checkpoints. Use putGeometry for structural replacements.
    std::string editGeometry(WriteTxn&, const std::string& objectUuid,
                             const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision,
                             const std::vector<GeometryEdgeEdit>& edgeEdits = {});
    std::vector<EdgeRecord> affectedGeometryEditEdges(const Txn&, const std::string& objectUuid,
                             const std::vector<GeometryArrayEdit>&,
                             const std::string& expectedRevision) const;
    std::optional<GeometryRevision> currentGeometry(const Txn& txn, const std::string& objectUuid) const;
    std::optional<GeometryRevision> geometryRevision(const Txn& txn, const std::string& revisionUuid) const;
    std::vector<GeometryRevision> geometryHistory(const Txn& txn, const std::string& objectUuid) const;
    // Byte ranges allow bounded loading without materializing a whole array.
    // Every touched chunk is verified against its SHA-256 content address.
    std::vector<uint8_t> readGeometryArray(const Txn& txn, const std::string& revisionUuid,
        const std::string& arrayName, int64_t byteOffset = 0,
        const std::optional<int64_t>& byteCount = std::nullopt) const;

    // Batched byte ranges [offset,count], in caller order, sharing verified chunks.
    std::vector<std::vector<uint8_t>> readGeometryRanges(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const;

    // One packed buffer, caller revision order (including duplicates).
    // Dtype and trailing dimensions must match. All chunks remain verified.
    PackedGeometryArray readGeometryArrays(const Txn&, const std::vector<std::string>& revisionUuids,
        const std::string& arrayName) const;

    // Dense row-major float32 XYZ rectangle from a patch points array. Exact
    // bytes; handles tiled and legacy layouts. Coordinates/counts are vertices.
    // Rectangle/range reads may reuse verified bytes on a pooled reader until
    // its next snapshot observes a commit. Writes always use fresh readers.
    std::vector<uint8_t> readGeometryRectangle(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns) const;

    // [row,column,rows,columns], in caller order, sharing manifest/chunk reads.
    std::vector<std::vector<uint8_t>> readGeometryRectangles(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, const std::vector<std::array<int64_t, 4>>& rectangles) const;

    // Approval: one generic call for objects and edges. Stamps the
    // current generation without bumping it. State must be
    // "pending", "approved" or "rejected".
    void setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state);

    // Open-ended metadata: single-key atomic updates via SQLite JSON1.
    // jsonValue is JSON text (e.g. "\"reviewed\"", "3", "true").
    // Both bump generation and reset approval, like any content change.
    void setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue);
    void removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key);

    std::vector<JournalEntry> journalSince(const Txn& txn, int64_t afterSeq) const;

    EdgeGeometryState edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const;
    // Publish corrected anchors and exact checked revision bindings together.
    // Reject changed edge generations or endpoint heads, even for a no-op.
    // Preserves other edge metadata, resets approval, and journals one change.
    // Savepoint-protected; identical anchors/bindings are a complete no-op.
    void repairEdgeGeometry(WriteTxn&, const EdgeGeometryRepair&);


    // Edges. addEdge normalises endpoint order (swapping anchors to
    // match) and enforces (kind, endpoints, anchors) uniqueness.
    std::string addEdge(WriteTxn& txn, const EdgeRecord& record);
    // Refresh anchors/tags/meta while retaining UUID, endpoints, kind and history.
    // Changes reset approval; revision bindings remain stale until explicitly checked.
    void updateEdge(WriteTxn& txn, const EdgeRecord& record);
    std::optional<EdgeRecord> edgeByUuid(const Txn& txn, const std::string& uuid) const;
    void removeEdge(WriteTxn& txn, const std::string& uuid);
    std::vector<EdgeRecord> edgesOf(const Txn& txn, const std::string& objectUuid) const;
    std::vector<EdgeRecord> edgesByKind(const Txn& txn, const std::string& edgeKind) const;
    // Projection always includes uuid/kind/a/b; empty fields means identities.
    std::vector<EdgeRecord> selectEdges(const Txn&, const EdgeSelection& = {},
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const;
    // Both endpoints must belong to the selected objects. Reads topology,
    // generations and geometry heads from the caller's consistent snapshot.
    GraphReadView graphReadView(const Txn&, const ObjectSelection& = {},
                               const EdgeSelection& = {}) const;
    // Neighbour object UUIDs, optionally filtered by edge kind and/or
    // neighbour object kind.
    std::vector<std::string> neighbours(const Txn& txn, const std::string& objectUuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const;

    // Indexed traversal, filtered by edgeKindFilter when non-empty. When
    // cancel is given and becomes true, the traversal aborts with a
    // "cancelled" runtime_error (SQLite progress handler during SQL,
    // checked each BFS step in memory); safe to set from another thread.
    std::vector<std::string> componentOf(const Txn& txn, const std::string& objectUuid,
                                         const std::vector<std::string>& edgeKindFilter = {},
                                         const std::atomic_bool* cancel = nullptr) const;
    std::vector<std::string> reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                             const std::vector<std::string>& edgeKindFilter = {},
                                             const std::atomic_bool* cancel = nullptr) const;

    // No fields argument loads complete records. With fields, only requested
    // top-level fields plus uuid/kind are loaded; other C++ members stay default.
    // An empty fields vector requests identities only.
    std::vector<ObjectRecord> selectObjects(const Txn&, const ObjectSelection& = {},
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const;
    // Requires selection.bbox. Same filters as selectObjects, but pagination
    // counts regions, ordered by object UUID, array name and start indices.
    // Managed geometry returns tiles/blocks; external objects return boxes.
    std::vector<SpatialRegion> spatialRegions(const Txn&, const ObjectSelection&) const;
    // Metadata only, with geometry UUID/hash references; payloads stay in the store.
    std::string exportJson(const Txn&, const JsonExportOptions& = {}) const;
    void importJson(WriteTxn&, const std::string&);

    // Resolve a file-backed patch/fiber/point_collection by its source UUID.
    // First verify storageRef against meta.json or the source JSON, then search
    // datasetRoot recursively if the hint is stale. The default root is
    // the DB directory; an explicit root also rebases relative hints.
    // Returns an absolute path, or nullopt for unknown/unsupported/missing
    // objects. Throws on ambiguous fallback matches or traversal errors.
    // Discovery skips hidden directories and Zarr stores, does not follow
    // directory symlinks, and reads JSON only (no geometry). No DB writes
    // or persistent cache: subsequent calls see renamed/replaced files.
    std::optional<std::filesystem::path> resolveStoragePath(
        const Txn& txn, const std::string& objectUuid,
        const std::optional<std::filesystem::path>& datasetRoot = std::nullopt) const;

    // Flush the WAL into the main DB file (safe copying). Must not be
    // called while a WriteTxn is live on this handle.
    void checkpoint();

private:
    struct Impl;
    static std::unique_ptr<Impl> openImpl(const std::filesystem::path&, const OpenOptions&, bool create);
    explicit ObjectStore(std::unique_ptr<Impl>);
    void* connection(const Txn&) const;
    std::unique_ptr<Impl> impl_;
    friend class Txn;
};

} // namespace vc::objects
