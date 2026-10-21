#pragma once

// SQLite object database with immutable geometry and relationship metadata.

#include "vc_object_store/objects/Geometry.hpp"
#include "vc_object_store/objects/FiberRegion.hpp"
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

// Generation, stamps and approval are library-managed and ignored on write.
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
    // Library-managed lifecycle, independent of ordinary JSON metadata.
    bool retired = false;
    std::string lifecycleEpoch;
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
    // False for sync baselines whose original metadata history was not copied.
    bool metadataComplete = true;
    // UTC Unix microseconds assigned once per atomic publication, monotonically
    // increasing within this store. Missing only for untimed sync baselines.
    std::optional<int64_t> publishedAtUs;
    bool deleted = false;
    ObjectRecord record;
};

// Edge changes revise both endpoints atomically; unchanged membership is inherited.
// Pins are immutable objectRevision() UUIDs, never moving heads.
struct RelationshipRevision {
    std::string uuid;
    EdgeRecord record;
    std::string aRevision, bRevision;
    std::optional<std::string> aGeometryRevision, bGeometryRevision;
};

struct SelectedRelationship {
    RelationshipRevision relationship;
    // Both selected endpoints contain this version and its dependencies still match.
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
    bool includeRetired = false;
};
struct EdgeSelection {
    std::optional<std::string> kind, tag;
    std::vector<std::string> approvalStates; // empty means all states
    // nullopt means unrestricted; an explicitly empty set matches no edges.
    std::optional<std::vector<std::string>> endpointUuids, uuids;
    bool bothEndpoints = false; // induced subgraph vs incident edges
    std::optional<int64_t> limit;
    std::optional<std::string> afterUuid; // exclusive, UUID-ordered cursor
    bool includeRetired = false; // includes lifecycle-inactive recorded edges
    // nullopt: all kinds; false/true: ordinary/adjacent fiber_branch only.
    // A missing adjacent metadata flag is an ordinary fiber branch.
    std::optional<bool> adjacent;
    // Shared membership/dependencies against latest heads. Requires a ReadTxn.
    bool applicableOnly = false;
};

// Snapshot topology in selection order. Every edge, including self/parallel edges,
// retains its identity and has two CSR incidences.
struct GraphReadView {
    std::vector<std::string> objectUuids, edgeUuids;
    std::vector<std::optional<std::string>> geometryRevisions;
    std::vector<int64_t> generations;
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

// Snapshot repair in edgeByUuid() endpoint order. Null revisions mean no geometry.
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
    // Typed, bounded geometry reads. Include store/GeometryViews.hpp to use.
    GeometrySnapshot geometrySnapshot();
    WriteTxn write();
    // Check supplied heads before editing; require every changed object/endpoint.
    // Commit publishes one revision per changed object. Missing expectations or
    // conflicts abort the whole transaction, even if caught.
    WriteTxn write(const RevisionSelection& expected);
    std::optional<ObjectRevision> currentObjectRevision(const Txn&, const std::string& objectUuid) const;
    std::optional<ObjectRevision> objectRevision(const Txn&, const std::string& revisionUuid) const;
    std::vector<ObjectRevision> objectHistory(const Txn&, const std::string& objectUuid) const;
    // Latest live revisions at/before this time; actions are indivisible.
    // Empty before the first publication; excludes deletions.
    RevisionSelection revisionSelectionAt(const Txn&, int64_t publishedAtUs, bool includeRetired = false) const;
    std::vector<RelationshipRevision> recordedRelationships(const Txn&, const std::string& objectRevisionUuid) const;
    // Explicit, closed selection: omitted endpoints are not selected. null
    // selects the latest head (unlike write(), where null expects absence).
    std::vector<SelectedRelationship> relationshipsAt(const Txn&, const RevisionSelection&) const;
    // After endpoint edits, validate anchors/metadataKeys and retain the relationship.
    // The caller decides domain meaning.
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

    // Create/update by UUID (mint one if empty). Updates bump generation, stamp
    // changed_*, reset approval and journal. Returns the UUID.
    std::string upsertObject(WriteTxn& txn, const ObjectRecord& record);
    std::optional<ObjectRecord> objectByUuid(const Txn& txn, const std::string& uuid) const;
    std::vector<ObjectRecord> objectsByKind(const Txn& txn, const std::string& kind,
                                            const std::optional<std::string>& approvalState = std::nullopt, bool includeRetired = false) const;
    int64_t objectCount(const Txn& txn, const std::string& kind, bool includeRetired = false) const;
    std::vector<ObjectRecord> objectsByTag(const Txn& txn, const std::string& tag, bool includeRetired = false) const;
    // Require write(expected). Decisions use keepRecordedRelationship,
    // updateEdge/retainRelationship, or removeEdge before commit. UUID lookup
    // always includes retired objects. Restore copies content into a new revision;
    // omitted source selects the last active revision. Links need explicit validation.
    void retireObject(WriteTxn&, const std::string& uuid);
    void restoreObject(WriteTxn&, const std::string& uuid,
                       const std::optional<std::string>& sourceRevision = std::nullopt);
    void deleteObject(WriteTxn& txn, const std::string& uuid);  // fails while edges or retained geometry reference it

    // Savepoint-protected replacement; null expectedRevision expects no geometry.
    // Stale writers fail; identical current content is a no-op. Every affected
    // edge requires an explicit retain/update/remove decision.
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
    // Aligned annotations for a geometry revision, derived from stable IDs.
    std::string geometryPointAnnotations(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName) const;

    // Dense row-major rectangle from any aligned patch array. Exact
    // bytes; handles tiled and linear layouts. Coordinates/counts are vertices.
    // Rectangle/range reads may reuse verified bytes on a pooled reader until
    // its next snapshot observes a commit. Writes always use fresh readers.
    std::vector<uint8_t> readGeometryRectangle(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns) const;

    // [row,column,rows,columns], in caller order, sharing manifest/chunk reads.
    std::vector<std::vector<uint8_t>> readGeometryRectangles(const Txn&, const std::string& revisionUuid,
        const std::string& arrayName, const std::vector<std::array<int64_t, 4>>& rectangles) const;

    // Stamp current generation without bumping it: pending, approved or rejected.
    void setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state);

    // Open-ended metadata: single-key atomic updates via SQLite JSON1.
    // All four geometry kinds use meta.coordinates = {volume_uuid, scale_factor}.
    // UUID is canonical hyphenated hex (or null); scale is positive finite (or
    // null), with stored XYZ = volume-voxel XYZ * scale_factor. Metadata only:
    // neither writes nor import transform geometry. Omission means unknown.
    // jsonValue is JSON text (e.g. "\"reviewed\"", "3", "true").
    // Both bump generation and reset approval, like any content change.
    void setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue);
    void removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key);

    std::vector<JournalEntry> journalSince(const Txn& txn, int64_t afterSeq) const;

    EdgeGeometryState edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const;
    // Atomically repair anchors/bindings, checking edge generation and endpoint heads.
    // Preserves other metadata, resets approval and journals. Identical input is a no-op.
    void repairEdgeGeometry(WriteTxn&, const EdgeGeometryRepair&);


    // Edges. addEdge normalises endpoint order (swapping anchors to
    // match) and enforces (kind, endpoints, anchors) uniqueness.
    std::string addEdge(WriteTxn& txn, const EdgeRecord& record);
    // Refresh anchors/tags/meta while retaining UUID, endpoints, kind and history.
    // Changes reset approval; revision bindings remain stale until explicitly checked.
    void updateEdge(WriteTxn& txn, const EdgeRecord& record);
    std::optional<EdgeRecord> edgeByUuid(const Txn& txn, const std::string& uuid) const;
    void removeEdge(WriteTxn& txn, const std::string& uuid);
    std::vector<EdgeRecord> edgesOf(const Txn& txn, const std::string& objectUuid, bool includeRetired = false) const;
    std::vector<EdgeRecord> edgesByKind(const Txn& txn, const std::string& edgeKind, bool includeRetired = false) const;
    // Projection always includes uuid/kind/a/b; empty fields means identities.
    std::vector<EdgeRecord> selectEdges(const Txn&, const EdgeSelection& = {},
        const std::optional<std::vector<std::string>>& fields = std::nullopt) const;
    // Both endpoints must belong to the selected objects. Reads topology,
    // generations and geometry heads from the caller's consistent snapshot.
    GraphReadView graphReadView(const Txn&, const ObjectSelection& = {},
                               const EdgeSelection& = {}) const;
    // Requires a ReadTxn. Closed selection, null revisions resolve to latest in this snapshot.
    // Always includes only applicable relationships. Retains isolated objects,
    // including explicitly selected retired objects (which have no applicable edges).
    GraphReadView graphReadViewAt(const Txn&, const RevisionSelection&,
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
    // One read snapshot; owned partial runs, no continuation or coverage cache.
    FiberRegionResult fiberRegion(const ReadTxn&, const FiberSlice&,
        const FiberRegionOptions& = {}, const FiberRegionCancel& cancel = {}) const;

    // Requires selection.bbox. Same filters as selectObjects, but pagination
    // counts regions, ordered by object UUID, array name and start indices.
    // Managed geometry returns tiles/blocks; external objects return boxes.
    std::vector<SpatialRegion> spatialRegions(const Txn&, const ObjectSelection&) const;
    // Metadata only, with geometry UUID/hash references; payloads stay in the store.
    std::string exportJson(const Txn&, const JsonExportOptions& = {}) const;
    void importJson(WriteTxn&, const std::string&);

    // Resolve patch/fiber/point_collection source UUIDs: verify storageRef, then
    // search datasetRoot (default: DB directory; also rebases relative hints).
    // Read JSON only; skip hidden/Zarr directories and directory symlinks. No cache
    // or writes. Return an absolute path or nullopt; ambiguity/traversal errors throw.
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
