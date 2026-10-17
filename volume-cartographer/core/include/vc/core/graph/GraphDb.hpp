#pragma once

// SQLite-backed unified geometry and metadata graph (see
// docs/plans/graph_db_initial_implementation.md). Objects and edges are
// generic records: `kind` is metadata on the record, never encoded in
// API verbs. Every record carries explicit tier-1 fields (uuid, kind,
// generation with author/hostname/timestamp stamps, approval state)
// plus an open-ended `meta` JSON object for caller-defined metadata.

#include "vc/core/util/Progress.hpp"
#include "vc/core/graph/Geometry.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::graph {

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

struct Anchor {
    std::optional<int64_t> index;
    std::optional<std::array<double, 3>> position;
    std::optional<std::array<double, 3>> direction;
};

struct EdgeRecord : RecordCommon {
    std::string aUuid, bUuid;  // endpoint order is normalised by addEdge
    Anchor aAnchor, bAnchor;
};

struct JournalEntry {
    int64_t seq = 0;
    std::string recordUuid;
    int64_t generation = 0;
    std::string author, hostname, timestamp, changeKind;
    // Approval snapshot after this change. Unknown for pre-v2 history
    // and deletion events; migration records the surviving current state.
    std::optional<std::string> approvalState, approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedGeneration;
};

struct OpenOptions {
    bool readOnly = false;
    int readerPoolSize = 4;
    int busyTimeoutMs = 5000;
    std::string author;  // required for read-write opens
    std::optional<std::string> hostnameOverride;  // tests / reproducible imports only
};

class GraphDb;
struct GraphDbImplAccess;

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
    Txn(GraphDb* db, void* handle, bool write) : db_(db), handle_(handle), write_(write) {}
    void release(bool commit);

private:
    friend class GraphDb;
    friend struct GraphDbImplAccess;
    GraphDb* db_ = nullptr;
    void* handle_ = nullptr;  // sqlite3*
    bool write_ = false;
};

class ReadTxn : public Txn {
public:
    ReadTxn(GraphDb* db, void* handle) : Txn(db, handle, false) {}
};

class WriteTxn : public Txn {
public:
    WriteTxn(GraphDb* db, void* handle) : Txn(db, handle, true) {}
};

// Threading model: one writer connection serialised by WriteTxn (at
// most one live at a time per GraphDb), plus a bounded pool of reader
// connections. ReadTxn/WriteTxn are thread-confined: begin, use and
// end a transaction on the same thread, and keep transactions short
// (a ReadTxn pins a WAL snapshot; a WriteTxn must not span file IO).
class GraphDb {
public:
    static GraphDb create(const std::filesystem::path& path, const OpenOptions& options);
    static GraphDb open(const std::filesystem::path& path, const OpenOptions& options);

    GraphDb(GraphDb&&) noexcept;
    GraphDb& operator=(GraphDb&&) noexcept;
    GraphDb(const GraphDb&) = delete;
    GraphDb& operator=(const GraphDb&) = delete;
    ~GraphDb();

    ReadTxn read();
    WriteTxn write();

    const std::filesystem::path& path() const;
    std::string hostname() const;
    std::string author() const;

    // Immutable geometry. A null expectedRevision means "expect no geometry";
    // pass the current revision UUID to replace it. Stale writers fail, even
    // when their proposed bytes match. Identical current content is a no-op.
    // The write is savepoint-protected and invalidates connected edge approvals.
    std::string putGeometry(WriteTxn& txn, const std::string& objectUuid,
                            const GeometryData& data,
                            const std::optional<std::string>& expectedRevision = std::nullopt);
    std::optional<GeometryRevision> currentGeometry(const Txn& txn, const std::string& objectUuid) const;
    std::optional<GeometryRevision> geometryRevision(const Txn& txn, const std::string& revisionUuid) const;
    std::vector<GeometryRevision> geometryHistory(const Txn& txn, const std::string& objectUuid) const;
    // Byte ranges allow bounded loading without materializing a whole array.
    // Every touched chunk is verified against its SHA-256 content address.
    std::vector<uint8_t> readGeometryArray(const Txn& txn, const std::string& revisionUuid,
        const std::string& arrayName, int64_t byteOffset = 0,
        const std::optional<int64_t>& byteCount = std::nullopt) const;
    EdgeGeometryState edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const;
    // Explicitly acknowledge that anchors/relationship have been revalidated
    // against the current endpoint revisions. Resets approval and journals.
    void revalidateEdgeGeometry(WriteTxn& txn, const std::string& edgeUuid);

    // Objects. upsertObject creates (minting a UUID when record.uuid is
    // empty) or updates by UUID; updates bump generation, restamp
    // changed_*, reset approval, and journal. Returns the UUID.
    std::string upsertObject(WriteTxn& txn, const ObjectRecord& record);
    // Refresh geometry metadata. Changed values bump generation, restamp
    // changed_*, reset approval and journal; identical values are a no-op.
    // An optional packed-store family fills an unset family; conflicts fail.
    void setTrackLength(WriteTxn& txn, const std::string& uuid, double length,
                        const std::optional<std::string>& family = std::nullopt);
    void setFiberLength(WriteTxn& txn, const std::string& uuid, double length);
    std::optional<ObjectRecord> objectByUuid(const Txn& txn, const std::string& uuid) const;
    std::vector<ObjectRecord> objectsByKind(const Txn& txn, const std::string& kind,
                                            const std::optional<std::string>& approvalState = std::nullopt) const;
    std::vector<ObjectRecord> objectsByTag(const Txn& txn, const std::string& tag) const;
    void deleteObject(WriteTxn& txn, const std::string& uuid);  // fails while edges or retained geometry reference it

    // Edges. addEdge normalises endpoint order (swapping anchors to
    // match) and enforces (kind, endpoints, anchors) uniqueness.
    std::string addEdge(WriteTxn& txn, const EdgeRecord& record);
    std::optional<EdgeRecord> edgeByUuid(const Txn& txn, const std::string& uuid) const;
    void removeEdge(WriteTxn& txn, const std::string& uuid);
    std::vector<EdgeRecord> edgesOf(const Txn& txn, const std::string& objectUuid) const;
    std::vector<EdgeRecord> edgesByKind(const Txn& txn, const std::string& edgeKind) const;
    // Neighbour object UUIDs, optionally filtered by edge kind and/or
    // neighbour object kind.
    std::vector<std::string> neighbours(const Txn& txn, const std::string& objectUuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const;

    // Traversal over materialised edge lists (all edges by default, or
    // only edges whose kind is in edgeKindFilter when non-empty). When
    // cancel is given and becomes true, the traversal aborts with a
    // "cancelled" runtime_error (SQLite progress handler during SQL,
    // checked each BFS step in memory); safe to set from another thread.
    std::vector<std::string> componentOf(const Txn& txn, const std::string& objectUuid,
                                         const std::vector<std::string>& edgeKindFilter = {},
                                         const std::atomic_bool* cancel = nullptr) const;
    std::vector<std::string> reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                             const std::vector<std::string>& edgeKindFilter = {},
                                             const std::atomic_bool* cancel = nullptr) const;

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

    // Managed geometry objects are excluded from this external-source audit.
    // Reports objects whose external data reference does not check out
    // (storage file missing, or data_generation ahead of generation).
    // Paths are resolved relative to the DB file's directory.
    std::vector<std::string> sweepExternalData(const Txn& txn) const;

    // Flush the WAL into the main DB file (safe copying). Must not be
    // called while a WriteTxn is live on this handle.
    void checkpoint();

    // JSON export/import of the metadata graph (records only; external
    // data referenced by UUID/ref). Export optionally restricts to one
    // object kind and/or one tag. Import runs in the caller's WriteTxn
    // Version 2 preserves UUIDs, generations and all record provenance.
    // Existing identical records are a no-op; conflicts are rejected (no merge).
    // Version 1 remains readable but cannot restore missing provenance.
    // Journal history is local and is not part of a graph snapshot.
    // Export refuses selected objects with managed geometry: use a SQLite
    // backup for a self-contained geometry database (including history).
    std::string exportJson(const Txn& txn, const std::optional<std::string>& objectKind = std::nullopt,
                           const std::optional<std::string>& tag = std::nullopt) const;
    void importJson(WriteTxn& txn, const std::string& jsonText);

private:
    friend class Txn;
    friend struct GraphDbImplAccess;
    struct Impl;
    static std::unique_ptr<Impl> openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create);
    explicit GraphDb(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

struct ImportReport {
    int64_t objectsImported = 0;
    int64_t edgesImported = 0;
    std::vector<std::string> problems;  // unresolved/one-sided entries, skipped files
};

// Scans fiberDir for vc3d_fiber JSON files (one object per file, kind
// "fiber") and imports the edges their branch entries assert. Branch
// targets are resolved by file name within the scanned set; unresolved
// and one-sided entries are reported, never silently dropped. Runs in
// the caller's WriteTxn. Idempotent: re-importing an unchanged file
// (same uuid + same source content) is a no-op for that fiber.
ImportReport importFibers(GraphDb& db, WriteTxn& txn, const std::filesystem::path& fiberDir,
                         const util::ProgressCallback& progress = {});

// Registers each immediate subdirectory of segmentsDir that contains a
// meta.json with a "uuid" as a kind="patch" object. Reads nothing else
// from segment directories (in particular, never overlapping.json)
// and creates no edges.
ImportReport importPatches(GraphDb& db, WriteTxn& txn, const std::filesystem::path& segmentsDir,
                         const util::ProgressCallback& progress = {});

// Registers tracks from a JSON interchange file: an array of objects
// with dataset_ref + external_key (the track identity) and optional
// name, family, point_count, length, bbox (6 numbers), tags, meta.
// This is the registration path for track datasets produced outside
// this repo (e.g. Spiral): the external producer emits this list, the
// graph mints and remembers a UUID per (dataset_ref, external_key).
// Idempotent: an unchanged identity keeps its UUID and generation; a
// changed entry updates in place. Geometry stays in the external
// dataset; only metadata lands here.
ImportReport importTrackList(GraphDb& db, WriteTxn& txn, const std::filesystem::path& trackListJson,
                         const util::ProgressCallback& progress = {});

}  // namespace vc::graph
