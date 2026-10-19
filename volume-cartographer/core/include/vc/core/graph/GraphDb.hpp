#pragma once
#include "vc/core/storage/RecordDb.hpp"
#include "vc/core/objects/ObjectStore.hpp"
namespace vc::graph {
using namespace vc::storage;
using vc::objects::ObjectStore;
struct EdgeGeometryState {
    std::optional<std::string> aRevision, bRevision;
    bool stale = false;
};
struct ObjectSelection {
    std::string objectUuid, storeUuid, revisionUuid;
};
struct ObjectUpdate {
    std::string objectUuid, selectedRevision, currentRevision;
};
// Graph-local membership records, edges and explicit endpoint revision bindings.
class GraphDb : public RecordDb {
public:
    static GraphDb create(const std::filesystem::path&, const OpenOptions&);
    static GraphDb open(const std::filesystem::path&, const OpenOptions&);
    void includeObject(WriteTxn&, ObjectStore&, const std::string& objectUuid, const std::string& revisionUuid);
    void advanceObject(WriteTxn&, ObjectStore&, const std::string& objectUuid,
                       const std::string& revisionUuid, const std::string& expectedRevision);
    std::optional<ObjectSelection> selectedObject(const Txn&, const std::string& objectUuid) const;
    std::vector<ObjectUpdate> availableUpdates(const Txn&, ObjectStore&) const;
    void setApproval(WriteTxn&, const std::string& recordUuid, const std::string& state) override;
    EdgeGeometryState edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const;
    // Explicitly acknowledge that anchors/relationship have been revalidated
    // against the current endpoint revisions. Resets approval and journals.
    void revalidateEdgeGeometry(WriteTxn& txn, const std::string& edgeUuid);

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


    // Metadata-only export refuses selections with object-store bindings.
    std::string exportJson(const Txn&, const std::optional<std::string>& objectKind = std::nullopt,
                           const std::optional<std::string>& tag = std::nullopt) const;
    void importJson(WriteTxn&, const std::string&);
private:
    explicit GraphDb(std::unique_ptr<Impl>);
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
