#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Progress.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects {
// A decoded current-format object. Geometry, metadata and native relationships
// are inserted together; relationship endpoints use explicit UUIDs.
struct GeometrySource {
    ObjectRecord object;
    GeometryData geometry;
    nlohmann::json source;
};
// How source files that predate the store's contracts are admitted. Defaults
// reject invalid patches/PCLs; UUID-less fibers use the fixed file-name namespace.
// Each option is an explicit, recorded repair.
struct SourceIngestOptions {
    // A UUID namespace admitting sources without canonical UUIDs: a tifxyz patch keyed
    // by its meta.json "uuid" or directory name, a PCL collection by "<document stem>:<id>",
    // a fiber file by its stem (as version-3 branches name targets). Each gets
    // uuid5(namespace, "<kind>:<key>"), so reingesting the same files yields the same
    // identities. Canonical source UUIDs are kept.
    std::optional<std::string> uuidNamespace;
    // Drop fiber branch declarations whose target file is absent, recording each in
    // the declaring fiber's meta.source.dropped_branches. Otherwise the import fails.
    bool dropDanglingBranches = false;
    // Renumber every PCL collection that shares a point ID with another collection
    // of its document, above the document's largest ID and in point-ID order. Each
    // renumbered point keeps its file ID as the point attribute "source_point_id".
    // Documents with point links are refused. Otherwise the import fails.
    bool renumberCollidingPointIds = false;
};

// Decode a native patch, point-collection file, or fiber file/collection/directory
// before opening a write transaction. Linked documents are loaded recursively;
// available UUIDs are existing endpoints whose stored contents remain authoritative.
// Explicit roots and duplicate identities are never silently skipped.
std::vector<GeometrySource> prepareGeometrySource(const std::string& kind,
    const std::filesystem::path&, const std::vector<std::string>& availableUuids = {},
    const SourceIngestOptions& options = {});

// Create new identities atomically in the caller's transaction. Re-import and
// retired identities are rejected. Checked callers must pin every relationship
// endpoint. Returns the number of relationships imported; XYZ is unchanged.
int64_t ingestGeometryBatch(ObjectStore&, WriteTxn&, std::vector<GeometrySource>,
    const std::string& coordinateContext = "{}");

struct GeometryIngestReport {
    int64_t ingested = 0, edgesImported = 0;
    std::vector<std::string> problems;
    // Repairs applied by SourceIngestOptions; informational, not failures.
    std::vector<std::string> notices;
};

// Read/validate files outside write transactions, then insert the selected batch
// atomically, including native topology. Only registered placeholders that have
// never had geometry are eligible. Existing/retired imports are errors, even if
// unchanged. Source data must be stable during ingestion; edit imported objects
// through the store APIs. There is no file refresh or reconciliation.
// coordinateContext is applied as for ingestGeometrySource.
GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db,
    const std::optional<std::filesystem::path>& datasetRoot = std::nullopt,
    const std::vector<std::string>& objectUuids = {}, const util::ProgressCallback& progress = {},
    const std::string& coordinateContext = "{}");

// Create new objects and ingest one source: a tifxyz patch directory,
// vc3d_fiber JSON, collection or directory, vc_pointcollections JSON (all collections), or packed tracks
// (.vctracks directory or DBM path with an existing packed mirror). Track
// identity is (dataset_ref, external_key), within the object store. Fiber/collection
// sources are inserted atomically; packed tracks are inserted individually with partial reports.
// Existing identities (including retired objects) and duplicate UUIDs within a
// batch are errors. Use ingestRegisteredGeometry for an initial placeholder load.
// coordinateContext supplies {volume_uuid, scale_factor} for any geometry kind.
// It fills unknown declarations and rejects source conflicts; XYZ is unchanged.
// options are as for prepareGeometrySource; tracks keep (dataset_ref, external_key)
// and ignore them.
GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind,
    const std::filesystem::path& source, const util::ProgressCallback& progress = {},
    const std::string& coordinateContext = "{}", const SourceIngestOptions& options = {});

} // namespace vc::objects
