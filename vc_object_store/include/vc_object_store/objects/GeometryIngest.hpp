#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Progress.hpp"

namespace vc::objects {
struct GeometryIngestReport {
    int64_t ingested = 0, edgesImported = 0;
    std::vector<std::string> problems;
};

// Read/validate files outside write transactions, then publish the selected batch
// atomically, including native topology. Only registered placeholders that have
// never had geometry are eligible. Existing/retired imports are errors, even if
// unchanged. Source data must be stable during ingestion; edit imported objects
// through the store APIs. There is no file refresh or reconciliation.
GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db,
    const std::optional<std::filesystem::path>& datasetRoot = std::nullopt,
    const std::vector<std::string>& objectUuids = {}, const util::ProgressCallback& progress = {});

// Create new objects and ingest one source: a tifxyz patch directory,
// vc3d_fiber JSON or directory, vc_pointcollections JSON (all collections), or packed tracks
// (.vctracks directory or DBM path with an existing packed mirror). Track
// identity is (dataset_ref, external_key), within the object store. Fiber/collection
// sources publish atomically; packed tracks publish individually with partial reports.
// Existing identities (including retired objects) and duplicate UUIDs within a
// batch are errors. Use ingestRegisteredGeometry for an initial placeholder load.
// coordinateContext supplies {volume_uuid, scale_factor} for any geometry kind.
// It fills unknown declarations and rejects source conflicts; XYZ is unchanged.
GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind,
    const std::filesystem::path& source, const util::ProgressCallback& progress = {},
    const std::string& coordinateContext = "{}");

} // namespace vc::objects
