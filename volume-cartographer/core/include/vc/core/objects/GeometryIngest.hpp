#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/util/Progress.hpp"

namespace vc::objects {
struct GeometryIngestReport {
    int64_t ingested = 0, unchanged = 0, edgesImported = 0;
    std::vector<std::string> problems;
};

// Read/validate files outside write transactions, then publish the selected batch
// atomically, including native topology when geometry is first ingested. Source
// data must be stable during ingestion. Existing DB annotations/topology are retained.
// Geometry changes affecting relationships require explicit edit decisions; ingestion refuses them.
GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db,
    const std::optional<std::filesystem::path>& datasetRoot = std::nullopt,
    const std::vector<std::string>& objectUuids = {}, const util::ProgressCallback& progress = {});

// Register missing objects and ingest one source: a tifxyz patch directory,
// vc3d_fiber JSON or directory, vc_pointcollections JSON (all collections), or packed tracks
// (.vctracks directory or DBM path with an existing packed mirror). Track
// identity is (dataset_ref, external_key), within the object store. Fiber/collection
// sources publish atomically; packed tracks publish individually with partial reports.
GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind,
    const std::filesystem::path& source, const util::ProgressCallback& progress = {});
} // namespace vc::objects
