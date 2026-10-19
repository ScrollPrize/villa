#pragma once
#include "vc/core/objects/ObjectStore.hpp"

namespace vc::objects {
struct GeometryIngestReport {
    int64_t ingested = 0, unchanged = 0;
    std::vector<std::string> problems;
};

// Read/validate files outside write transactions, then publish each object in
// its own atomic transaction. Reports partial batch success explicitly; failed
// objects keep their previous state. Source data must be stable during ingestion.
// Existing object metadata/tags/identities are retained. No link detection runs.
GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db,
    const std::optional<std::filesystem::path>& datasetRoot = std::nullopt,
    const std::vector<std::string>& objectUuids = {}, const util::ProgressCallback& progress = {});

// Register missing objects and ingest one source: a tifxyz patch directory,
// vc3d_fiber JSON, vc_pointcollections JSON (all collections), or packed tracks
// (.vctracks directory or DBM path with an existing packed mirror). Track
// identity is (dataset_ref, external_key), within the object store.
GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind,
    const std::filesystem::path& source, const util::ProgressCallback& progress = {});
} // namespace vc::objects
