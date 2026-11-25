#pragma once

#include "vc/lasagna/Dataset.hpp"
#include <vc_reader/catalog.hpp>
#include <vector>

namespace vc::lasagna {

enum class CatalogDatasetKind { Normal, FiberInference };

// Attachable catalog artifacts, deduplicated by URL, model and coordinate level.
[[nodiscard]] std::vector<std::size_t> catalogArtifacts(const vcr::catalog::OpenDataVolume& volume);

// The inner manifest is authoritative; catalog facts, when present, must agree.
[[nodiscard]] CatalogDatasetKind validateCatalogManifest(
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const LasagnaDatasetManifest& manifest);

struct CachedCatalogDataset {
    std::filesystem::path manifest;
    LasagnaDataset dataset;
    CatalogDatasetKind kind;
};

// Discover and materialize the manifest through the shared file cache, preserving
// its catalog identity in the remote marker. Validate metadata and channel arrays;
// voxel data stays lazy. The artifact must have a valid source coordinate level.
[[nodiscard]] CachedCatalogDataset cacheCatalogDataset(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const std::filesystem::path& remoteCacheRoot);

} // namespace vc::lasagna
