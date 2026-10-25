#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"
#include "OpenDataSurfaceStore.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class VolumePkg;

namespace vc3d::opendata {

struct OpenDataSampleDownloadProgress {
    int totalSegments = 0;
    int completedSegments = 0;
    int failedSegments = 0;
    int totalFiles = 0;
    int completedFiles = 0;
    int activeWorkers = 0;
    int totalWorkers = 0;
    std::string segmentId;
    std::string fileName;
    std::string status;
};

using OpenDataSampleProgressCallback =
    std::function<void(const OpenDataSampleDownloadProgress&)>;

struct OpenDataSampleProjectResult {
    int supportedVolumes = 0;
    int attachedVolumeEntries = 0;
    int skippedVolumes = 0;
    int failedVolumes = 0;
    int attachedLasagnaDatasets = 0;
    int attachedNormalGrids = 0;
    int supportedTifxyzSegments = 0;
    // Representations recorded in the sample's surface store (new ones counted in addedSurfaces).
    int importedSurfaces = 0;
    int addedSurfaces = 0;
    int failedTifxyzSegments = 0;
    std::string preferredVolumeId;
    std::vector<std::string> messages;
};

struct OpenDataNewProject {
    std::filesystem::path path;
    std::string name;
};

// selection optionally restricts volumes, derived representations and segments;
// nullptr attaches everything. newProject bypasses the cached full-sample project
// and saves a fresh JSON before returning an ordinarily persistent loaded package.
[[nodiscard]] std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const vcr::catalog::OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut = nullptr,
    const OpenDataSampleProgressCallback& progressCallback = {},
    const OpenDataResourceSelection* selection = nullptr,
    const OpenDataNewProject* newProject = nullptr);

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const vcr::catalog::OpenDataSample& sample,
    const OpenDataResourceSelection* selection = nullptr);

// Record the sample's segments (only `segmentIds` when given) as surfaces in
// its surface store.
void importOpenDataSampleSurfaces(
    const vcr::catalog::OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult& result,
    const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);

} // namespace vc3d::opendata
