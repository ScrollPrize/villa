#pragma once

#include "OpenDataManifest.hpp"
#include "OpenDataSegmentCache.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class VolumePkg;

namespace vc3d::opendata {

struct OpenDataSampleProjectResult {
    int supportedVolumes = 0;
    int attachedVolumeEntries = 0;
    int skippedVolumes = 0;
    int failedVolumes = 0;
    int attachedLasagnaDatasets = 0;
    int attachedNormalGrids = 0;
    int supportedTifxyzSegments = 0;
    int cachedTifxyzSegments = 0;
    int attachedSegmentEntries = 0;
    int skippedTifxyzSegments = 0;
    int failedTifxyzSegments = 0;
    int transformedTifxyzSegments = 0;
    int failedTransformedTifxyzSegments = 0;
    std::string preferredVolumeId;
    std::vector<std::string> messages;
};

// selection optionally restricts attached volumes and derived
// representations; nullptr attaches everything.
[[nodiscard]] std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut = nullptr,
    const OpenDataSampleProgressCallback& progressCallback = {},
    const OpenDataResourceSelection* selection = nullptr);

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const OpenDataResourceSelection* selection = nullptr);

void attachOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult& result,
    const OpenDataSampleProgressCallback& progressCallback = {});

} // namespace vc3d::opendata
