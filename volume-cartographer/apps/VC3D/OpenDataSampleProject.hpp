#pragma once

#include "OpenDataManifest.hpp"

#include <filesystem>
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
    std::string preferredVolumeId;
    std::vector<std::string> messages;
};

[[nodiscard]] std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut = nullptr);

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const OpenDataSample& sample);

} // namespace vc3d::opendata
