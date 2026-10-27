#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class VolumePkg;

namespace vc3d::opendata {

// The stage of opening a sample: "resolving-volumes", then "project-ready".
struct OpenDataSampleDownloadProgress {
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
    std::string preferredVolumeId;
    std::vector<std::string> messages;
};

struct OpenDataNewProject {
    std::filesystem::path path;
    std::string name;
};

// selection optionally restricts volumes and derived representations;
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

} // namespace vc3d::opendata
