#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>

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

// A project of a sample at a chosen path (instead of the sample's own).
struct OpenDataNewProject {
    std::filesystem::path path;
    std::string name;
};

// The sample's own project: <projects>/open_data/<sample>.vcproj.
[[nodiscard]] std::filesystem::path sampleProjectPath(const std::filesystem::path& projectsDirectory,
                                                      const vcr::catalog::OpenDataSample& sample);
// Create a project file and separate store for the sample.
void createSampleProjectFiles(const std::filesystem::path& path, const vcr::catalog::OpenDataSample& sample,
                              const std::string& name, const vc::objects::OpenOptions& options);
// The Open Data sample recorded in a project file.
[[nodiscard]] std::optional<std::string> openDataProjectSample(const VolumePkg& project);

// Populate the project file and the attached sample store with catalog
// surfaces, volumes, normal grids and Lasagna datasets,
// as `selection` restricts them (nullptr: everything); resources already there
// stay as they are. Returns the opened, resolved project.
[[nodiscard]] std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const std::shared_ptr<vc::objects::ObjectStoreSession>& session,
    const std::filesystem::path& projectPath,
    const vc::objects::AttachmentIdentity& project,
    const vcr::catalog::OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut = nullptr,
    const OpenDataSampleProgressCallback& progressCallback = {},
    const OpenDataResourceSelection* selection = nullptr);

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const vcr::catalog::OpenDataSample& sample,
    const OpenDataResourceSelection* selection = nullptr,
    const std::filesystem::path& remoteCacheRoot = {});

} // namespace vc3d::opendata
