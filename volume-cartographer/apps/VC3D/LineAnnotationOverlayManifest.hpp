#pragma once

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Manifest.hpp"

namespace vc3d::line_annotation
{

// The GUI needs only the channel names and base frame. It must never use
// LasagnaDataset::openLocation here: even CacheFirst can download or repair a
// cached manifest. The existing background prefetch validates remote files;
// a pending read leaves the overlay empty until that worker completes.
inline std::optional<vc::lasagna::LasagnaDatasetManifest> readOverlayManifest(
    const std::string& location,
    const std::filesystem::path& projectDirectory,
    const std::filesystem::path& remoteCacheRoot,
    bool remoteFetchFinished)
{
    const bool remote = vc::project::isLocationRemote(location);
    if (remote && !remoteFetchFinished) {
        return std::nullopt;
    }
    const auto path = remote
        ? remoteCacheRoot / vc::core::util::remoteFileCachePath(location)
        : vc::project::resolveLocalPath(location, projectDirectory);
    return vc::lasagna::LasagnaDatasetManifest::parseFile(path);
}

}  // namespace vc3d::line_annotation
