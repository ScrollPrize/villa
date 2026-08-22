#pragma once

#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace utils { class ZarrArray; }

namespace vc::lasagna {

inline constexpr const char* kLasagnaRemoteMarker = "lasagna-remote.json";

[[nodiscard]] bool isRemoteLasagnaLocation(std::string_view location);

struct LasagnaDatasetOpenOptions {
    double workingToBaseScale = 1.0;
    std::filesystem::path remoteCacheRoot;
    vc::HttpAuth remoteAuth;
    vc::core::util::RemoteFileCachePolicy cachePolicy = vc::core::util::RemoteFileCachePolicy::CacheFirst;
    vc::core::util::RemoteFileFetcher remoteFileFetcher;
    // False forces anonymous S3 access for both the manifest and its groups.
    bool discoverAwsCredentials = true;
};

struct MaterializedLasagnaManifest {
    std::filesystem::path path;
    std::string normalizedLocation;
    bool cacheHit = false;
};

[[nodiscard]] MaterializedLasagnaManifest materializeLasagnaManifest(const std::string& manifestLocation, const LasagnaDatasetOpenOptions& options);
[[nodiscard]] std::string lasagnaGroupSourceLocation(
    const LasagnaChannelGroup& group);

class LasagnaDataset {
public:
    explicit LasagnaDataset(LasagnaDatasetManifest manifest);

    static LasagnaDataset open(
        const std::filesystem::path& manifestPath,
        LasagnaDatasetOpenOptions options = {});
    static LasagnaDataset openLocation(
        const std::string& manifestLocation,
        LasagnaDatasetOpenOptions options = {});

    [[nodiscard]] const LasagnaDatasetManifest& manifest() const noexcept;
    [[nodiscard]] bool hasNormalSource() const noexcept;
    [[nodiscard]] const std::filesystem::path& normalSourcePath() const;

private:
    LasagnaDatasetManifest manifest_;
};

// Open a channel group's Zarr through the local filesystem or, for a
// manifest-backed catalog cache, through its persistent read-through store.
[[nodiscard]] utils::ZarrArray openLasagnaChannelArray(
    const LasagnaDatasetManifest& manifest,
    const LasagnaChannelGroup& group,
    std::size_t dtypeSize = 1);

} // namespace vc::lasagna
