#pragma once

#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace vcr { class MemoryCache; }
namespace vcr::zarr { struct OpenedArray; }

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

// A channel group's Zarr array opened through vc_reader, with `cache` as its
// memory cache: a local array, the pooled view of a local prediction store
// (.native_pool.json; see openPooledPredictionArray), or a remote array cached
// under the group's remote cache root.
[[nodiscard]] vcr::zarr::OpenedArray openLasagnaGroupArray(
    const LasagnaChannelGroup& group, std::shared_ptr<vcr::MemoryCache> cache);

} // namespace vc::lasagna
