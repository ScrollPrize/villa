#pragma once

#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
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

// The directory a remote group's read-through store keeps its Zarr array's
// chunks in, byte for byte; nullopt for a local group.
[[nodiscard]] std::optional<std::filesystem::path> lasagnaChannelArrayCacheDirectory(
    const LasagnaChannelGroup& group);

struct LasagnaArrayCacheProgress {
    std::size_t done = 0;
    std::size_t total = 0;
};

// Fetch every object of a remote group's Zarr array (an S3 listing of it)
// through its read-through store, and copy its metadata beside the chunks, so
// lasagnaChannelArrayCacheDirectory reads as a complete local directory Zarr
// for readers that know nothing of the store. A marker there records the
// objects; a later call that finds them all still cached fetches nothing, and
// one that finds some evicted fetches those again. Returns false when
// `cancelled` stops it; throws when an object cannot be listed, fetched or
// kept (the cache's size limit can refuse chunks). `progress` is called from
// worker threads.
bool cacheLasagnaChannelArray(
    const LasagnaChannelGroup& group,
    const std::function<void(const LasagnaArrayCacheProgress&)>& progress = {},
    const std::function<bool()>& cancelled = {});

} // namespace vc::lasagna
