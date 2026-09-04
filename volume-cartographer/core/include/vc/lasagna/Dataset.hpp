#pragma once

#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <cstdint>
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

// Process-wide counters for the remote read-through store (all
// PersistentHttpStore instances). Logical objects, not HTTP attempts: one
// "owned" per fetch that went to the origin, one "joined" per caller that
// waited on another caller's in-flight fetch, one "fromDisk" per disk-cache
// hit. Monotonic; consumers take before/after deltas. Overlapping work
// (metrics, other panes) is included in a delta, so treat it as a process
// total, not an exact per-operation count.
struct RemoteStoreStats {
    std::uint64_t objectsOwned = 0;
    std::uint64_t objectsJoined = 0;
    std::uint64_t objectsFromDisk = 0;
    std::uint64_t bytesOwned = 0;
    std::uint64_t ownerNanoseconds = 0;
    std::uint64_t failures = 0;
};
[[nodiscard]] RemoteStoreStats remoteStoreStats() noexcept;

// Open a channel group's Zarr through the local filesystem or, for a
// manifest-backed catalog cache, through its persistent read-through store.
[[nodiscard]] utils::ZarrArray openLasagnaChannelArray(
    const LasagnaDatasetManifest& manifest,
    const LasagnaChannelGroup& group,
    std::size_t dtypeSize = 1);

} // namespace vc::lasagna
