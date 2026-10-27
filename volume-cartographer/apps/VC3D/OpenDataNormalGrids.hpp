#pragma once
#include <vc_reader/catalog.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class VolumePkg;
namespace vc::project { struct Entry; }

namespace vc3d::opendata {

struct OpenDataResourceSelection;

inline constexpr int kNormalGridsDownloadWorkers = 16;

// Pairing of an open-data volume with its normal-grids artifact. `url` is the
// resolved https prefix of the artifact, or empty when the volume has none.
struct OpenDataNormalGridsInfo {
    std::string sampleId;
    std::string volumeId;
    std::string url;
    int sourceCoordinateLevel = 0;
    std::size_t artifactIndex = 0;  // in the volume's artifacts
};

[[nodiscard]] std::vector<OpenDataNormalGridsInfo> normalGridsArtifacts(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume);

// The catalog normal grids a normal-grid entry caches (the URL of its remote
// marker); nullopt for other entries.
[[nodiscard]] std::optional<vcr::catalog::OpenDataVolumeView> normalGridsCatalogView(
    const VolumePkg& pkg, const vc::project::Entry& entry);

// <remoteCacheRoot>/normal_grids/<sampleId>/<volumeId>/L<level>-<url hash>
[[nodiscard]] std::filesystem::path normalGridsCacheDir(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId,
    const std::string& volumeId,
    int sourceCoordinateLevel,
    std::string_view artifactUrl);

// True when the directory looks like a complete normal-grid store
// (metadata.json plus xy/, xz/, yz/).
[[nodiscard]] bool isCachedNormalGridsDir(const std::filesystem::path& dir);

// Set up streaming normal grids for every volume of this sample that has a
// normal-grids artifact. Returns the number of entries attached. selection
// optionally restricts volumes and representations; nullptr attaches all.
int attachOpenDataNormalGrids(VolumePkg& pkg,
                              const vcr::catalog::OpenDataSample& sample,
                              const std::filesystem::path& remoteCacheRoot,
                              const OpenDataResourceSelection* selection = nullptr);

// On-disk cache state of one volume's normal grids.
struct NormalGridsCacheState {
    bool hasArtifact = false;
    bool complete = false;        // full download finished (completion marker)
    std::uint64_t cachedBytes = 0;
    std::uint64_t totalBytes = 0; // known only after a full download
    int cachedFiles = 0;
};

[[nodiscard]] NormalGridsCacheState normalGridsCacheState(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume);

struct NormalGridsDownloadProgress {
    int totalFiles = 0;
    int completedFiles = 0;
    int failedFiles = 0;
    std::uint64_t totalBytes = 0;
    std::uint64_t completedBytes = 0;
    std::string fileName;
    std::string status;  // "listing" | "downloading"
};
using NormalGridsProgressCallback =
    std::function<void(const NormalGridsDownloadProgress&)>;

// Enumerate the artifact's S3 prefix (anonymous ListObjectsV2) and download all
// objects with `workers` threads into normalGridsCacheDir(...). The download
// lands in a temp dir and is published atomically. Returns the final cache dir,
// or an empty path on failure/cancellation (errorOut explains why).
[[nodiscard]] std::filesystem::path downloadOpenDataNormalGrids(
    const OpenDataNormalGridsInfo& info,
    const std::filesystem::path& remoteCacheRoot,
    int workers = kNormalGridsDownloadWorkers,
    const NormalGridsProgressCallback& progress = {},
    const std::atomic<bool>* cancelRequested = nullptr,
    std::string* errorOut = nullptr);

} // namespace vc3d::opendata
