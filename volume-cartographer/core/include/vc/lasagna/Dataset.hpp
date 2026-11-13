#pragma once

#include <vc_reader/native/reader_types.hpp>
#include "vc/lasagna/Manifest.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace vcr { class MemoryCache; }
namespace vcr::zarr { struct OpenedArray; }

namespace vc::lasagna {

inline constexpr const char* kLasagnaRemoteMarker = "lasagna-remote.json";


struct LasagnaDatasetOpenOptions {
    double workingToBaseScale = 1.0;
    std::filesystem::path remoteCacheRoot;
    bool refreshManifest = false;  // read a remote manifest again even when it is cached
    // False forces anonymous S3 access for both the manifest and its groups.
    bool discoverAwsCredentials = true;
};

struct MaterializedLasagnaManifest {
    std::filesystem::path path;
    std::string location;  // where requests went: the location as given, query kept, no slash ending its path
    bool cacheHit = false;
};

[[nodiscard]] MaterializedLasagnaManifest materializeLasagnaManifest(const std::string& manifestLocation, const LasagnaDatasetOpenOptions& options);
[[nodiscard]] std::string lasagnaGroupSourceLocation(
    const LasagnaChannelGroup& group);

// For a local copy of a remote dataset's manifest (opened beside its
// lasagna-remote.json marker), fetch the provenance record the manifest names
// (its "provenance" file, e.g. inference.json) beside it unless already
// there, as the dataset's identity reads it; nothing for any other manifest.
// The fetcher, empty by default, replaces the download in tests: it writes
// the object at `location` to `temporary`.
using LasagnaFileFetcher =
    std::function<void(const std::string& location, const std::filesystem::path& temporary)>;
void cacheLasagnaProvenance(const LasagnaDatasetManifest& manifest,
                            const LasagnaFileFetcher& fetcher = {});

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

private:
    LasagnaDatasetManifest manifest_;
};

// A channel group's Zarr array opened through vc_reader, with `cache` as its
// memory cache: a local array, or a remote array cached under the group's remote cache root.
[[nodiscard]] vcr::zarr::OpenedArray openLasagnaGroupArray(
    const LasagnaChannelGroup& group, std::shared_ptr<vcr::MemoryCache> cache,
    std::size_t readerThreads=0);

// Select predicted sheet normals while preserving the compound channel layout.
[[nodiscard]] LasagnaDataset fiberSheetNormalDataset(const LasagnaDataset& prediction);

// The directory cacheLasagnaChannelArray keeps a remote group's Zarr array
// in, byte for byte; nullopt for a local group. It sits under the group's
// remote cache root, apart from the vc_reader cache of the same array (whose
// chunks may be stored decoded).
[[nodiscard]] std::optional<std::filesystem::path> lasagnaChannelArrayCacheDirectory(
    const LasagnaChannelGroup& group);

struct LasagnaArrayCacheProgress {
    std::size_t done = 0;
    std::size_t total = 0;
};

// Fetch every object of a remote group's Zarr array (an S3 listing of it)
// byte for byte, so lasagnaChannelArrayCacheDirectory reads as a complete
// local directory Zarr. The objects are written outside the cache budget's
// accounting, so its eviction cannot take them while this process runs. A
// marker there records the objects; a later call that finds them all still
// cached fetches nothing, and one that finds some evicted fetches those again.
// Returns false when `cancelled` stops it; throws when an object cannot be
// listed, fetched or kept. `progress` is called from worker threads.
bool cacheLasagnaChannelArray(
    const LasagnaChannelGroup& group,
    const std::function<void(const LasagnaArrayCacheProgress&)>& progress = {},
    const std::function<bool()>& cancelled = {});

} // namespace vc::lasagna
